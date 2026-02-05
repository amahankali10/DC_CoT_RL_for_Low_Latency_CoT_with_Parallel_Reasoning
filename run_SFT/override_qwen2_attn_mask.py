from typing import Optional, Union

import torch
from torch import nn
from transformers.utils import logging
import transformers.integrations.flex_attention

from transformers.models.qwen2 import (
    Qwen2Model,
    Qwen2PreTrainedModel,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast
)
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack

from parse_claude_completions_for_new_parallel_generation import create_attention_mask_from_segment_lengths

logger = logging.get_logger(__name__)

class Qwen2ModelOverride(Qwen2Model):
    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flex_attention":
            assert len(attention_mask.shape) == 4, "attention_mask should be a 4D tensor"
            return attention_mask
        
        if self.config._attn_implementation == "sdpa":
            assert len(attention_mask.shape) == 4, "attention_mask should be a 4D tensor"
            return attention_mask

        return super()._update_causal_mask(
            attention_mask=attention_mask,
            input_tensor=input_tensor,
            cache_position=cache_position,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
        )


def segment_length_attn_mask_wrapper(segment_lengths_tensor):
    batch_size, _  = segment_lengths_tensor.shape
    list_2d = segment_lengths_tensor.cpu().tolist()
    attn_mask_list = []
    max_len = -1
    for idx in range(batch_size):
        current_segment_lengths = list_2d[idx]
        true_num_segments = 0
        for segment_length in current_segment_lengths:
            if segment_length == -1:
                break
            true_num_segments += 1
        current_segment_lengths = current_segment_lengths[:true_num_segments]
        attn_mask = create_attention_mask_from_segment_lengths(current_segment_lengths)
        attn_mask_list.append(attn_mask)
        total_len = sum(current_segment_lengths)    
        max_len = max(max_len, total_len)
    
    # Pad all attention masks to (max_len, max_len)
    padded_attn_mask_list = []
    for attn_mask in attn_mask_list:
        current_len = attn_mask.size(0)
        pad_size = max_len - current_len
        if pad_size > 0:
            right_pad = torch.zeros((current_len, pad_size), dtype=torch.bool)
            attn_mask_padded_right = torch.cat([attn_mask, right_pad], dim=1)
            bottom_pad = torch.zeros((pad_size, max_len), dtype=torch.bool)
            padded_attn_mask = torch.cat([attn_mask_padded_right, bottom_pad], dim=0)
        else:
            padded_attn_mask = attn_mask
        padded_attn_mask_list.append(padded_attn_mask)
    
    full_attn_mask = torch.stack(padded_attn_mask_list, dim=0) # (B, L, L)
    full_attn_mask = full_attn_mask.unsqueeze(1) # (B, 1, L, L)
    return full_attn_mask


class Qwen2ForCausalLMOverride(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs
    ) -> CausalLMOutputWithPast:
        logger.warning_once("CUSTOM FORWARD CALLED")
        segment_lengths_tensor = kwargs.get("segment_lengths")
        assert segment_lengths_tensor is not None, "segment_lengths_tensor is required"
        attention_mask = segment_length_attn_mask_wrapper(segment_lengths_tensor).to(input_ids.device)
        attention_mask = torch.where(
            attention_mask, 0, torch.finfo(torch.bfloat16).min
        )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )