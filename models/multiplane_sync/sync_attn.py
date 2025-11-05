from typing import Union, Optional
from einops import rearrange
import torch


def cube_sync_self_attn_processor(self):
    def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        
        # hidden_states.shape: [B*M, HW, C]
        m = 6
        hidden_states = rearrange(hidden_states, '(b m) hw c -> b (m hw) c', m=m)
        hidden_states = self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = rearrange(hidden_states, 'b (m hw) c -> (b m) hw c', m=m)
        return hidden_states
    
    return forward


def cube_sync_attn_processor(self):
    import inspect
    from diffusers.models.attention_processor import logger
    
    def forward(
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        m = 6
        orig_shape = hidden_states.shape
        if hidden_states.ndim == 3:
            hidden_states = rearrange(hidden_states, '(b m) hw c -> b (m hw) c', m=m)
        else:
            assert hidden_states.ndim == 4, f'Expected 3D or 4D input, but got shape {hidden_states.shape}!'
            hidden_states = rearrange(hidden_states, '(b m) c h w -> b c (m h) w', m=m)
        
        if encoder_hidden_states is not None:
            encoder_hidden_states = rearrange(encoder_hidden_states, '(b m) hw c -> b (m hw) c', m=m)
        
        hidden_states = self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        
        if len(orig_shape) == 3:
            hidden_states = rearrange(hidden_states, 'b (m hw) c -> (b m) hw c', m=m)
        else:
            h, w = orig_shape[-2:]
            
            hidden_states = rearrange(hidden_states, 'b c (m h) w -> (b m) c h w', m=m, h=h, w=w)

        if encoder_hidden_states is not None:
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b (m hw) c -> (b m) hw c', m=m)
        
        return hidden_states
    
    return forward
