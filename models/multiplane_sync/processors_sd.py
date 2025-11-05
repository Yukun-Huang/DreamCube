from diffusers.utils import check_min_version
import torch.nn as nn
from diffusers.models.attention_processor import Attention

from .sync_attn import cube_sync_attn_processor
from .sync_conv2d import cube_sync_conv2d_processor
from .sync_norm import cube_sync_gn_processor
from .utils import safe_setattr

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.2")


####################################################################################################################################
def switch_custom_processors_for_vae(model, enable_sync_gn: bool, enable_sync_conv2d: bool, enable_sync_attn: bool):
    for name, module in model.named_modules():
        if isinstance(module, nn.GroupNorm):
            if hasattr(module, 'forward_w_sync_gn'):
                if enable_sync_gn:
                    module.forward = module.forward_w_sync_gn
                else:
                    module.forward = module.forward_wo_sync_gn
        
        elif isinstance(module, nn.Conv2d):
            if hasattr(module, 'forward_w_sync_conv2d'):
                if enable_sync_conv2d:
                    module.forward = module.forward_w_sync_conv2d
                else:
                    module.forward = module.forward_wo_sync_conv2d

        elif isinstance(module, Attention):
            if hasattr(module, 'forward_w_sync_attn'):
                if enable_sync_attn:
                    module.forward = module.forward_w_sync_attn
                else:
                    module.forward = module.forward_wo_sync_attn


####################################################################################################################################
def apply_custom_processors_for_unet(
    model,
    enable_sync_self_attn: bool = True,
    enable_sync_cross_attn: bool = False,
    enable_sync_conv2d: bool = False,
    enable_sync_gn: bool = False,
    rot_inv_conv2d_mode: str = 'none',
):
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            if enable_sync_self_attn and not module.is_cross_attention:
                safe_setattr(module, 'forward_wo_sync_self_attn', module.forward)
                module.forward = cube_sync_attn_processor(module)
                safe_setattr(module, 'forward_w_sync_self_attn', module.forward)

            if enable_sync_cross_attn and module.is_cross_attention:
                safe_setattr(module, 'forward_wo_sync_cross_attn', module.forward)
                module.forward = cube_sync_attn_processor(module)
                safe_setattr(module, 'forward_w_sync_cross_attn', module.forward)

        elif isinstance(module, nn.Conv2d) and enable_sync_conv2d:
            if module.kernel_size == (1, 1):
                continue
            safe_setattr(module, 'forward_wo_sync_conv2d', module.forward)
            module.forward = cube_sync_conv2d_processor(module, rot_inv_mode=rot_inv_conv2d_mode)
            safe_setattr(module, 'forward_w_sync_conv2d', module.forward)

        elif isinstance(module, nn.GroupNorm) and enable_sync_gn:
            safe_setattr(module, 'forward_wo_sync_gn', module.forward)
            module.forward = cube_sync_gn_processor(module)
            safe_setattr(module, 'forward_w_sync_gn', module.forward)


def apply_custom_processors_for_vae(
    model,
    mode: str = 'all',
    enable_sync_gn: bool = True,
    enable_sync_conv2d: bool = False,
    enable_sync_attn: bool = False,
    rot_inv_conv2d_mode: str = 'none',
):
    assert mode in ('all', 'encoder_only', 'decoder_only', 'none')
    if mode == 'none':
        print('Disable sync processors for VAE!')
        return
    
    for name, module in model.named_modules():
        if mode == 'encoder_only' and not name.startswith('encoder'):
            continue
        if mode == 'decoder_only' and not name.startswith('decoder'):
            continue

        if isinstance(module, nn.GroupNorm) and enable_sync_gn:
            if 'attentions' in name and enable_sync_attn:
                continue  # otherwise will cause error of unmatched shapes
            safe_setattr(module, 'forward_wo_sync_gn', module.forward)
            module.forward = cube_sync_gn_processor(module)
            safe_setattr(module, 'forward_w_sync_gn', module.forward)

        elif isinstance(module, Attention) and enable_sync_attn:
            assert not module.is_cross_attention, 'Cross attention should not occur in VAE!'
            safe_setattr(module, 'forward_wo_sync_attn', module.forward)
            module.forward = cube_sync_attn_processor(module)
            safe_setattr(module, 'forward_w_sync_attn', module.forward)

        elif isinstance(module, nn.Conv2d) and enable_sync_conv2d:
            if module.kernel_size == (1, 1):
                continue
            safe_setattr(module, 'forward_wo_sync_conv2d', module.forward)
            module.forward = cube_sync_conv2d_processor(module, rot_inv_mode=rot_inv_conv2d_mode)
            safe_setattr(module, 'forward_w_sync_conv2d', module.forward)
