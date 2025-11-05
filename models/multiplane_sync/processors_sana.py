import torch.nn as nn
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import get_2d_sincos_pos_embed

from .sync_attn import cube_sync_attn_processor
from .sync_norm import cube_sync_gn_processor
from .sync_conv2d import cube_sync_conv2d_processor
from .utils import safe_setattr


def get_patch_embed_forward(self):
    def forward(latent):
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)
        if self.pos_embed is None:
            return latent.to(latent.dtype)
        # Interpolate or crop positional embeddings as needed
        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)
        else:
            if self.height != height or self.width != width:
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                    device=latent.device,
                    output_type="pt",
                )
                pos_embed = pos_embed.float().unsqueeze(0)
            else:
                pos_embed = self.pos_embed
        return (latent + pos_embed).to(latent.dtype)
    return forward


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


def apply_custom_processors_for_transformer(
    model,
    enable_sync_self_attn: bool = True,
    enable_sync_cross_attn: bool = False,
    enable_sync_conv2d: bool = False,
    enable_sync_gn: bool = False,
):
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            if enable_sync_self_attn and not module.is_cross_attention:
                setattr(module, 'forward_wo_sync_self_attn', module.forward)
                module.forward = cube_sync_attn_processor(module)
                setattr(module, 'forward_w_sync_self_attn', module.forward)
            elif enable_sync_cross_attn and module.is_cross_attention:
                print(name, type(module))

        elif isinstance(module, nn.Conv2d) and enable_sync_conv2d:
            if module.kernel_size == (1, 1):
                continue
            safe_setattr(module, 'forward_wo_sync_conv2d', module.forward)
            module.forward = cube_sync_conv2d_processor(module, impl='ref')
            safe_setattr(module, 'forward_w_sync_conv2d', module.forward)

        elif isinstance(module, nn.GroupNorm) and enable_sync_gn:
            print(name, type(module))
