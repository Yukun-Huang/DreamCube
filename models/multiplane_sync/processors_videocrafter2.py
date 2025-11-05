import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.cube import pad_cube
from torch import Tensor
from einops import rearrange, repeat, einsum
from typing import Optional, List, Tuple
from diffusers.utils import check_min_version
from diffusers.models.attention_processor import Attention

from .sync_conv2d import adaptive_convolution, rotation_invariant_kernels, calculate_edge_weights
from .utils import safe_setattr

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.2")


####################################################################################################################################
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
        f = 16
        orig_shape = hidden_states.shape
        if hidden_states.ndim == 3:
            hidden_states = rearrange(hidden_states, '(b m f) hw c -> (b f) (m hw) c', m=m, f=f)
        else:
            assert 0, f'Expected 3D input, but got shape {hidden_states.shape}!'
            # assert hidden_states.ndim == 4, f'Expected 3D or 4D input, but got shape {hidden_states.shape}!'
            # hidden_states = rearrange(hidden_states, '(b m) c h w -> b c (m h) w', m=m)
        
        if encoder_hidden_states is not None:
            try:
                encoder_hidden_states = rearrange(encoder_hidden_states, '(b m) n c -> b (m n) c', m=m)
            except:
                print(encoder_hidden_states.shape)
            # assert 0, f'encoder_hidden_states.shape = {encoder_hidden_states.shape}. Not implemented yet!'
        
        hidden_states = self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        
        if len(orig_shape) == 3:
            hidden_states = rearrange(hidden_states, '(b f) (m hw) c -> (b m f) hw c', m=m, f=f)
        else:
            assert 0, f'Expected 3D input, but got shape {hidden_states.shape}!'
            # h, w = orig_shape[-2:]
            # hidden_states = rearrange(hidden_states, 'b c (m h) w -> (b m) c h w', m=m, h=h, w=w)

        return hidden_states
    
    return forward


def cube_sync_conv2d_processor(self, rot_inv_mode='none', enable_cube_padding=True):

    def forward(input: torch.Tensor) -> torch.Tensor:
        # padding
        padding = self.padding
        assert padding[0] == padding[1], 'Only support square padding!'

        m = 6
        f = 16

        if padding[0] > 0:
            if enable_cube_padding:
                input = rearrange(input, '(b m f) c h w -> (b f) m c h w', m=m, f=f)
                if input.nelement() > 8000000:
                    input = pad_cube(input, padding[0], impl='ref')
                else:
                    input = pad_cube(input, padding[0], impl='cuda')
                input = rearrange(input, '(b f) m c h w -> (b m f) c h w', m=m, f=f)
            else:
                if self.padding_mode != 'zeros':
                    input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
                else:
                    input = F.pad(input, self._reversed_padding_repeated_twice, mode='constant', value=0)

        # rot-inv conv2d
        if rot_inv_mode != 'none':
            assert rot_inv_mode in ('input', 'kernel', 'shift', 'shift_max') \
              or rot_inv_mode.startswith('sum') or rot_inv_mode.startswith('rot'), \
                f'Invalid rot_inv_mode: {rot_inv_mode}!'

            if rot_inv_mode == 'input':

                assert input.ndim == 4 and input.shape[0] % 6 == 0, f'Invalid input shape: {input.shape}!'
                input = rearrange(input, '(b m) c h w -> b m c h w', m=6)
                
                input_1 = rearrange(input[:, :4], 'b m c h w -> (b m) c h w')
                input_2 = rearrange(input[:, 4:], 'b m c h w -> (b m) c h w')

                output_1 = F.conv2d(input_1, self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_1 = rearrange(output_1, '(b m) c h w -> b m c h w', m=4)

                # output_2a = F.conv2d(torch.rot90(input_2, dims=(2, 3), k=0), self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                # output_2b = F.conv2d(torch.rot90(input_2, dims=(2, 3), k=1), self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                # output_2c = F.conv2d(torch.rot90(input_2, dims=(2, 3), k=2), self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                # output_2d = F.conv2d(torch.rot90(input_2, dims=(2, 3), k=3), self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                
                input_2 = torch.rot90(input_2, dims=(2, 3), k=2)
                output_2a = F.conv2d(input_2, self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_2a = torch.rot90(output_2a, dims=(2, 3), k=-2)
                output_2b = output_2a
                output_2c = output_2a
                output_2d = output_2a

                outputs_2 = torch.stack([output_2a, output_2b, output_2c, output_2d], dim=1)  # [BM, 4, C, H, W]
                outputs_2 = rearrange(outputs_2, '(b m) e c h w -> b m e c h w', m=2)         # [B, M, 4, C, H, W]

                outputs_2 = torch.mean(outputs_2, dim=2, keepdim=False)  # [B, M, C, H, W]

                # weights = calculate_edge_weights(outputs_2, one_hot=True)  # [B, 4, H, W]
                # outputs_2_up = einsum(outputs_2[:, 0], weights[:, [2, 1, 0, 3]], 'b e c h w, b e h w -> b c h w').unsqueeze(1)
                # outputs_2_down = einsum(outputs_2[:, 1], weights[:, [0, 3, 2, 1]], 'b e c h w, b e h w -> b c h w').unsqueeze(1)

                output = torch.cat([output_1, outputs_2], dim=1)
                # output = torch.cat([output_1, outputs_2_up, outputs_2_down], dim=1)
                output = rearrange(output, 'b m c h w -> (b m) c h w')

            elif rot_inv_mode == 'kernel':
                assert input.ndim == 4 and input.shape[0] % 6 == 0, f'Invalid input shape: {input.shape}!'
                input = rearrange(input, '(b m) c h w -> b m c h w', m=6)
                
                input_1 = rearrange(input[:, :4], 'b m c h w -> (b m) c h w')
                input_2 = rearrange(input[:, 4:6], 'b m c h w -> (b m) c h w')

                output_1 = F.conv2d(input_1, self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_1 = rearrange(output_1, '(b m) c h w -> b m c h w', m=4)

                weight_rot_inv = rotation_invariant_kernels(self.weight, mode='90')
                output_2 = F.conv2d(input_2, weight_rot_inv, self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_2 = rearrange(output_2, '(b m) c h w -> b m c h w', m=2)

                output = torch.cat([output_1, output_2], dim=1)
                output = rearrange(output, 'b m c h w -> (b m) c h w')

            elif rot_inv_mode.startswith('sum'):
                # sum_max: one_hot
                # sum_avg: 1/4 for each
                # sum_smooth: ...
                one_hot = (rot_inv_mode == 'sum_max')
                two_hot = (rot_inv_mode == 'sum_max2')

                assert input.ndim == 4 and input.shape[0] % 6 == 0, f'Invalid input shape: {input.shape}!'
                input = rearrange(input, '(b m) c h w -> b m c h w', m=6)
                
                input_1 = rearrange(input[:, :4], 'b m c h w -> (b m) c h w')
                input_2 = rearrange(input[:, 4:], 'b m c h w -> (b m) c h w')

                output_1 = F.conv2d(input_1, self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_1 = rearrange(output_1, '(b m) c h w -> b m c h w', m=4)

                output_2a = F.conv2d(input_2, torch.rot90(self.weight, dims=(2, 3), k=0), self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_2b = F.conv2d(input_2, torch.rot90(self.weight, dims=(2, 3), k=1), self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_2c = F.conv2d(input_2, torch.rot90(self.weight, dims=(2, 3), k=2), self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_2d = F.conv2d(input_2, torch.rot90(self.weight, dims=(2, 3), k=3), self.bias, self.stride, 'valid', self.dilation, self.groups)

                outputs_2 = torch.stack([output_2a, output_2b, output_2c, output_2d], dim=1)  # [BM, 4, C, H, W]
                outputs_2 = rearrange(outputs_2, '(b m) e c h w -> b m e c h w', m=2)         # [B, M, 4, C, H, W]

                weights = calculate_edge_weights(outputs_2, one_hot=one_hot, two_hot=two_hot)  # [B, 4, H, W]
                outputs_2_up = einsum(outputs_2[:, 0], weights[:, [2, 1, 0, 3]], 'b e c h w, b e h w -> b c h w').unsqueeze(1)
                outputs_2_down = einsum(outputs_2[:, 1], weights[:, [0, 3, 2, 1]], 'b e c h w, b e h w -> b c h w').unsqueeze(1)

                output = torch.cat([output_1, outputs_2_up, outputs_2_down], dim=1)
                output = rearrange(output, 'b m c h w -> (b m) c h w')
            
            elif rot_inv_mode.startswith('rot'):
                assert input.ndim == 4 and input.shape[0] % 6 == 0, f'Invalid input shape: {input.shape}!'
                input = rearrange(input, '(b m) c h w -> b m c h w', m=6)
                
                input_1 = rearrange(input[:, :4], 'b m c h w -> (b m) c h w')
                input_2 = rearrange(input[:, 4:5], 'b m c h w -> (b m) c h w')
                input_3 = rearrange(input[:, 5:6], 'b m c h w -> (b m) c h w')

                output_1 = F.conv2d(input_1, self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_1 = rearrange(output_1, '(b m) c h w -> b m c h w', m=4)

                discrete = (rot_inv_mode == 'rot_max')

                output_2 = adaptive_convolution(input_2, self.weight, self.bias, padding=padding, stride=self.stride,
                                                dilation=self.dilation, groups=self.groups, inverse=False, discrete=discrete)
                output_2 = rearrange(output_2, '(b m) c h w -> b m c h w', m=1)

                output_3 = adaptive_convolution(input_3, self.weight, self.bias, padding=padding, stride=self.stride,
                                                dilation=self.dilation, groups=self.groups, inverse=True, discrete=discrete)
                output_3 = rearrange(output_3, '(b m) c h w -> b m c h w', m=1)

                output = torch.cat([output_1, output_2, output_3], dim=1)
                output = rearrange(output, 'b m c h w -> (b m) c h w')
            
            elif rot_inv_mode.startswith('shift'):
                assert input.ndim == 4 and input.shape[0] % 6 == 0, f'Invalid input shape: {input.shape}!'
                input = rearrange(input, '(b m) c h w -> b m c h w', m=6)
                
                input_1 = rearrange(input[:, :4], 'b m c h w -> (b m) c h w')
                input_2 = rearrange(input[:, 4:5], 'b m c h w -> (b m) c h w')
                input_3 = rearrange(input[:, 5:6], 'b m c h w -> (b m) c h w')

                output_1 = F.conv2d(input_1, self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_1 = rearrange(output_1, '(b m) c h w -> b m c h w', m=4)

                discrete = (rot_inv_mode == 'shift_max')

                output_2 = adaptive_convolution(input_2, self.weight, self.bias, padding=padding, stride=self.stride,
                                                dilation=self.dilation, groups=self.groups, inverse=False, discrete=discrete, shift=True)
                output_2 = rearrange(output_2, '(b m) c h w -> b m c h w', m=1)

                output_3 = adaptive_convolution(input_3, self.weight, self.bias, padding=padding, stride=self.stride,
                                                dilation=self.dilation, groups=self.groups, inverse=True, discrete=discrete, shift=True)
                output_3 = rearrange(output_3, '(b m) c h w -> b m c h w', m=1)

                output = torch.cat([output_1, output_2, output_3], dim=1)
                output = rearrange(output, 'b m c h w -> (b m) c h w')
            
            else:
                raise NotImplementedError(f'Invalid rot_inv_mode: {rot_inv_mode}!')
        
        else:
            output = F.conv2d(input, self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)

        return output
    
    return forward


def cube_sync_gn_processor(self):
    def forward(input: Tensor) -> Tensor:
        """
        Shape:
        - Input: (B*M, C, F, H, W) or (B*M*F, C, H, W) or (B*M*F, C, H*W)
        - Output: (B*M, C, F, H, W) or (B*M*F, C, H, W) or (B*M*F, C, H*W)
        """
        m = 6
        f = 16

        if input.ndim == 5:
            input = rearrange(input, '(b m) c f h w -> b m c f h w', m=m, f=f)
        elif input.ndim == 4:
            input = rearrange(input, '(b m f) c h w -> (b f) m c h w', m=m, f=f)
        elif input.ndim == 3:
            input = rearrange(input, '(b m f) c hw -> (b f) m c hw', m=m, f=f)
        else:
            raise ValueError(f'Unsupported input shape: {input.shape}')

        input = [input[:, i] for i in range(m)]  # M * [B, C, F, H, W] or [BF, C, H, W]

        used_dtype = torch.float32
        b, dtype, device = input[0].shape[0], input[0].dtype, input[0].device
        input = [tile.to(used_dtype) for tile in input]
        shapes, tmp_tiles, num_elements = list(), list(), 0
        for tile in input:
            hw = tile.shape[2:]
            shapes.append(hw)
            tmp_tile = rearrange(tile, 'b (g c) ... -> b g (c ...)', g=self.num_groups)
            tmp_tiles.append(tmp_tile)
            num_elements = num_elements + tmp_tile.shape[-1]
        mean, var = (
            torch.zeros((b, self.num_groups, 1), dtype=used_dtype, device=device),
            torch.zeros((b, self.num_groups, 1), dtype=used_dtype, device=device)
        )

        for tile in tmp_tiles:
            mean = mean + tile.mean(-1, keepdim=True) * float(tile.shape[-1] / num_elements)
            # Unbiased variance estimation
            var = var + (
                ((tile - mean) ** 2) * (tile.shape[-1] / (tile.shape[-1] - 1))
            ).mean(-1, keepdim=True) * float(tile.shape[-1] / num_elements)

        input = []
        for shape, tile in zip(shapes, tmp_tiles):
            if len(shape) == 3:
                f, h, w = shape
                tile = rearrange((tile - mean) / (var + self.eps).sqrt(), 'b g (c f h w) -> b (g c) f h w', f=f, h=h, w=w)
                input.append(tile * self.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + self.bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            elif len(shape) == 2:
                h, w = shape
                tile = rearrange((tile - mean) / (var + self.eps).sqrt(), 'bf g (c h w) -> bf (g c) h w', h=h, w=w)
                input.append(tile * self.weight.unsqueeze(-1).unsqueeze(-1) + self.bias.unsqueeze(-1).unsqueeze(-1))
            elif len(shape) == 1:
                hw = shape[0]
                tile = rearrange((tile - mean) / (var + self.eps).sqrt(), 'b g (c hw) -> b (g c) hw', hw=hw)
                input.append(tile * self.weight.unsqueeze(-1) + self.bias.unsqueeze(-1))
            else:
                raise NotImplementedError(f'Unsupported shape: {shape}')
        
        input = torch.stack([tile.to(dtype) for tile in input], dim=1)
        if input.ndim == 6:
            input = rearrange(input, 'b m c f h w -> (b m) c f h w', m=m, f=f)
        elif input.ndim == 5:
            input = rearrange(input, '(b f) m c h w -> (b m f) c h w', m=m, f=f)
        elif input.ndim == 4:
            input = rearrange(input, '(b f) m c hw -> (b m f) c hw', m=m, f=f)
        else:
            raise ValueError(f'Unsupported input shape: {input.shape}')

        return input

    return forward


def cube_sync_conv2d_processor_for_vae(self, rot_inv_mode='none', enable_cube_padding=True):

    def forward(input: torch.Tensor) -> torch.Tensor:
        # padding
        padding = self.padding
        assert padding[0] == padding[1], 'Only support square padding!'

        m = 6
        f = 1

        if padding[0] > 0:
            if enable_cube_padding:
                input = rearrange(input, '(b m f) c h w -> (b f) m c h w', m=m, f=f)
                if input.nelement() > 8000000:
                    input = pad_cube(input, padding[0], impl='ref')
                else:
                    input = pad_cube(input, padding[0], impl='cuda')
                input = rearrange(input, '(b f) m c h w -> (b m f) c h w', m=m, f=f)
            else:
                if self.padding_mode != 'zeros':
                    input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
                else:
                    input = F.pad(input, self._reversed_padding_repeated_twice, mode='constant', value=0)

        # rot-inv conv2d
        if rot_inv_mode != 'none':
            assert rot_inv_mode in ('input', 'kernel', 'shift', 'shift_max') \
              or rot_inv_mode.startswith('sum') or rot_inv_mode.startswith('rot'), \
                f'Invalid rot_inv_mode: {rot_inv_mode}!'

            if rot_inv_mode == 'input':

                assert input.ndim == 4 and input.shape[0] % 6 == 0, f'Invalid input shape: {input.shape}!'
                input = rearrange(input, '(b m) c h w -> b m c h w', m=6)
                
                input_1 = rearrange(input[:, :4], 'b m c h w -> (b m) c h w')
                input_2 = rearrange(input[:, 4:], 'b m c h w -> (b m) c h w')

                output_1 = F.conv2d(input_1, self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_1 = rearrange(output_1, '(b m) c h w -> b m c h w', m=4)

                # output_2a = F.conv2d(torch.rot90(input_2, dims=(2, 3), k=0), self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                # output_2b = F.conv2d(torch.rot90(input_2, dims=(2, 3), k=1), self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                # output_2c = F.conv2d(torch.rot90(input_2, dims=(2, 3), k=2), self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                # output_2d = F.conv2d(torch.rot90(input_2, dims=(2, 3), k=3), self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                
                input_2 = torch.rot90(input_2, dims=(2, 3), k=2)
                output_2a = F.conv2d(input_2, self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_2a = torch.rot90(output_2a, dims=(2, 3), k=-2)
                output_2b = output_2a
                output_2c = output_2a
                output_2d = output_2a

                outputs_2 = torch.stack([output_2a, output_2b, output_2c, output_2d], dim=1)  # [BM, 4, C, H, W]
                outputs_2 = rearrange(outputs_2, '(b m) e c h w -> b m e c h w', m=2)         # [B, M, 4, C, H, W]

                outputs_2 = torch.mean(outputs_2, dim=2, keepdim=False)  # [B, M, C, H, W]

                # weights = calculate_edge_weights(outputs_2, one_hot=True)  # [B, 4, H, W]
                # outputs_2_up = einsum(outputs_2[:, 0], weights[:, [2, 1, 0, 3]], 'b e c h w, b e h w -> b c h w').unsqueeze(1)
                # outputs_2_down = einsum(outputs_2[:, 1], weights[:, [0, 3, 2, 1]], 'b e c h w, b e h w -> b c h w').unsqueeze(1)

                output = torch.cat([output_1, outputs_2], dim=1)
                # output = torch.cat([output_1, outputs_2_up, outputs_2_down], dim=1)
                output = rearrange(output, 'b m c h w -> (b m) c h w')

            elif rot_inv_mode == 'kernel':
                assert input.ndim == 4 and input.shape[0] % 6 == 0, f'Invalid input shape: {input.shape}!'
                input = rearrange(input, '(b m) c h w -> b m c h w', m=6)
                
                input_1 = rearrange(input[:, :4], 'b m c h w -> (b m) c h w')
                input_2 = rearrange(input[:, 4:6], 'b m c h w -> (b m) c h w')

                output_1 = F.conv2d(input_1, self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_1 = rearrange(output_1, '(b m) c h w -> b m c h w', m=4)

                weight_rot_inv = rotation_invariant_kernels(self.weight, mode='90')
                output_2 = F.conv2d(input_2, weight_rot_inv, self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_2 = rearrange(output_2, '(b m) c h w -> b m c h w', m=2)

                output = torch.cat([output_1, output_2], dim=1)
                output = rearrange(output, 'b m c h w -> (b m) c h w')

            elif rot_inv_mode.startswith('sum'):
                # sum_max: one_hot
                # sum_avg: 1/4 for each
                # sum_smooth: ...
                one_hot = (rot_inv_mode == 'sum_max')
                two_hot = (rot_inv_mode == 'sum_max2')

                assert input.ndim == 4 and input.shape[0] % 6 == 0, f'Invalid input shape: {input.shape}!'
                input = rearrange(input, '(b m) c h w -> b m c h w', m=6)
                
                input_1 = rearrange(input[:, :4], 'b m c h w -> (b m) c h w')
                input_2 = rearrange(input[:, 4:], 'b m c h w -> (b m) c h w')

                output_1 = F.conv2d(input_1, self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_1 = rearrange(output_1, '(b m) c h w -> b m c h w', m=4)

                output_2a = F.conv2d(input_2, torch.rot90(self.weight, dims=(2, 3), k=0), self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_2b = F.conv2d(input_2, torch.rot90(self.weight, dims=(2, 3), k=1), self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_2c = F.conv2d(input_2, torch.rot90(self.weight, dims=(2, 3), k=2), self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_2d = F.conv2d(input_2, torch.rot90(self.weight, dims=(2, 3), k=3), self.bias, self.stride, 'valid', self.dilation, self.groups)

                outputs_2 = torch.stack([output_2a, output_2b, output_2c, output_2d], dim=1)  # [BM, 4, C, H, W]
                outputs_2 = rearrange(outputs_2, '(b m) e c h w -> b m e c h w', m=2)         # [B, M, 4, C, H, W]

                weights = calculate_edge_weights(outputs_2, one_hot=one_hot, two_hot=two_hot)  # [B, 4, H, W]
                outputs_2_up = einsum(outputs_2[:, 0], weights[:, [2, 1, 0, 3]], 'b e c h w, b e h w -> b c h w').unsqueeze(1)
                outputs_2_down = einsum(outputs_2[:, 1], weights[:, [0, 3, 2, 1]], 'b e c h w, b e h w -> b c h w').unsqueeze(1)

                output = torch.cat([output_1, outputs_2_up, outputs_2_down], dim=1)
                output = rearrange(output, 'b m c h w -> (b m) c h w')
            
            elif rot_inv_mode.startswith('rot'):
                assert input.ndim == 4 and input.shape[0] % 6 == 0, f'Invalid input shape: {input.shape}!'
                input = rearrange(input, '(b m) c h w -> b m c h w', m=6)
                
                input_1 = rearrange(input[:, :4], 'b m c h w -> (b m) c h w')
                input_2 = rearrange(input[:, 4:5], 'b m c h w -> (b m) c h w')
                input_3 = rearrange(input[:, 5:6], 'b m c h w -> (b m) c h w')

                output_1 = F.conv2d(input_1, self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_1 = rearrange(output_1, '(b m) c h w -> b m c h w', m=4)

                discrete = (rot_inv_mode == 'rot_max')

                output_2 = adaptive_convolution(input_2, self.weight, self.bias, padding=padding, stride=self.stride,
                                                dilation=self.dilation, groups=self.groups, inverse=False, discrete=discrete)
                output_2 = rearrange(output_2, '(b m) c h w -> b m c h w', m=1)

                output_3 = adaptive_convolution(input_3, self.weight, self.bias, padding=padding, stride=self.stride,
                                                dilation=self.dilation, groups=self.groups, inverse=True, discrete=discrete)
                output_3 = rearrange(output_3, '(b m) c h w -> b m c h w', m=1)

                output = torch.cat([output_1, output_2, output_3], dim=1)
                output = rearrange(output, 'b m c h w -> (b m) c h w')
            
            elif rot_inv_mode.startswith('shift'):
                assert input.ndim == 4 and input.shape[0] % 6 == 0, f'Invalid input shape: {input.shape}!'
                input = rearrange(input, '(b m) c h w -> b m c h w', m=6)
                
                input_1 = rearrange(input[:, :4], 'b m c h w -> (b m) c h w')
                input_2 = rearrange(input[:, 4:5], 'b m c h w -> (b m) c h w')
                input_3 = rearrange(input[:, 5:6], 'b m c h w -> (b m) c h w')

                output_1 = F.conv2d(input_1, self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_1 = rearrange(output_1, '(b m) c h w -> b m c h w', m=4)

                discrete = (rot_inv_mode == 'shift_max')

                output_2 = adaptive_convolution(input_2, self.weight, self.bias, padding=padding, stride=self.stride,
                                                dilation=self.dilation, groups=self.groups, inverse=False, discrete=discrete, shift=True)
                output_2 = rearrange(output_2, '(b m) c h w -> b m c h w', m=1)

                output_3 = adaptive_convolution(input_3, self.weight, self.bias, padding=padding, stride=self.stride,
                                                dilation=self.dilation, groups=self.groups, inverse=True, discrete=discrete, shift=True)
                output_3 = rearrange(output_3, '(b m) c h w -> b m c h w', m=1)

                output = torch.cat([output_1, output_2, output_3], dim=1)
                output = rearrange(output, 'b m c h w -> (b m) c h w')
            
            else:
                raise NotImplementedError(f'Invalid rot_inv_mode: {rot_inv_mode}!')
        
        else:
            output = F.conv2d(input, self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)

        return output
    
    return forward


def cube_sync_gn_processor_for_vae(self):
    def forward(input: Tensor) -> Tensor:
        """
        Shape:
        - Input: (B*M, C, F, H, W) or (B*M*F, C, H, W) or (B*M*F, C, H*W)
        - Output: (B*M, C, F, H, W) or (B*M*F, C, H, W) or (B*M*F, C, H*W)
        """
        m = 6
        f = 1

        if input.ndim == 5:
            input = rearrange(input, '(b m) c f h w -> b m c f h w', m=m, f=f)
        elif input.ndim == 4:
            input = rearrange(input, '(b m f) c h w -> (b f) m c h w', m=m, f=f)
        elif input.ndim == 3:
            input = rearrange(input, '(b m f) c hw -> (b f) m c hw', m=m, f=f)
        else:
            raise ValueError(f'Unsupported input shape: {input.shape}')

        input = [input[:, i] for i in range(m)]  # M * [B, C, F, H, W] or [BF, C, H, W]

        used_dtype = torch.float32
        b, dtype, device = input[0].shape[0], input[0].dtype, input[0].device
        input = [tile.to(used_dtype) for tile in input]
        shapes, tmp_tiles, num_elements = list(), list(), 0
        for tile in input:
            hw = tile.shape[2:]
            shapes.append(hw)
            tmp_tile = rearrange(tile, 'b (g c) ... -> b g (c ...)', g=self.num_groups)
            tmp_tiles.append(tmp_tile)
            num_elements = num_elements + tmp_tile.shape[-1]
        mean, var = (
            torch.zeros((b, self.num_groups, 1), dtype=used_dtype, device=device),
            torch.zeros((b, self.num_groups, 1), dtype=used_dtype, device=device)
        )

        for tile in tmp_tiles:
            mean = mean + tile.mean(-1, keepdim=True) * float(tile.shape[-1] / num_elements)
            # Unbiased variance estimation
            var = var + (
                ((tile - mean) ** 2) * (tile.shape[-1] / (tile.shape[-1] - 1))
            ).mean(-1, keepdim=True) * float(tile.shape[-1] / num_elements)

        input = []
        for shape, tile in zip(shapes, tmp_tiles):
            if len(shape) == 3:
                f, h, w = shape
                tile = rearrange((tile - mean) / (var + self.eps).sqrt(), 'b g (c f h w) -> b (g c) f h w', f=f, h=h, w=w)
                input.append(tile * self.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + self.bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            elif len(shape) == 2:
                h, w = shape
                tile = rearrange((tile - mean) / (var + self.eps).sqrt(), 'bf g (c h w) -> bf (g c) h w', h=h, w=w)
                input.append(tile * self.weight.unsqueeze(-1).unsqueeze(-1) + self.bias.unsqueeze(-1).unsqueeze(-1))
            elif len(shape) == 1:
                hw = shape[0]
                tile = rearrange((tile - mean) / (var + self.eps).sqrt(), 'b g (c hw) -> b (g c) hw', hw=hw)
                input.append(tile * self.weight.unsqueeze(-1) + self.bias.unsqueeze(-1))
            else:
                raise NotImplementedError(f'Unsupported shape: {shape}')
        
        input = torch.stack([tile.to(dtype) for tile in input], dim=1)
        if input.ndim == 6:
            input = rearrange(input, 'b m c f h w -> (b m) c f h w', m=m, f=f)
        elif input.ndim == 5:
            input = rearrange(input, '(b f) m c h w -> (b m f) c h w', m=m, f=f)
        elif input.ndim == 4:
            input = rearrange(input, '(b f) m c hw -> (b m f) c hw', m=m, f=f)
        else:
            raise ValueError(f'Unsupported input shape: {input.shape}')

        return input

    return forward


####################################################################################################################################
def apply_custom_processors_for_unet(
    model,
    enable_sync_self_attn: bool = False,
    enable_sync_cross_attn: bool = False,
    enable_sync_conv2d: bool = False,
    enable_sync_gn: bool = True,
    rot_inv_conv2d_mode: str = 'none',
):
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            if 'transformer_in' in name or 'temp_attentions' in name:
                continue
            if enable_sync_self_attn and not module.is_cross_attention:
                safe_setattr(module, 'forward_wo_sync_self_attn', module.forward, mode='ignore')
                module.forward = cube_sync_attn_processor(module)
                safe_setattr(module, 'forward_w_sync_self_attn', module.forward, mode='overwrite')
            if enable_sync_cross_attn and module.is_cross_attention:
                safe_setattr(module, 'forward_wo_sync_cross_attn', module.forward, mode='ignore')
                module.forward = cube_sync_attn_processor(module)
                safe_setattr(module, 'forward_w_sync_cross_attn', module.forward, mode='overwrite')
            
        elif isinstance(module, nn.Conv2d) and enable_sync_conv2d:
            if module.kernel_size == (1, 1):
                continue
            safe_setattr(module, 'forward_wo_sync_conv2d', module.forward, mode='ignore')
            module.forward = cube_sync_conv2d_processor(module, rot_inv_mode=rot_inv_conv2d_mode)
            safe_setattr(module, 'forward_w_sync_conv2d', module.forward, mode='overwrite')

        elif isinstance(module, nn.GroupNorm) and enable_sync_gn:
            safe_setattr(module, 'forward_wo_sync_gn', module.forward, mode='ignore')
            module.forward = cube_sync_gn_processor(module,)
            safe_setattr(module, 'forward_w_sync_gn', module.forward, mode='overwrite')


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

        elif isinstance(module, Attention) and enable_sync_attn:
            assert not module.is_cross_attention, 'Cross attention should not occur in VAE!'
            safe_setattr(module, 'forward_wo_sync_attn', module.forward, mode='ignore')
            module.forward = cube_sync_attn_processor(module)
            safe_setattr(module, 'forward_w_sync_attn', module.forward, mode='overwrite')

        elif isinstance(module, nn.Conv2d) and enable_sync_conv2d:
            if module.kernel_size == (1, 1):
                continue
            safe_setattr(module, 'forward_wo_sync_conv2d', module.forward, mode='ignore')
            module.forward = cube_sync_conv2d_processor_for_vae(module, rot_inv_mode=rot_inv_conv2d_mode)
            safe_setattr(module, 'forward_w_sync_conv2d', module.forward, mode='overwrite')

        if isinstance(module, nn.GroupNorm) and enable_sync_gn:
            if 'attentions' in name and enable_sync_attn:
                continue  # otherwise will cause error of unmatched shapes
            safe_setattr(module, 'forward_wo_sync_gn', module.forward, mode='ignore')
            module.forward = cube_sync_gn_processor_for_vae(module)
            safe_setattr(module, 'forward_w_sync_gn', module.forward, mode='overwrite')
