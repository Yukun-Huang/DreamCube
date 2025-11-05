from einops import rearrange
import torch
from torch import Tensor
import torch.nn.functional as F


def cube_sync_gn_processor(self):
    def forward(input: Tensor) -> Tensor:
        """
        Shape:
        - Input: (B*M, C, H, W) or (B*M, C, HW)
        - Output: (B*M, C, H, W) or (B*M, C, HW)
        """
        m = 6

        input = rearrange(input, '(b m) ... -> b m ...', m=m)
        input = [input[:, i] for i in range(m)]  # M * [B, C, H, W] or [B, C, HW]

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

        for tile in tmp_tiles:
            # Unbiased variance estimation
            var = var + (
                    ((tile - mean) ** 2) * (tile.shape[-1] / (tile.shape[-1] - 1))
            ).mean(-1, keepdim=True) * float(tile.shape[-1] / num_elements)

        input = []
        for shape, tile in zip(shapes, tmp_tiles):
            if len(shape) == 2:
                h, w = shape
                tile = rearrange((tile - mean) / (var + self.eps).sqrt(), 'b g (c h w) -> b (g c) h w', h=h, w=w)
                input.append(tile * self.weight.unsqueeze(-1).unsqueeze(-1) + self.bias.unsqueeze(-1).unsqueeze(-1))
            elif len(shape) == 1:
                hw = shape[0]
                tile = rearrange((tile - mean) / (var + self.eps).sqrt(), 'b g (c hw) -> b (g c) hw', hw=hw)
                input.append(tile * self.weight.unsqueeze(-1) + self.bias.unsqueeze(-1))
            else:
                raise NotImplementedError(f'Unsupported shape: {shape}')
        
        input = torch.stack([tile.to(dtype) for tile in input], dim=1)
        input = rearrange(input, 'b m ... -> (b m) ...', m=m)
        return input

    return forward


def cube_video_sync_gn_processor(self):
    def forward(input: Tensor) -> Tensor:
        """
        Shape:
        - Input: (B*M, C, F, H, W) or (B*M*F, C, H, W)
        - Output: (B*M, C, F, H, W) or (B*M*F, C, H, W)
        """
        m = 6

        if input.ndim == 5:
            input = rearrange(input, '(b m) c f h w -> b m c f h w', m=m)
        elif input.ndim == 4:
            input = rearrange(input, '(b m f) c h w -> (b f) m c h w', m=m)
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

        for tile in tmp_tiles:
            # Unbiased variance estimation
            var = var + (
                    ((tile - mean) ** 2) * (tile.shape[-1] / (tile.shape[-1] - 1))
            ).mean(-1, keepdim=True) * float(tile.shape[-1] / num_elements)

        input = []
        for shape, tile in zip(shapes, tmp_tiles):
            if len(shape) == 2:
                h, w = shape
                tile = rearrange((tile - mean) / (var + self.eps).sqrt(), 'b g (c h w) -> b (g c) h w', h=h, w=w)
                input.append(tile * self.weight.unsqueeze(-1).unsqueeze(-1) + self.bias.unsqueeze(-1).unsqueeze(-1))
            elif len(shape) == 3:
                f, h, w = shape
                tile = rearrange((tile - mean) / (var + self.eps).sqrt(), 'b g (c f h w) -> b (g c) f h w', f=f, h=h, w=w)
                input.append(tile * self.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + self.bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            else:
                raise NotImplementedError(f'Unsupported shape: {shape}')
        
        input = torch.stack([tile.to(dtype) for tile in input], dim=1)
        input = rearrange(input, 'b m ... -> (b m) ...', m=m)
        return input

    return forward


def cube_sync_ln_processor(self):

    def forward(input: Tensor) -> Tensor:
        print('input:', input.shape)
        print('normalized_shape:', self.normalized_shape)
        output = F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)
        print('output:', output.shape)
        return output
    
    return forward
