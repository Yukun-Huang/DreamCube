from typing import Union, Optional
import numpy as np
from einops import rearrange, repeat, einsum
from diffusers.utils import check_min_version
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.resnet import ResnetBlock2D, Upsample2D
from diffusers.models.attention_processor import Attention
from diffusers.utils.torch_utils import randn_tensor
from utils.cube import pad_cube, unpad_cube

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.2")


####################################################################################################################################
def rotation_invariant_kernels(weight: torch.Tensor, mode: str) -> torch.Tensor:
    """
    参数:
        weight (torch.Tensor): 输入的卷积核权重，形状为 [C_in, C_out, K, K]
    
    返回:
        torch.Tensor: 旋转后的卷积核权重，形状相同。
    """
    _, _, K1, K2 = weight.shape
    assert K1 == K2, f"Invalid weight shape: {weight.shape}"
    new_weight = weight.clone()
    if mode == '90':
        for i in range(1, 4):
            new_weight += torch.rot90(weight, i, [2, 3])
        new_weight *= 0.25
    elif mode == '180':
        new_weight += torch.rot90(weight, 2, [2, 3])
        new_weight *= 0.5
    else:
        raise NotImplementedError(f"Invalid mode: {mode}")
    return new_weight


def convert_to_one_hot(weights: torch.Tensor, multiple: bool = True) -> torch.Tensor:
    """
    将权重矩阵转换为 one-hot 形式。对于每个像素，选择最大权重对应的索引位置为 1，其他位置为 0。
    :param weights: 形状为 [B, 4, H, W] 的权重矩阵
    :return: 转换后的 one-hot 矩阵，形状为 [B, 4, H, W]
    """
    if multiple:
        # 找到每个像素位置的最大权重值
        max_values, _ = torch.max(weights, dim=1, keepdim=True)  # [B, 1, H, W]
        # 比较每个像素位置的权重是否等于最大值
        one_hot = (weights == max_values).to(weights.dtype)  # [B, 4, H, W]
        # 对每个像素位置的 one-hot 值进行归一化
        sum_one_hot = one_hot.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        one_hot = one_hot / sum_one_hot
    else:
        # 使用 torch.argmax 找到每个像素位置最大权重的索引
        max_indices = torch.argmax(weights, dim=1)  # [B, H, W]
        # 创建一个与 weights 同形状的全零张量
        one_hot = torch.zeros_like(weights)
        # 对于每个像素，设置最大索引位置为 1
        one_hot.scatter_(1, max_indices.unsqueeze(1), 1)
    
    return one_hot


def calculate_edge_weights(tensor: torch.Tensor, one_hot: bool = False) -> torch.Tensor:
    # 获取输入tensor的尺寸
    B, *_, H, W = tensor.shape
    
    
    # 计算每个像素到四个边的距离
    top_distance = torch.arange(H).view(1, 1, H, 1).expand(B, 1, H, W) / (H - 1)
    bottom_distance = torch.arange(H).view(1, 1, H, 1).expand(B, 1, H, W) / (H - 1)
    left_distance = torch.arange(W).view(1, 1, 1, W).expand(B, 1, H, W) / (W - 1)
    right_distance = torch.arange(W).view(1, 1, 1, W).expand(B, 1, H, W) / (W - 1)
    
    top_weight = 1.0 - top_distance  # 计算与上边的距离并取反
    bottom_weight = bottom_distance  # 计算与下边的距离并取反
    left_weight = 1.0 - left_distance  # 计算与左边的距离并取反
    right_weight = right_distance  # 计算与右边的距离并取反
    
    # 权重归一化，确保总和为1
    sum_weight = top_weight + bottom_weight + left_weight + right_weight
    weights = torch.concat([top_weight, right_weight, bottom_weight, left_weight], dim=1)

    weights = (weights / sum_weight).to(tensor.device)  # [B, 4, H, W]
    
    if one_hot:
        weights = convert_to_one_hot(weights, multiple=True)

    return weights


def compute_theta_map(H, W, device, mode, inverse=False, bias=None, degree=True):
    """ 计算每个像素点相对于中心的角度矩阵
    Returns:
        theta: [H, W] 角度矩阵
    """
    assert H == W, f'H and W should be equal, but got {H} and {W}!'
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    center_y, center_x = (H - 1) / 2, (W - 1) / 2
    theta = torch.atan2(y - center_y, x - center_x)  # 计算弧度

    if mode == 'flux':
        if not inverse:
            theta = (theta - (3 * np.pi) / 4)
        else:
            theta = ((3 * np.pi) / 4 - theta)

    elif mode == 'conv2d':
        if not inverse:
            theta = (theta - torch.pi / 2)
        else:
            theta = (theta + torch.pi / 2)

        if bias is not None:
            theta = theta + (bias * torch.pi / 180 if degree else bias)

    else:
        raise NotImplementedError(f'Invalid mode: {mode}!')

    theta = theta % (2 * torch.pi)

    if H % 2 == 1 and W % 2 == 1:
        theta[H//2, W//2] = np.pi
    
    return torch.rad2deg(theta) if degree else theta


def rotate_patches(patches, theta):
    """
    对 [B, C, K, K, H, W] 形状的 patches 进行 2D 旋转
    :param patches: Tensor, shape [B, C, K, K, H, W]
    :param theta: Tensor, shape [H, W], 角度（弧度制）
    :return: 旋转后的 patches，形状不变
    """
    B, C, K, _, H, W = patches.shape
    device = patches.device

    # 生成标准网格坐标 [-1, 1] 归一化
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, K, device=device),
                            torch.linspace(-1, 1, K, device=device), indexing='ij')
    grid = torch.stack((xs, ys), dim=-1)  # 形状 [K, K, 2]
    
    # 计算旋转矩阵
    cos_t = torch.cos(theta)  # [H, W]
    sin_t = torch.sin(theta)  # [H, W]
    
    # 旋转变换 [H, W, 2, 2]
    R = torch.stack([torch.stack([cos_t, -sin_t], dim=-1),
                     torch.stack([sin_t, cos_t], dim=-1)], dim=-2)
    
    # 旋转后的 grid [H, W, K, K, 2]
    rotated_grid = torch.einsum('hwab,ijb->hwija', R, grid)
    
    # 添加 batch 维度并转换为 [-1, 1] 坐标系
    rotated_grid = rotated_grid.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1)  # [B, H, W, K, K, 2]
    
    # `grid_sample` 需要输入形状为 [B, C, H, W]，因此转换 patches
    patches_reshaped = patches.permute(0, 4, 5, 1, 2, 3).reshape(B * H * W, C, K, K)
    rotated_grid = rotated_grid.reshape(B * H * W, K, K, 2)
    
    # 进行插值
    rotated_patches = F.grid_sample(
        patches_reshaped, rotated_grid,
        mode='nearest', padding_mode='border', align_corners=True)
    
    # 还原形状
    rotated_patches = rotated_patches.view(B, H, W, C, K, K).permute(0, 3, 4, 5, 1, 2)
    
    return rotated_patches


def adaptive_convolution(
    input: torch.Tensor, weight: torch.Tensor, bias=None,
    padding=0, stride=1, dilation=1, groups=1, inverse=False,
    discrete=False,
):
    """ 进行基于 theta 角度自适应旋转的卷积 """
    B, C, H, W = input.shape
    C_out, C_in, K, K = weight.shape
    device = input.device

    assert stride[0] == stride[1] and dilation[0] == dilation[1] and groups == 1, \
        f'Only support equal strides, equal dilations, groups=1, but got {stride}, {dilation}, {groups}'
    
    H_out = (H - dilation[0] * (K - 1) - 1) // stride[0] + 1
    W_out = (W - dilation[0] * (K - 1) - 1) // stride[0] + 1
    
    patches = F.unfold(input, kernel_size=(K, K), stride=stride, dilation=dilation)
    patches = patches.view(B, C_in, K, K, H_out, W_out)

    theta_map = compute_theta_map(H_out, W_out, device, mode='conv2d', degree=False, inverse=inverse)
    if discrete:
        theta_map = ((theta_map + (torch.pi/4)) % (torch.pi*2)) // (torch.pi/2) * (torch.pi/2)
    
    patches = rotate_patches(patches, theta_map)

    output = einsum(
        patches,
        weight,
        'b c_in k1 k2 h w, c_out c_in k1 k2 -> b c_out h w',
    )

    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output


####################################################################################################################################
def cube_sync_gn_processor(self):
    def forward(tiles):
        """
        Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)
        """
        m = 6

        tiles = rearrange(tiles, '(b m) ... -> b m ...', m=m)
        tiles = [tiles[:, i] for i in range(m)]  # M * [B, C, H, W] or [B, C, HW]

        used_dtype = torch.float32
        b, dtype, device = tiles[0].shape[0], tiles[0].dtype, tiles[0].device
        tiles = [tile.to(used_dtype) for tile in tiles]
        shapes, tmp_tiles, num_elements = list(), list(), 0
        for tile in tiles:
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

        tiles = []
        for shape, tile in zip(shapes, tmp_tiles):
            if len(shape) == 2:
                h, w = shape
                tile = rearrange((tile - mean) / (var + self.eps).sqrt(), 'b g (c h w) -> b (g c) h w', h=h, w=w)
                tiles.append(tile * self.weight.unsqueeze(-1).unsqueeze(-1) + self.bias.unsqueeze(-1).unsqueeze(-1))
            else:
                assert len(shape) == 1, f'Invalid shape: {shape}'
                hw = shape[0]
                tile = rearrange((tile - mean) / (var + self.eps).sqrt(), 'b g (c hw) -> b (g c) hw', hw=hw)
                tiles.append(tile * self.weight.unsqueeze(-1) + self.bias.unsqueeze(-1))
        
        tiles = torch.stack([tile.to(dtype) for tile in tiles], dim=1)
        tiles = rearrange(tiles, 'b m ... -> (b m) ...', m=m)
        return tiles

    return forward


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


def cube_sync_conv2d_processor(self, enable_rot_inv=False, rot_inv_mode='avg', cube_padding_impl='cuda'):

    def forward(input: torch.Tensor) -> torch.Tensor:
        padding = self.padding
        pad_h, pad_w = padding
        assert pad_h == pad_w and pad_h > 0, 'Only support square padding!'
        input = pad_cube(input, pad_h, impl=cube_padding_impl)

        if enable_rot_inv:
            assert rot_inv_mode in ('sum_max', 'sum_avg', 'rot', 'rot_max', 'kernel'), f'Invalid rot_inv_mode: {rot_inv_mode}!'

            if rot_inv_mode.startswith('sum'):
                one_hot = (rot_inv_mode == 'sum_max')

                assert input.ndim == 4 and input.shape[0] % 6 == 0, f'Invalid input shape: {input.shape}!'
                input = rearrange(input, '(b m) c h w -> b m c h w', m=6)
                
                input_1 = rearrange(input[:, :4], 'b m c h w -> (b m) c h w')
                input_2 = rearrange(input[:, 4:], 'b m c h w -> (b m) c h w')

                output_1 = F.conv2d(input_1, self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_1 = rearrange(output_1, '(b m) c h w -> b m c h w', m=4)

                output_2a = F.conv2d(input_2, self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_2b = F.conv2d(input_2, torch.rot90(self.weight, dims=(2, 3), k=1), self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_2c = F.conv2d(input_2, torch.rot90(self.weight, dims=(2, 3), k=2), self.bias, self.stride, 'valid', self.dilation, self.groups)
                output_2d = F.conv2d(input_2, torch.rot90(self.weight, dims=(2, 3), k=3), self.bias, self.stride, 'valid', self.dilation, self.groups)
                outputs_2 = torch.stack([output_2a, output_2b, output_2c, output_2d], dim=1)  # [BM, 4, C, H, W]
                outputs_2 = rearrange(outputs_2, '(b m) e c h w -> b m e c h w', m=2)         # [B, M, 4, C, H, W]

                weights = calculate_edge_weights(outputs_2, one_hot=one_hot)  # [B, 4, H, W]
                outputs_2_up = einsum(outputs_2[:, 0], weights[:, [2, 1, 0, 3]], 'b e c h w, b e h w -> b c h w').unsqueeze(1)
                outputs_2_down = einsum(outputs_2[:, 1], weights[:, [0, 3, 2, 1]], 'b e c h w, b e h w -> b c h w').unsqueeze(1)

                output = torch.cat([output_1, outputs_2_up, outputs_2_down], dim=1)
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
        
        else:
            output = F.conv2d(input, self.weight, self.bias, self.stride, 'valid', self.dilation, self.groups)

        return output
    
        # return unpad_cube(output, pad_h)  # no need to manually unpad because the Conv2d op already does it

    return forward


def setup_sync_conv2d_processor(module, **kwargs):
    padding = module.padding
    if isinstance(padding, tuple) and padding[0] == padding[1]:
        if padding[0] == 0:
            return
        assert not hasattr(module, 'forward_wo_sync_conv2d') and \
            not hasattr(module, 'forward_w_sync_conv2d'), 'Already applied sync Conv2d processor!'
        module.forward_wo_sync_conv2d = module.forward
        module.forward = cube_sync_conv2d_processor(module, **kwargs)
        module.forward_w_sync_conv2d = module.forward
    else:
        print(f'[Warning] Only support square padding for sync conv2d, but got {padding} from {module}!')


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
def apply_custom_processors_for_transformer(
    model,
    enable_sync_self_attn: bool = True,
    enable_sync_cross_attn: bool = False,
    enable_sync_gn: bool = False,
):
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            if enable_sync_self_attn and not module.is_cross_attention:
                module.forward_wo_sync_self_attn = module.forward
                module.forward = cube_sync_attn_processor(module)
                module.forward_w_sync_self_attn = module.forward

            if enable_sync_cross_attn and module.is_cross_attention:
                module.forward_wo_sync_cross_attn = module.forward
                module.forward = cube_sync_attn_processor(module)
                module.forward_w_sync_cross_attn = module.forward

        elif isinstance(module, nn.GroupNorm) and enable_sync_gn:
            assert not hasattr(module, 'forward_wo_sync_gn') and \
                not hasattr(module, 'forward_w_sync_gn'), 'Already applied sync GN processor!'
            module.forward_wo_sync_gn = module.forward
            module.forward = cube_sync_gn_processor(module)
            module.forward_w_sync_gn = module.forward


def apply_custom_processors_for_unet(
    model,
    enable_sync_self_attn: bool = True,
    enable_sync_cross_attn: bool = False,
    enable_sync_conv2d: bool = False,
    enable_sync_gn: bool = False,
    enable_rot_inv_conv2d: bool = False,
    rot_inv_conv2d_mode: str = 'avg',
    cube_padding_impl: str = 'cuda',
):
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            if enable_sync_self_attn and not module.is_cross_attention:
                module.forward_wo_sync_self_attn = module.forward
                module.forward = cube_sync_attn_processor(module)
                module.forward_w_sync_self_attn = module.forward

            if enable_sync_cross_attn and module.is_cross_attention:
                module.forward_wo_sync_cross_attn = module.forward
                module.forward = cube_sync_attn_processor(module)
                module.forward_w_sync_cross_attn = module.forward

        elif isinstance(module, nn.Conv2d) and enable_sync_conv2d:
            setup_sync_conv2d_processor(
                module,
                enable_rot_inv=enable_rot_inv_conv2d,
                rot_inv_mode=rot_inv_conv2d_mode,
                cube_padding_impl=cube_padding_impl,
            )

        elif isinstance(module, nn.GroupNorm) and enable_sync_gn:
            assert not hasattr(module, 'forward_wo_sync_gn') and \
                not hasattr(module, 'forward_w_sync_gn'), 'Already applied sync GN processor!'
            module.forward_wo_sync_gn = module.forward
            module.forward = cube_sync_gn_processor(module)
            module.forward_w_sync_gn = module.forward


def apply_custom_processors_for_vae(
    model,
    mode: str = 'all',
    enable_sync_gn: bool = True,
    enable_sync_conv2d: bool = False,
    enable_sync_attn: bool = False,
    enable_rot_inv_conv2d: bool = False,
    rot_inv_conv2d_mode: str = 'avg',
    cube_padding_impl: str = 'cuda',
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
            
            assert not hasattr(module, 'forward_wo_sync_gn') and \
                not hasattr(module, 'forward_w_sync_gn'), 'Already applied sync GN processor!'
            module.forward_wo_sync_gn = module.forward
            module.forward = cube_sync_gn_processor(module)
            module.forward_w_sync_gn = module.forward
        
        elif isinstance(module, Attention) and enable_sync_attn:
            assert not module.is_cross_attention, 'Cross attention should not occur in VAE!'
            module.forward_wo_sync_attn = module.forward
            module.forward = cube_sync_attn_processor(module)
            module.forward_w_sync_attn = module.forward

        elif isinstance(module, nn.Conv2d) and enable_sync_conv2d:
            setup_sync_conv2d_processor(
                module,
                enable_rot_inv=enable_rot_inv_conv2d,
                rot_inv_mode=rot_inv_conv2d_mode,
                cube_padding_impl=cube_padding_impl,
            )
