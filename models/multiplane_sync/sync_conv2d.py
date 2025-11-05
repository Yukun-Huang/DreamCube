from typing import Union, Optional
import numpy as np
from einops import rearrange, repeat, einsum
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.cube import pad_cube, unpad_cube


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


def convert_to_two_hot(weights: torch.Tensor) -> torch.Tensor:
    """
    将输入的权重矩阵中每个位置上最小的两个值置0，然后重新归一化。

    :param weights: 输入的权重矩阵，形状为 [B, 4, H, W]
    :return: 处理后的权重矩阵，形状为 [B, 4, H, W]
    """
    # 获取输入形状
    B, C, H, W = weights.shape
    
    # 找到每个位置上最小的两个值的位置
    topk = torch.topk(weights, k=2, dim=1, largest=False)
    
    # 创建一个掩码，将最小的两个值置0
    mask = torch.ones_like(weights)
    mask.scatter_(1, topk.indices, 0)
    
    # 将最小的两个值置0
    weights = weights * mask
    
    # 重新归一化
    sum_weights = weights.sum(dim=1, keepdim=True)
    normalized_weights = weights / (sum_weights + 1e-10)  # 加上一个小的常数以避免除以零

    return normalized_weights


def calculate_edge_weights(tensor: torch.Tensor, one_hot: bool = False, two_hot: bool = False) -> torch.Tensor:
    # 获取输入tensor的尺寸
    B, *_, H, W = tensor.shape
    
    
    # 计算每个像素到四个边的距离
    top_distance = torch.arange(H).view(1, 1, H, 1).expand(B, 1, H, W) / (H - 1)
    bottom_distance = torch.arange(H).view(1, 1, H, 1).expand(B, 1, H, W) / (H - 1)
    left_distance = torch.arange(W).view(1, 1, 1, W).expand(B, 1, H, W) / (W - 1)
    right_distance = torch.arange(W).view(1, 1, 1, W).expand(B, 1, H, W) / (W - 1)
    
    top_weight = 1.0 - top_distance  # 计算与上边的距离并取反
    bottom_weight = bottom_distance  # 计算与下边的距离
    left_weight = 1.0 - left_distance  # 计算与左边的距离并取反
    right_weight = right_distance  # 计算与右边的距离
    
    # 权重归一化，确保总和为1
    sum_weight = top_weight + bottom_weight + left_weight + right_weight
    weights = torch.concat([top_weight, right_weight, bottom_weight, left_weight], dim=1)

    weights = (weights / sum_weight).to(tensor.device)  # [B, 4, H, W]
    
    if one_hot:
        weights = convert_to_one_hot(weights, multiple=True)
    
    elif two_hot:
        weights = convert_to_two_hot(weights)

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


def rotate_patches_by_shifts(patches, k):
    """
    patches: [B, C, K, K, H, W]
    theta: int
    """
    assert patches.ndim == 6, f'Invalid patches shape: {patches.shape}!'
    patches = torch.rot90(patches, k=k, dims=(2, 3))
    return patches


def adaptive_convolution(
    input: torch.Tensor, weight: torch.Tensor, bias=None,
    padding=0, stride=(1, 1), dilation=(1, 1), groups=1, inverse=False,
    discrete=False, shift=False,
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
    
    if shift:
        patches = rotate_patches_by_shifts(patches, k=2)
    else:
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
def cube_sync_conv2d_processor(self, rot_inv_mode='none', enable_cube_padding=True, impl='cuda'):

    def forward(input: torch.Tensor) -> torch.Tensor:
        # padding
        padding = self.padding
        assert padding[0] == padding[1], 'Only support square padding!'
        if padding[0] > 0:
            if enable_cube_padding:
                input = pad_cube(input, padding[0], impl=impl)
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
