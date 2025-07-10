import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
from einops import rearrange, repeat
import numpy as np
from typing import Optional, Literal, Union
import open3d as o3d
from pytorch3d.transforms import matrix_to_quaternion
import gsplat


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


# Create Ray Directions. Right-Handed Coordinate System, X: left, Y: up, Z: forward
def equi_unit_rays(h: int, w: int, device):
    u = (torch.arange(w, device=device).float() + 0.5) / w
    v = (torch.arange(h, device=device).float() + 0.5) / h
    vv, uu = torch.meshgrid(v, u, indexing="ij")   # (H,W) order here

    phi   = uu * 2 * torch.pi - torch.pi  # [-pi, +pi]
    theta = torch.pi / 2 - vv * torch.pi  # [pi/2, -pi/2]

    x = - torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta)
    z = torch.cos(theta) * torch.cos(phi)
    return torch.stack((x, y, z), dim=-1)  # (h,w,3)


def cube_unit_rays(h: int, w: int, device):
    assert h == w, "Cube rays require equal height and width"
    u = (torch.arange(w, device=device).float() + 0.5) / w
    v = (torch.arange(h, device=device).float() + 0.5) / h
    vv, uu = torch.meshgrid(v, u, indexing="ij")   # (H, W) order here
    rays = []
    for i in range(6):
        # +Z
        if i == 0:
            x = 1 - uu * 2
            y = 1 - vv * 2
            z = torch.ones_like(vv)
        # -X
        elif i == 1:
            x = -torch.ones_like(vv)
            y = 1 - vv * 2
            z = 1 - uu * 2
        # -Z
        elif i == 2:
            x = uu * 2 - 1
            y = 1 - vv * 2
            z = -torch.ones_like(vv)
        # +X
        elif i == 3:
            x = torch.ones_like(vv)
            y = 1 - vv * 2
            z = uu * 2 - 1
        # +Y
        elif i == 4:
            x = 1 - uu * 2
            y = torch.ones_like(vv)
            z = vv * 2 - 1
        # -Y
        elif i == 5:
            x = 1 - uu * 2
            y = -torch.ones_like(vv)
            z = - vv * 2 + 1

        rays.append(torch.stack((x, y, z), dim=-1))
    
    rays = torch.stack(rays, dim=0)  # (6, h, w, 3)
    rays = rays / torch.linalg.norm(rays, dim=-1, keepdim=True)  # Normalize rays
    
    return rays


# Equirectangular-to-3D
def convert_rgbd_equi_to_mesh(
    rgb: torch.Tensor,                       # (H, W, 3) RGB image
    distance: torch.Tensor,                  # (H, W) Distance map
    rays: Optional[torch.Tensor] = None,     # (H, W, 3) Ray directions (unit vectors ideally)
    mask: Optional[torch.Tensor] = None,     # (H, W) Optional boolean mask
    max_size: Optional[int] = None,          # Max dimension for resizing
    device: Optional[Literal["cuda", "cpu"]] = None, # Computation device
    save_path: Optional[str] = None,
    closed_boundary: bool = True,
) -> o3d.geometry.TriangleMesh:
    """
    Converts panoramic RGBD data (image, distance, rays) into an Open3D mesh.

    Args:
        image: Input RGB image tensor (H, W, 3), uint8 or float [0, 255].
        distance: Input distance map tensor (H, W).
        rays: Input ray directions tensor (H, W, 3). Assumed to originate from (0,0,0).
        mask: Optional boolean mask tensor (H, W). True values indicate regions to potentially exclude.
        max_size: Maximum size (height or width) to resize inputs to.
        device: The torch device ('cuda' or 'cpu') to use for computations.

    Returns:
        An Open3D TriangleMesh object.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "Image must be HxWx3"
    assert distance.ndim == 2, "Distance must be HxW"
    assert rgb.shape[:2] == distance.shape[:2], "Input shapes must match"
    if rays is not None:
        assert rgb.shape[:2] == rays.shape[:2], "Input shapes must match"
        assert rays.ndim == 3 and rays.shape[2] == 3, "Rays must be HxWx3"
    if mask is not None:
        assert mask.ndim == 2 and mask.shape[:2] == rgb.shape[:2], "Mask shape must match"
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    rgb = rgb.to(device)
    distance = distance.to(device)
    
    if rays is None:
        rays = equi_unit_rays(rgb.shape[0], rgb.shape[1], device)
    rays = rays.to(device)
    
    if mask is not None:
        mask = mask.to(device)

    H, W = distance.shape
    if max_size is not None and max(H, W) > max_size:
        scale = max_size / max(H, W)
    else:
        scale = 1.0

    rgb_nchw = rgb.permute(2, 0, 1).unsqueeze(0)
    distance_nchw = distance.unsqueeze(0).unsqueeze(0)
    rays_nchw = rays.permute(2, 0, 1).unsqueeze(0)

    rgb_nchw = rgb_nchw / 255.0  # Normalize RGB to [0, 1]

    rgb_resized = F.interpolate(
        rgb_nchw,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False
    ).squeeze(0).permute(1, 2, 0)

    distance_resized = F.interpolate(
        distance_nchw,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False
    ).squeeze(0).squeeze(0)

    rays_resized_nchw = F.interpolate(
        rays_nchw,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False
    )
    
    # IMPORTANT: Renormalize ray directions after interpolation
    rays_resized = rays_resized_nchw.squeeze(0).permute(1, 2, 0)
    rays_norm = torch.linalg.norm(rays_resized, dim=-1, keepdim=True)
    rays_resized = rays_resized / (rays_norm + 1e-8)

    if mask is not None:
        mask_resized = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(), # Needs float for interpolation
            scale_factor=scale,
            mode="bilinear", # Or 'nearest' if sharp boundaries are critical
            align_corners=False,
            recompute_scale_factor=False
        ).squeeze(0).squeeze(0)
        mask_resized = mask_resized > 0.5 # Convert back to boolean
    else:
        mask_resized = None

    H_new, W_new = distance_resized.shape # Get new dimensions

    # --- Calculate 3D Vertices ---
    # Vertex position = origin + distance * ray_direction
    # Assuming origin is (0, 0, 0)
    distance_flat = distance_resized.reshape(-1, 1)     # (H*W, 1)
    rays_flat = rays_resized.reshape(-1, 3)             # (H*W, 3)
    vertices = distance_flat * rays_flat                # (H*W, 3)
    vertex_colors = rgb_resized.reshape(-1, 3)  # (H*W, 3)

    # --- Generate Mesh Faces (Triangles from Quads) ---
    if not closed_boundary:
        row_indices = torch.arange(0, H_new - 1, device=device)
        col_indices = torch.arange(0, W_new - 1, device=device)
        row = row_indices.repeat(W_new - 1)
        col = col_indices.repeat_interleave(H_new - 1)

        tl = row * W_new + col      # Top-left
        tr = tl + 1                 # Top-right
        bl = tl + W_new             # Bottom-left
        br = bl + 1                 # Bottom-right
    else:
        row_indices = torch.arange(0, H_new - 1, device=device)
        col_indices = torch.arange(0, W_new, device=device)
        row = row_indices.repeat(W_new)
        col = col_indices.repeat_interleave(H_new - 1)

        tl = row * W_new + col                      # Top-left
        tr = row * W_new + (col + 1) % W_new        # Top-right
        bl = tl + W_new                             # Bottom-left
        br = (row + 1) * W_new + (col + 1) % W_new  # Bottom-right
        
    # Apply mask if provided
    if mask_resized is not None:
        mask_tl = mask_resized[row, col]
        mask_tr = mask_resized[row, col + 1]
        mask_bl = mask_resized[row + 1, col]
        mask_br = mask_resized[row + 1, col + 1]

        quad_keep_mask = ~(mask_tl | mask_tr | mask_bl | mask_br)

        keep_indices = quad_keep_mask.nonzero(as_tuple=False).squeeze(-1)
        tl = tl[keep_indices]
        tr = tr[keep_indices]
        bl = bl[keep_indices]
        br = br[keep_indices]

    # --- Create Triangles ---
    tri1 = torch.stack([tl, tr, bl], dim=1)
    tri2 = torch.stack([tr, br, bl], dim=1)
    faces = torch.cat([tri1, tri2], dim=0)

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors.cpu().numpy())
    mesh_o3d.remove_unreferenced_vertices()
    mesh_o3d.remove_degenerate_triangles()

    if save_path is not None:
        o3d.io.write_triangle_mesh(save_path, mesh_o3d)

    return mesh_o3d


def convert_rgbd_equi_to_3dgs(
    rgb: torch.Tensor,                       # (H, W, 3) RGB image
    distance: torch.Tensor,                  # (H, W) Distance map
    rays: Optional[torch.Tensor] = None,     # (H, W, 3) Ray directions (unit vectors ideally)
    mask: Optional[torch.Tensor] = None,     # (H, W) Optional boolean mask
    max_size: Optional[int] = None,          # Max dimension for resizing
    dis_threshold=0.,
    epsilon=1e-3,
    scale_rate=1.0,
    save_path: Optional[str] = None,
) -> nn.ParameterDict:
    """
    Given an equirectangular RGB-D image, back-project each pixel to a 3D point
    and compute the corresponding 3D Gaussian covariance so that the projection covers 1 pixel.

    Parameters:
        rgb (H x W x 3): RGB image as torch.Tensor, uint8
        distance (H x W): Distance map (in meters) as torch.Tensor, float32
        rays (H x W x 3): Ray directions as torch.Tensor, float32
        epsilon (float): Small Z-scale for the splat in ray direction

    Returns:
        centers (N x 3): 3D positions of splats
        covariances (N x 3 x 3): 3D Gaussian covariances of splats
        colors (N x 3): RGB values of splats
        opacities (N x 1): Opacities of splats
        scales (N x 3): Scales of splats
        rotations (N x 4): Rotations of splats
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "Image must be HxWx3"
    assert distance.ndim == 2, "Distance must be HxW"
    assert rgb.shape[:2] == distance.shape[:2], "Input shapes must match"
    if rays is not None:
        assert rgb.shape[:2] == rays.shape[:2], "Input shapes must match"
        assert rays.ndim == 3 and rays.shape[2] == 3, "Rays must be HxWx3"
    if mask is not None:
        assert mask.ndim == 2 and mask.shape[:2] == rgb.shape[:2], "Mask shape must match"
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

    H, W = rgb.shape[:2]
    device = rgb.device

    if max_size is not None and max(H, W) > max_size:
        scale = max_size / max(H, W)
    else:
        scale = 1.0

    rgb_nchw = rgb.permute(2, 0, 1).unsqueeze(0)
    distance_nchw = distance.unsqueeze(0).unsqueeze(0)

    rgb_nchw = rgb_nchw / 255.0  # Normalize RGB to [0, 1]

    rgb = F.interpolate(
        rgb_nchw,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False
    ).squeeze(0).permute(1, 2, 0)

    distance = F.interpolate(
        distance_nchw,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False
    ).squeeze(0).squeeze(0)

    if rays is None:
        rays = equi_unit_rays(rgb.shape[0], rgb.shape[1], device)
        rays[..., [0, 1]] *= -1
    
    valid_mask = distance > dis_threshold
    rays_flat = rays.view(-1, 3)
    rgbs_flat = rgb.view(-1, 3)
    distance_flat = distance.view(-1)
    valid_rays = rays_flat[valid_mask.view(-1)]
    valid_rgbs = rgbs_flat[valid_mask.view(-1)]
    valid_distance = distance_flat[valid_mask.view(-1)]
    centers = valid_rays * valid_distance[:, None]

    delta_phi = 2 * torch.pi / rgb.shape[1]
    delta_theta = torch.pi / rgb.shape[0]
    sigma_x = valid_distance * delta_phi * scale_rate
    sigma_y = valid_distance * delta_theta * scale_rate
    sigma_z = torch.ones_like(valid_distance) * epsilon * scale_rate

    S = torch.stack([sigma_x, sigma_y, sigma_z], dim=1)
    covariances = torch.einsum('ni,nj->nij', S, S)  # Sigma = S @ S.T        # (N, 3, 3)

    up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device).expand_as(valid_rays)
    x_axis = torch.nn.functional.normalize(torch.cross(up, valid_rays, dim=-1), dim=1)
    fallback_up = torch.tensor([1, 0, 0], dtype=torch.float32, device=device).expand_as(valid_rays)
    degenerate_mask = torch.isnan(x_axis).any(dim=1)
    x_axis[degenerate_mask] = torch.nn.functional.normalize(torch.cross(fallback_up[degenerate_mask], valid_rays[degenerate_mask], dim=-1), dim=1)
    y_axis = torch.nn.functional.normalize(torch.cross(valid_rays, x_axis, dim=-1), dim=1)
    z_axis = valid_rays

    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (N, 3, 3)

    # Step 5: apply covariance transformation: Sigma = R S S^T R^T
    S_matrices = torch.zeros((S.shape[0], 3, 3), device=device)
    S_matrices[:, 0, 0] = S[:, 0]
    S_matrices[:, 1, 1] = S[:, 1]
    S_matrices[:, 2, 2] = S[:, 2]

    covariances = R @ S_matrices @ S_matrices.transpose(1, 2) @ R.transpose(1, 2)

    sh0 = rearrange(rgb_to_sh(valid_rgbs), 'n c -> n 1 c')  # (N, 1, 3)
    shN = repeat(torch.zeros_like(sh0), 'n 1 c -> n k c', k=0)  # (N, 9, 3)
    
    def inverse_sigmoid(x):
        return torch.log(x/(1-x))

    inverse_scaling_activation = torch.log    
    inverse_opacity_activation = inverse_sigmoid
    
    scales = inverse_scaling_activation(S)
    alphas = torch.ones((centers.shape[0],), device=device) * 0.99
    opacities = inverse_opacity_activation(alphas)
    
    quats = matrix_to_quaternion(R)
    
    # print(centers.shape)
    # print(sh0.shape)
    # print(centers.min(), centers.max())
    # print(scales.min(), scales.max())
    # print(opacities.min(), opacities.max())

    splats = nn.ParameterDict({
        "means": nn.Parameter(centers, requires_grad=False),
        "sh0": nn.Parameter(sh0, requires_grad=False),
        "shN": nn.Parameter(shN, requires_grad=False),
        "scales": nn.Parameter(scales, requires_grad=False),
        "quats": nn.Parameter(quats, requires_grad=False),
        "opacities": nn.Parameter(opacities, requires_grad=False),
    })
    
    if save_path is not None:
        gsplat.export_splats(
            means=splats["means"],
            sh0=splats["sh0"],
            shN=splats["shN"],
            scales=splats["scales"],
            quats=splats["quats"],
            opacities=splats["opacities"],
            format=osp.splitext(save_path)[-1].lower().lstrip('.'),
            save_to=save_path,
        )
    
    return splats


def convert_rgbd_equi_to_pcd(
    rgb: torch.Tensor,                       # (H, W, 3) RGB image
    distance: torch.Tensor,                  # (H, W) Distance map
    rays: Optional[torch.Tensor] = None,     # (H, W, 3) Ray directions (unit vectors ideally)
    mask: Optional[torch.Tensor] = None,     # (H, W) Optional boolean mask
    max_size: Optional[int] = None,          # Max dimension for resizing
    device: Optional[Literal["cuda", "cpu"]] = None, # Computation device
    save_path: Optional[str] = None,
) -> o3d.geometry.TriangleMesh:
    """
    Converts panoramic RGBD data (image, distance, rays) into an Open3D mesh.

    Args:
        image: Input RGB image tensor (H, W, 3), uint8 or float [0, 255].
        distance: Input distance map tensor (H, W).
        rays: Input ray directions tensor (H, W, 3). Assumed to originate from (0,0,0).
        mask: Optional boolean mask tensor (H, W). True values indicate regions to potentially exclude.
        max_size: Maximum size (height or width) to resize inputs to.
        device: The torch device ('cuda' or 'cpu') to use for computations.

    Returns:
        An Open3D TriangleMesh object.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "Image must be HxWx3"
    assert distance.ndim == 2, "Distance must be HxW"
    assert rgb.shape[:2] == distance.shape[:2], "Input shapes must match"
    if rays is not None:
        assert rgb.shape[:2] == rays.shape[:2], "Input shapes must match"
        assert rays.ndim == 3 and rays.shape[2] == 3, "Rays must be HxWx3"
    if mask is not None:
        assert mask.ndim == 2 and mask.shape[:2] == rgb.shape[:2], "Mask shape must match"
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    rgb = rgb.to(device)
    distance = distance.to(device)
    
    if rays is None:
        rays = equi_unit_rays(rgb.shape[0], rgb.shape[1], device)
    rays = rays.to(device)
    
    if mask is not None:
        mask = mask.to(device)

    H, W = distance.shape
    if max_size is not None and max(H, W) > max_size:
        scale = max_size / max(H, W)
    else:
        scale = 1.0

    rgb_nchw = rgb.permute(2, 0, 1).unsqueeze(0)
    distance_nchw = distance.unsqueeze(0).unsqueeze(0)
    rays_nchw = rays.permute(2, 0, 1).unsqueeze(0)

    rgb_nchw = rgb_nchw / 255.0  # Normalize RGB to [0, 1]

    rgb_resized = F.interpolate(
        rgb_nchw,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False
    ).squeeze(0).permute(1, 2, 0)

    distance_resized = F.interpolate(
        distance_nchw,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False
    ).squeeze(0).squeeze(0)

    rays_resized_nchw = F.interpolate(
        rays_nchw,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False
    )
    
    # IMPORTANT: Renormalize ray directions after interpolation
    rays_resized = rays_resized_nchw.squeeze(0).permute(1, 2, 0)
    rays_norm = torch.linalg.norm(rays_resized, dim=-1, keepdim=True)
    rays_resized = rays_resized / (rays_norm + 1e-8)

    if mask is not None:
        mask_resized = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(), # Needs float for interpolation
            scale_factor=scale,
            mode="bilinear", # Or 'nearest' if sharp boundaries are critical
            align_corners=False,
            recompute_scale_factor=False
        ).squeeze(0).squeeze(0)
        mask_resized = mask_resized > 0.5 # Convert back to boolean
    else:
        mask_resized = None

    H_new, W_new = distance_resized.shape # Get new dimensions

    # --- Calculate 3D Vertices ---
    # Vertex position = origin + distance * ray_direction
    # Assuming origin is (0, 0, 0)
    distance_flat = distance_resized.reshape(-1, 1)     # (H*W, 1)
    rays_flat = rays_resized.reshape(-1, 3)             # (H*W, 3)
    points = distance_flat * rays_flat                # (H*W, 3)
    colors = rgb_resized.reshape(-1, 3)  # (H*W, 3)

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())  # 设置颜色（确保颜色值在[0, 1]之间）

    if save_path is not None:
        o3d.io.write_point_cloud(save_path, pcd_o3d)

    return pcd_o3d


# Cubemap-to-3D
def generate_cubemap_triangles(H: int, W: int) -> torch.Tensor:
    """
    Generate triangle indices for a closed cube mesh from a CubeMap of shape [6, H, W].
    Returns: [N, 3] triangle index tensor
    """
    M = 6
    all_triangles = []

    # --- Vectorized per-face triangles ---
    grid_y, grid_x = torch.meshgrid(torch.arange(H - 1), torch.arange(W - 1), indexing='ij')
    grid_y = grid_y.reshape(-1)
    grid_x = grid_x.reshape(-1)

    v0 = grid_y * W + grid_x
    v1 = grid_y * W + (grid_x + 1)
    v2 = (grid_y + 1) * W + grid_x
    v3 = (grid_y + 1) * W + (grid_x + 1)

    tris1 = torch.stack([v0, v1, v2], dim=1)
    tris2 = torch.stack([v2, v1, v3], dim=1)
    face_tris = torch.cat([tris1, tris2], dim=0)

    for f in range(M):
        all_triangles.append(face_tris + f * H * W)

    # --- Face edge connections ---
    def get_edge(face, edge):
        if edge == 'top':
            return torch.arange(W) + face * H * W
        if edge == 'bottom':
            return torch.arange(W) + face * H * W + (H - 1) * W
        if edge == 'left':
            return torch.arange(H) * W + face * H * W
        if edge == 'right':
            return torch.arange(H) * W + face * H * W + (W - 1)

    edge_pairs = [
        (0, 'right', 1, 'left', False),
        (1, 'right', 2, 'left', False),
        (2, 'right', 3, 'left', False),
        (3, 'right', 0, 'left', False),

        (0, 'top', 4, 'bottom', False),
        (1, 'top', 4, 'right', True),
        (2, 'top', 4, 'top', True),
        (3, 'top', 4, 'left', False),

        (0, 'bottom', 5, 'top', False),
        (1, 'bottom', 5, 'right', False),
        (2, 'bottom', 5, 'bottom', True),
        (3, 'bottom', 5, 'left', True),
    ]

    for fA, eA, fB, eB, rev in edge_pairs:
        edgeA = get_edge(fA, eA)
        edgeB = get_edge(fB, eB)
        if rev:
            edgeB = edgeB.flip(0)

        v0 = edgeA[:-1]
        v1 = edgeA[1:]
        v2 = edgeB[:-1]
        v3 = edgeB[1:]

        tris1 = torch.stack([v0, v1, v2], dim=1)
        tris2 = torch.stack([v2, v1, v3], dim=1)
        all_triangles.append(tris1)
        all_triangles.append(tris2)

    # --- Add corner sealing triangles ---
    def vid(f, y, x):
        return f * H * W + y * W + x

    corners = [
        # (left, up, front)
        [vid(3, 0, W - 1), vid(4, H - 1, 0), vid(0, 0, 0)],
        # (front, up, right)
        [vid(0, 0, W - 1), vid(4, H - 1, W - 1), vid(1, 0, 0)],
        # (right, up, back)
        [vid(1, 0, W - 1), vid(4, 0, W - 1), vid(2, 0, 0)],
        # (back, up, left)
        [vid(2, 0, W - 1), vid(4, 0, 0), vid(3, 0, 0)],

        # (left, down, back)
        [vid(3, H - 1, 0), vid(5, H - 1, 0), vid(2, H - 1, W - 1)],
        # (back, down, right)
        [vid(2, H - 1, 0), vid(5, H - 1, W - 1), vid(1, H - 1, W - 1)],
        # (right, down, front)
        [vid(1, H - 1, 0), vid(5, 0, W - 1), vid(0, H - 1, W - 1)],
        # (front, down, left)
        [vid(0, H - 1, 0), vid(5, 0, 0), vid(3, H - 1, W - 1)],
    ]

    all_triangles.extend([torch.tensor(corner, dtype=torch.long).unsqueeze(0) for corner in corners])

    return torch.cat(all_triangles, dim=0).long()


def convert_rgbd_cube_to_mesh(
    rgb: torch.Tensor,                       # (M, H, W, 3) RGB image, [0, 255]
    distance: torch.Tensor,                  # (M, H, W) Distance map
    rays: Optional[torch.Tensor] = None,     # (M, H, W, 3) Ray directions (unit vectors ideally)
    mask: Optional[torch.Tensor] = None,     # (M, H, W) Optional boolean mask
    max_size: Optional[int] = None,          # Max dimension for resizing
    device: Optional[Literal["cuda", "cpu"]] = None, # Computation device
    save_path: Optional[str] = None,
    closed_boundary: bool = True,
) -> o3d.geometry.TriangleMesh:
    """
    Converts panoramic RGBD data (image, distance, rays) into an Open3D mesh.

    Args:
        rgb: Input RGB image tensor (M, H, W, 3), uint8 or float, [0, 255].
        distance: Input distance map tensor (M, H, W), meters.
        rays: Input ray directions tensor (M, H, W, 3). Assumed to originate from (0, 0, 0).
        mask: Optional boolean mask tensor (M, H, W). True values indicate regions to potentially exclude.
        max_size: Maximum size (height or width) to resize inputs to.
        device: The torch device ('cuda' or 'cpu') to use for computations.

    Returns:
        An Open3D TriangleMesh object.
    """
    assert rgb.ndim == 4 and rgb.shape[-1] == 3, "Image must be MxHxWx3"
    assert distance.ndim == 3, "Distance must be MxHxW"
    assert rgb.shape[:3] == distance.shape[:3], "Input shapes must match"
    assert rgb.shape[1] == rgb.shape[2], "Input image must be square (HxH)"
    if rays is not None:
        assert rgb.shape[:3] == rays.shape[:3], "Input shapes must match"
        assert rays.ndim == 4 and rays.shape[-1] == 3, "Rays must be MxHxWx3"
    if mask is not None:
        assert mask.ndim == 3 and mask.shape[:3] == rgb.shape[:3], "Mask shape must match"
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    rgb = rgb.to(device)
    distance = distance.to(device)
    
    if mask is not None:
        mask = mask.to(device)

    _, H, W = distance.shape
    if max_size is not None and max(H, W) > max_size:
        scale = max_size / max(H, W)
    else:
        scale = 1.0

    rgb = rgb / 255.0  # Normalize RGB to [0, 1]

    rgb = F.interpolate(
        rearrange(rgb, 'm h w c -> m c h w'),
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )
    rgb = rearrange(rgb, 'm c h w -> m h w c')

    distance = F.interpolate(
        rearrange(distance, 'm h w -> m 1 h w'),
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False
    )
    distance = rearrange(distance, 'm 1 h w -> m h w')

    if rays is None:
        rays = cube_unit_rays(rgb.shape[1], rgb.shape[2], device)
    else:
        rays = rays.to(device)
        rays = F.interpolate(
            rearrange(rays, 'm h w c -> m c h w'),
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        rays = rearrange(rays, 'm c h w -> m h w c')
        # IMPORTANT: Renormalize ray directions after interpolation
        rays_norm = torch.linalg.norm(rays, dim=-1, keepdim=True)
        rays = rays / (rays_norm + 1e-8)
    
    if mask is not None:
        mask = F.interpolate(
            mask.unsqueeze(1).float(), # Needs float for interpolation
            scale_factor=scale,
            mode="bilinear", # Or 'nearest' if sharp boundaries are critical
            align_corners=False,
            recompute_scale_factor=False
        ).squeeze(1)
        mask = mask > 0.5 # Convert back to boolean

    _, H_new, W_new = distance.shape # Get new dimensions

    # --- Calculate 3D Vertices ---
    # Vertex position = origin + distance * ray_direction
    # Assuming origin is (0, 0, 0)
    distance_flat = distance.reshape(-1, 1)     # (M*H*W, 1)
    rays_flat = rays.reshape(-1, 3)             # (M*H*W, 3)
    vertices = distance_flat * rays_flat        # (M*H*W, 3)
    vertex_colors = rgb.reshape(-1, 3)          # (M*H*W, 3)

    # --- Generate Mesh Faces (Triangles from Quads for Cubemap) ---
    faces = []
    for face_idx in range(6):  # Iterate over the 6 faces of the cubemap
        if not closed_boundary or face_idx >= 4:
            row_indices = torch.arange(0, H_new - 1, device=device)
            col_indices = torch.arange(0, W_new - 1, device=device)
            row = row_indices.repeat(W_new - 1)
            col = col_indices.repeat_interleave(H_new - 1)

            offsets = face_idx * (H_new * W_new)  # Offset for each face

            tl = offsets + row * W_new + col      # Top-left
            tr = tl + 1                           # Top-right
            bl = tl + W_new                       # Bottom-left
            br = bl + 1                           # Bottom-right
        else:
            row_indices = torch.arange(0, H_new - 1, device=device)
            col_indices = torch.arange(0, W_new, device=device)
            row = row_indices.repeat(W_new)
            col = col_indices.repeat_interleave(H_new - 1)

            offset = H_new * W_new
            start = face_idx * (H_new * W_new)  # Offset for each face

            if face_idx in (0, 1, 2):
                tl = start + row * W_new + col                               # Top-left
                tr = start + row * W_new + (col + 1) % W_new + (col + 1) // W_new * offset # Top-right
                bl = tl + W_new                                                 # Bottom-left
                br = start + (row + 1) * W_new + (col + 1) % W_new + (col + 1) // W_new * offset  # Bottom-right
            else:
                tl = start + row * W_new + col                               # Top-left
                tr = start + row * W_new + (col + 1) % W_new + (col + 1) // W_new * (-3 * offset) # Top-right
                bl = tl + W_new                                                 # Bottom-left
                br = start + (row + 1) * W_new + (col + 1) % W_new + (col + 1) // W_new * (-3 * offset)  # Bottom-right

        # Apply mask if provided
        if mask is not None:
            mask_face = mask[face_idx]
            mask_tl = mask_face[row, col]
            mask_tr = mask_face[row, col + 1]
            mask_bl = mask_face[row + 1, col]
            mask_br = mask_face[row + 1, col + 1]

            quad_keep_mask = ~(mask_tl | mask_tr | mask_bl | mask_br)

            keep_indices = quad_keep_mask.nonzero(as_tuple=False).squeeze(-1)
            tl = tl[keep_indices]
            tr = tr[keep_indices]
            bl = bl[keep_indices]
            br = br[keep_indices]

        # --- Create Triangles ---
        tri1 = torch.stack([tl, tr, bl], dim=1)
        tri2 = torch.stack([tr, br, bl], dim=1)
        faces.append(torch.cat([tri1, tri2], dim=0))

    faces = torch.cat(faces, dim=0)  # Combine faces from all cubemap sides

    faces = generate_cubemap_triangles(H_new, W_new)

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors.cpu().numpy())
    mesh_o3d.remove_unreferenced_vertices()
    mesh_o3d.remove_degenerate_triangles()

    if save_path is not None:
        o3d.io.write_triangle_mesh(save_path, mesh_o3d)

    return mesh_o3d


def convert_rgbd_cube_to_3dgs(
    rgb: torch.Tensor,                       # (M, H, W, 3) RGB image, [0, 255]
    distance: torch.Tensor,                  # (M, H, W) Distance map
    rays: Optional[torch.Tensor] = None,     # (M, H, W, 3) Ray directions (unit vectors ideally)
    mask: Optional[torch.Tensor] = None,     # (M, H, W) Optional boolean mask
    max_size: Optional[int] = None,          # Max dimension for resizing
    dis_threshold: float = 0.,
    epsilon: float = 1e-3,
    scale_rate: float = 1.0,
    device: Optional[Literal["cuda", "cpu"]] = None, # Computation device
    save_path: Optional[str] = None,
) -> nn.ParameterDict:
    """
    Given an equirectangular RGB-D image, back-project each pixel to a 3D point
    and compute the corresponding 3D Gaussian covariance so that the projection covers 1 pixel.

    Parameters:
        rgb: Input RGB image tensor (M, H, W, 3), uint8 or float, [0, 255].
        distance: Input distance map tensor (M, H, W), meters.
        rays: Input ray directions tensor (M, H, W, 3). Assumed to originate from (0, 0, 0).
        mask: Optional boolean mask tensor (M, H, W). True values indicate regions to potentially exclude.
        epsilon (float): Small Z-scale for the splat in ray direction

    Returns:
        centers (N x 3): 3D positions of splats
        covariances (N x 3 x 3): 3D Gaussian covariances of splats
        colors (N x 3): RGB values of splats
        opacities (N x 1): Opacities of splats
        scales (N x 3): Scales of splats
        rotations (N x 4): Rotations of splats
    """
    assert rgb.ndim == 4 and rgb.shape[-1] == 3, "Image must be MxHxWx3"
    assert distance.ndim == 3, "Distance must be MxHxW"
    assert rgb.shape[:3] == distance.shape[:3], "Input shapes must match"
    assert rgb.shape[1] == rgb.shape[2], "Input image must be square (HxH)"
    if rays is not None:
        assert rgb.shape[:3] == rays.shape[:3], "Input shapes must match"
        assert rays.ndim == 4 and rays.shape[-1] == 3, "Rays must be MxHxWx3"
    if mask is not None:
        assert mask.ndim == 3 and mask.shape[:3] == rgb.shape[:3], "Mask shape must match"
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

    _, H, W, _ = rgb.shape
    if device is None:
        device = rgb.device

    if max_size is not None and max(H, W) > max_size:
        scale = max_size / max(H, W)
    else:
        scale = 1.0

    rgb = rgb / 255.0  # Normalize RGB to [0, 1]

    rgb = F.interpolate(
        rearrange(rgb, 'm h w c -> m c h w'),
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False
    )
    rgb = rearrange(rgb, 'm c h w -> m h w c')

    distance = F.interpolate(
        rearrange(distance, 'm h w -> m 1 h w'),
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False
    )
    distance = rearrange(distance, 'm 1 h w -> m h w')

    if rays is None:
        rays = cube_unit_rays(rgb.shape[1], rgb.shape[2], device)
        rays[..., [0, 1]] *= -1
    
    valid_mask = distance > dis_threshold
    rays_flat = rays.reshape(-1, 3)
    rgbs_flat = rgb.reshape(-1, 3)
    distance_flat = distance.reshape(-1)
    valid_rays = rays_flat[valid_mask.view(-1)]
    valid_rgbs = rgbs_flat[valid_mask.view(-1)]
    valid_distance = distance_flat[valid_mask.view(-1)]
    centers = valid_rays * valid_distance[:, None]

    delta_phi = scale_rate / rays.shape[1]
    delta_theta = scale_rate / rays.shape[2]
    sigma_x = valid_distance * delta_phi
    sigma_y = valid_distance * delta_theta
    sigma_z = torch.ones_like(valid_distance) * epsilon

    S = torch.stack([sigma_x, sigma_y, sigma_z], dim=1)
    covariances = torch.einsum('ni,nj->nij', S, S)  # Sigma = S @ S.T        # (N, 3, 3)

    up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device).expand_as(valid_rays)
    x_axis = torch.nn.functional.normalize(torch.cross(up, valid_rays, dim=-1), dim=1)
    fallback_up = torch.tensor([1, 0, 0], dtype=torch.float32, device=device).expand_as(valid_rays)
    degenerate_mask = torch.isnan(x_axis).any(dim=1)
    x_axis[degenerate_mask] = torch.nn.functional.normalize(torch.cross(fallback_up[degenerate_mask], valid_rays[degenerate_mask], dim=-1), dim=1)
    y_axis = torch.nn.functional.normalize(torch.cross(valid_rays, x_axis, dim=-1), dim=1)
    z_axis = valid_rays

    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (N, 3, 3)

    # Step 5: apply covariance transformation: Sigma = R S S^T R^T
    S_matrices = torch.zeros((S.shape[0], 3, 3), device=device)
    S_matrices[:, 0, 0] = S[:, 0]
    S_matrices[:, 1, 1] = S[:, 1]
    S_matrices[:, 2, 2] = S[:, 2]

    covariances = R @ S_matrices @ S_matrices.transpose(1, 2) @ R.transpose(1, 2)

    sh0 = rearrange(rgb_to_sh(valid_rgbs), 'n c -> n 1 c')  # (N, 1, 3)
    shN = repeat(torch.zeros_like(sh0), 'n 1 c -> n k c', k=0)  # (N, 9, 3)
    
    def inverse_sigmoid(x):
        return torch.log(x/(1-x))

    inverse_scaling_activation = torch.log    
    inverse_opacity_activation = inverse_sigmoid
    
    scales = inverse_scaling_activation(S)
    alphas = torch.ones((centers.shape[0],), device=device) * 0.99
    opacities = inverse_opacity_activation(alphas)
    
    quats = matrix_to_quaternion(R)
    
    splats = nn.ParameterDict({
        "means": nn.Parameter(centers, requires_grad=False),
        "sh0": nn.Parameter(sh0, requires_grad=False),
        "shN": nn.Parameter(shN, requires_grad=False),
        "scales": nn.Parameter(scales, requires_grad=False),
        "quats": nn.Parameter(quats, requires_grad=False),
        "opacities": nn.Parameter(opacities, requires_grad=False),
    })
    
    if save_path is not None:
        gsplat.export_splats(
            means=splats["means"],
            sh0=splats["sh0"],
            shN=splats["shN"],
            scales=splats["scales"],
            quats=splats["quats"],
            opacities=splats["opacities"],
            format=osp.splitext(save_path)[-1].lower().lstrip('.'),
            save_to=save_path,
        )
    
    return splats


def convert_rgbd_cube_to_pcd(
    rgb: torch.Tensor,                       # (M, H, W, 3) RGB image, [0, 255]
    distance: torch.Tensor,                  # (M, H, W) Distance map
    rays: Optional[torch.Tensor] = None,     # (M, H, W, 3) Ray directions (unit vectors ideally)
    mask: Optional[torch.Tensor] = None,     # (M, H, W) Optional boolean mask
    max_size: Optional[int] = None,          # Max dimension for resizing
    device: Optional[Literal["cuda", "cpu"]] = None, # Computation device
    save_path: Optional[str] = None,
) -> o3d.geometry.PointCloud:
    """
    Converts panoramic RGBD data (image, distance, rays) into an Open3D mesh.

    Args:
        rgb: Input RGB image tensor (M, H, W, 3), uint8 or float, [0, 255].
        distance: Input distance map tensor (M, H, W), meters.
        rays: Input ray directions tensor (M, H, W, 3). Assumed to originate from (0, 0, 0).
        mask: Optional boolean mask tensor (M, H, W). True values indicate regions to potentially exclude.
        max_size: Maximum size (height or width) to resize inputs to.
        device: The torch device ('cuda' or 'cpu') to use for computations.

    Returns:
        An Open3D TriangleMesh object.
    """
    assert rgb.ndim == 4 and rgb.shape[-1] == 3, "Image must be MxHxWx3"
    assert distance.ndim == 3, "Distance must be MxHxW"
    assert rgb.shape[:3] == distance.shape[:3], "Input shapes must match"
    assert rgb.shape[1] == rgb.shape[2], "Input image must be square (HxH)"
    if rays is not None:
        assert rgb.shape[:3] == rays.shape[:3], "Input shapes must match"
        assert rays.ndim == 4 and rays.shape[-1] == 3, "Rays must be MxHxWx3"
    if mask is not None:
        assert mask.ndim == 3 and mask.shape[:3] == rgb.shape[:3], "Mask shape must match"
        assert mask.dtype == torch.bool, "Mask must be a boolean tensor"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    rgb = rgb.to(device)
    distance = distance.to(device)
    
    if mask is not None:
        mask = mask.to(device)

    _, H, W = distance.shape
    if max_size is not None and max(H, W) > max_size:
        scale = max_size / max(H, W)
    else:
        scale = 1.0

    rgb = rgb / 255.0  # Normalize RGB to [0, 1]

    rgb = F.interpolate(
        rearrange(rgb, 'm h w c -> m c h w'),
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )
    rgb = rearrange(rgb, 'm c h w -> m h w c')

    distance = F.interpolate(
        rearrange(distance, 'm h w -> m 1 h w'),
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False
    )
    distance = rearrange(distance, 'm 1 h w -> m h w')

    if rays is None:
        rays = cube_unit_rays(rgb.shape[1], rgb.shape[2], device)
    else:
        rays = rays.to(device)
        rays = F.interpolate(
            rearrange(rays, 'm h w c -> m c h w'),
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        rays = rearrange(rays, 'm c h w -> m h w c')
        # IMPORTANT: Renormalize ray directions after interpolation
        rays_norm = torch.linalg.norm(rays, dim=-1, keepdim=True)
        rays = rays / (rays_norm + 1e-8)
    
    if mask is not None:
        mask = F.interpolate(
            mask.unsqueeze(1).float(), # Needs float for interpolation
            scale_factor=scale,
            mode="bilinear", # Or 'nearest' if sharp boundaries are critical
            align_corners=False,
            recompute_scale_factor=False
        ).squeeze(1)
        mask = mask > 0.5 # Convert back to boolean

    # --- Calculate 3D Vertices ---
    # Vertex position = origin + distance * ray_direction
    # Assuming origin is (0, 0, 0)
    distance_flat = distance.reshape(-1, 1)     # (M*H*W, 1)
    rays_flat = rays.reshape(-1, 3)             # (M*H*W, 3)
    points = distance_flat * rays_flat        # (M*H*W, 3)
    colors = rgb.reshape(-1, 3)          # (M*H*W, 3)

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())  # 设置颜色（确保颜色值在[0, 1]之间）

    if save_path is not None:
        o3d.io.write_point_cloud(save_path, pcd_o3d)

    return pcd_o3d