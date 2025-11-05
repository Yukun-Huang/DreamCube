import os
import cv2
import numpy as np
from PIL import Image
import torch
import math
import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Any, Union, List, Optional
from utils.pers import npers2pers
from external import py360convert
from external import equilib

from utils.ops.resample_padding import resample_padding


def pad_cube(cube: torch.Tensor, padding: int, backwards: bool = False, impl: str = 'cuda') -> torch.Tensor:
    """
    Args:
        cube: torch.Tensor, [b*m, c, h, w] or [b, m, c, h, w]
        padding: int
    """
    # no padding
    if padding <= 0:
        return cube
    # dimension check
    h, w = cube.shape[-2:]
    assert h == w
    if cube.ndim == 5:
        assert cube.shape[1] == 6
        cube_pad = rearrange(cube, 'b m c h w -> (b m) c h w')
    else:
        assert cube.ndim == 4 and cube.shape[0] % 6 == 0, "cube must be [b*m, c, h, w] or [b, m, c, h, w], got {}".format(cube.shape)
        cube_pad = cube
    # resample padding
    dtype = cube_pad.dtype
    if dtype == torch.bfloat16 or dtype == torch.float16:
        cast_to_float32 = True
        cube_pad = cube_pad.to(torch.float32)
    else:
        cast_to_float32 = False
    # pre-padding
    cube_pad = F.pad(cube_pad, [padding] * 4, mode='reflect')
    cube_pad = resample_padding(cube_pad, sampling_rate=h, backwards=backwards, impl=impl)
    if cast_to_float32:
        cube_pad = cube_pad.to(dtype)
    # return
    if cube.ndim == 5:
        cube_pad = rearrange(cube_pad, '(b m) c h w -> b m c h w', m=6)
    return cube_pad


def unpad_cube(cube_pad: torch.Tensor, padding: int):
    """
    Args:
        cube: torch.Tensor, [b*m, c, h, w] or [b, m, c, h, w]
        padding: int
    """
    if padding <= 0:
        return cube_pad
    return cube_pad[..., padding:-padding, padding:-padding]


def rotate_cube(cube: torch.Tensor, yaw: int):
    """
    Args:
        cube: torch.Tensor, [b*m, c, h, w] or [b, m, c, h, w]
        yaw: int
    """
    if yaw % 360 == 0:
        return cube
    # dimension check
    c, h, w = cube.shape[-3:]
    if cube.ndim == 4:
        assert cube.shape[0] % 6 == 0
        horizon = cube.reshape(-1, 6, c, h, w)
    else:
        horizon = cube
    assert h == w and horizon.ndim == 5
    bs = horizon.shape[0]
    horizon = rearrange(horizon, 'b m c h w -> b c h (m w)')
    # rotate
    equi = equilib.cube2equi(horizon, cube_format='horizon', height=h*2, width=w*4, clip_output=False)
    if bs == 1 and equi.ndim == 3:
        equi = equi.unsqueeze(0)  # fix equilib's auto squeeze
    _, _, h_pano, w_pano = equi.shape
    equi = torch.roll(equi, shifts=int(w_pano * yaw / 360), dims=3)
    rots = [{'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0} for _ in range(bs)]
    horizon = equilib.equi2cube(equi, rots=rots, w_face=w, cube_format='horizon', clip_output=False)
    # return
    horizon = horizon.reshape(bs, c, h, 6, w)
    if cube.ndim == 4:
        return rearrange(horizon, 'b c h m w -> (b m) c h w')
    else:
        return rearrange(horizon, 'b c h m w -> b m c h w')


def rotate_cube_by_shifts(cube: torch.Tensor | np.ndarray, shifts: int):
    """
    Args:
        cube: torch.Tensor [b*m, c, h, w] or [b, m, c, h, w] | np.ndarray, [b*m, h, w, c] or [b, m, h, w, c]
        shifts: int
    """
    if shifts == 0:
        return cube
    # dimension check
    orig_ndim = cube.ndim
    if cube.ndim == 4:
        assert cube.shape[0] % 6 == 0
        cube = rearrange(cube, '(b m) ... -> b m ...', m=6)
    h, w = cube.shape[-2:] if isinstance(cube, torch.Tensor) else cube.shape[-3:-1]
    assert h == w and cube.ndim == 5, "Invalid input shape: {}".format(cube.shape)
    # rotate
    if isinstance(cube, torch.Tensor):
        cube = cube.clone()
        cube[:, :4] = torch.roll(cube[:, :4], shifts=shifts, dims=1)
        cube[:, 4] = torch.rot90(cube[:, 4], k=shifts, dims=(2, 3))
        cube[:, 5] = torch.rot90(cube[:, 5], k=-shifts, dims=(2, 3))
    else:
        cube = np.copy(cube)
        cube[:, :4] = np.roll(cube[:, :4], shift=shifts, axis=1)
        cube[:, 4] = np.rot90(cube[:, 4], k=shifts, axes=(1, 2))
        cube[:, 5] = np.rot90(cube[:, 5], k=-shifts, axes=(1, 2))
    # return
    if orig_ndim == 4:
        cube = rearrange(cube, 'b m ... -> (b m) ...')
    return cube


def crop_to_fov(
        image: Image.Image | np.ndarray,
        original_fov_x: float,
        original_fov_y: float, 
        target_fov_x: float = 90,
        target_fov_y: float = 90,
    ) -> Image.Image:
    """
    Crops an image to simulate reduced horizontal and vertical field of views (FOV) based on angles.
    
    Parameters:
    - image: PIL.Image.Image - The input image in PIL format.
    - original_fov_x: float - The original horizontal field of view in degrees (e.g., 120).
    - original_fov_y: float - The original vertical field of view in degrees.
    - target_fov_x: float - The target horizontal field of view in degrees (default is 90).
    - target_fov_y: float - The target vertical field of view in degrees (default is 90).
    
    Returns:
    - PIL.Image.Image - The cropped image with the target horizontal and vertical FOVs.
    """
    # Calculate crop ratios based on tan(fov / 2)
    crop_ratio_x = math.tan(math.radians(target_fov_x / 2)) / math.tan(math.radians(original_fov_x / 2))
    crop_ratio_y = math.tan(math.radians(target_fov_y / 2)) / math.tan(math.radians(original_fov_y / 2))
    if isinstance(image, Image.Image):
        img_width, img_height = image.size
    else:
        img_height, img_width = image.shape[:2]

    # Calculate target dimensions after cropping
    target_width = int(np.round(img_width * crop_ratio_x))
    target_height = int(np.round(img_height * crop_ratio_y))
    
    # Ensure target dimensions are within the original image dimensions
    if target_width >= img_width or target_height >= img_height:
        raise ValueError("Target FOVs cannot be greater than or equal to the original FOVs")

    # Calculate cropping box to center the crop area
    left = (img_width - target_width) // 2
    upper = (img_height - target_height) // 2
    right = left + target_width
    lower = upper + target_height
    
    # Crop and return the image with adjusted FOV
    if isinstance(image, Image.Image):
        cropped_image = image.crop((left, upper, right, lower))
    else:
        cropped_image = image[upper:lower, left:right].copy()
    
    return cropped_image


def get_dice_mask(L):
    """
    Args:
        L: cube face length, int
    Returns:
        mask: np.ndarray, (3L, 4L), uint8
    """
    # Height and width of the mask
    H = 3 * L
    W = 4 * L
    
    # Create an empty mask with uint8 type
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # Define the faces of the cubemap (dice)
    face_indices = {
        'top':    (0, L, L, 2*L),
        'left':   (L, 0, 2*L, L),
        'front':  (L, L, 2*L, 2*L),
        'right':  (L, 2*L, 2*L, 3*L),
        'back':   (L, 3*L, 2*L, 4*L),
        'bottom': (2*L, L, 3*L, 2*L)
    }
    
    # Assign values to the mask for each face
    for face_index, (row_start, col_start, row_end, col_end) in enumerate(face_indices.values(), start=1):
        mask[row_start:row_end, col_start:col_end] = face_index
    
    return mask


def concat_dice_mask(dice: np.ndarray | Image.Image) -> Any:
    """
    Args:
        dice: np.ndarray, (H, W, C), uint8 | Image.Image
    Returns:
        dice: np.ndarray, (H, W, C+1), uint8 | Image.Image
    """
    if isinstance(dice, Image.Image):
        from_PIL = True
        dice = np.array(dice)
    else:
        from_PIL = False

    H, W, C = dice.shape

    assert H // 3 == W // 4, f"The input image should be a dice (cubemap) image, but got shape {dice.shape}"
    assert dice.dtype == np.uint8, f"The input image should be a dice (cubemap) image, but got dtype {dice.dtype}"
    
    L = H // 3
    mask = get_dice_mask(L)
    mask[mask > 0] = 255
    
    dice = np.concatenate([dice, mask[:, :, None]], axis=-1)
    if from_PIL:
        dice = Image.fromarray(dice)
    return dice


def images_to_equi_and_dice(
    images: Union[np.ndarray, torch.Tensor],
    equi_height: Optional[int] = None,
    equi_width: Optional[int] = None,
    impl: str = 'crop',
):
    """
    Args:
        images: np.ndarray (B, 6, H, W, C) | torch.Tensor (B, 6, C, H, W)
    Returns:
        equis: np.ndarray, (B, H, W, C)
        dices: np.ndarray, (B, H, W, C)
    """
    if isinstance(images, torch.Tensor):
        images = rearrange(images.cpu().numpy(), '... c h w -> ... h w c')
    
    if equi_height is None:
        equi_height = images.shape[-3] * 2
    
    if equi_width is None:
        equi_width = images.shape[-2] * 4
    
    equis, dices = [], []
    for views in images:
        assert len(views) == 6, "Expected 6 views for cubemap representation"
        if impl == 'pers':
            fovs = np.array([90.0] * 6, dtype=np.float32)
            thetas = np.array([0.0, 90.0, 180.0, 270.0, 0.0, 0.0], dtype=np.float32)
            phis = np.array([0.0, 0.0, 0.0, 0.0, 90.0, -90.0], dtype=np.float32)
            cubemap = Cubemap.from_perspective(views, fovs=fovs, thetas=thetas, phis=phis)
        else:
            cubemap = Cubemap.from_cubediffusion(views, 'list')
        dice = cubemap.cube_all2all(cubemap.faces, cubemap.cube_format, 'dice').astype(images.dtype)
        equi = cubemap.to_equirectangular(h=equi_height, w=equi_width).equirectangular.astype(images.dtype)
        dices.append(dice)
        equis.append(equi)
    
    equis = np.stack(equis, axis=0)  # np.ndarray, (B, H, W, C)
    dices = np.stack(dices, axis=0)  # np.ndarray, (B, H, W, C)

    return equis, dices


class Cubemap:
    cube_fov = [90, 90, 90, 90, 90, 90]
    cube_theta = [0, 90, 180, 270, 0, 0]
    cube_phi = [0, 0, 0, 0, 90, -90]

    def __init__(self, faces, cube_format, as_horizon=True):
        self.faces = faces  # np.ndarray, (H, H*6, 3)
        self.cube_format = cube_format
        if as_horizon:
            self.as_horizon()
    
    @classmethod
    def from_mp3d_skybox(cls, mp3d_skybox_path, scene, view, subfolder='matterport_skybox_images'):
        keys = ['U', 'L', 'F', 'R', 'B', 'D']
        images = {}
        for idx, key in enumerate(keys):
            img_path = os.path.join(mp3d_skybox_path, scene, subfolder, f"{view}_skybox{idx}_sami.jpg")
            images[key] = np.array(Image.open(img_path))
        images['U'] = np.flip(images['U'], 0)
        images['U'] = np.rot90(images['U'], 1)
        images['R'] = np.flip(images['R'], 1)
        images['B'] = np.flip(images['B'], 1)
        images['D'] = np.rot90(images['D'], 1)
        return cls(images, 'dict')

    @classmethod
    def from_cubediffusion(cls, faces, cube_format, from_equilib=True, fov=None):
        """
        Args:
            faces: np.ndarray, (6, H, W, C), uint8
            cube_format: str, 'list', 'dict', 'dice', 'horizon'
        """
        if fov is not None:
            assert cube_format == 'list'
            faces = [crop_to_fov(face, fov, fov, 90, 90) for face in faces]
        faces = Cubemap.cube_all2all_equilib(faces, cube_format, target_cube_format='horizon', from_equilib=from_equilib)
        return cls(faces, 'horizon')
    
    @classmethod
    def from_perspective(cls, images, fovs, thetas, phis, from_equilib=True, keep_res=False):
        """
        Args:
            images: np.ndarray, (M, H, W, C), uint8
            fovs: np.ndarray, (M,), float
            thetas: np.ndarray, (M,), float
            phis: np.ndarray, (M,), float
        """
        cube_fov = cls.cube_fov
        cube_theta = cls.cube_theta
        cube_phi = cls.cube_phi

        _, height, width, _ = images.shape
        fov = fovs[0].item()

        if not keep_res:
            crop_ratio = math.tan(math.radians(90.0 / 2)) / math.tan(math.radians(fov / 2))
            cube_res = int(np.round(height * crop_ratio))
        else:
            cube_res = height
        
        faces = [
            npers2pers(images, fovs, thetas, phis, cube_fov[i], cube_theta[i], cube_phi[i], cube_res, cube_res, overlap=True)
            for i in range(6)
        ]
        
        faces = Cubemap.cube_all2all_equilib(faces, 'list', target_cube_format='horizon', from_equilib=from_equilib)
        return cls(faces, 'horizon')

    @staticmethod
    def cube_all2all(faces, cube_format, target_cube_format='horizon'):
        # already in target format
        if cube_format == target_cube_format:
            return faces
        # all to horizon
        if cube_format == 'list':
            faces = py360convert.cube_list2h(faces)
        elif cube_format == 'dict':
            faces = py360convert.cube_dict2h(faces)
        elif cube_format == 'dice':
            faces = py360convert.cube_dice2h(faces)
        else:
            assert cube_format == 'horizon', f'unknown cube_format: {cube_format}'
        assert faces.ndim == 3, faces.shape  # (H, W, 3)
        assert faces.shape[0] * 6 == faces.shape[1], faces.shape  # (H, W, 3) = (H, 6H, 3)
        # horizon to all
        if target_cube_format == 'list':
            faces = py360convert.cube_h2list(faces)
        elif target_cube_format == 'dict':
            faces = py360convert.cube_h2dict(faces)
        elif target_cube_format == 'dice':
            faces = py360convert.cube_h2dice(faces)
        else:
            assert target_cube_format == 'horizon', f'unknown target_cube_format: {target_cube_format}'
        return faces

    @staticmethod
    def cube_all2all_equilib(faces, cube_format, target_cube_format='horizon', to_equilib=False, from_equilib=False):
        assert not (to_equilib and from_equilib), 'to_equilib and from_equilib cannot be True at the same time'
        faces = Cubemap.cube_all2all(faces, cube_format, target_cube_format='dict')
        new_faces = {}  # avoid inplace modification
        new_faces.update(faces)
        if to_equilib or from_equilib:
            new_faces['R'] = np.flip(new_faces['R'], 1)
            new_faces['B'] = np.flip(new_faces['B'], 1)
            new_faces['U'] = np.flip(new_faces['U'], 0)
        new_faces = Cubemap.cube_all2all(new_faces, cube_format='dict', target_cube_format=target_cube_format)
        return new_faces

    def as_horizon(self) -> None:
        self.faces = self.cube_all2all(self.faces, self.cube_format, target_cube_format='horizon')
        self.cube_format = 'horizon'

    def to_equirectangular(self, h, w, mode='bilinear', backend='equilib'):
        faces = self.faces
        cube_format = self.cube_format
        if backend == 'equilib':
            faces = self.cube_all2all_equilib(faces, cube_format, target_cube_format='horizon', to_equilib=True)
            faces = rearrange(faces, 'h w c -> c h w')
            equirectangular = equilib.cube2equi(faces, 'horizon', h, w, mode)
            equirectangular = rearrange(equirectangular, 'c h w -> h w c')
        elif backend == 'py360convert':
            equirectangular = py360convert.c2e(faces, h, w, mode, cube_format)
            raise AssertionError('Do not use py360convert for cube2equi transformation! It is not accurate!')
        else:
            raise NotImplementedError(f'unknown backend: {backend}')
        from utils.equi import Equirectangular
        return Equirectangular(equirectangular)

    def to_perspective(
        self,
        fov, theta, phi,
        height=None, width=None,
        use_resize=True, scale=None,
        output_format='numpy',
    ) -> np.ndarray:
        """ Converts a cubemap to a perspective image using a specified field of view (FOV) and orientation angles.
        Args:
            - fov: float - Field of view in degrees.
            - theta: float - Rotation angle around the y-axis in degrees.
            - phi: float - Rotation angle around the x-axis in degrees.
            - height: int - Height of the output image (default is None).
            - width: int - Width of the output image (default is None).
            - use_resize: bool - Whether to resize the output image to the specified height and width (default is True).
            - scale: float - Scaling factor for the output image (default is None).
            - mask_input: bool - Whether to mask input pixels or to mask output pixels (default is True).
        Returns:
            - np.ndarray - Perspective image with the specified FOV and orientation angles.
        """
        faces = self.cube_all2all_equilib(self.faces, self.cube_format, target_cube_format='list', to_equilib=True)

        cube_fov = self.cube_fov
        cube_phi = self.cube_phi
        cube_theta = self.cube_theta
        cube_height, cube_width, _ = faces[0].shape

        target_size = (width, height) if (use_resize and height and width) else None

        if height is None or use_resize:
            height = cube_height
        if width is None or use_resize:
            width = cube_width
        
        if scale is not None and use_resize:
            height = int(height * scale)
            width = int(width * scale)
        
        image = npers2pers(faces, cube_fov, cube_theta, cube_phi, fov, theta, phi, height, width, overlap=False)
        
        if use_resize and target_size:
            current_size = (image.shape[1], image.shape[0])
            if current_size != target_size:
                ori_ndim = image.ndim
                image = cv2.resize(image, target_size)
                if image.ndim == 2 and ori_ndim == 3:
                    image = image[:, :, None]
        
        if output_format == 'pil':
            return Image.fromarray(image)
        else:
            assert output_format == 'numpy'
            return image

    def to_perspective_mvdiffusion(self, fov, theta, phi, hw=None, align=True) -> np.ndarray:
        from utils.camera import get_K_R

        # Borrowed from MVDiffusion
        faces = self.cube_all2all(self.faces, self.cube_format, target_cube_format='dict')

        images = []
        for k in ['U', 'L', 'F', 'R', 'B', 'D']:
            image = faces[k]
            if align:
                if k == 'U':
                    image = np.rot90(image, 3)
                    image = np.flip(image, 0)
                elif k == 'R' or k == 'B':
                    image = np.flip(image, 1)
                elif k == 'D':
                    image = np.rot90(image, 3)
            images.append(image)
        
        vx = [-90, 270, 0, 90, 180, -90]
        vy = [90, 0, 0, 0, 0, -90]
        
        img_combine = np.zeros(images[0].shape).astype(np.uint8)
        
        min_theta = 10000
        for i, img in enumerate(images):
            _theta = vx[i] - theta
            _phi = vy[i] - phi

            if i == 2 and theta > 270:
                _theta = max(360 - theta, _theta)
            if _phi == 0 and np.absolute(_theta) > 90:
                continue

            if i > 0 and i < 5 and np.absolute(_theta) < min_theta:
                min_theta = _theta

            im_h, im_w, _ = img.shape
            K, R = get_K_R(fov, _theta, _phi, im_h, im_w)
            homo_matrix = K @ R @ np.linalg.inv(K)
            img_warp1 = cv2.warpPerspective(img, homo_matrix, (im_w, im_h))
            if i == 0:
                img_warp1[im_h//2:] = 0
            elif i == 5:
                img_warp1[:im_h//2] = 0

            img_combine += img_warp1
        
        if hw is not None:
            img_combine = cv2.resize(img_combine, (hw[1], hw[0]))
        return img_combine

    def to_PIL_image(self, cube_format='horizon') -> Image.Image:
        assert cube_format in ('horizon', 'dice'), f'Unsupported cube_format {cube_format} for PIL.Image.Image'
        faces = self.cube_all2all(self.faces, self.cube_format, target_cube_format=cube_format)
        faces = faces.astype(np.uint8)
        if faces.shape[-1] == 1:
            faces = faces.squeeze(-1)
        return Image.fromarray(faces)

    def flip(self, flip=True) -> None:
        if flip:
            faces = self.cube_all2all(self.faces, cube_format=self.cube_format, target_cube_format='dict')
            for k in ['F', 'B', 'U', 'D']:
                faces[k] = np.flip(faces[k], 1)
            faces['R'], faces['L'] = faces['L'], faces['R']
            self.faces = faces
            self.cube_format = 'dict'
            # self.as_horizon()

    def save(self, save_path, cube_format='horizon'):
        self.to_PIL_image(cube_format).save(save_path)

    def save_as_video(self, save_path, seconds: int = 8, fps: int = 25, fov: float = 90.0, phi: float = 0.0, height: int = 512, width: int = 512):
        from utils.video import VideoWriterPyAV
        num_frames = seconds * fps
        video_writer = VideoWriterPyAV(save_path, fps=fps, kbitrate=5_000)
        for i in range(num_frames):
            theta = i * 360 / num_frames
            frame = self.to_perspective(fov=fov, theta=theta, phi=phi, height=height, width=width, output_format='pil')
            video_writer.write(frame.convert('RGB'))
        video_writer.release()

    def show(self, cube_format='dice') -> None:
        self.to_PIL_image(cube_format).show()


if __name__ == '__main__':
    import argparse
    from utils.camera import icosahedron_sample_camera, horizon_sample_camera, random_sample_camera

    parser = argparse.ArgumentParser(description="Stitch Matterport3D Skybox")
    parser.add_argument("--mp3d_skybox_path", required=True,
                        help="Matterport3D mp3d_skybox path")
    parser.add_argument("--label_data_path", required=True,
                        help="Matterport3DLayoutAnnotation label_data path")
    parser.add_argument("--scene", default=None,
                        help="scene id", type=str)
    parser.add_argument("--view", default=None,
                        help="view id", type=str)
    args = parser.parse_args()

    cubemap = Cubemap.from_mp3d_skybox(args.mp3d_skybox_path, args.scene, args.view)
    equirectangular = cubemap.to_equirectangular(1024, 2048)
    equirectangular.save(os.path.join('debug', 'equirectangular.jpg'))

    theta, phi = icosahedron_sample_camera()
    theta, phi = np.rad2deg(theta), np.rad2deg(phi)
    for i, (t, p) in enumerate(zip(theta, phi)):
        perspective = equirectangular.to_perspective((90, 90), t, p, (512, 512))
        Image.fromarray(perspective.astype(np.uint8)).save(os.path.join('debug', f"icosahedron_{i}.jpg"))

    theta, phi = horizon_sample_camera(8)
    theta, phi = np.rad2deg(theta), np.rad2deg(phi)
    for i, (t, p) in enumerate(zip(theta, phi)):
        perspective = equirectangular.to_perspective((90, 90), t, p, (512, 512))
        Image.fromarray(perspective.astype(np.uint8)).save(os.path.join('debug', f"horizon_{i}.jpg"))

    theta, phi = random_sample_camera(20)
    theta, phi = np.rad2deg(theta), np.rad2deg(phi)
    for i, (t, p) in enumerate(zip(theta, phi)):
        perspective = equirectangular.to_perspective((90, 90), t, p, (512, 512))
        Image.fromarray(perspective.astype(np.uint8)).save(os.path.join('debug', f"random_{i}.jpg"))

    layout_json = os.path.join(args.label_data_path, f"{args.scene}_{args.view}_label.json")
    layout = Layout.from_json(layout_json)
    layout.render_to_files('debug')
