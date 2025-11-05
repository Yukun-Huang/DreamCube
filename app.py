import os
import os.path as osp
import argparse
from PIL import Image
from typing import Optional, List, Dict, Union, Any
import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from einops import repeat, rearrange

from models.dreamcube import DreamCubeDepthPipeline
from models.multiplane_sync_legacy import apply_custom_processors_for_vae, apply_custom_processors_for_unet

from utils.cube import Cubemap
from utils.depth import z_distance_to_depth, depth_to_z_distance, DepthVisualizer
from utils.pano_to_3d import convert_rgbd_equi_to_3dgs, convert_rgbd_equi_to_mesh, convert_rgbd_cube_to_3dgs, convert_rgbd_cube_to_mesh

PANO_TO_3D_MODES = ('3D from RGB-D Cubemap', '3D from RGB-D Equirectangular')


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


def postprocess_rgb(images: np.ndarray) -> dict:
    """
    Postprocess the generated images.
    Args:
        images: (bs, m, h, w, 3), np.uint8
    Returns:
        dict: postprocessed images in different formats
    """
    outputs = {}
    equis, dices = images_to_equi_and_dice(images)
    outputs['equi'] = equis  # (bs, h, w, 3), np.uint8
    outputs['dice'] = dices  # (bs, h, w, 3), np.uint8
    return outputs


def postprocess_depth(depths: np.ndarray) -> dict:
    """
    Postprocess the generated images.
    Args:
        depths: (bs, m, h, w, 1), np.float32
    Returns:
        dict: postprocessed depths in different formats
    """
    val_min, val_max = depths.min() - 1e-8, depths.max() + 1e-8
    
    depths = rearrange(depths, 'b m h w c -> (b m) h w c')
    depths_vis = DepthVisualizer.visualize_depth(depths, val_min, val_max)
    depths_vis_np = rearrange(np.array(depths_vis), '(b m) h w c -> b m h w c', m=6)
    # depths_16bit = DepthVisualizer.export_depth_to_16bit_png(depths, val_min, val_max)
    # depths_16bit_np = rearrange(np.array(depths_16bit)[..., None], '(b m) h w c -> b m h w c', m=6)

    equis_depth_vis, dices_depth_vis = images_to_equi_and_dice(depths_vis_np)
    # depths_16bit_np = depths_16bit_np.astype(np.float64)
    # equis_depth_16bit, dices_depth_16bit = images_to_equi_and_dice(depths_16bit_np)
    # equis_depth_16bit = equis_depth_16bit[:, :, :, 0].clip(0, 65535).astype(np.uint16)
    # dices_depth_16bit = dices_depth_16bit[:, :, :, 0].clip(0, 65535).astype(np.uint16)

    depths_raw = rearrange(np.array(depths), '(b m) h w c -> b m h w c', m=6)
    pano_depth_raw, _ = images_to_equi_and_dice(depths_raw)
    pano_depth_raw = pano_depth_raw[:, :, :, 0].clip(0, 65535).astype(np.uint16)

    return {
        'equi_depth_vis': equis_depth_vis,
        'dice_depth_vis': dices_depth_vis,
        # 'pano_depth_16bit': equis_depth_16bit,
        # 'dice_depth_16bit': dices_depth_16bit,
        'equi_depth_raw': pano_depth_raw,
    }


def get_examples(example_dir: str = './assets') -> list:
    examples = []

    for folder in sorted(os.listdir(example_dir)):
        if not folder.startswith('example'):
            continue
        
        image_path = osp.join(example_dir, folder, 'input_image.png')
        depth_path = osp.join(example_dir, folder, 'input_depth.png')
        prompt_path = osp.join(example_dir, folder, 'input_prompts.txt')

        if not osp.exists(image_path) or not osp.exists(depth_path):
            raise FileNotFoundError(f"Image or depth file not found in {folder}: {image_path}, {depth_path}")
        if not osp.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found in {folder}: {prompt_path}")

        with open(prompt_path, 'r') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        
        if len(prompts) != 6:
            raise ValueError(f"Expected 6 prompts, got {len(prompts)} in {prompt_path}")
        
        default_mode= PANO_TO_3D_MODES[0]

        examples.append(prompts + [image_path, depth_path] + [default_mode])

    return examples


def build_pipeline(
    ckpt_path: str = 'KevinHuang/DreamCube',
    device: Optional[torch.device] = None,
    local_files_only: bool = False,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = DreamCubeDepthPipeline.from_pretrained(ckpt_path, local_files_only=local_files_only)
    
    apply_custom_processors_for_unet(
            pipe.unet,
            enable_sync_self_attn=True,
            enable_sync_cross_attn=False,
            enable_sync_gn=True,
            enable_sync_conv2d=True,
            cube_padding_impl='cuda',
        )

    apply_custom_processors_for_vae(
            pipe.vae,
            mode='all',
            enable_sync_gn=True,
            enable_sync_conv2d=True,
            enable_sync_attn=True,
            cube_padding_impl='cuda',
        )

    pipe = pipe.to(device)
    
    return pipe


def prepare_inputs(
    image: Union[Image.Image, str],
    depth: Union[Image.Image, str],
    prompts: List[str],
    device: torch.device,
    height: int = 512,
    width: int = 512,
    pers_prompt_prefix: str = 'This is one view of a scene. {pers_prompt}',
    pers_up_prompt_prefix: str = 'This a upward view of a scene. {pers_prompt}',
    pers_down_prompt_prefix: str = 'This a downward view of a scene. {pers_prompt}',
):
    # Image
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    if image.size[0] != image.size[1]:
        image.crop((0, 0, min(image.size), min(image.size)))
    if image.size[0] != width or image.size[1] != height:
        image = image.resize((width, height))
    image = np.asarray(image, dtype=np.float32) / 127.5 - 1

    # Depth
    if isinstance(depth, str):
        depth = Image.open(depth)
    if depth.size[0] != depth.size[1]:
        depth.crop((0, 0, min(depth.size), min(depth.size)))
    if depth.size[0] != width or depth.size[1] != height:
        depth = depth.resize((width, height))
    depth = np.asarray(depth, dtype=np.float32)

    # To Z-Axis Depth
    depth = rearrange(depth, 'h w -> 1 h w 1')
    depth = depth_to_z_distance(depth, fov_x=90.0,fov_y=90.0)
    depth = rearrange(depth, '1 h w 1 -> h w')
    
    # To Tensor
    cube_rgbs = repeat(torch.from_numpy(image).to(device), 'h w c -> 1 m c h w', m=6)
    cube_depths = repeat(torch.from_numpy(depth).to(device), 'h w -> 1 m 1 h w', m=6)
    
    # Mask
    cube_masks = torch.ones_like(cube_depths, dtype=bool)
    cube_masks[:, 0, :, :, :] = False

    # Prompt
    cube_prompts = []
    for i in range(len(prompts)):
        if i in (0, 1, 2, 3):
            cube_prompts.append(pers_prompt_prefix.format(pers_prompt=prompts[i]))
        elif i in (4,):
            cube_prompts.append(pers_up_prompt_prefix.format(pers_prompt=prompts[i]))
        elif i in (5,):
            cube_prompts.append(pers_down_prompt_prefix.format(pers_prompt=prompts[i]))
        else:
            raise ValueError(f"Unknown prompt index: {i}")

    return cube_rgbs, cube_depths, cube_masks, cube_prompts


def inference(
    pipe: Any,
    image: Union[Image.Image, str],
    depth: Union[Image.Image, str],
    prompts: List[str],
    height: int = 512,
    width: int = 512,
    output_type='np',
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    normalize_scale: float = 0.6,
    **kwargs,
):
    """
    Inference function for the DreamCube model.
    Args:
        cube_rgbs (torch.Tensor): Input RGB images of shape (B, M, C, H, W).
        cube_depths (torch.Tensor): Input depth maps of shape (B, M, 1, H, W).
        cube_masks (torch.Tensor): Input masks of shape (B, M, 1, H, W).
        prompts (List[str]): List of prompts for the model.
        height (int): Height of the output image.
        width (int): Width of the output image.
        output_type (str): Type of output ('np' for numpy arrays).
        num_inference_steps (int): Number of inference steps.
        guidance_scale (float): Guidance scale for the model.
        normalize_scale (float): Scale for normalization.
        **kwargs: Additional keyword arguments.
    """
    # pre-process
    cube_rgbs, cube_depths, cube_masks, cube_prompts = prepare_inputs(
        image=image,
        depth=depth,
        prompts=prompts,
        device=pipe.device,
        height=height,
        width=width,
    )

    print(cube_prompts)
    
    # inference
    prediction = pipe(
        cube_rgbs=cube_rgbs,
        cube_depths=cube_depths,
        cube_masks=cube_masks,
        prompt=cube_prompts,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        output_type=output_type,
        normalize_scale=normalize_scale,
        **kwargs,
    )

    images_pred, depths_pred = prediction.images, prediction.depths

    # post-process
    if output_type == 'np':
        images_pred = (images_pred * 255).round().astype("uint8")
        images_pred = rearrange(images_pred, '(b m) ... -> b m ...', m=6)  # np.ndarray, (B, M, H, W, C), uint8
        depths_pred = rearrange(depths_pred, '(b m) ... -> b m ...', m=6)  # np.ndarray, (B, M, H, W, C), float32

    return {
        'images': images_pred,
        'depths': depths_pred,
        'normals': None,
    }


def run_example(
    prompt_front: str,
    prompt_right: str,
    prompt_back: str,
    prompt_left: str,
    prompt_top: str,
    prompt_bottom: str,
    image: Union[Image.Image, str],
    depth: Union[Image.Image, str],
    mode: str,
    save_dir: str = './outputs/gradio',
    max_equi_size: Optional[int] = 1024,
    max_cube_size: Optional[int] = 256,
    save_all: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)

    prompts = [prompt_front, prompt_right, prompt_back, prompt_left, prompt_top, prompt_bottom]
    
    assert mode in PANO_TO_3D_MODES, f"Mode must be one of {PANO_TO_3D_MODES}, got {mode}"
    
    with torch.inference_mode():
        with autocast('cuda'):
            predictions = inference(
                pipe,
                image=image,
                depth=depth,
                prompts=prompts,
            )

    images_pred = predictions['images']  # np.ndarray, (B, M, H, W, C), uint8
    depths_pred = predictions['depths']  # np.ndarray, (B, M, H, W, 1), float32
    depths_pred = z_distance_to_depth(depths_pred, fov_x=90.0, fov_y=90.0)

    # RGB Visualization
    if images_pred is not None:
        assert images_pred.shape[0] == 1, "Batch size must be 1 for inference"
        postprocessed = postprocess_rgb(images_pred)
        equi_rgb = Image.fromarray(postprocessed['equi'][0])
        dice_rgb = Image.fromarray(postprocessed['dice'][0])
        equi_rgb.save(osp.join(save_dir, 'output_equi_rgb.png'))
        if save_all:
            dice_rgb.save(osp.join(save_dir, 'output_dice_rgb.png'))

    # Depth Visualization
    if depths_pred is not None:
        assert depths_pred.shape[0] == 1, "Batch size must be 1 for inference"
        postprocessed_depth = postprocess_depth(depths_pred)
        equi_depth_raw = Image.fromarray(postprocessed_depth['equi_depth_raw'][0])
        equi_depth_vis = Image.fromarray(postprocessed_depth['equi_depth_vis'][0])
        dice_depth_vis = Image.fromarray(postprocessed_depth['dice_depth_vis'][0])
        
        equi_depth_raw.save(osp.join(save_dir, 'output_equi_depth.png'))
        equi_depth_vis.save(osp.join(save_dir, 'output_equi_depth_vis.png'))
        if save_all:
            dice_depth_vis.save(osp.join(save_dir, 'output_dice_depth_vis.png'))
    
    # 3D Reconstuction
    mesh_path = osp.join(save_dir, 'output_mesh.obj')
    splat_path = osp.join(save_dir, 'output_3dgs.splat')

    device = pipe.device

    if mode == '3D from RGB-D Equirectangular':
        rgb = torch.tensor(np.array(equi_rgb), device=device)
        distance = torch.tensor(np.array(equi_depth_raw), device=device) / 1000.0  # Convert mm to m
        convert_rgbd_equi_to_mesh(
            rgb=rgb,
            distance=distance,
            max_size=max_equi_size,
            save_path=mesh_path,
        )
        convert_rgbd_equi_to_3dgs(
            rgb=rgb,
            distance=distance,
            max_size=max_equi_size,
            save_path=splat_path,
        )
    elif mode == '3D from RGB-D Cubemap':
        rgb = torch.tensor(images_pred[0], device=device, dtype=torch.float32)
        distance = torch.tensor(depths_pred[0, ..., 0], device=device, dtype=torch.float32) / 1000.0  # Convert mm to m
        convert_rgbd_cube_to_mesh(
            rgb=rgb,
            distance=distance,
            max_size=max_cube_size,
            save_path=mesh_path,
        )
        convert_rgbd_cube_to_3dgs(
            rgb=rgb,
            distance=distance,
            max_size=max_cube_size,
            save_path=splat_path,
        )
    
    dice_outputs = [dice_rgb, dice_depth_vis]
    equi_outputs = [equi_rgb, equi_depth_vis]
    scene_outputs = [mesh_path, splat_path]

    return dice_outputs + equi_outputs + scene_outputs


def create_gradio_demo():
    import gradio as gr

    input_prompts = [gr.Textbox(label=f"Text Prompt ({view} View)") for view in ["Front", "Right", "Back", "Left", "Up", "Down"]]
    input_image = [gr.Image(type="filepath", image_mode="RGB", height=512, width=512, label="Input RGB Image (Front View)")]
    input_depth = [gr.Image(type="filepath", image_mode=None, height=512, width=512, label="Input Depth Image (Front View)")]
    input_dropdown = [gr.Dropdown(choices=PANO_TO_3D_MODES, label='Pano-to-3D Mode')]

    output_cubemaps = [
        gr.Image(label="Generated Cubemap (RGB)"),
        gr.Image(label="Generated Cubemap (Euclidean Depth)"),
    ]
    
    output_equirects = [
        gr.Image(label="Converted Equirectangular (RGB)"),
        gr.Image(label="Converted Equirectangular (Euclidean Depth)"),
    ]

    camera_position = (0, 90, 0.4)
    output_3d_models = [
        gr.Model3D(
            display_mode="wireframe",
            camera_position=camera_position,
            label="Reconstructed 3D Scene (Mesh)",
            pan_speed=5.0,
            height=512,
        ),

        gr.Model3D(
            label="Reconstructed 3D Scene (3DGS)",
            height=512,
        )
    ]

    examples = get_examples()

    title = "DreamCube Gradio Demo"
    description = "This demo generates RGB-D cubemaps and 3D scenes (both mesh and 3dgs) from front-view RGB-D inputs. " \
        "Please see our <a href='https://yukun-huang.github.io/DreamCube/' target='_blank'>project page</a> for more details.<br>" \
        "Note that the 3D scenes are reconstructed from downsampled RGB-D panoramas, which enables faster rendering but lower visual quality."

    return gr.Interface(
        fn=run_example,
        inputs=input_prompts + input_image + input_depth + input_dropdown,
        outputs=output_cubemaps + output_equirects + output_3d_models,
        examples=examples,
        title=title,
        description=description,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DreamCube Online Demo")
    parser.add_argument('--use-gradio', action='store_true', help='Use Gradio for the demo')
    args = parser.parse_args()

    print('Building DreamCube model...')
    pipe = build_pipeline('KevinHuang/DreamCube', local_files_only=False)
    print('DreamCube model built successfully!')

    if args.use_gradio:
        demo = create_gradio_demo()
        demo.queue(max_size=1)
        demo.launch(server_name="0.0.0.0", server_port=7422)
    else:
        examples = get_examples()
        for i, example in enumerate(examples):
            outputs = run_example(
                *example,
                save_dir=f'./outputs/example_{i+1}',
                save_all=True,
                max_equi_size=None,
                max_cube_size=None,
            )
