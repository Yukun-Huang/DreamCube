import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from einops import rearrange, repeat
from dataclasses import dataclass

from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDPMScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps, rescale_noise_cfg, CLIPTextModel, XLA_AVAILABLE
)
from diffusers.pipelines.stable_diffusion.pipeline_output import BaseOutput
from diffusers.utils import deprecate

from utils.camera import skybox_sample_camera
from utils.depth import AbsoluteDepthScaler
from utils.pers import prepare_positions
from models.multiplane_sync.processors_sd import switch_custom_processors_for_vae


@dataclass
class DreamCubePipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        depths (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        nsfw_content_detected (`List[bool]`)
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray, None] = None
    depths: Union[List[PIL.Image.Image], np.ndarray, None] = None
    normals: Union[List[PIL.Image.Image], np.ndarray, None] = None
    nsfw_content_detected: Optional[List[bool]] = None


class DreamCubeVAEMixin():
    def pack(self, inputs: torch.Tensor):
        shape = inputs.shape
        dtype = inputs.dtype
        if len(shape) != 4:
            inputs = inputs.reshape(-1, *shape[-3:])
        return inputs.to(self.vae.dtype), {'shape': shape, 'dtype': dtype}
    
    def unpack(self, outputs: torch.Tensor, shape: torch.Size, dtype: torch.dtype):
        if len(shape) != 4:
            outputs = outputs.reshape(*shape[:-3], *outputs.shape[-3:])
        return outputs.to(dtype)

    def vae_encode(self, images: torch.Tensor):
        latents = self.vae.encode(images)
        latents = latents.latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents
    
    def vae_decode(self, latents: torch.Tensor):
        latents = (1 / self.vae.config.scaling_factor * latents)
        images = self.vae.decode(latents, return_dict=True).sample
        return images

    def encode_cube_images(self, images: torch.Tensor):
        images, inputs_info = self.pack(images)
        latents = self.vae_encode(images)
        latents = self.unpack(latents, **inputs_info)
        return latents

    def decode_cube_latents(self, latents: torch.Tensor):
        latents, inputs_info = self.pack(latents)
        images = self.vae_decode(latents)
        images = self.unpack(images, **inputs_info)
        return images


class DreamCubeDepthPipeline(StableDiffusionPipeline, DreamCubeVAEMixin):

    def prepare_cube_latents(
        self, cube_rgbs, cube_depths, cube_masks,
        dtype, device, normalize_scale, is_training=False,
        position_type: str = 'xyz', use_geo_cond=True,
    ):
        bs, m, _, h, w = cube_rgbs.shape

        if not is_training:
            switch_custom_processors_for_vae(self.vae, enable_sync_gn=False, enable_sync_conv2d=False, enable_sync_attn=False)
        
        cube_depths = cube_depths.to(dtype)
        cube_visible_masks = cube_masks.logical_not()  # True: valid, False: invalid
        assert cube_visible_masks.any(), 'At least one pixel should be valid.'
        cube_depths, min_value, max_value = AbsoluteDepthScaler.normalize(cube_depths, scale=normalize_scale, mask_for_minmax=cube_visible_masks)

        cube_rgbs = rearrange(cube_rgbs, 'b m c h w -> (b m) c h w')
        cube_depths = rearrange(cube_depths, 'b m 1 h w -> (b m) 1 h w')
        cube_masks = rearrange(cube_masks, 'b m 1 h w -> (b m) 1 h w')
        min_value = repeat(min_value, 'b -> (b m)', m=m)
        max_value = repeat(max_value, 'b -> (b m)', m=m)

        cube_masked_rgbs = cube_rgbs.clone()
        if not is_training:
            cube_masked_rgbs[cube_masks.expand(-1, 3, -1, -1)] = 0.0
        masked_rgb_latents = self.encode_cube_images(cube_masked_rgbs)  # [B*M, C, H, W]

        if is_training:
            cube_depths = repeat(cube_depths, 'bm 1 h w -> bm 3 h w')
            depth_latents = self.encode_cube_images(cube_depths)  # [B*M, C, H, W]
        else:
            if use_geo_cond:
                cube_masked_depths = cube_depths.clone()
                cube_masked_depths[cube_masks] = 0.0
                cube_masked_depths = repeat(cube_masked_depths, 'bm 1 h w -> bm 3 h w')
                depth_latents = self.encode_cube_images(cube_masked_depths)  # [B*M, C, H, W]
            else:
                depth_latents = torch.zeros_like(masked_rgb_latents)

        h_latent, w_latent = masked_rgb_latents.shape[-2:]
        latent_masks = F.interpolate(cube_masks.to(dtype), (h_latent, w_latent), mode='nearest')
        # latent_masks = (latent_masks > 0.5).expand(-1, 4, -1, -1)

        mask_latents = latent_masks.clone()  # [B*M, 1, H, W]
        # mask_latents = cube_masks.to(dtype) * 2 - 1
        # mask_latents = repeat(mask_latents, 'bm 1 h w -> bm 3 h w')
        # mask_latents = self.encode_cube_images(mask_latents)  # [B*M, C, H, W]

        if not is_training:
            switch_custom_processors_for_vae(self.vae, enable_sync_gn=True, enable_sync_conv2d=True, enable_sync_attn=True)
        
        cameras = {}
        theta, phi = skybox_sample_camera(degree=True)
        cameras['fov'] = repeat(cube_rgbs.new_tensor([90.0] * 6), 'm -> b m', b=bs, m=m)
        cameras['theta'] = repeat(cube_rgbs.new_tensor(theta), 'm -> b m', b=bs, m=m)
        cameras['phi'] = repeat(cube_rgbs.new_tensor(phi), 'm -> b m', b=bs, m=m)

        positions = prepare_positions(
            height=masked_rgb_latents.shape[-2],
            width=masked_rgb_latents.shape[-1],
            cameras=cameras,
            dtype=dtype,
            device=device,
            output_type=position_type,
        )
        positions = repeat(positions, 'm c h w -> (bs m) c h w', bs=bs, m=m)

        if is_training:
            return masked_rgb_latents, depth_latents, mask_latents, positions, latent_masks
        else:
            return masked_rgb_latents, depth_latents, mask_latents, positions, latent_masks, min_value, max_value

    @torch.no_grad()
    def __call__(
        self,
        cube_rgbs: Optional[torch.FloatTensor],
        cube_depths: Optional[torch.FloatTensor],
        cube_masks: Optional[torch.FloatTensor],
        normalize_scale: float = 0.6,
        output_condition_source: str = 'gt',
        position_type: str = 'xyz',
        # ------------------------
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. [Modified] Prepare latent variables
        """
            cube_rgbs: [B, M, 3, H, W], torch.FloatTensor, [-1.0, 1.0]
            cube_depths: [B, M, 1, H, W], torch.FloatTensor, absolute depth
            cube_normals: [B, M, 3, H, W], torch.FloatTensor, [-1.0, 1.0]
            cube_masks: [B, M, 1, H, W], torch.BoolTensor, [0, 1], False: available, True: unavailable

            latent_inputs:
                masked_rgbs: 4
                noisy_rgbs: 4
                masked_depths: 4
                noisy_depths: 4
                masks: 4
                positions: 3
            total: 23
        """
        dtype = self.unet.dtype
        # dtype = cube_rgbs.dtype
        bs, m, _, _, _ = cube_rgbs.shape

        masked_rgb_latents, masked_depth_latents, mask_latents, positions, latent_masks, min_value, max_value = \
            self.prepare_cube_latents(
                cube_rgbs, cube_depths, cube_masks, dtype, device, normalize_scale, is_training=False,
                position_type=position_type,
            )
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.unet.config.out_channels,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )

        noisy_rgb_latents = latent_masks * latents[:, 0:4, :, :] + (1 - latent_masks) * masked_rgb_latents
        noisy_depth_latents = latent_masks * latents[:, 4:8, :, :] + (1 - latent_masks) * masked_depth_latents
        latents = torch.cat([noisy_rgb_latents, noisy_depth_latents], dim=1)

        if self.do_classifier_free_guidance:
            mask_latents = torch.cat([mask_latents] * 2) 
            positions = torch.cat([positions] * 2)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # concat condition inputs
                latents[:, 0:4, :, :] = latent_masks * latents[:, 0:4, :, :] + (1 - latent_masks) * masked_rgb_latents
                latents[:, 4:8, :, :] = latent_masks * latents[:, 4:8, :, :] + (1 - latent_masks) * masked_depth_latents

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # concat extra inputs
                latent_model_input = torch.cat([latent_model_input, mask_latents, positions], dim=1)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()
        
        images, depths = latents[:, 0:4], latents[:, 4:8]
        if output_condition_source == 'gt':
            images = latent_masks * images + (1 - latent_masks) * masked_rgb_latents
            depths = latent_masks * depths + (1 - latent_masks) * masked_depth_latents
        else:
            assert output_condition_source == 'pred', f'Invalid output_condition_source: {output_condition_source}'

        if not output_type == "latent":
            images = self.vae.decode(images / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            images, has_nsfw_concept = self.run_safety_checker(images, device, prompt_embeds.dtype)
            depths = self.vae.decode(depths / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            depths = depths.mean(dim=1, keepdim=True)
        else:
            has_nsfw_concept = None
        
        if not output_type == "latent":
        
            if has_nsfw_concept is None:
                do_denormalize = [True] * images.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            images = self.image_processor.postprocess(images, output_type=output_type, do_denormalize=do_denormalize)

            # min_value = torch.zeros_like(min_value)
            # max_value = torch.ones_like(max_value)
            depths = AbsoluteDepthScaler.denormalize(depths, normalize_scale, min_value, max_value)
            if output_type == "np":
                depths = rearrange(depths.float().cpu().numpy(), '(b m) 1 h w -> (b m) h w 1', b=bs, m=m)
            elif output_type == "pil":
                depths = self.marigold_image_processor.export_depth_to_16bit_png(
                    depths, val_min=depths.min().item(), val_max=depths.max().item() + 1e-6)
            else:
                assert output_type == "pt"
            
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (images, depths, None, has_nsfw_concept)

        return DreamCubePipelineOutput(images=images, depths=depths, normals=None, nsfw_content_detected=has_nsfw_concept)
