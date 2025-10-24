from typing import Optional, Union
import gc

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
    StableVideoDiffusionPipeline,
)
from diffusers.utils import logging
from kornia.utils import create_meshgrid
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@torch.no_grad()
def normalize_point_map(point_map, valid_mask):
    # T,H,W,3 T,H,W
    norm_factor = (point_map[..., 2] * valid_mask.float()).mean() / (valid_mask.float().mean() + 1e-8)
    norm_factor = norm_factor.clip(min=1e-3)
    return point_map / norm_factor

def point_map_xy2intrinsic_map(point_map_xy):
    # *,h,w,2
    height, width = point_map_xy.shape[-3], point_map_xy.shape[-2]
    assert height % 2 == 0
    assert width % 2 == 0
    mesh_grid = create_meshgrid(
        height=height,
        width=width,
        normalized_coordinates=True,
        device=point_map_xy.device,
        dtype=point_map_xy.dtype
    )[0] # h,w,2
    assert mesh_grid.abs().min() > 1e-4
    # *,h,w,2
    mesh_grid = mesh_grid.expand_as(point_map_xy)
    nc = point_map_xy.mean(dim=-2).mean(dim=-2) # *, 2
    nc_map = nc[..., None, None, :].expand_as(point_map_xy)
    nf = ((point_map_xy - nc_map) / mesh_grid).mean(dim=-2).mean(dim=-2)
    nf_map = nf[..., None, None, :].expand_as(point_map_xy)
    # print((mesh_grid * nf_map + nc_map - point_map_xy).abs().max())

    return torch.cat([nc_map, nf_map], dim=-1)

def robust_min_max(tensor, quantile=0.99):
    T, H, W = tensor.shape
    min_vals = []
    max_vals = []
    for i in range(T):
        min_vals.append(torch.quantile(tensor[i], q=1-quantile, interpolation='nearest').item())
        max_vals.append(torch.quantile(tensor[i], q=quantile, interpolation='nearest').item())
    return min(min_vals), max(max_vals) 

class GeometryCrafterDetermPipeline(StableVideoDiffusionPipeline):

    @torch.inference_mode()
    def encode_video(
        self,
        video: torch.Tensor,
        chunk_size: int = 14,
    ) -> torch.Tensor:
        """
        :param video: [b, c, h, w] in range [-1, 1], the b may contain multiple videos or frames
        :param chunk_size: the chunk size to encode video
        :return: image_embeddings in shape of [b, 1024]
        """

        video_224 = _resize_with_antialiasing(video.float(), (224, 224))
        video_224 = (video_224 + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        embeddings = []
        for i in range(0, video_224.shape[0], chunk_size):
            emb = self.feature_extractor(
                images=video_224[i : i + chunk_size],
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values.to(video.device, dtype=video.dtype)
            embeddings.append(self.image_encoder(emb).image_embeds)  # [b, 1024]

        embeddings = torch.cat(embeddings, dim=0)  # [t, 1024]
        return embeddings

    @torch.inference_mode()
    def encode_vae_video(
        self,
        video: torch.Tensor,
        chunk_size: int = 14,
    ):
        """
        :param video: [b, c, h, w] in range [-1, 1], the b may contain multiple videos or frames
        :param chunk_size: the chunk size to encode video
        :return: vae latents in shape of [b, c, h, w]
        """
        video_latents = []
        for i in range(0, video.shape[0], chunk_size):
            video_latents.append(
                self.vae.encode(video[i : i + chunk_size]).latent_dist.mode()
            )
        video_latents = torch.cat(video_latents, dim=0)
        return video_latents

    
    @torch.inference_mode()
    def produce_priors(self, prior_model, frame, chunk_size=8, low_memory_usage=False):
        T, _, H, W = frame.shape 
        # frame = (frame + 1) / 2
        pred_point_maps = []
        pred_masks = []
        for i in range(0, len(frame), chunk_size):
            pred_p, pred_m = prior_model.forward_image(
                frame[i:i+chunk_size].to(self._execution_device) if low_memory_usage else frame[i:i+chunk_size]
            )
            pred_point_maps.append(pred_p.cpu() if low_memory_usage else pred_p)
            pred_masks.append(pred_m.cpu() if low_memory_usage else pred_m)
        pred_point_maps = torch.cat(pred_point_maps, dim=0)
        pred_masks = torch.cat(pred_masks, dim=0)
        
        pred_masks = pred_masks.float() * 2 - 1
        
        # T,H,W,3 T,H,W
        pred_point_maps = normalize_point_map(pred_point_maps, pred_masks > 0)

        pred_disps = 1.0 / pred_point_maps[..., 2].clamp_min(1e-3)
        pred_disps = pred_disps * (pred_masks > 0)
        min_disparity, max_disparity = robust_min_max(pred_disps)
        pred_disps = ((pred_disps - min_disparity) / (max_disparity - min_disparity+1e-4)).clamp(0, 1)
        pred_disps = pred_disps * 2 - 1

        pred_point_maps[..., :2] = pred_point_maps[..., :2] / (pred_point_maps[..., 2:3] + 1e-7)
        pred_point_maps[..., 2] = torch.log(pred_point_maps[..., 2] + 1e-7) * (pred_masks > 0) # [x/z, y/z, log(z)]

        pred_intr_maps = point_map_xy2intrinsic_map(pred_point_maps[..., :2]).permute(0,3,1,2) # T,H,W,2      
        pred_point_maps = pred_point_maps.permute(0,3,1,2)
        
        return pred_disps, pred_masks, pred_point_maps, pred_intr_maps
    
    @torch.inference_mode()
    def encode_point_map(self, point_map_vae, disparity, valid_mask, point_map, intrinsic_map, chunk_size=8, low_memory_usage=False):
        T, _, H, W = point_map.shape
        latents = []

        psedo_image = disparity[:, None].repeat(1,3,1,1)
        intrinsic_map = torch.norm(intrinsic_map[:, 2:4], p=2, dim=1, keepdim=False)

        for i in range(0, T, chunk_size):
            latent_dist = self.vae.encode(
                psedo_image[i : i + chunk_size].to(dtype=self.vae.dtype, device=self._execution_device) if low_memory_usage else psedo_image[i : i + chunk_size].to(self.vae.dtype)
            ).latent_dist
            latent_dist = point_map_vae.encode(       
                torch.cat([
                    intrinsic_map[i:i+chunk_size, None].to(self._execution_device) if low_memory_usage else intrinsic_map[i:i+chunk_size, None],
                    point_map[i:i+chunk_size, 2:3].to(self._execution_device) if low_memory_usage else point_map[i:i+chunk_size, 2:3], 
                    disparity[i:i+chunk_size, None].to(self._execution_device) if low_memory_usage else disparity[i:i+chunk_size, None], 
                    valid_mask[i:i+chunk_size, None].to(self._execution_device) if low_memory_usage else valid_mask[i:i+chunk_size, None], 
                ], dim=1),
                latent_dist
            )
            if isinstance(latent_dist, DiagonalGaussianDistribution):
                latent = latent_dist.mode()
            else:
                latent = latent_dist
            
            assert isinstance(latent, torch.Tensor)    
            latents.append(latent)
        latents = torch.cat(latents, dim=0)
        latents = latents * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def decode_point_map(
        self, 
        point_map_vae, 
        latents, 
        chunk_size=8, 
        force_projection=True, 
        force_fixed_focal=True, 
        use_extract_interp=False, 
        need_resize=False, 
        height=None, 
        width=None, 
        low_memory_usage=False
    ):
        T = latents.shape[0]
        rec_intrinsic_maps = []
        rec_depth_maps = []
        rec_valid_masks = []
        for i in range(0, T, chunk_size):
            lat = latents[i:i+chunk_size] 
            rec_imap, rec_dmap, rec_vmask = point_map_vae.decode(  
                lat,           
                num_frames=lat.shape[0],
            )
            rec_intrinsic_maps.append(rec_imap.cpu() if low_memory_usage else rec_imap)
            rec_depth_maps.append(rec_dmap.cpu() if low_memory_usage else rec_dmap)
            rec_valid_masks.append(rec_vmask.cpu() if low_memory_usage else rec_vmask)
        
        rec_intrinsic_maps = torch.cat(rec_intrinsic_maps, dim=0)
        rec_depth_maps = torch.cat(rec_depth_maps, dim=0)
        rec_valid_masks = torch.cat(rec_valid_masks, dim=0)
        
        if need_resize:
            # transform the log-depth to depth domain for bilinear interpolation
            rec_depth_maps = F.interpolate(rec_depth_maps, (height, width), mode='nearest-exact') if use_extract_interp else \
                F.interpolate(rec_depth_maps.clamp_max(10).exp(), (height, width), mode='bilinear', align_corners=False).log()
            rec_valid_masks = F.interpolate(rec_valid_masks, (height, width), mode='nearest-exact') if use_extract_interp else \
                F.interpolate(rec_valid_masks, (height, width), mode='bilinear', align_corners=False)
            rec_intrinsic_maps = F.interpolate(rec_intrinsic_maps, (height, width), mode='bilinear', align_corners=False)

        H, W = rec_intrinsic_maps.shape[-2], rec_intrinsic_maps.shape[-1]
        mesh_grid = create_meshgrid(
            H, W, 
            normalized_coordinates=True
        ).to(rec_intrinsic_maps.device, rec_intrinsic_maps.dtype, non_blocking=True)
        # 1,h,w,2
        rec_intrinsic_maps = torch.cat([rec_intrinsic_maps * W / np.sqrt(W**2+H**2), rec_intrinsic_maps * H / np.sqrt(W**2+H**2)], dim=1) # t,2,h,w
        mesh_grid = mesh_grid.permute(0,3,1,2)
        rec_valid_masks = rec_valid_masks.squeeze(1) > 0

        if force_projection:
            if force_fixed_focal:
                nfx = (rec_intrinsic_maps[:, 0, :, :] * rec_valid_masks.float()).mean() / (rec_valid_masks.float().mean() + 1e-4) 
                nfy = (rec_intrinsic_maps[:, 1, :, :] * rec_valid_masks.float()).mean() / (rec_valid_masks.float().mean() + 1e-4) 
                rec_intrinsic_maps = torch.tensor([nfx, nfy], device=rec_intrinsic_maps.device)[None, :, None, None].repeat(T, 1, 1, 1)    
            else:
                nfx = (rec_intrinsic_maps[:, 0, :, :] * rec_valid_masks.float()).mean(dim=[-1, -2]) / (rec_valid_masks.float().mean(dim=[-1, -2]) + 1e-4) 
                nfy = (rec_intrinsic_maps[:, 1, :, :] * rec_valid_masks.float()).mean(dim=[-1, -2]) / (rec_valid_masks.float().mean(dim=[-1, -2]) + 1e-4) 
                rec_intrinsic_maps = torch.stack([nfx, nfy], dim=-1)[:, :, None, None]
            # t,2,1,1

        rec_point_maps = torch.cat([rec_intrinsic_maps * mesh_grid, rec_depth_maps], dim=1).permute(0,2,3,1)
        xy, z = rec_point_maps.split([2, 1], dim=-1)
        z = torch.clamp_max(z, 10) # for numerical stability
        z = torch.exp(z)
        rec_point_maps = torch.cat([xy * z, z], dim=-1)

        return rec_point_maps, rec_valid_masks


    @torch.no_grad()
    def __call__(
        self,
        video: Union[np.ndarray, torch.Tensor],
        point_map_vae,
        prior_model,
        height: int = 576,
        width: int = 1024,
        window_size: Optional[int] = 14,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        overlap: int = 4,
        force_projection: bool = True,
        force_fixed_focal: bool = True,
        use_extract_interp: bool = False,
        track_time: bool = False,
        low_memory_usage: bool = False,
        **kwargs
    ):
        # video: in shape [t, h, w, c] if np.ndarray or [t, c, h, w] if torch.Tensor, in range [0, 1]
        
        # 0. Define height and width for preprocessing

        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose(0, 3, 1, 2))
        else:
            assert isinstance(video, torch.Tensor)

        height = height or video.shape[-2]
        width = width or video.shape[-1]
        original_height = video.shape[-2]
        original_width = video.shape[-1]
        num_frames = video.shape[0]
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else 8
        if num_frames <= window_size:
            window_size = num_frames
            overlap = 0
        stride = window_size - overlap

        # 1. Check inputs. Raise error if not correct
        assert height % 64 == 0 and width % 64 == 0
        if original_height != height or original_width != width:
            need_resize = True
        else:
            need_resize = False

        # 2. Define call parameters
        batch_size = 1
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = 1.0

        if track_time:
            print(f'Processing video shape : {list(video.shape)}')
            start_event = torch.cuda.Event(enable_timing=True)
            prior_event = torch.cuda.Event(enable_timing=True)
            encode_event = torch.cuda.Event(enable_timing=True)
            denoise_event = torch.cuda.Event(enable_timing=True)
            decode_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        # 3. Compute prior latents under original resolutions
        pred_disparity, pred_valid_mask, pred_point_map, pred_intrinsic_map = self.produce_priors(
            prior_model, 
            video.to(torch.float32) if low_memory_usage else video.to(device=device, dtype=torch.float32),
            chunk_size=decode_chunk_size,
            low_memory_usage=low_memory_usage
        ) # T,H,W T,H,W T,3,H,W T,2,H,W

        if need_resize:
            pred_disparity = F.interpolate(pred_disparity.unsqueeze(1), (height, width), mode='bilinear', align_corners=False).squeeze(1)
            pred_valid_mask = F.interpolate(pred_valid_mask.unsqueeze(1), (height, width), mode='bilinear', align_corners=False).squeeze(1)
            # transform the log-depth to depth domain for bilinear interpolation
            pred_point_map = torch.cat([
                F.interpolate(pred_point_map[:, 0:2], (height, width), mode='bilinear', align_corners=False),
                F.interpolate(pred_point_map[:, 2:3].clamp_max(10).exp(), (height, width), mode='bilinear', align_corners=False).log()
            ], dim=1)
            pred_intrinsic_map = F.interpolate(pred_intrinsic_map, (height, width), mode='bilinear', align_corners=False)

        if track_time:
            prior_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(prior_event)
            print(f"Elapsed time for computing per-frame prior: {elapsed_time_ms} ms")
        else:
            gc.collect()
            torch.cuda.empty_cache()
            


        # 3. Encode input video
        if need_resize:
            video = F.interpolate(video, (height, width), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)

        video = video.to(device=device, dtype=self.dtype)
        video = video * 2.0 - 1.0  # [0,1] -> [-1,1], in [t, c, h, w]
        
        
        video_embeddings = self.encode_video(video, chunk_size=decode_chunk_size).unsqueeze(0)


        # 4. Encode input image using VAE

        # pdb.set_trace()
        needs_upcasting = (
            self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        )
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        video_latents = self.encode_vae_video(
            video.to(self.vae.dtype),
            chunk_size=decode_chunk_size,
        ).unsqueeze(0).to(video_embeddings.dtype)  # [1, t, c, h, w]
        
        if low_memory_usage:
            del video
            torch.cuda.empty_cache()

        prior_latents = self.encode_point_map(
            point_map_vae,
            pred_disparity, 
            pred_valid_mask, 
            pred_point_map, 
            pred_intrinsic_map, 
            chunk_size=decode_chunk_size,
            low_memory_usage=low_memory_usage
        ).unsqueeze(0).to(video_embeddings.dtype) # 1,T,C,H,W

        if track_time:
            encode_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = prior_event.elapsed_time(encode_event)
            print(f"Elapsed time for encode prior and frames: {elapsed_time_ms} ms")
        else:
            gc.collect()
            torch.cuda.empty_cache()

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            7,
            127,
            noise_aug_strength,
            video_embeddings.dtype,
            batch_size,
            1,
            False,
        )  # [1 or 2, 3]
        added_time_ids = added_time_ids.to(device)

        # 6. Prepare timesteps
        timestep = 1.6378
        self._num_timesteps = 1

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents_init = prior_latents  # [1, t, c, h, w]
        latents_all = None

        idx_start = 0
        if overlap > 0:
            weights = torch.linspace(0, 1, overlap, device=device)
            weights = weights.view(1, overlap, 1, 1, 1)
        else:
            weights = None

        while idx_start < num_frames - overlap:
            idx_end = min(idx_start + window_size, num_frames)
            # 9. Denoising loop
            # latents_init = latents_init.flip(1)
            latents = latents_init[:, idx_start:idx_end]
            video_latents_current = video_latents[:, idx_start:idx_end]
            video_embeddings_current = video_embeddings[:, idx_start:idx_end]

            latent_model_input = torch.cat(
                [latents, video_latents_current], dim=2
            )

            model_pred = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=video_embeddings_current,
                added_time_ids=added_time_ids,
                return_dict=False,
            )[0]

            c_out = -1
            latents = model_pred * c_out

            if latents_all is None:
                latents_all = latents.clone()
            else:
                if overlap > 0:
                    latents_all[:, -overlap:] = latents[
                        :, :overlap
                    ] * weights + latents_all[:, -overlap:] * (1 - weights)
                latents_all = torch.cat([latents_all, latents[:, overlap:]], dim=1)

            idx_start += stride

        latents_all = 1 / self.vae.config.scaling_factor * latents_all.squeeze(0).to(torch.float32)

        if low_memory_usage:
            del latents
            del prior_latents
            del latent_model_input
            del latents_init
            torch.cuda.empty_cache()


        if track_time:
            denoise_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = encode_event.elapsed_time(denoise_event)
            print(f"Elapsed time for denoise latent: {elapsed_time_ms} ms")
        else:
            gc.collect()
            torch.cuda.empty_cache()

        point_map, valid_mask = self.decode_point_map(
            point_map_vae, 
            latents_all, 
            chunk_size=decode_chunk_size, 
            force_projection=force_projection,
            force_fixed_focal=force_fixed_focal,
            use_extract_interp=use_extract_interp, 
            need_resize=need_resize, 
            height=original_height, 
            width=original_width,
            low_memory_usage=low_memory_usage 
        )

        if track_time:
            decode_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = denoise_event.elapsed_time(decode_event)
            print(f"Elapsed time for decode latent: {elapsed_time_ms} ms")
        else:
            gc.collect()
            torch.cuda.empty_cache()

        self.maybe_free_model_hooks()
        # t,h,w,3   t,h,w
        return point_map, valid_mask
