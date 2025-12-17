"""
Wonder3D Guidance Module

This module implements the Wonder3D guidance system for multi-view 3D generation.
It provides functionality for training and inference using diffusion models with
LoRA adapters for efficient fine-tuning.
"""

import time
import os
import math
import random
import shutil
import logging
import inspect
import argparse
import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, Normalize, ToPILImage
from torchvision.utils import make_grid, save_image

import numpy as np
import cv2
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import transformers
import accelerate
import diffusers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from peft import LoraConfig, PeftConfig, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file

from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from .wonder3d_utils.base import BaseModule
from .wonder3d_utils.misc import C, cleanup, parse_version, get_rank
from .wonder3d_utils.typing import *
from threestudio.models.networks import get_encoding, get_mlp
from .wonder3d_utils.wonder3D_plus.mv_diffusion_30.models.unet_mv2d_condition import (
    UNetMV2DConditionModel,
)
from .wonder3d_utils.wonder3D_plus.mv_diffusion_30.data.single_image_dataset import (
    SingleImageDataset as MVDiffusionDataset,
)
from .wonder3d_utils.wonder3D_plus.mv_diffusion_30.pipelines.pipeline_mvdiffusion_image import (
    MVDiffusionImagePipeline,
)

from .utils.seed import setup_seed
from .utils.multi_port import find_free_port
from .utils.assign_cfg import assign_signle_cfg
from .utils.distributed import generalized_all_gather, all_reduce
from .utils.video_op import save_i2vgen_video, save_i2vgen_video_safe
from .utils.registry_class import INFER_ENGINE, MODEL, EMBEDDER, AUTO_ENCODER, DIFFUSION
from .utils.camera_utils import get_camera
from .utils.transforms import Compose, CenterCropWide, ToTensor, Normalize, Resize
from .modules.autoencoder import AutoencoderKL
from .modules.diffusions import DiffusionDDIM
from .modules.unet import UNetSD_I2VGen
from .modules.clip_embedder import FrozenOpenCLIPTtxtVisualEmbedder


# Constants
DEFAULT_IMAGE_SIZE = 256
CONDITION_DROP_RATE = 0.05
LORA_R = 4
LORA_ALPHA = LORA_R
LORA_DROPOUT = 0.05
CAM_TYPE_EMB = [1, 0]
COLOR_CLASS = [0, 1]
NORMAL_CLASS = [1, 0]


class Normalize_train(object):
    """
    Normalization transform for training data.
    
    Normalizes RGB images using ImageNet statistics and handles both
    3D and 4D tensor inputs.
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        Initialize normalization parameters.
        
        Args:
            mean: Mean values for normalization (default: ImageNet mean)
            std: Standard deviation values for normalization (default: ImageNet std)
        """
        self.mean = mean
        self.std = std

    def __call__(self, rgb):
        """
        Apply normalization to RGB tensor.
        
        Args:
            rgb: Input RGB tensor (H, W, C) or (B, H, W, C)
            
        Returns:
            Normalized tensor with channels first
        """
        rgb = rgb.permute(2, 0, 1)
        rgb.clamp_(0, 1)
        if not isinstance(self.mean, torch.Tensor):
            self.mean = rgb.new_tensor(self.mean).view(-1)
        if not isinstance(self.std, torch.Tensor):
            self.std = rgb.new_tensor(self.std).view(-1)
        if rgb.dim() == 4:
            rgb.sub_(self.mean.view(1, -1, 1, 1)).div_(self.std.view(1, -1, 1, 1))
        elif rgb.dim() == 3:
            rgb.sub_(self.mean.view(-1, 1, 1)).div_(self.std.view(-1, 1, 1))
        return rgb


def _i(tensor, t, x):
    """
    Index tensor using timestep t and format the output according to x.
    
    Args:
        tensor: Tensor to index
        t: Timestep indices
        x: Reference tensor for shape and device
        
    Returns:
        Indexed tensor with shape matching x
    """
    shape = (x.size(0),) + (1,) * (x.ndim - 1)
    if tensor.device != x.device:
        tensor = tensor.to(x.device)
    return tensor[t].view(shape).to(x)


class ToWeightsDType(nn.Module):
    """
    Wrapper module to convert output to specified dtype.
    
    Used for maintaining consistent data types in model pipelines.
    """

    def __init__(self, module: nn.Module, dtype: torch.dtype):
        """
        Initialize the wrapper.
        
        Args:
            module: Module to wrap
            dtype: Target dtype for output
        """
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        """
        Forward pass with dtype conversion.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor converted to target dtype
        """
        return self.module(x).to(self.dtype)


class AttrDict(dict):
    """
    Dictionary subclass that allows attribute-style access.
    
    Enables accessing dictionary keys as attributes (e.g., dict.key instead of dict['key']).
    """

    def __getattr__(self, item):
        """Get attribute using dictionary lookup."""
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        """Set attribute using dictionary assignment."""
        self[key] = value


@threestudio.register("wonder3d-guidance")
class Wonder3dGuidance(BaseModule):
    """
    Wonder3D Guidance Module for multi-view 3D generation.
    
    This class implements the guidance system for Wonder3D, supporting both
    training and inference modes with LoRA fine-tuning capabilities.
    """

    @dataclass
    class Config(BaseModule.Config):
        """Configuration class for Wonder3D guidance."""

        revision: Optional[str] = "null"
        save_dir: str = "./example_images"
        seed: Optional[int] = 100
        validation_dataset: str = "none"
        validation_batch_size: int = 1
        dataloader_num_workers: int = 4
        trainable_modules: Any = None
        local_rank: int = 1

        pipe_kwargs: dict = field(default_factory=lambda: {})
        pipe_validation_kwargs: dict = field(default_factory=lambda: {})
        unet_from_pretrained_kwargs: dict = field(default_factory=lambda: {})
        validation_guidance_scales: list = field(default_factory=lambda: {})
        validation_grid_nrow: int = 6
        camera_embedding_lr_mult: float = 0.01
        snr_gamma: float = 0.01

        num_views: int = 4
        camera_embedding_type: str = "projection"
        pred_type: str = "joint"  # joint, or ablation

        enable_xformers_memory_efficient_attention: bool = True
        cond_on_normals: bool = False
        cond_on_colors: bool = True
        guidance_scale: float = 7.5
        do_classifier_free_guidance: bool = False
        use_gan_clip: bool = False

    cfg: Config

    def configure(self) -> None:
        """
        Configure the Wonder3D guidance module.
        
        Loads and initializes all necessary models including:
        - CLIP image encoder
        - VAE encoder/decoder
        - UNet with LoRA adapters
        - Noise scheduler
        - Inference pipeline
        """
        threestudio.info(f"Loading Stable Diffusion ...")
        cfg = self.cfg

        if cfg.seed is not None:
            set_seed(cfg.seed)

        # Load models
        self._load_models(cfg)
        
        # Setup schedulers
        self._setup_schedulers(cfg)
        
        # Setup LoRA
        self._setup_lora(cfg)
        
        # Create inference pipeline
        self._create_pipeline(cfg)

    def _load_models(self, cfg):
        """Load all required models."""
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            cfg.pretrained_model_name_or_path,
            subfolder="image_encoder",
            revision=cfg.revision,
            local_files_only=True,
        )
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            cfg.pretrained_model_name_or_path,
            subfolder="feature_extractor",
            revision=cfg.revision,
        )
        self.vae = AutoencoderKL.from_pretrained(
            cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision
        )
        self.unet = UNetMV2DConditionModel.from_pretrained(
            cfg.pretrained_unet_path,
            subfolder="unet",
            revision=cfg.revision,
            **cfg.unet_from_pretrained_kwargs,
        )

        # Freeze base models
        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.weight_dtype = torch.float32

        # Move models to GPU
        device = "cuda"
        self.image_encoder.to(device, dtype=self.weight_dtype)
        self.vae.to(device, dtype=self.weight_dtype)
        self.unet.to(device, dtype=self.weight_dtype)
        self.num_views = 4

    def _setup_schedulers(self, cfg):
        """Setup noise schedulers."""
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            cfg.pretrained_model_name_or_path, subfolder="scheduler"
        )
        if not self.cfg.use_ori_gen:
            self.noise_scheduler.set_timesteps(4, device=self.device)
        else:
            self.noise_scheduler.set_timesteps(50, device=self.device)

        self.do_classifier_free_guidance = cfg.do_classifier_free_guidance
        timesteps_list = [999, 749, 599, 299]
        self.noise_scheduler.timesteps = torch.tensor(timesteps_list).to(self.device)
        self.use_ori_gen = self.cfg.use_ori_gen

        # Fixed scheduler for inference
        self.fixed_noise_scheduler = DDPMScheduler.from_pretrained(
            cfg.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.fixed_noise_scheduler.set_timesteps(1000, device=self.device)
        self.alphas = self.fixed_noise_scheduler.alphas_cumprod.to(self.device)

    def _setup_lora(self, cfg):
        """Setup LoRA adapters."""
        unet_lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.unet.eval()
        self.unet = get_peft_model(self.unet, unet_lora_config)

        # Load LoRA adapter
        peft_model_id=cfg.peft_model_id
        self.unet.load_adapter(adapter_name="lora", model_id=peft_model_id)
        self.unet.set_adapter("lora")

        # Create fixed UNet for inference
        self.fixed_unet = UNetMV2DConditionModel.from_pretrained(
            cfg.pretrained_unet_path,
            subfolder="unet",
            revision=cfg.revision,
            **cfg.unet_from_pretrained_kwargs,
        )
        self.fixed_unet = get_peft_model(self.fixed_unet, unet_lora_config)
        self.fixed_unet.eval()

        # Enable gradients for LoRA parameters
        for name, param in self.unet.named_parameters():
            if "lora" in name:
                param.requires_grad_(True)

        self.generator = torch.Generator(device=self.unet.device).manual_seed(cfg.seed)

    def _create_pipeline(self, cfg):
        """Create inference pipeline."""
        self.pipeline = MVDiffusionImagePipeline(
            image_encoder=self.image_encoder,
            feature_extractor=self.feature_extractor,
            vae=self.vae,
            unet=self.fixed_unet,
            safety_checker=None,
            scheduler=DDPMScheduler.from_pretrained(
                cfg.pretrained_model_name_or_path, subfolder="scheduler"
            ),
            **cfg.pipe_kwargs,
        )
        self.pipeline.set_progress_bar_config(disable=True)

    def _prepare_camera_embeddings(self, elevations_cond, elevations, azimuths):
        """
        Prepare camera embeddings from angles.
        
        Args:
            elevations_cond: Conditioning elevation angles
            elevations: Target elevation angles
            azimuths: Azimuth angles
            
        Returns:
            Camera embeddings tensor
        """
        elevations_cond = torch.deg2rad(elevations_cond)
        elevations = torch.deg2rad(elevations)
        azimuths = torch.deg2rad(azimuths)

        elevations_cond = torch.cat([elevations_cond] * self.cfg.num_views, dim=0)
        cam_type_emb = torch.tensor(CAM_TYPE_EMB).expand(self.num_views, -1).cuda().unsqueeze(0)
        
        camera_embeddings = torch.stack(
            [elevations_cond.unsqueeze(0), elevations.unsqueeze(0), azimuths.unsqueeze(0)], dim=-1
        )
        camera_embeddings = torch.cat((camera_embeddings, cam_type_emb), dim=-1)
        
        return camera_embeddings

    def _prepare_task_embeddings(self, use_normal=False):
        """
        Prepare task embeddings for color/normal prediction.
        
        Args:
            use_normal: Whether to include normal task embeddings
            
        Returns:
            Task embeddings tensor
        """
        if use_normal:
            normal_task_embeddings = (
                torch.stack([torch.tensor(NORMAL_CLASS).float()] * self.cfg.num_views, dim=0)
                .cuda()
                .unsqueeze(0)
            )
            color_task_embeddings = (
                torch.stack([torch.tensor(COLOR_CLASS).float()] * self.cfg.num_views, dim=0)
                .cuda()
                .unsqueeze(0)
            )
            task_embeddings = torch.cat([normal_task_embeddings, color_task_embeddings], dim=0)
        else:
            color_task_embeddings = (
                torch.stack([torch.tensor(COLOR_CLASS).float()] * self.cfg.num_views, dim=0)
                .cuda()
                .unsqueeze(0)
            )
            task_embeddings = torch.cat([color_task_embeddings], dim=0)
        
        return task_embeddings

    def _encode_images_for_clip(self, imgs_in):
        """
        Encode images for CLIP processing.
        
        Args:
            imgs_in: Input images
            
        Returns:
            CLIP image embeddings
        """
        imgs_in_proc = TF.resize(
            imgs_in,
            (self.feature_extractor.crop_size["height"], self.feature_extractor.crop_size["width"]),
            interpolation=InterpolationMode.BICUBIC,
        )
        clip_image_mean = (
            torch.as_tensor(self.feature_extractor.image_mean)[:, None, None]
            .to(self.device, dtype=torch.float32)
        )
        clip_image_std = (
            torch.as_tensor(self.feature_extractor.image_std)[:, None, None]
            .to(self.device, dtype=torch.float32)
        )
        imgs_in_proc = ((imgs_in_proc.float() - clip_image_mean) / clip_image_std).to(
            self.weight_dtype
        )
        return self.image_encoder(imgs_in_proc).image_embeds.unsqueeze(1)

    def _encode_images_to_latents(self, imgs, is_condition=False):
        """
        Encode images to VAE latents.
        
        Args:
            imgs: Input images
            is_condition: Whether these are conditioning images
            
        Returns:
            Encoded latents
        """
        if imgs.shape[1] == 3:
            imgs_normalized = imgs * 2.0 - 1.0
            if is_condition:
                latents = self.vae.encode(imgs_normalized).latent_dist.mode()
            else:
                latents = self.vae.encode(imgs_normalized).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        else:
            latents = imgs
        return latents

    def _apply_camera_embedding_encoding(self, camera_task_embeddings, cfg):
        """
        Apply camera embedding encoding (sin/cos).
        
        Args:
            camera_task_embeddings: Camera task embeddings
            cfg: Configuration
            
        Returns:
            Encoded camera embeddings
        """
        if cfg.camera_embedding_type == "e_de_da_sincos":
            return torch.cat(
                [torch.sin(camera_task_embeddings), torch.cos(camera_task_embeddings)], dim=-1
            )
        else:
            raise NotImplementedError(
                f"Camera embedding type {cfg.camera_embedding_type} not implemented"
            )

    def _apply_classifier_free_guidance(
        self, noisy_latents, image_embeddings, cond_vae_embeddings, camera_task_embeddings
    ):
        """
        Apply classifier-free guidance by duplicating inputs.
        
        Args:
            noisy_latents: Noisy latents
            image_embeddings: Image embeddings
            cond_vae_embeddings: Conditioning VAE embeddings
            camera_task_embeddings: Camera task embeddings
            
        Returns:
            Tuple of (noisy_latents, image_embeddings, cond_vae_embeddings, camera_task_embeddings)
        """
        noisy_latents = torch.cat([noisy_latents] * 2)
        image_embeddings = torch.cat([torch.zeros_like(image_embeddings), image_embeddings])
        cond_vae_embeddings = torch.cat(
            [torch.zeros_like(cond_vae_embeddings), cond_vae_embeddings]
        )
        camera_task_embeddings = torch.cat([camera_task_embeddings] * 2)
        
        # Reshape for cross-domain attention
        noisy_latents = self.reshape_to_cd_input(noisy_latents)
        image_embeddings = self.reshape_to_cd_input(image_embeddings)
        camera_task_embeddings = self.reshape_to_cd_input(camera_task_embeddings)
        cond_vae_embeddings = self.reshape_to_cd_input(cond_vae_embeddings)
        
        return noisy_latents, image_embeddings, cond_vae_embeddings, camera_task_embeddings

    def _apply_conditioning_dropout(
        self, cond_vae_embeddings, image_embeddings, bnm, Nv, use_normal=False
    ):
        """
        Apply conditioning dropout for classifier-free guidance training.
        
        Args:
            cond_vae_embeddings: Conditioning VAE embeddings
            image_embeddings: Image embeddings
            bnm: Batch size
            Nv: Number of views
            use_normal: Whether using normal maps
            
        Returns:
            Tuple of (masked_cond_vae_embeddings, masked_image_embeddings)
        """
        random_p = torch.rand(bnm, device=cond_vae_embeddings.device, generator=self.generator)

        # Sample masks for conditioning images
        image_mask_dtype = cond_vae_embeddings.dtype
        image_mask = 1 - (
            (random_p >= CONDITION_DROP_RATE).to(image_mask_dtype)
            * (random_p < 3 * CONDITION_DROP_RATE).to(image_mask_dtype)
        )
        image_mask = image_mask.reshape(bnm, 1, 1, 1, 1).repeat(1, Nv, 1, 1, 1)
        image_mask = rearrange(image_mask, "B Nv C H W -> (B Nv) C H W")
        if use_normal:
            image_mask = torch.cat([image_mask] * 2, dim=0)
        cond_vae_embeddings = image_mask * cond_vae_embeddings

        # Sample masks for CLIP embeddings
        clip_mask_dtype = image_embeddings.dtype
        clip_mask = 1 - ((random_p < 2 * CONDITION_DROP_RATE).to(clip_mask_dtype))
        clip_mask = clip_mask.reshape(bnm, 1, 1, 1).repeat(1, Nv, 1, 1)
        clip_mask = rearrange(clip_mask, "B Nv M C -> (B Nv) M C")
        if use_normal:
            clip_mask = torch.cat([clip_mask] * 2, dim=0)
        image_embeddings = clip_mask * image_embeddings

        return cond_vae_embeddings, image_embeddings

    def get_t_plus(
        self,
        t: Float[Tensor, "B"],
        num_train_timesteps,
        min_step,
        plus_ratio=0,
        plus_random=False,
    ):
        """
        Compute t_plus timesteps for advanced sampling strategies.
        
        Args:
            t: Current timesteps
            num_train_timesteps: Total number of training timesteps
            min_step: Minimum timestep value
            plus_ratio: Ratio for computing t_plus
            plus_random: Whether to use random t_plus
            
        Returns:
            Adjusted timesteps t_plus
        """
        t_plus = plus_ratio * (t - min_step)
        if plus_random:
            t_plus = (t_plus * torch.rand(*t.shape, device=self.device)).to(torch.long)
        else:
            t_plus = t_plus.to(torch.long)
        t_plus = t + t_plus
        t_plus = torch.clamp(t_plus, 1, num_train_timesteps - 1)
        return t_plus

    def _encode_image(self, image_pil, device, num_images_per_prompt, do_classifier_free_guidance):
        """
        Encode input images to embeddings and latents.
        
        Args:
            image_pil: PIL images to encode
            device: Target device
            num_images_per_prompt: Number of images per prompt
            do_classifier_free_guidance: Whether to use classifier-free guidance
            
        Returns:
            Tuple of (image_embeddings, image_latents)
        """
        dtype = next(self.image_encoder.parameters()).dtype

        image_pt = self.feature_extractor(images=image_pil, return_tensors="pt").pixel_values
        image_pt = image_pt.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image_pt).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # Duplicate image embeddings for each generation per prompt
        image_embeddings = image_embeddings.repeat(num_images_per_prompt, 1, 1)

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        image_pt = torch.stack([TF.to_tensor(img) for img in image_pil], dim=0).to(device).to(dtype)
        image_pt = image_pt * 2.0 - 1.0
        image_latents = (
            self.vae.encode(image_pt).latent_dist.mode() * self.vae.config.scaling_factor
        )
        image_latents = image_latents.repeat(num_images_per_prompt, 1, 1, 1)

        if do_classifier_free_guidance:
            image_latents = torch.cat([torch.zeros_like(image_latents), image_latents])

        return image_embeddings, image_latents

    def obtain_bg_colors(self, gb_normal, mask, bg_colors="blue"):
        """
        Apply background colors to normal maps based on mask.
        
        Args:
            gb_normal: Normal map tensor
            mask: Mask tensor indicating valid regions
            bg_colors: Background color type ("blue", "black", or "white")
            
        Returns:
            Normal map with background colors applied
        """
        indices = mask != 0.0
        
        if bg_colors == "blue":
            _a = torch.zeros_like(gb_normal)
            gb_normal[..., :1] = (gb_normal[..., :1] - 1.0) * 2.0
            gb_normal[..., :1][indices.expand(-1, -1, -1, 1)] *= -1
            gb_normal[..., :1] = gb_normal[..., :1] + 1.0
            _a[..., 2] = 1.0
            _a = (_a + 1.0) / 2.0
        elif bg_colors == "black":
            _a = torch.zeros_like(gb_normal)
        elif bg_colors == "white":
            _a = torch.ones_like(gb_normal)
            gb_normal[..., :1] = (gb_normal[..., :1] - 1.0) * 2.0
            gb_normal[..., :1][indices.expand(-1, -1, -1, 1)] *= -1
            gb_normal[..., :1] = gb_normal[..., :1] + 1.0
        else:
            raise NotImplementedError(f"Background color {bg_colors} not implemented")

        _b = (gb_normal + 1.0) / 2.0
        gb_normal_aa = torch.lerp(_a, _b, mask.float())
        gb_normal_aa.requires_grad_(True)
        return gb_normal_aa

    def train_with_asd(
        self,
        batch,
        cond_img,
        elevations_cond,
        elevations,
        azimuths,
        use_gen=True,
        use_normal=False,
        index=None,
    ):
        """
        Training step with ASD (Adaptive Score Distillation) loss.
        
        Args:
            batch: Training batch data
            cond_img: Conditioning images
            elevations_cond: Conditioning elevation angles
            elevations: Target elevation angles
            azimuths: Azimuth angles
            use_gen: Whether to use generated images
            use_normal: Whether to use normal maps
            index: Optional timestep index
            
        Returns:
            Dictionary containing loss values
        """
        cond_img = F.interpolate(
            cond_img, size=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), mode="bilinear", align_corners=False
        )

        do_classifier_free_guidance = self.do_classifier_free_guidance
        cfg = self.cfg

        # Prepare camera and task embeddings
        camera_embeddings = self._prepare_camera_embeddings(elevations_cond, elevations, azimuths)
        cond_img = torch.cat([cond_img] * self.cfg.num_views, dim=0)

        # Prepare input and output images
        if use_gen:
            imgs_in, colors_out, normals_out = (
                cond_img.unsqueeze(0),
                batch["asd_rgb"].unsqueeze(0),
                batch["asd_normal"].unsqueeze(0),
            )
        else:
            imgs_in, colors_out, normals_out = (
                cond_img.unsqueeze(0),
                batch["gt_rgb"].unsqueeze(0),
                batch["gt_normal"].unsqueeze(0),
            )

        imgs_out = torch.cat([colors_out], dim=0)
        task_embeddings = self._prepare_task_embeddings(use_normal=False)
        camera_task_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)

        if do_classifier_free_guidance:
            camera_task_embeddings = torch.cat([camera_task_embeddings] * 2)

        # Reshape and encode
        imgs_in, imgs_out = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W"), rearrange(
            imgs_out, "B Nv C H W -> (B Nv) C H W"
        )
        camera_task_embeddings = rearrange(camera_task_embeddings, "B Nv Nce -> (B Nv) Nce")
        camera_task_embeddings = self._apply_camera_embedding_encoding(camera_task_embeddings, cfg)

        imgs_in, imgs_out, camera_task_embeddings = (
            imgs_in.to(self.weight_dtype).to(imgs_out.device),
            imgs_out.to(self.weight_dtype),
            camera_task_embeddings.to(self.weight_dtype),
        )

        cond_vae_embeddings = self._encode_images_to_latents(imgs_in, is_condition=True)
        latents = self._encode_images_to_latents(imgs_out, is_condition=False)
        image_embeddings = self._encode_images_for_clip(imgs_in)

        # Add noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        if index is None:
            timesteps = torch.randint(
                int(0.5 * 1000),
                int(0.98 * 1000 + 1),
                (cfg.num_views // cfg.num_views,),
                device=latents.device,
            ).repeat_interleave(cfg.num_views * (bsz // cfg.num_views))
        else:
            timesteps = self.fixed_noise_scheduler.timesteps[index].repeat_interleave(
                cfg.num_views * (bsz // cfg.num_views)
            )

        timesteps = timesteps.long()
        noisy_latents = self.fixed_noise_scheduler.add_noise(latents, noise, timesteps)

        # Compute t_plus for ASD
        t_plus = self.get_t_plus(
            timesteps, num_train_timesteps=1000, min_step=int(0.02 * 1000)
        ).to(self.device)
        noisy_latents_second = self.fixed_noise_scheduler.add_noise(latents, noise, t_plus)

        # Apply classifier-free guidance
        if do_classifier_free_guidance:
            noisy_latents, image_embeddings, cond_vae_embeddings, camera_task_embeddings = (
                self._apply_classifier_free_guidance(
                    noisy_latents, image_embeddings, cond_vae_embeddings, camera_task_embeddings
                )
            )
            noisy_latents_second = torch.cat([noisy_latents_second] * 2)
            timesteps_input = torch.cat([timesteps] * 2)
            t_plus = torch.cat([t_plus] * 2)
        else:
            timesteps_input = timesteps

        # Forward passes
        latent_model_input = torch.cat([noisy_latents, cond_vae_embeddings], dim=1)
        latent_model_input_second = torch.cat([noisy_latents_second, cond_vae_embeddings], dim=1)

        model_pred = self.fixed_unet(
            latent_model_input,
            timesteps_input,
            encoder_hidden_states=image_embeddings,
            class_labels=camera_task_embeddings,
        ).sample.float()

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = self.reshape_to_cfg_output(model_pred).chunk(2)
            model_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        model_pred_second = self.fixed_unet(
            latent_model_input_second,
            t_plus,
            encoder_hidden_states=image_embeddings,
            class_labels=camera_task_embeddings,
        ).sample.float()

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = self.reshape_to_cfg_output(model_pred_second).chunk(2)
            model_pred_second = noise_pred_uncond + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # Compute ASD loss
        if not cfg.use_dmd:
            w = (1 - self.alphas[timesteps]).view(-1, 1, 1, 1)
            grad = w[: model_pred.shape[0]] * (model_pred - model_pred_second)
        else:
            alpha = (self.alphas[timesteps] ** 0.5).view(-1, 1, 1, 1)
            sigma = ((1 - self.alphas[timesteps]) ** 0.5).view(-1, 1, 1, 1)
            latent_first = (latents - sigma * model_pred) / alpha
            latent_second = (latents - sigma * model_pred_second) / alpha
            w = torch.abs(latents - latent_first).mean(dim=(1, 2, 3))
            w = w.view(-1, 1, 1, 1)
            grad = (latent_second - latent_first) / (w + 0.1)

        grad = torch.nan_to_num(grad)
        target = (latents - grad).detach()
        loss_asd = 0.5 * F.mse_loss(latents, target, reduction="mean")

        return {"loss_asd": loss_asd}

    def train_one_step(
        self,
        batch,
        cond_img,
        elevations_cond,
        elevations,
        azimuths,
        use_gen=False,
        use_normal=False,
        index=None,
    ):
        """
        Single training step for the diffusion model.
        
        Args:
            batch: Training batch data
            cond_img: Conditioning images
            elevations_cond: Conditioning elevation angles
            elevations: Target elevation angles
            azimuths: Azimuth angles
            use_gen: Whether to use generated images
            use_normal: Whether to use normal maps
            index: Optional timestep index
            
        Returns:
            Dictionary containing training outputs and losses
        """
        batch["gt_rgb_new"] = F.interpolate(
            batch["gt_rgb"], size=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), mode="bilinear", align_corners=False
        )
        batch["gt_normal_new"] = F.interpolate(
            batch["gt_normal"], size=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), mode="bilinear", align_corners=False
        )

        cond_img = F.interpolate(
            cond_img.clone(), size=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), mode="bilinear", align_corners=False
        )

        cfg = self.cfg

        # Prepare camera and task embeddings
        camera_embeddings = self._prepare_camera_embeddings(elevations_cond, elevations, azimuths)
        cond_img = torch.cat([cond_img] * self.cfg.num_views, dim=0)

        imgs_in, colors_out, normals_out = (
            cond_img.unsqueeze(0),
            batch["gen_rgb"].unsqueeze(0),
            batch["gt_normal"].unsqueeze(0),
        )

        bnm, Nv = imgs_in.shape[:2]
        imgs_out = torch.cat([colors_out], dim=0)
        task_embeddings = self._prepare_task_embeddings(use_normal=False)
        camera_task_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)

        # Reshape and encode
        imgs_in, imgs_out = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W"), rearrange(
            imgs_out, "B Nv C H W -> (B Nv) C H W"
        )
        camera_task_embeddings = rearrange(camera_task_embeddings, "B Nv Nce -> (B Nv) Nce")
        camera_task_embeddings = self._apply_camera_embedding_encoding(camera_task_embeddings, cfg)

        imgs_in, imgs_out, camera_task_embeddings = (
            imgs_in.to(self.weight_dtype).to(imgs_out.device),
            imgs_out.to(self.weight_dtype),
            camera_task_embeddings.to(self.weight_dtype),
        )

        cond_vae_embeddings = self._encode_images_to_latents(imgs_in, is_condition=True)
        latents = self._encode_images_to_latents(imgs_out, is_condition=False)
        image_embeddings = self._encode_images_for_clip(imgs_in)

        # Add noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        if index is None:
            timesteps = self.noise_scheduler.timesteps[
                torch.randint(0, 4, (cfg.num_views // cfg.num_views,), device=latents.device)
            ].repeat_interleave(cfg.num_views * (bsz // cfg.num_views))
        else:
            timesteps = self.noise_scheduler.timesteps[index].repeat_interleave(
                cfg.num_views * (bsz // cfg.num_views)
            )

        if imgs_out.shape[1] == 3:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        else:
            noisy_latents = latents

        # Apply conditioning dropout
        cond_vae_embeddings, image_embeddings = self._apply_conditioning_dropout(
            cond_vae_embeddings, image_embeddings, bnm, Nv, use_normal
        )

        # Forward pass
        latent_model_input = torch.cat([noisy_latents, cond_vae_embeddings], dim=1)
        model_pred = self.unet(
            latent_model_input,
            timesteps,
            encoder_hidden_states=image_embeddings,
            class_labels=camera_task_embeddings,
        ).sample

        # Denoise step
        step_output = self.noise_scheduler.step(model_pred, timesteps[0].cpu(), noisy_latents)
        latents_prev = step_output.prev_sample
        latents_0 = step_output.pred_original_sample

        # Decode to image space
        image = self.vae.decode(latents_0 / self.vae.config.scaling_factor, return_dict=False)[0]
        image = (image + 1.0) / 2.0
        image = image.clamp(0, 1)

        # Compute SNR-based loss weights
        snr = self.compute_snr(timesteps)
        mse_loss_weights = (
            torch.stack([snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        )

        output = {
            "imgs_final": image,
            "noise_pred": model_pred,
            "input_latents": latents,
            "input_imgs": imgs_out,
            "x_0": latents_0,
            "noise": noise,
            "mse_loss_weights": mse_loss_weights,
        }
        if noise is not None:
            output["x_prev"] = latents_prev

        return output

    def reshape_to_cd_input(self, input):
        """
        Reshape input for cross-domain attention.
        
        Args:
            input: Input tensor to reshape
            
        Returns:
            Reshaped tensor
        """
        input_norm_uc, input_rgb_uc, input_norm_cond, input_rgb_cond = torch.chunk(
            input, dim=0, chunks=4
        )
        return torch.cat([input_norm_uc, input_norm_cond, input_rgb_uc, input_rgb_cond], dim=0)

    def eval_one_step(
        self,
        batch,
        cond_img,
        elevations_cond,
        elevations,
        azimuths,
        use_gen=False,
        use_normal=False,
        index=None,
    ):
        """
        Single evaluation step for inference.
        
        Args:
            batch: Batch data
            cond_img: Conditioning images
            elevations_cond: Conditioning elevation angles
            elevations: Target elevation angles
            azimuths: Azimuth angles
            use_gen: Whether to use generated images
            use_normal: Whether to use normal maps
            index: Optional timestep index
            
        Returns:
            Dictionary containing evaluation outputs
        """
        peft_model_id = "/home/xinyue_liang/lxy/ScaleDreamer_v1-main/outputs/gd2d/2new_train_wonder3d_asd@20250130-154711/save/10000"
        self.unet.load_adapter(adapter_name="lora", model_id=peft_model_id)
        self.unet.set_adapter("lora")

        cond_img = F.interpolate(
            cond_img, size=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), mode="bilinear", align_corners=False
        )

        do_classifier_free_guidance = self.do_classifier_free_guidance
        cfg = self.cfg

        # Prepare camera and task embeddings
        camera_embeddings = self._prepare_camera_embeddings(elevations_cond, elevations, azimuths)
        cond_img = torch.cat([cond_img] * self.cfg.num_views, dim=0)

        # Prepare input and output images
        if use_gen:
            if use_normal:
                imgs_in, colors_out, normals_out = (
                    cond_img.unsqueeze(0),
                    batch["gen_rgb"].unsqueeze(0),
                    batch["gen_normal"].unsqueeze(0),
                )
            else:
                imgs_in, colors_out, normals_out = (
                    cond_img.unsqueeze(0),
                    batch["gen_rgb"].unsqueeze(0),
                    batch["gt_normal"].unsqueeze(0),
                )
        else:
            imgs_in, colors_out, normals_out = (
                cond_img.unsqueeze(0),
                batch["gt_rgb"].unsqueeze(0),
                batch["gt_normal"].unsqueeze(0),
            )

        bnm, Nv = imgs_in.shape[:2]

        if use_normal:
            imgs_in = torch.cat([imgs_in] * 2, dim=0)
            imgs_out = torch.cat([normals_out, colors_out], dim=0)
            camera_embeddings = torch.cat([camera_embeddings] * 2, dim=0)
            task_embeddings = self._prepare_task_embeddings(use_normal=True)
        else:
            imgs_out = torch.cat([colors_out], dim=0)
            task_embeddings = self._prepare_task_embeddings(use_normal=False)

        camera_task_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)

        if do_classifier_free_guidance:
            camera_task_embeddings = torch.cat([camera_task_embeddings] * 2)

        # Reshape and encode
        imgs_in, imgs_out = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W"), rearrange(
            imgs_out, "B Nv C H W -> (B Nv) C H W"
        )
        camera_task_embeddings = rearrange(camera_task_embeddings, "B Nv Nce -> (B Nv) Nce")
        camera_task_embeddings = self._apply_camera_embedding_encoding(camera_task_embeddings, cfg)

        imgs_in, imgs_out, camera_task_embeddings = (
            imgs_in.to(self.weight_dtype).to(imgs_out.device),
            imgs_out.to(self.weight_dtype),
            camera_task_embeddings.to(self.weight_dtype),
        )

        cond_vae_embeddings = self._encode_images_to_latents(imgs_in, is_condition=True)

        if imgs_out.shape[1] == 3:
            imgs_out = F.interpolate(
                imgs_out, size=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), mode="bilinear", align_corners=False
            )
            latents = self._encode_images_to_latents(imgs_out, is_condition=False)
        else:
            imgs_out = F.interpolate(imgs_out, size=(32, 32), mode="bilinear", align_corners=False)
            latents = imgs_out

        image_embeddings = self._encode_images_for_clip(imgs_in)

        # Add noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        if index is None:
            timesteps = self.noise_scheduler.timesteps[
                torch.randint(0, 4, (cfg.num_views // cfg.num_views,), device=latents.device)
            ].repeat_interleave(cfg.num_views * (bsz // cfg.num_views))
        else:
            timesteps = self.noise_scheduler.timesteps[index].repeat_interleave(
                cfg.num_views * (bsz // cfg.num_views)
            )

        timesteps = timesteps.long()

        if imgs_out.shape[1] == 3:
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        else:
            noisy_latents = latents

        # Apply classifier-free guidance
        if do_classifier_free_guidance:
            noisy_latents = torch.cat([noisy_latents] * 2)
            timesteps = torch.cat([timesteps] * 2)
            cond_vae_embeddings = torch.cat(
                [torch.zeros_like(cond_vae_embeddings), cond_vae_embeddings]
            )
            image_embeddings = torch.cat([torch.zeros_like(image_embeddings), image_embeddings])
            noisy_latents, image_embeddings, cond_vae_embeddings, camera_task_embeddings = (
                self._apply_classifier_free_guidance(
                    noisy_latents, image_embeddings, cond_vae_embeddings, camera_task_embeddings
                )
            )

        # Forward pass
        latent_model_input = torch.cat([noisy_latents, cond_vae_embeddings], dim=1)
        model_pred = self.unet(
            latent_model_input,
            timesteps,
            encoder_hidden_states=image_embeddings,
            class_labels=camera_task_embeddings,
        ).sample

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = self.reshape_to_cfg_output(model_pred).chunk(2)
            noisy_latents_cfg = self.reshape_to_cfg_output(noisy_latents)
            model_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            _, noisy_latents = noisy_latents_cfg.chunk(2)
            _, timesteps = timesteps.chunk(2)

        # Denoise step
        step_output = self.noise_scheduler.step(model_pred, timesteps[0].cpu(), noisy_latents)
        latents_prev = step_output.prev_sample
        latents_0 = step_output.pred_original_sample

        # Decode to image space
        image = self.vae.decode(latents_0 / self.vae.config.scaling_factor, return_dict=False)[0]
        image = (image + 1.0) / 2.0

        output = {
            "imgs_final": image,
            "noise_pred": model_pred,
            "input_latents": latents,
            "x_0": latents_0,
            "noise": noise,
        }
        if noise is not None:
            output["x_prev"] = latents_prev

        return output

    def reshape_to_cfg_output(self, output):
        """
        Reshape output for classifier-free guidance.
        
        Args:
            output: Output tensor to reshape
            
        Returns:
            Reshaped tensor
        """
        output_norm_uc, output_norm_cond, output_rgb_uc, output_rgb_cond = torch.chunk(
            output, dim=0, chunks=4
        )
        return torch.cat(
            [output_norm_uc, output_rgb_uc, output_norm_cond, output_rgb_cond], dim=0
        )

    def log_validation_joint(self, cond_img, elevations_cond, elevations, azimuths):
        """
        Log validation results for joint prediction.
        
        Args:
            cond_img: Conditioning images
            elevations_cond: Conditioning elevation angles
            elevations: Target elevation angles
            azimuths: Azimuth angles
            
        Returns:
            Generated images
        """
        camera_embeddings = self._prepare_camera_embeddings(elevations_cond, elevations, azimuths)
        cond_img = torch.cat([cond_img.unsqueeze(0)] * self.cfg.num_views, dim=0).unsqueeze(0)
        cfg = self.cfg
        start_time = time.time()

        imgs_in = torch.cat([cond_img] * 2, dim=0)
        camera_embeddings = torch.cat([camera_embeddings] * 2, dim=0)
        task_embeddings = self._prepare_task_embeddings(use_normal=True)
        camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)

        imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")
        camera_embeddings = rearrange(camera_embeddings, "B Nv Nce -> (B Nv) Nce")

        with torch.autocast("cuda"):
            for guidance_scale in cfg.validation_guidance_scales:
                out = self.pipeline(
                    imgs_in,
                    camera_embeddings,
                    generator=self.generator,
                    guidance_scale=guidance_scale,
                    output_type="pt",
                    num_images_per_prompt=1,
                    **cfg.pipe_validation_kwargs,
                )
                out = out.images

        end_time = time.time()
        print("pipeline:", end_time - start_time)
        torch.cuda.empty_cache()

        return out

    def init_unet(self):
        """
        Initialize UNet model from pretrained checkpoint.
        
        Loads pretrained weights and sets up the model for training.
        """
        if self.unet is None:
            model = UNetSD_I2VGen(**dict(self.cfg.UNet))
            pretrained_model_name_or_path = "/data/lxy/VideoMV/pretrained_models/i2v_882000.pth"

            checkpoint_dict = torch.load(pretrained_model_name_or_path, map_location="cpu")
            state_dict = checkpoint_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            model = model.to(self.device)
            model.train()
            torch.cuda.empty_cache()
            self.unet = model
            for name, param in model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True

    def init(self):
        """
        Initialize diffusion process state.
        
        Sets up initial noise and timesteps for DDIM sampling.
        """
        noise = torch.randn(
            [
                1,
                4,
                self.cfg.max_frames,
                int(self.cfg.resolution[1] / self.cfg.scale),
                int(self.cfg.resolution[0] / self.cfg.scale),
            ]
        )
        self.xt = noise.to(self.device)
        self.t_idx = 0
        self.noise = noise
        self.t_steps = (
            (1 + torch.arange(0, self.num_timesteps, self.num_timesteps // self.cfg.ddim_timesteps))
            .clamp(0, self.num_timesteps - 1)
            .flip(0)
        )

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        """
        Set minimum and maximum timestep steps for training.
        
        Args:
            min_step_percent: Minimum step percentage
            max_step_percent: Maximum step percentage
        """
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @property
    def pipe(self):
        """Get the diffusion pipeline."""
        return self.submodules.pipe

    @property
    def pipe_lora(self):
        """Get the LoRA-enabled diffusion pipeline."""
        return self.submodules.pipe_lora

    def encode_images_gt(self, img):
        """
        Encode ground truth images to latent space and extract features.
        
        Args:
            img: Input images
            
        Returns:
            Tuple of (latents, visual_features) or (latents, visual_features, dino_features)
        """
        with torch.no_grad():
            img_tensor = F.resize(
                img, (self.cfg.vit_resolution[0], self.cfg.vit_resolution[1])
            ).to(img.device)
            normalize = Normalize(mean=self.cfg.vit_mean, std=self.cfg.vit_std)
            img_tensor = normalize(img_tensor)
            image_tensor = img_tensor.to(img.device)

            y_visual, y_text, y_words = self.clip_encoder(image=image_tensor, text="")

            if self.cfg.use_dino:
                features_dict = self.dino_model.forward_features(image_tensor)
                features = features_dict["x_norm_patchtokens"]
                features = features.view(24, 16, 16, 384).permute(0, 3, 1, 2)
                features = self.upsample(features)

            local_image = None
            for i in range(img.shape[0]):
                encoded = self.autoencoder.encode_firsr_stage(
                    img[i].unsqueeze(0), self.cfg.scale_factor
                ).detach()
                if local_image is None:
                    local_image = encoded
                else:
                    local_image = torch.cat([local_image, encoded], dim=0)

            if self.cfg.use_dino:
                return local_image, features, y_visual
            return local_image, y_visual

    def encode_images_normal(self, img):
        """
        Encode normal maps to latent space.
        
        Args:
            img: Input normal maps
            
        Returns:
            Encoded latent representations
        """
        with torch.no_grad():
            img_tensor = F.resize(
                img, (self.cfg.vit_resolution[0], self.cfg.vit_resolution[1])
            ).to(img.device)
            normalize = Normalize(mean=self.cfg.vit_mean, std=self.cfg.vit_std)
            img_tensor = normalize(img_tensor)

            local_image = None
            for i in range(img.shape[0]):
                encoded = self.autoencoder.encode_firsr_stage(
                    img[i].unsqueeze(0), self.cfg.scale_factor
                ).detach()
                if local_image is None:
                    local_image = encoded
                else:
                    local_image = torch.cat([local_image, encoded], dim=0)

            return local_image

    def add_noise(self, img):
        """
        Add noise to images for diffusion process.
        
        Args:
            img: Input images
            
        Returns:
            Noised images
        """
        noised_img = self.diffusion.add_noise(
            img, self.t_recur, self.alphas_prev_recur, self.sigmas_recur, self.eps_recur, self.noise
        )
        self.xt = noised_img
        return noised_img

    def encode_image(self, img):
        """
        Encode a single image for inference.
        
        Args:
            img: PIL Image to encode
            
        Returns:
            Tuple of encoded features and metadata
        """
        train_trans = Compose(
            [
                CenterCropWide(size=self.cfg.resolution),
                ToTensor(),
                Normalize(mean=self.cfg.mean, std=self.cfg.std),
            ]
        )

        vit_trans = Compose(
            [
                CenterCropWide(size=(self.cfg.resolution[0], self.cfg.resolution[0])),
                Resize(self.cfg.vit_resolution),
                ToTensor(),
                Normalize(mean=self.cfg.vit_mean, std=self.cfg.vit_std),
            ]
        )

        with torch.no_grad():
            image_tensor = vit_trans(img)
            image_tensor = image_tensor.unsqueeze(0)
            y_visual, y_text, y_words = self.clip_encoder(image=image_tensor, text="")
            y_visual = y_visual.unsqueeze(1)

        _, _, zero_y = self.clip_encoder(text="")
        _, _, zero_y_negative = self.clip_encoder(text="")
        zero_y, zero_y_negative = zero_y.detach(), zero_y_negative.detach()
        black_image_feature = torch.zeros([1, 1, 1024])

        fps_tensor = torch.tensor([self.cfg.target_fps], dtype=torch.long, device=self.device)
        image_id_tensor = train_trans([img]).to(self.device)
        local_image = self.autoencoder.encode_firsr_stage(
            image_id_tensor, self.cfg.scale_factor
        ).detach()
        local_image = local_image.unsqueeze(2).repeat_interleave(repeats=self.cfg.max_frames, dim=2)
        return y_words, y_visual, local_image, fps_tensor, zero_y_negative, black_image_feature

    def deconde_image(self, video_data):
        """
        Decode video data from latent space to image space.
        
        Args:
            video_data: Video data in latent space (B, C, F, H, W)
            
        Returns:
            Decoded video data in image space
        """
        video_data = 1.0 / self.cfg.scale_factor * video_data
        video_data = rearrange(video_data, "b c f h w -> (b f) c h w")
        chunk_size = min(self.cfg.decoder_bs, video_data.shape[0])
        video_data_list = torch.chunk(video_data, video_data.shape[0] // chunk_size, dim=0)
        decode_data = []
        for vd_data in video_data_list:
            gen_frames = self.autoencoder.decode(vd_data)
            decode_data.append(gen_frames)
        video_data = torch.cat(decode_data, dim=0)
        video_data = rearrange(video_data, "(b f) c h w -> b c f h w", b=self.cfg.batch_size)
        vid_mean = torch.tensor(self.cfg.mean, device=video_data.device).view(1, -1, 1, 1, 1)
        vid_std = torch.tensor(self.cfg.std, device=video_data.device).view(1, -1, 1, 1, 1)

        video_data = video_data.mul_(vid_std).add_(vid_mean)
        video_data = video_data.clamp_(0, 1)

        return video_data
