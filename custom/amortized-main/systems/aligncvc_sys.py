import os
import time
import math
from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from PIL import Image
from einops import rearrange

import torchvision.transforms as transforms

from lpips import LPIPS
from pytorch_msssim import MS_SSIM
import vision_aided_loss

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import get_rank


# =========================
# Utility
# =========================

def get_image_paths(folder):
    exts = (".jpg", ".png", ".jpeg", ".webp")
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ]


# =========================
# Main System
# =========================

@threestudio.register("aligncvc-system-clean")
class AlignCVCSystem(BaseLift3DSystem):

    @dataclass
    class Config(BaseLift3DSystem.Config):
        train_guidance: bool = True
        visualize_samples: bool = False

    cfg: Config

    # =========================
    # Setup
    # =========================

    def configure(self):
        super().configure()

        self.lpips = LPIPS(net="alex").cuda().eval()
        self.ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)

        self.writer = None
        self.tensorboard_interval = 50

        if self.cfg.guidance.use_gan_clip:
            self.net_disc = vision_aided_loss.Discriminator(
                cv_type="clip",
                loss_type="multilevel_sigmoid_s",
                device="cuda",
            ).cuda()
            self.net_disc.requires_grad_(True)

        if self.cfg.train_guidance:
            self.guidance = threestudio.find(
                self.cfg.guidance_type
            )(self.cfg.guidance)

        self._setup_default_infer_image()

    def _setup_default_infer_image(self):
        img = Image.new("RGB", (256, 256), (255, 255, 255))
        self.infer_img = transforms.ToTensor()(img)

    # =========================
    # Core Helpers
    # =========================

    def _render_2dgs(self, batch):
        out_gs, _ = self.geometry.gs_network.render(
            batch["tar_rgb"],
            batch["Ks"],
            batch["cam_poses_lgm"],
            batch["fovy"],
            batch["elevation"],
            batch["camera_distances"],
            -batch["azimuth"],
        )
        rgb = out_gs["image"][0].permute(0, 2, 3, 1)
        alpha = out_gs["alpha"][0]
        return rgb, alpha

    def _prepare_gs_batch(self, batch, imgs, x0):
        batch["tar_rgb"] = imgs.unsqueeze(0)
        batch["tar_latents"] = x0.unsqueeze(0)

        batch["Ks"] = (
            self.FOV_to_intrinsics(batch["fovy"][0])
            .unsqueeze(0)
            .repeat(4, 1, 1)
            .unsqueeze(0)
            .float()
        )

    # =========================
    # Forward (Inference)
    # =========================

    @torch.no_grad()
    def forward(self, batch, noise=None, time_step=0):
        out = {}

        infer_img = self.infer_img.to(batch["gt_rgb"].device)
        elevation = batch["elevation"]
        azimuth = -batch["azimuth"] + batch["azimuth"][2]

        use_normal = self.guidance.cfg.use_normal

        output = self.guidance.eval_one_step(
            batch=batch,
            cond_img=infer_img.unsqueeze(0),
            elevations_cond=elevation[2:3],
            elevations=elevation,
            azimuths=azimuth,
            index=time_step,
            use_gen=noise is not None,
            use_normal=use_normal,
        )

        imgs = output["imgs_final"]
        x0 = output["x_0"]

        if use_normal:
            rgb = imgs[4:]
        else:
            rgb = imgs

        self._prepare_gs_batch(batch, rgb, x0)
        rgb_render, alpha = self._render_2dgs(batch)

        out["rgb"] = rgb_render
        out["gt_rgb"] = batch["gt_rgb"]

        return out

    # =========================
    # Forward (Training)
    # =========================

    def forward_gen(self, batch, test_batch=None):
        out = {}

        input_index = 2
        infer_img = batch["gt_rgb"][input_index:input_index+1]

        elevation = batch["elevation"]
        azimuth = -batch["azimuth"] + batch["azimuth"][input_index]

        use_normal = self.guidance.cfg.use_normal

        # ---- Diffusion GT ----
        output_gt = self.guidance.train_one_step(
            batch=batch,
            cond_img=infer_img,
            elevations_cond=elevation[input_index:input_index+1],
            elevations=elevation,
            azimuths=azimuth,
            use_normal=use_normal,
        )

        imgs = output_gt["imgs_final"]
        x0 = output_gt["x_0"]

        if use_normal:
            rgb = imgs[4:]
        else:
            rgb = imgs

        # ---- 2DGS ----
        self._prepare_gs_batch(batch, rgb, x0)
        rgb_render, _ = self._render_2dgs(batch)

        out["rgb"] = rgb_render
        out["gt_rgb"] = batch["gt_rgb"]

        # ---- ASD ----
        if self.cfg.guidance.use_asd:
            batch["asd_rgb"] = rgb
            asd = self.guidance.train_with_asd(
                batch=batch,
                cond_img=infer_img,
                elevations_cond=elevation[input_index:input_index+1],
                elevations=elevation,
                azimuths=azimuth,
                use_normal=use_normal,
            )
            out["loss_asd"] = asd["loss_asd"]

        return out

    # =========================
    # Training Step
    # =========================

    def training_step(self, batch, batch_idx):
        if self.writer is None:
            self.writer = SummaryWriter(self.get_save_dir())

        batch, supervise = batch
        out = self.forward_gen(batch, supervise)

        rgb = out["rgb"]
        gt_rgb = supervise["gt_rgb"].permute(0, 2, 3, 1)

        loss = 0.0

        # ---- GAN ----
        if self.cfg.guidance.use_gan_clip:
            loss_g = self.net_disc(rgb.permute(0, 3, 1, 2), for_G=True).mean()
            loss_d_r = self.net_disc(gt_rgb.permute(0, 3, 1, 2), for_real=True).mean()
            loss_d_f = self.net_disc(rgb.permute(0, 3, 1, 2), for_real=False).mean()
            loss_gan = (loss_g + loss_d_r + loss_d_f) * self.cfg.loss.lambda_gan
            loss += loss_gan

        # ---- ASD ----
        if "loss_asd" in out:
            loss += out["loss_asd"] * self.cfg.loss.lambda_asd

        self.log("train/loss", loss)
        return {"loss": loss}