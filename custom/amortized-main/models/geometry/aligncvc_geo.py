import os
from dataclasses import dataclass, field
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import grid_sample
import skimage
from torch.nn import TransformerEncoderLayer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threestudio
from threestudio.models.mesh import Mesh
from threestudio.models.geometry.base import BaseImplicitGeometry, contract_to_unisphere
from threestudio.utils.misc import broadcast, get_rank, C
from threestudio.utils.typing import *
from threestudio.utils.ops import get_activation
from threestudio.models.networks import get_encoding, get_mlp
from einops import rearrange
import matplotlib.pyplot as plt
#from torchsparse.tensor import SparseTensor

from torchviz import make_dot

import trimesh
import time

from .lgm.models import LGM
from .lgm.options import AllConfigs
from .lgm.utils import get_rays, grid_distortion, orbit_camera_jitter

from safetensors.torch import load_file
import tyro




@threestudio.register("aligncvc-geo")
class AlignCVC(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
       
        output_size: int = 128
        fovy: float=39.6
        zfar: float=2.5
        znear: float=0.5
        n_view: int=32
        init_ckpt: str=""


  
    def configure(self) -> None:
        super().configure()
        #opt = tyro.cli(AllConfigs)
        

        self.gs_network= LGM()#(opt)
   
     
        ckpt = load_file(self.cfg.init_ckpt, device='cpu')
   
        self.gs_network.load_state_dict(ckpt, strict=False)
        self.gs_network= self.gs_network.train()

   
        



    def forward_gaussians(self, images):
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)

        x = self.unet(images) # [B*24, 14, h, w]
        x = self.conv(x) # [B*24, 14, h, w]

        x = x.reshape(B, 32, 14, 64,64) # hard code: 24??
        

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
        
        pos = self.pos_act(x[..., 0:3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = x[..., 11:]

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        
        return gaussians

   