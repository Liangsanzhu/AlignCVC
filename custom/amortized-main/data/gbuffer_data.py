import bisect
import math
import random
from dataclasses import dataclass, field
from threestudio.utils.misc import get_rank

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
import os
import cv2
from diso import DiffMC

import threestudio
from threestudio import register
from threestudio.models.mesh import Mesh
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_device
from threestudio.utils.ops import (
    get_full_projection_matrix,
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *
#计算camerapos的一些函数
import json
from PIL import Image
from torchvision import transforms
import torchvision

from mesh_to_sdf import mesh_to_voxels

import trimesh
from pytorch3d.renderer import (
    RasterizationSettings,
    TexturesVertex,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
)
from pytorch3d.renderer import MeshRasterizer
from pytorch3d.renderer.cameras import look_at_view_transform, OrthographicCameras, CamerasBase

def get_camera(dist,azim,elev,w2c=None, eye=None,fov_in_degrees=60, focal_length=1 / (2**0.5), cam_type='fov'):
    # pytorch3d expects transforms as row-vectors, so flip rotation: https://github.com/facebookresearch/pytorch3d/issues/1183
    """
      center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(camera_positions.shape[0], 1)

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0
        flip_yz = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                                [0, 0, 0, 1]])
        if c2w.ndim == 3:
            flip_yz = flip_yz.unsqueeze(0)
        c2w = torch.matmul(
            flip_yz.to(c2w), c2w)
    """
    
    if w2c is None:
        R, T = look_at_view_transform(dist, elev, azim)#, up=((0, 0,1),))
        conversion_matrix = torch.tensor([[1, 0, 0],
                                  [0, 0, 1],
                                  [0, -1, 0]]).float()
      

        
        # 坐标系转换
        #R = torch.matmul(conversion_matrix,R.T).T# 旋转矩阵转换
        #T = torch.matmul(conversion_matrix, T.T).T  # 平移矩阵转换
        w2c = torch.cat([R[0].T, T[0, :, None]], dim=1)
    R = w2c[:3, :3].t()[None, ...]
    T = w2c[:3, 3][None, ...]
    if cam_type == 'fov':
        camera = FoVPerspectiveCameras(device=w2c.device, R=R, T=T, fov=fov_in_degrees, degrees=True)
    else:
        focal_length = 1 / focal_length
        camera = FoVOrthographicCameras(device=w2c.device, R=R, T=T, min_x=-focal_length, max_x=focal_length, min_y=-focal_length, max_y=focal_length)
    return camera

@dataclass
class ObjDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_batch_size: int = 1
    n_val_views: int = 32 #val用的view数目
    n_test_views: int = 32 #test用的view数目
    num_sample:int=4 #每一次train用的view数目
    rgb_dir: str=""
    json_dir: str=""
    normal_dir : str=""
    depth_dir : str=""
    sdf_dir : str=""
    num_views: int=32 #数据集里每个三维模型总共渲染的view数目
    resolution: int=256 
    index: int=0
    fov: float=45
    dist: float=1.7
    elev: float=5
    infer: bool=False
    sdf_json_dir:str=""



class ObjIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any, mode="train") -> None:
        super().__init__()
        self.cfg: ObjDataModuleConfig = cfg
     
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        self.sdf_json_dir=self.cfg.sdf_json_dir

        self.batch_size: int = self.batch_sizes[0]
        
        self.rgb_dir = self.cfg.rgb_dir
        self.json_dir = self.cfg.json_dir
        self.normal_dir = self.cfg.normal_dir
        self.depth_dir = self.cfg.depth_dir
        self.sdf_dir = self.cfg.sdf_dir


        self.num_views = self.cfg.num_views
        self.num_sample = self.cfg.num_sample

        self.bg_color = (255,255,255)
        self.bg_blue_color= (127,127,255)
        self.resolution=self.cfg.resolution
        self.glb_name=self.cfg.index
        self.get_first_frame=0
        self.max_frames=24
        with open(self.sdf_json_dir, 'r') as file:
            self.index_data = json.load(file)


        image_transforms = [torchvision.transforms.Resize(self.resolution)]

        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x * 2. - 1.)])#, 'c h w -> h w c'))])
        self.tform = torchvision.transforms.Compose(image_transforms)

        self.heights=self.cfg.resolution
        self.widths=self.cfg.resolution
      
        self.mode=mode
        # 获取所有 .obj 文件的路径列表
        self.paths = self.get_rgb_file_paths(self.json_dir)
        self.path_len=len(self.paths)
        self.diffmc = DiffMC(dtype=torch.float32) # or dtype=torch.float64

        if mode=="train":
            self.path_begin=0 #850%(int(self.path_len*0.9)%self.path_len) #0
            self.path_end=231#int(self.path_len*0.9)%self.path_len#self.path_begin+1 #int(self.path_len*0.9)%self.path_len
        else:
            self.path_begin=int(self.path_len*0.9+1)%self.path_len+2
            self.path_end=int(self.path_len)#%self.path_len
           

        self.path_count=0
        self.path_len=self.path_end-self.path_begin

        #self.gt_data=self.load_gts(self.glb_name)

    def get_mvp_matrix(
        self,c2w: Float[Tensor, "B 4 4"], proj_mtx: Float[Tensor, "B 4 4"]
    ) -> Float[Tensor, "B 4 4"]:
        # calculate w2c from c2w: R' = Rt, t' = -Rt * t
        # mathematically equivalent to (c2w)^-1
        w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
        w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
        w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
        w2c[:, 3, 3] = 1.0
        # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
        mvp_mtx = proj_mtx @ w2c
        return mvp_mtx,w2c
    def get_rgb_file_paths(self,root_dir):
        #读取rgb的列表
        obj_file_paths = []
        with open(root_dir, 'r', encoding='utf-8') as file:
            lines_list = file.readlines()

        lines_without_newline = [line.strip() for line in lines_list]
        
        return lines_without_newline  
   
 
    def process_im_old(self, im):
        im = im.convert("RGB")
        return self.tform(im)
    

    def convert_opengl_to_blender(self,camera_matrix):
        if isinstance(camera_matrix, np.ndarray):
            # Construct transformation matrix to convert from OpenGL space to Blender space
            flip_yz = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0],
                                [0, 0, 0, 1]])
            camera_matrix_blender = np.dot(flip_yz, camera_matrix)
        else:
            #修改
            # Construct transformation matrix to convert from OpenGL space to Blender space
            flip_yz = torch.tensor([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                                [0, 0, 0, 1]])
          
            if camera_matrix.ndim == 3:
                flip_yz = flip_yz.unsqueeze(0)
            camera_matrix_blender = torch.matmul(
                flip_yz.to(camera_matrix), camera_matrix)
        return camera_matrix_blender
    def get_projection_matrix(
        self, fovy: Float[Tensor, "B"], aspect_wh: float, near: float, far: float
    ) -> Float[Tensor, "B 4 4"]:
        batch_size = fovy.shape[0]
        proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
        #print(proj_mtx.shape,torch.tan(fovy / 2.0).shape)
        
        proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
        proj_mtx[:, 1, 1] = -1.0 / torch.tan(
            fovy / 2.0
        )  # add a negative sign here as the y axis is flipped in nvdiffrast output
        proj_mtx[:, 2, 2] = -(far + near) / (far - near)
        proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
        proj_mtx[:, 3, 2] = -1.0
        return proj_mtx
    def get_projection_matrix_new(
        self, fovy: Float[Tensor, "B"], aspect_wh: float, near: float, far: float,w2c,
    ) -> Float[Tensor, "B 4 4"]:
        batch_size = fovy.shape[0]
        w=256
        f = (w / 2) /torch.tan(fovy[0] / 2.0)
    
        # 计算主点的坐标
        cx = w/ 2
        cy = w / 2
        
        # 构建内部参数矩阵
        intrinsic_matrix = torch.tensor([[f, 0, cx],
                                     [0, f, cy],
                                     [0, 0, 1]])
        affine_mat = np.eye(4)
        affine_mat[:3, :4] = intrinsic_matrix[:3, :3] @ w2c[:3, :4]
        return affine_mat
    def load_K_Rt_from_P(self,filename, P=None):
        if P is None:
            lines = open(filename).read().splitlines()
            if len(lines) == 4:
                lines = lines[1:]
            lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
            P = np.asarray(lines).astype(np.float32).squeeze()

        out = cv2.decomposeProjectionMatrix(P)
        K = out[0]
        R = out[1]
        t = out[2]

        K = K / K[2, 2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()  # ? why need transpose here
        pose[:3, 3] = (t[:3] / t[3])[:, 0]

        return intrinsics, pose  # ! return cam2world matrix here
    def getProjectionMatrix(self, znear, zfar, fovX, fovY): ##Hui: copy from graphics_utils.py; simplified calcuation
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 1 / tanHalfFovX
        P[1, 1] = 1 / tanHalfFovY
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P
    
    
    def orbit_camera(self,elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
        # radius: scalar
        # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
        # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
        # return: [4, 4], camera pose matrix
        if is_degree:
            elevation = np.deg2rad(elevation)
            azimuth = np.deg2rad(azimuth-90)
        x = radius * np.cos(elevation) * np.sin(azimuth)
        y = - radius * np.sin(elevation)
        z = radius * np.cos(elevation) * np.cos(azimuth)
        if target is None:
            target = np.zeros([3], dtype=np.float32)
        campos = np.array([x, y, z]) + target  # [3]
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = self.look_at(campos, target, opengl)
        T[:3, 3] = campos
        return T
    def make_gs_camera(self,c2w, elevation, azimuth, radius,fovy, is_degree=True, target=None, opengl=False):
        camera_centers=None
        full_proj_transforms=None
        world_view_transforms=None
        for j in range(len(elevation)):
            i=(len(elevation))%len(elevation)
            pose = self.orbit_camera(-elevation[i], azimuth[i], radius[i],is_degree,target=None, opengl=False)
            #pose=c2w[i]

            gs_w2c = np.linalg.inv(pose)

            world_view_transform = torch.tensor(gs_w2c).transpose(0, 1)
            projection_matrix = (
                self.getProjectionMatrix(
                    znear=0.5, zfar=2.5, fovX=float(fovy[0]), fovY=float(fovy[0])
                )
                .transpose(0, 1)
            )
            full_proj_transform = world_view_transform @ projection_matrix
            camera_center = -torch.tensor(pose[:3, 3]).unsqueeze(0)
            if camera_centers is None:
                camera_centers=camera_center
            else:
                camera_centers=torch.cat([camera_centers,camera_center],dim=0)
                
            full_proj_transform=full_proj_transform.unsqueeze(0)
            world_view_transform=world_view_transform.unsqueeze(0)
            if full_proj_transforms is None:
                full_proj_transforms=full_proj_transform
            else:
                full_proj_transforms=torch.cat([full_proj_transforms,full_proj_transform],dim=0)
            if world_view_transforms is None:
                world_view_transforms=world_view_transform
            else:
                world_view_transforms=torch.cat([world_view_transforms,world_view_transform],dim=0)
        return world_view_transforms,full_proj_transforms,camera_centers

       
    def safe_normalize(self,x, eps=1e-20):
        return x / self.length(x, eps)
    def length(self, x, eps=1e-20):
        if isinstance(x, np.ndarray):
            return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
        else:
            return torch.sqrt(torch.clamp(self.dot(x, x), min=eps))
    def look_at(self, campos, target, opengl=True):
        # campos: [N, 3], camera/eye position
        # target: [N, 3], object to look at
        # return: [N, 3, 3], rotation matrix
        if not opengl:
            # camera forward aligns with -z
            forward_vector = self.safe_normalize(target - campos)
            up_vector = np.array([0, 1, 0], dtype=np.float32)
            right_vector = self.safe_normalize(np.cross(forward_vector, up_vector))
            up_vector = self.safe_normalize(np.cross(right_vector, forward_vector))
        else:
            # camera forward aligns with +z
            forward_vector = self.safe_normalize(campos - target)
            up_vector = np.array([0, 1, 0], dtype=np.float32)
            right_vector = self.safe_normalize(np.cross(up_vector, forward_vector))
            up_vector = self.safe_normalize(np.cross(forward_vector, right_vector))
        R = np.stack([right_vector, -up_vector, forward_vector], axis=1)
        return R

    def dot(self,x, y):
        if isinstance(x, np.ndarray):
            return np.sum(x * y, -1, keepdims=True)
        else:
            return torch.sum(x * y, -1, keepdim=True)
    def create_camera_from_angle(self,elevation_deg, azimuth_deg, fovy_deg, camera_distances, device='cuda'):
        #主要修改了这里，elevation_deg, azimuth_deg, fovy_deg, camera_distances
        #跟MVDream对应，旋转矩阵计算方式相同
        '''
        :param elevation_deg: elevation angle of the camera
        :param azimuth_deg: azimuth angle of the camera
        :param fovy_deg: fovy angle of the camera
        :param camera_distances: distance from the camera to the origin
        :param device:
        :return:
        '''
        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(camera_positions.shape[0], 1)

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0
        flip_yz = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                                [0, 0, 0, 1]])
        if c2w.ndim == 3:
            flip_yz = flip_yz.unsqueeze(0)
        c2w = torch.matmul(
            flip_yz.to(c2w), c2w)
        fovy = fovy_deg * math.pi / 180

        proj_mtx: Float[Tensor, "B 4 4"] = self.get_projection_matrix(
            fovy, 1, 0.1,100.0
        ) 
       
        
        mvp_mtx,w2c = self.get_mvp_matrix(c2w, proj_mtx)
        frame=elevation_deg.shape[0]
        #c2w=c2w.reshape(1,frame,16)
        mvp_mtx=self.convert_opengl_to_blender(mvp_mtx)

        world_view_transform,full_proj_transform,camera_center=self.make_gs_camera(c2w,elevation_deg, azimuth_deg, fovy, camera_distances)
        results={}
        #results['camera_data']=camera_data
        results['cam_view'] = world_view_transform.unsqueeze(0)
        results['cam_view_proj'] = full_proj_transform.unsqueeze(0)
        results['cam_pos'] = camera_center.unsqueeze(0)
        gs_data = results
        return mvp_mtx,camera_positions,w2c,c2w,gs_data,proj_mtx
    
    def rigid_transform(self,xyz, transform):
        """Applies a rigid transform (c2w) to an (N, 3) pointcloud.
        """
        device = xyz.device
        xyz_h = torch.cat([xyz, torch.ones((len(xyz), 1)).to(device)], dim=1)  # (N, 4)
        xyz_t_h = (transform @ xyz_h.T).T  # * checked: the same with the below

        return xyz_t_h[:, :3]

    def get_view_frustum(self,min_depth, max_depth, size, cam_intr, c2w):
        """Get corners of 3D camera view frustum of depth image
        """
        device = cam_intr.device
        im_h, im_w = size
        im_h = int(im_h)
        im_w = int(im_w)
        view_frust_pts = torch.stack([
            (torch.tensor([0, 0, im_w, im_w, 0, 0, im_w, im_w]).to(device) - cam_intr[0, 2]) * torch.tensor(
                [min_depth, min_depth, min_depth, min_depth, max_depth, max_depth, max_depth, max_depth]).to(device) /
            cam_intr[0, 0],
            (torch.tensor([0, im_h, 0, im_h, 0, im_h, 0, im_h]).to(device) - cam_intr[1, 2]) * torch.tensor(
                [min_depth, min_depth, min_depth, min_depth, max_depth, max_depth, max_depth, max_depth]).to(device) /
            cam_intr[1, 1],
            torch.tensor([min_depth, min_depth, min_depth, min_depth, max_depth, max_depth, max_depth, max_depth]).to(
                device)
        ])
        view_frust_pts = view_frust_pts.type(torch.float32)
        view_frust_pts = self.rigid_transform(view_frust_pts.T, c2w).T
        return view_frust_pts
    def get_boundingbox(self, img_hw, intrinsics, extrinsics):
        """
        # get the minimum bounding box of all visual hulls
        :param img_hw:
        :param intrinsics:
        :param extrinsics:
        :param near_fars:
        :return:
        """

        bnds = torch.zeros((3, 2))
        bnds[:, 0] = np.inf
        bnds[:, 1] = -np.inf

        
        num = extrinsics.shape[0]

        for i in range(num):
            if not isinstance(intrinsics, torch.Tensor):
                cam_intr = torch.tensor(intrinsics)
                w2c = torch.tensor(extrinsics[i])
                c2w = torch.inverse(w2c)
            else:
                cam_intr = intrinsics
                w2c = extrinsics[i]
                c2w = torch.inverse(w2c)
            min_depth, max_depth =0.5,2.5
            # todo: check the coresponding points are matched
            view_frust_pts = self.get_view_frustum(min_depth, max_depth, img_hw, cam_intr, c2w)
            bnds[:, 0] = torch.min(bnds[:, 0], torch.min(view_frust_pts, dim=1)[0])
            bnds[:, 1] = torch.max(bnds[:, 1], torch.max(view_frust_pts, dim=1)[0])

        center = torch.tensor(((bnds[0, 1] + bnds[0, 0]) / 2, (bnds[1, 1] + bnds[1, 0]) / 2,
                            (bnds[2, 1] + bnds[2, 0]) / 2))

        lengths = bnds[:, 1] - bnds[:, 0]

        max_length, _ = torch.max(lengths, dim=0)
        radius = max_length / 2

        return center, radius, bnds
    def cal_scale_mat(self, img_hw, intrinsics, extrinsics,factor=1.):
        center, radius, _ = self.get_boundingbox(img_hw, intrinsics, extrinsics)
        radius = radius * factor
        scale_mat = np.diag([radius, radius, radius, 1.0])
        scale_mat[:3, 3] = center.cpu().numpy()
        scale_mat = scale_mat.astype(np.float32)

        return scale_mat, 1. / radius.cpu().numpy()
    def unity2blender(self,normal):
        normal_clone = normal.copy()
        normal_clone[...,0] = -normal[...,-1]
        normal_clone[...,1] = -normal[...,0]
        normal_clone[...,2] = normal[...,1]

        return normal_clone

    def blender2midas(self,img):
        '''Blender: rub
        midas: lub
        '''
        img[...,0] = -img[...,0]
        img[...,1] = -img[...,1]
        img[...,-1] = -img[...,-1]
        return img
    def process_im_np(self,im_array):
        # 对原过程中的 Image 操作进行等效的 NumPy 实现
        # 示例：假设 process_im 只包含简单的亮度调整
        im_array = (im_array.astype(float)/255.0)  # 示例处理
        return torch.tensor(im_array)
    def read_camera_matrix_single(self,json_file):
        with open(json_file, 'r', encoding='utf8') as reader:
            json_content = json.load(reader)

        cond_camera_matrix = np.eye(4)
        cond_camera_matrix[:3, 0] = np.array(json_content['x'])
        cond_camera_matrix[:3, 1] = -np.array(json_content['y'])
        cond_camera_matrix[:3, 2] = -np.array(json_content['z'])
        cond_camera_matrix[:3, 3] = np.array(json_content['origin'])


        camera_matrix = np.eye(4)
        camera_matrix[:3, 0] = np.array(json_content['x'])
        camera_matrix[:3, 1] = np.array(json_content['y'])
        camera_matrix[:3, 2] = np.array(json_content['z'])
        camera_matrix[:3, 3] = np.array(json_content['origin'])
        cond_camera_matrix=camera_matrix
        rotation_matrix = camera_matrix[:3, :3]

        # 计算相机的仰角（俯视角度）
        direction_vector = cond_camera_matrix[:3, 2]
    
        # 计算方位角 Azimuth (从x轴开始，逆时针方向)
        azimuth = np.arctan2(direction_vector[1], direction_vector[0])
        
        # 计算仰角 Elevation (从水平面开始，向上为正)
        # 注意这里使用arcsin，因为arccos可能会导致精度问题
        # 另外，由于arcsin的输出范围是[-π/2, π/2]，所以直接用arcsin计算仰角
        elevation = np.arcsin(direction_vector[2])
    

        # 将弧度转换为角度
        elevation = -np.degrees(elevation)
        azimuth = np.degrees(azimuth)       
        camera_distance = torch.tensor(cond_camera_matrix).inverse()[2, 3]


        y_fov=json_content['y_fov']
        y_fov=math.degrees(y_fov)

        return camera_matrix, cond_camera_matrix, y_fov,elevation,camera_distance,azimuth
    
    def load_gts(self,index):
        json_path = self.paths[index]
        
        parts = json_path.split('/')

                # 获取最后两节的路径
                
        last_two_parts = '/'.join(parts[-2:])
        sdf_path=self.sdf_dir+"/"+self.index_data[last_two_parts].split(".")[0]+"/"+"sdf.npy"
        print(sdf_path)
        prefix = json_path
       

        rgb_path = [os.path.join(prefix, "{:05d}/{:05d}.png".format(frame_idx, frame_idx)) for frame_idx in range(24)]
        ab_path = [os.path.join(prefix, "{:05d}/{:05d}_albedo.png".format(frame_idx, frame_idx)) for frame_idx in range(24)]

        camera_path = [os.path.join(prefix, "{:05d}/{:05d}.json".format(frame_idx, frame_idx)) for frame_idx in range(24)]
        normal_path = [os.path.join(prefix, "{:05d}/{:05d}_nd.exr".format(frame_idx, frame_idx)) for frame_idx in range(24)]

     
     
       
        #print(normal.shape,rgb_im.shape)
      
        i=0
       
        filename=json_path

        fovs=[]      
        rgb_tensors_in=[]
        normal_tensors_in=[]
        depth_tensors_in=[]

        azimuths=[]
        distances=[]
        elevations=[]
        mvp_mtx=[]

    
        #print(rgb_path)     



        #mesh = trimesh.load('/data/lxy/ScaleDreamer_v1-main/load/75fcd5d2b39b44eabd18042148b165b2.obj')
        #sdf_path="/data/lxy/ScaleDreamer_v1-main/che.npy"
        try:
            voxels = np.load(sdf_path)
            print(sdf_path)
            voxels=torch.tensor(voxels).float()
            voxels[voxels!=voxels]=-1.0
        except:
            sdf_path="/data/lxy/ScaleDreamer_v1-main/che.npy"
            voxels = np.load(sdf_path)
            print(sdf_path)
            voxels=torch.tensor(voxels).float()
        #print(voxels.shape,voxels[voxels==0])

        for j in range(self.num_views):

            i = int((-j+self.num_views/2+self.num_views)%self.num_views)

            # 读取 RGB 图像并调整尺寸
            try:
                rgb_im = cv2.imread(rgb_path[i], cv2.IMREAD_UNCHANGED)
                b, g, r, a = cv2.split(rgb_im)
            except:
                try:
                    rgb_im = cv2.imread(ab_path[i], cv2.IMREAD_UNCHANGED)
                    b, g, r, a = cv2.split(rgb_im)
                    print("Error:ab")
                except:

                    temp="/home/xinyue_liang/lxy/gbuffer/data/4/30373/00000/00000.png"
                    rgb_im = cv2.imread(temp, cv2.IMREAD_UNCHANGED)
                    b, g, r, a = cv2.split(rgb_im)
                    print("Error:temp")

            # 将BGR重组为RGB，同时保留Alpha通道
            rgb_im = cv2.merge([r, g, b, a])

            rgb_im = cv2.resize(rgb_im, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)

            # 更换 RGB 背景
            mask = rgb_im[:,:,3] != 0  # 使用alpha通道作为遮罩
            bg_color = np.array(self.bg_color).reshape([1, 1, 3]).astype(rgb_im.dtype)
            rgb_im[:,:,:3][~mask] = bg_color[0][0]
            rgb_im = rgb_im[:,:,:3]  # 移除alpha通道
            
            # 处理 RGB 图像
            rgb_im = self.process_im_np(rgb_im)
            rgb_tensors_in.append(rgb_im)
            #print(normal_path[i])
            # 类似地处理 Normal 和 Depth 图像
            try:
                img_nd = cv2.imread(normal_path[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
                #print(img_nd.shape)
                b, g, r, a = cv2.split(img_nd)
                # 将BGR重组为RGB，同时保留Alpha通道
                img_nd = cv2.merge([r, g, b, a])

                img_nd = cv2.resize(img_nd, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)

                normal = img_nd[..., :3]  # in [-1, 1], bg is [0, 0, 1]
                depth = img_nd[..., 3:4] 
                depth=torch.cat([torch.tensor(depth)]*3,dim=-1).float()
                #print(normal)
                normal=(normal+1.0)/2.0
            except:
                normal=rgb_im
                depth=rgb_im

            normal_tensors_in.append(torch.tensor(normal).float())
            depth_tensors_in.append(torch.tensor(depth).float())

            camera_matrix, cond_camera_matrix, y_fov,elevation,camera_distance,azimuth = self.read_camera_matrix_single(camera_path[i])
            

            #print(y_fov,elevation,camera_distance,azimuth)
            distance=camera_distance
            elevation=elevation
            fov=y_fov
            
            # 添加其他元数据
            fovs.append(torch.tensor(fov))
            elevations.append(torch.tensor(elevation))
            azimuth = 360 * (-i + self.num_views) / self.num_views
            azimuths.append(torch.tensor(azimuth))
            distances.append(torch.tensor(distance))

        rgb_tensors_in =torch.stack(rgb_tensors_in, dim=0).float().permute(0, 3, 1,2) # (Nv, 3, H, W)
        normal_tensors_in =torch.stack(normal_tensors_in, dim=0).float().permute(0,3, 1,2) # (Nv, 3, H, W)
        depth_tensors_in =torch.stack(depth_tensors_in, dim=0).float().permute(0,3, 1,2) # (Nv, 3, H, W)

        
        rgb_tensors_in[rgb_tensors_in != rgb_tensors_in] = 0.
        normal_tensors_in[normal_tensors_in != normal_tensors_in] = 0.
        depth_tensors_in[depth_tensors_in != depth_tensors_in] = 0.

        elevations = torch.stack(elevations, dim=0).float()#.squeeze(1)
        azimuths =torch.stack(azimuths, dim=0).float()#.squeeze(1)
        distances = torch.stack(distances, dim=0).float()

        #计算旋转矩阵
        fovs = torch.stack(fovs, dim=0).float()

        mvp_mtx,camera_positions,w2c,c2w,gs_data,proj_mtx=self.create_camera_from_angle(elevations, azimuths, fovs, distances)
        py3d_cams=[]
        for j in range(self.num_views):
            i=(-j+12-6)%self.num_views
            py3d_cam=get_camera(distances[i], azimuths[i],elevations[i],None,eye=camera_positions[i],fov_in_degrees=fovs[i], focal_length=1 / (2**0.5), cam_type='fov')
            py3d_cams.append(py3d_cam)
        w=32
        f = (w / 2) /torch.tan(torch.deg2rad(fovs[0]) / 2.0)
    
        # 计算主点的坐标
        cx = w/ 2
        cy = w / 2
        
        # 构建内部参数矩阵
        intrinsic = torch.tensor([[f, 0, cx],
                                     [0, f, cy],
                                     [0, 0, 1]])
        affine_mats=[]
        """
        scale_mat,scale_factor = self.cal_scale_mat(
            img_hw=[w, w],
            intrinsics=intrinsic,
            extrinsics=w2c,
            factor=1.0)
        """
     
      
        for i in range(self.num_views):
            """
            P = intrinsic[:3, :3] @ w2c[i][ :3, :4] @ scale_mat
            P = P.cpu().numpy()[:3, :4]

            # - should use load_K_Rt_from_P() to obtain c2w
            c2w_new = self.load_K_Rt_from_P(None, P)[1]
            w2c_new = np.linalg.inv(c2w_new)
            """
         
            affine_mat = np.eye(4)
            affine_mat[:3, :4] = intrinsic[:3, :3] @ w2c[i, :3, :4]
            affine_mats.append(torch.tensor(affine_mat))
            #light_pos暂时跟camera_pos相同
        affine_mats = torch.stack(affine_mats, dim=0).float()


        light_positions=camera_positions
            
       

        out =  {
            'height':self.heights,
            'width':self.widths,
            'light_positions':light_positions,
            'camera_positions':camera_positions,
            'elevation': elevations,
            'azimuth': azimuths,
            'camera_distances': distances,
            'gt_rgb': rgb_tensors_in,
            'gt_normal': normal_tensors_in,
            'gt_depth': depth_tensors_in,
            'filename': filename,
            "fovy": fovs,
            "index": index,
            "mvp_mtx":mvp_mtx,
            "w2c":w2c,
            "c2w":c2w,
            "proj_mtx":proj_mtx,
            "gs_data":gs_data,
            "gt_sdf":voxels,
            "affine_mats":affine_mats,
            "py3d_cams":py3d_cams
            
        }

        return out
        
    
     
    def load_infer(self,distance,elevation,fov):
        

        #读取json中的参数（可以改成直接传参数）
      
        fovs=[]      
        rgb_tensors_in=[]
        normal_tensors_in=[]
        depth_tensors_in=[]

        azimuths=[]
        distances=[]
        elevations=[]
        mvp_mtx=[]

       

        #print(rgb_path)     



        #mesh = trimesh.load('/data/lxy/ScaleDreamer_v1-main/load/75fcd5d2b39b44eabd18042148b165b2.obj')
        #sdf_path="/data/lxy/ScaleDreamer_v1-main/che.npy"
      

        for j in range(self.num_views):
            i = (j) % self.num_views
         
            
            # 添加其他元数据
            fovs.append(torch.tensor(fov))
            elevations.append(torch.tensor(elevation))
            azimuth = 360 * (i) / self.num_views
            azimuths.append(torch.tensor(azimuth))
            distances.append(torch.tensor(distance))

        elevations = torch.stack(elevations, dim=0).float()#.squeeze(1)
        azimuths =torch.stack(azimuths, dim=0).float()#.squeeze(1)
        distances = torch.stack(distances, dim=0).float()

        #计算旋转矩阵
        fovs = torch.stack(fovs, dim=0).float()

        mvp_mtx,camera_positions,w2c,c2w,gs_data,proj_mtx=self.create_camera_from_angle(elevations, azimuths, fovs, distances)
        py3d_cams=[]
        for j in range(self.num_views):
            i=(-j+16-8)%32
            py3d_cam=get_camera(distances[i], azimuths[i],elevations[i],None,eye=camera_positions[i],fov_in_degrees=fovs[i], focal_length=1 / (2**0.5), cam_type='fov')
            py3d_cams.append(py3d_cam)

       
        

        #light_pos暂时跟camera_pos相同
        light_positions=camera_positions
            
       

        out =  {
            'height':self.heights,
            'width':self.widths,
            'light_positions':light_positions,
            'camera_positions':camera_positions,
            'elevation': elevations,
            'azimuth': azimuths,
            'camera_distances': distances,
            "fovy": fovs,
            "mvp_mtx":mvp_mtx,
            "w2c":w2c,
            "c2w":c2w,
            "gs_data":gs_data,
            "proj_mtx":proj_mtx,
            "py3d_cams":py3d_cams

        }

        return out
        
    
     


    
    def __iter__(self):
        if self.mode=="train":
            size=self.path_end-self.path_begin
            rank=get_rank()
            while True:
                for path_name in range(self.path_begin,self.path_end):
                    yield {"path_name":(path_name+rank-self.path_begin)%size+self.path_begin}
        elif self.mode=="val":
            size=self.path_end-self.path_begin
            rank=get_rank()
            for path_name in range(self.path_begin,self.path_end):
                yield {"path_name":(path_name+rank-self.path_begin)%size+self.path_begin}
                break
        else:
            yield {"path_name":-1}
        


  
    def collate(self, batch): #-> Dict[str, Any]:
        #选取其中self.num_sample个渲染，否则可能遇到显存不够的情况
        index = int(self.num_views/self.num_sample)*torch.arange(self.num_sample)
        
        if self.mode=="test":
            gt_data=self.load_gts(batch["path_name"])
            """
            gt_data=self.load_infer(self.cfg.dist,self.cfg.elev,self.cfg.fov)
            return  {
            'height':self.heights,
            'width':self.widths,
            'light_positions':gt_data["light_positions"][index],
            'camera_positions':gt_data["camera_positions"][index],
            'elevation': gt_data["elevation"][index],
            'azimuth': gt_data["azimuth"][index],
            'camera_distances': gt_data["camera_distances"][index],
            "gs_data":gt_data["gs_data"],
            "fovy": gt_data["fovy"][index],
            "mvp_mtx":gt_data["mvp_mtx"][index],
            "w2c":gt_data["w2c"][index],
            "c2w":gt_data["c2w"][index],
            "proj_mtx":gt_data["proj_mtx"][index],

            "py3d_cams":gt_data["py3d_cams"],}
            """
        elif self.mode=="train":
            gt_data=self.load_gts(batch["path_name"])

        elif self.mode=="val":
            gt_data=self.load_gts(batch[0]["path_name"])
        
       
         #torch.randperm(self.num_views)[:self.num_sample]
        #print("my_mvp:",self.gt_data["mvp_mtx"][index])

        return  {
            'height':self.heights,
            'width':self.widths,
            'light_positions':gt_data["light_positions"][index],
            'camera_positions':gt_data["camera_positions"][index],
            'elevation': gt_data["elevation"][index],
            'azimuth': gt_data["azimuth"][index],
            'camera_distances': gt_data["camera_distances"][index],
            'gt_rgb': gt_data["gt_rgb"][index],
            'gt_normal': gt_data["gt_normal"][index],
            "gt_sdf":gt_data["gt_sdf"],
            'gt_depth': gt_data["gt_depth"][index],
            "gs_data":gt_data["gs_data"],
            "c2w":gt_data["c2w"][index],
            'filename': gt_data["filename"],
            "fovy": gt_data["fovy"][index],
            "index": index,
            "num_sample": self.num_sample,

            "proj_mtx":gt_data["proj_mtx"][index],
            "mvp_mtx":gt_data["mvp_mtx"][index],
            "w2c":gt_data["w2c"][index],
            "affine_mats":gt_data["affine_mats"][index],
            "py3d_cams":gt_data["py3d_cams"],



        } 
        #return self.gt_data


@register("gbuffer-datamodule")
class ObjDataModule(pl.LightningDataModule):
    cfg: ObjDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(ObjDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = ObjIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = ObjIterableDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset =ObjIterableDataset(self.cfg)# ObjIterableDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=torch.cuda.device_count(),  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=None, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
