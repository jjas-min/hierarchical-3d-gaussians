import os
import torch
import torchvision
from random import randint
from utils.loss_utils import ssim
from gaussian_renderer import render_post
import sys
from tqdm import tqdm
from utils.image_utils import psnr
from torch import nn
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
from plyfile import PlyData, PlyElement
from lpipsPyTorch import lpips
from gaussian_hierarchy._C import load_hierarchy, expand_to_size, get_interpolation_weights
import math
from tqdm import tqdm

def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)

class GaussianModel:
    def __init__(self):
        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._opacity = None
        self._scaling = None
        self._rotation = None
        self.nodes = None
        self.boxes = None
    
    def load_hier(self, path):
        pos, shs, alphas, scales, rot, nodes, boxes = load_hierarchy(path)
        self._xyz = nn.Parameter(torch.tensor(pos).clone().detach().cuda().requires_grad_(True))
        shs_tensor = torch.tensor(shs).clone().detach().cuda()
        self._features_dc = nn.Parameter(shs_tensor[:,:,:1].clone().detach().requires_grad_(True))
        self._features_rest = nn.Parameter(shs_tensor[:,:,1:16].clone().detach().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(alphas).clone().detach().cuda().requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales).clone().detach().cuda().requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rot).clone().detach().cuda().requires_grad_(True))
        self.nodes = nodes
        self.boxes = boxes
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append(f'f_dc_{i}')
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append(f'f_rest_{i}')
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append(f'scale_{i}')
        for i in range(self._rotation.shape[1]):
            l.append(f'rot_{i}')
        return l

    def render_splats(self, camera_center, tau=3.0):
        render_indices = torch.zeros(self._xyz.size(0)).int().cuda()
        parent_indices = torch.zeros(self._xyz.size(0)).int().cuda()
        nodes_for_render_indices = torch.zeros(self._xyz.size(0)).int().cuda()
        interpolation_weights = torch.zeros(self._xyz.size(0)).float().cuda()
        num_siblings = torch.zeros(self._xyz.size(0)).int().cuda()

        tanfovx = math.tan(60 * 0.5)  # Assuming a 60 degree FOV
        threshold = (2 * (tau + 0.5)) * tanfovx / 1024  # Assuming an image width of 1024

        to_render = expand_to_size(
            self.nodes,
            self.boxes,
            threshold,
            camera_center,
            torch.zeros((3)),
            render_indices,
            parent_indices,
            nodes_for_render_indices)
        
        indices = render_indices[:to_render].int().contiguous()
        node_indices = nodes_for_render_indices[:to_render].contiguous()

        get_interpolation_weights(
            node_indices,
            threshold,
            self.nodes,
            self.boxes,
            camera_center.cpu(),
            torch.zeros((3)),
            interpolation_weights,
            num_siblings
        )

        return indices

    def save_ply(self, path, camera_center, tau=3.0):
        mkdir_p(os.path.dirname(path))

        indices = self.render_splats(camera_center, tau)
        xyz = self._xyz[indices].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc[indices].detach().cpu().numpy().reshape(indices.shape[0], -1)
        f_rest = self._features_rest[indices].detach().cpu().numpy().reshape(indices.shape[0], -1)
        opacities = self._opacity[indices].detach().cpu().numpy().reshape(-1, 1)
        scale = self._scaling[indices].detach().cpu().numpy()
        rotation = self._rotation[indices].detach().cpu().numpy()

        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

def convert_hier_to_ply(base_path):
    model = GaussianModel()
    hier_path = os.path.join(base_path, "output")
    if os.path.exists(os.path.join(hier_path, "merged.hier")):
        model.load_hier(os.path.join(hier_path, "merged.hier"))
        ply_path = os.path.join(base_path, "output", "output.ply")
        camera_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        model.save_ply(ply_path, camera_center)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert hier files to PLY files.")
    parser.add_argument("base_path", type=str, help="Base path containing folders with hier files.")
    args = parser.parse_args()

    convert_hier_to_ply(args.base_path)
