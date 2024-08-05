import os
import torch
from torch import nn
import numpy as np
from plyfile import PlyData, PlyElement
from gaussian_hierarchy._C import load_hierarchy

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
    
    def load_hier(self, path):
        pos, shs, alphas, scales, rot, nodes, boxes = load_hierarchy(path)
        self._xyz = nn.Parameter(torch.tensor(pos).cuda().requires_grad_(True))
        shs_tensor = torch.tensor(shs).cuda()
        self._features_dc = nn.Parameter(shs_tensor[:,:,:1].requires_grad_(True))
        self._features_rest = nn.Parameter(shs_tensor[:,:,1:16].requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(alphas).cuda().requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales).cuda().requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rot).cuda().requires_grad_(True))
    
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

    def filter_large_splats(self, xyz, normals, f_dc, f_rest, opacities, scale, rotation, scale_threshold=0.1):
        scale_magnitudes = np.linalg.norm(scale, axis=1)
        mask = scale_magnitudes < scale_threshold

        return xyz[mask], normals[mask], f_dc[mask], f_rest[mask], opacities[mask], scale[mask], rotation[mask]

    def save_ply(self, path, scale_threshold=0.1):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().cpu().numpy().reshape(self._features_dc.shape[0], -1)
        f_rest = self._features_rest.detach().cpu().numpy().reshape(self._features_rest.shape[0], -1)
        opacities = self._opacity.detach().cpu().numpy().reshape(-1, 1)
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # Filter large splats
        xyz, normals, f_dc, f_rest, opacities, scale, rotation = self.filter_large_splats(
            xyz, normals, f_dc, f_rest, opacities, scale, rotation, scale_threshold
        )

        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

def convert_hier_to_ply(base_path, scale_threshold=0.1):
    model = GaussianModel()
    hier_path = os.path.join(base_path, "output")
    if os.path.exists(os.path.join(hier_path, "merged.hier")):
        model.load_hier(os.path.join(hier_path, "merged.hier"))
        ply_path = os.path.join(base_path, "output", "output.ply")
        model.save_ply(ply_path, scale_threshold)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert hier files to PLY files.")
    parser.add_argument("base_path", type=str, help="Base path containing folders with hier files.")
    parser.add_argument("--scale_threshold", type=float, default=0.1, help="Scale threshold to filter large splats.")
    args = parser.parse_args()

    convert_hier_to_ply(args.base_path, args.scale_threshold)
