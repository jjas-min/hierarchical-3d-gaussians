import os
import torch
from torch import nn
import numpy as np
from plyfile import PlyData, PlyElement
from gaussian_hierarchy._C import load_hierarchy, write_hierarchy

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
    
    def load_heir(self, path):
        xyz, shs_all, alpha, scales, rots, nodes, boxes = load_hierarchy(path)
        self._xyz = nn.Parameter(xyz.cuda().requires_grad_(True))
        self._features_dc = nn.Parameter(shs_all.cuda()[:,:1,:].requires_grad_(True))
        self._features_rest = nn.Parameter(shs_all.cuda()[:,1:16,:].requires_grad_(True))
        self._opacity = nn.Parameter(alpha.cuda().requires_grad_(True))
        self._scaling = nn.Parameter(scales.cuda().requires_grad_(True))
        self._rotation = nn.Parameter(rots.cuda().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

def convert_hier_to_ply(base_path):
    # GaussianModel 인스턴스 생성
    model = GaussianModel()
    # Output 폴더 내의 hier 파일을 로드하여 PLY 파일로 변환
    heir_path = os.path.join(base_path, "output")
    if os.path.exists(os.path.join(heir_path, "merge.hier")):
        model.load_heir(os.path.join(heir_path, "merge.hier"))

    ply_path = os.path.join(base_path, "output", "output.ply")
    model.save_ply(ply_path)


# 예시 사용법
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PT files to PLY files.")
    parser.add_argument("base_path", type=str, help="Base path containing folders with PT files.")
    args = parser.parse_args()

    convert_hier_to_ply(args.base_path)
