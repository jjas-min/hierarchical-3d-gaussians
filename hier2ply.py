import os
import torch
from torch import nn
import numpy as np
from plyfile import PlyData, PlyElement
from gaussian_hierarchy._C import load_hierarchy, expand_to_size, get_interpolation_weights

def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)

class GaussianModel(nn.Module):
    def __init__(self):
        super(GaussianModel, self).__init__()
        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._opacity = None
        self._scaling = None
        self._rotation = None
    
    def load_hier(self, path):
        xyz, shs_all, alpha, scales, rots, nodes, boxes = load_hierarchy(path)
        self._xyz = nn.Parameter(xyz.cuda().requires_grad_(True))
        self._features_dc = nn.Parameter(shs_all.cuda()[:,:1,:].requires_grad_(True))
        self._features_rest = nn.Parameter(shs_all.cuda()[:,1:16,:].requires_grad_(True))
        self._opacity = nn.Parameter(alpha.cuda().requires_grad_(True))
        self._scaling = nn.Parameter(scales.cuda().requires_grad_(True))
        self._rotation = nn.Parameter(rots.cuda().requires_grad_(True))
        self.nodes = nodes
        self.boxes = boxes
    
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

    def save_ply(self, path, threshold=0.01):
        mkdir_p(os.path.dirname(path))

        # Prepare tensors for expand_to_size and get_interpolation_weights
        render_indices = torch.zeros(self._xyz.size(0)).int().cuda()
        parent_indices = torch.zeros(self._xyz.size(0)).int().cuda()
        nodes_for_render_indices = torch.zeros(self._xyz.size(0)).int().cuda()
        interpolation_weights = torch.zeros(self._xyz.size(0)).float().cuda()
        num_siblings = torch.zeros(self._xyz.size(0)).int().cuda()

        # Use expand_to_size and get_interpolation_weights as in the render_set function
        to_render = expand_to_size(
            self.nodes,
            self.boxes,
            threshold,
            torch.zeros((3)).cuda(),  # Assuming the camera center is at the origin
            torch.zeros((3)),
            render_indices,
            parent_indices,
            nodes_for_render_indices
        )

        indices = render_indices[:to_render].int().contiguous()
        node_indices = nodes_for_render_indices[:to_render].contiguous()

        get_interpolation_weights(
            node_indices,
            threshold,
            self.nodes,
            self.boxes,
            torch.zeros((3)).cpu(),
            torch.zeros((3)),
            interpolation_weights,
            num_siblings
        )

        # Extract the relevant data using the render indices
        xyz = self._xyz[indices].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc[indices].detach().cpu().numpy()
        f_rest = self._features_rest[indices].detach().cpu().numpy()
        opacities = self._opacity[indices].detach().cpu().numpy()
        scale = self._scaling[indices].detach().cpu().numpy()
        rotation = self._rotation[indices].detach().cpu().numpy()

        # Combine the data into a single array for ply export
        f_dc_reshaped = f_dc.reshape(f_dc.shape[0], -1)
        f_rest_reshaped = f_rest.reshape(f_rest.shape[0], -1)
        attributes = np.concatenate((xyz, normals, f_dc_reshaped, f_rest_reshaped, opacities.reshape(-1, 1), scale, rotation), axis=1)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

def convert_hier_to_ply(base_path):
    # GaussianModel 인스턴스 생성
    model = GaussianModel()
    # Output 폴더 내의 hier 파일을 로드하여 PLY 파일로 변환
    hier_path = os.path.join(base_path, "output")
    if os.path.exists(os.path.join(hier_path, "merged.hier")):
        model.load_hier(os.path.join(hier_path, "merged.hier"))
        ply_path = os.path.join(base_path, "output", "output.ply")
        model.save_ply(ply_path)

# 예시 사용법
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert hier files to PLY files.")
    parser.add_argument("base_path", type=str, help="Base path containing folders with hier files.")
    args = parser.parse_args()

    convert_hier_to_ply(args.base_path)
