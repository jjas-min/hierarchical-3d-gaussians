import os
import torch
from torch import nn
import numpy as np
from plyfile import PlyData, PlyElement
from gaussian_hierarchy._C import load_hierarchy, get_interpolation_weights

torch.autograd.set_detect_anomaly(True)

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
        try:
            xyz, shs_all, alpha, scales, rots, nodes, boxes = load_hierarchy(path)
            self._xyz = nn.Parameter(xyz.cuda().requires_grad_(False))
            self._features_dc = nn.Parameter(shs_all.cuda()[:,:1,:].requires_grad_(False))
            self._features_rest = nn.Parameter(shs_all.cuda()[:,1:16,:].requires_grad_(False))
            self._opacity = nn.Parameter(alpha.cuda().requires_grad_(False))
            self._scaling = nn.Parameter(scales.cuda().requires_grad_(False))
            self._rotation = nn.Parameter(rots.cuda().requires_grad_(False))
            self.nodes = nodes.cuda()
            self.boxes = boxes.cuda()
        except Exception as e:
            print(f"Error loading hierarchy: {e}")
            raise

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

        try:
            # Prepare tensors for render hierarchy
            render_indices = torch.zeros(self._xyz.size(0)).int().cuda()
            parent_indices = torch.zeros(self._xyz.size(0)).int().cuda()
            nodes_for_render_indices = torch.zeros(self._xyz.size(0)).int().cuda()
            interpolation_weights = torch.zeros(self._xyz.size(0)).float().cuda()
            num_siblings = torch.zeros(self._xyz.size(0)).int().cuda()

            # Initialize camera settings (assuming a default camera position)
            camera_center = torch.zeros((3)).cuda()
            camera_forward = torch.zeros((3)).cuda()

            # Determine the indices to render based on threshold
            to_render = 0
            for i in range(self.nodes.size(0)):
                box_center = (self.boxes[i, 0, :3] + self.boxes[i, 1, :3]) / 2  # Only use the first 3 elements
                dist_to_center = torch.norm(camera_center - box_center.cuda())
                if dist_to_center < threshold:
                    render_indices[to_render] = i
                    nodes_for_render_indices[to_render] = self.nodes[i, 0]
                    to_render += 1

            indices = render_indices[:to_render].int().contiguous()
            node_indices = nodes_for_render_indices[:to_render].contiguous()

            get_interpolation_weights(
                node_indices,
                threshold,
                self.nodes,
                self.boxes,
                camera_center.cpu(),
                camera_forward,
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

            # Debugging: Print shape of final attributes
            print("Shape of attributes: ", attributes.shape)

            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path)
        except Exception as e:
            print(f"Error saving PLY: {e}")
            raise

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
