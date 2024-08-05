import os
import torch
from torch import nn
import numpy as np
from plyfile import PlyData, PlyElement
from gaussian_hierarchy._C import load_hierarchy, expand_to_size, get_interpolation_weights

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
            # Prepare tensors for expand_to_size and get_interpolation_weights
            render_indices = torch.zeros(self._xyz.size(0)).int().cuda()
            parent_indices = torch.zeros(self._xyz.size(0)).int().cuda()
            nodes_for_render_indices = torch.zeros(self._xyz.size(0)).int().cuda()
            interpolation_weights = torch.zeros(self._xyz.size(0)).float().cuda()
            num_siblings = torch.zeros(self._xyz.size(0)).int().cuda()

            # Print debug information
            print("nodes:", self.nodes.shape, self.nodes.dtype)
            print("boxes:", self.boxes.shape, self.boxes.dtype)
            print("threshold:", threshold)
            print("render_indices:", render_indices.shape, render_indices.dtype)
            print("parent_indices:", parent_indices.shape, parent_indices.dtype)
            print("nodes_for_render_indices:", nodes_for_render_indices.shape, nodes_for_render_indices.dtype)
            print("torch.zeros((3)).cuda():", torch.zeros((3)).cuda().shape, torch.zeros((3)).cuda().dtype)

            # Print sample data
            print("Sample nodes:", self.nodes[:5])
            print("Sample boxes:", self.boxes[:5])
            print("Sample render_indices:", render_indices[:5])
            print("Sample parent_indices:", parent_indices[:5])
            print("Sample nodes_for_render_indices:", nodes_for_render_indices[:5])

            # Use expand_to_size and get_interpolation_weights as in the render_set function
            camera_center = torch.zeros((3)).cuda()

            # Try-catch block to catch specific errors in expand_to_size
            try:
                to_render = expand_to_size(
                    self.nodes,
                    self.boxes,
                    threshold,
                    camera_center,
                    camera_center,  # Assuming camera forward is also at the origin
                    render_indices,
                    parent_indices,
                    nodes_for_render_indices
                )
            except Exception as e:
                print(f"Error in expand_to_size: {e}")
                raise

            # Debugging: Print shapes of all arrays after expand_to_size
            print("to_render:", to_render)
            print("render_indices after expand_to_size:", render_indices[:to_render])
            print("parent_indices after expand_to_size:", parent_indices[:to_render])
            print("nodes_for_render_indices after expand_to_size:", nodes_for_render_indices[:to_render])

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

            # Debugging: Print shapes of all arrays
            print("Shape of xyz: ", self._xyz[indices].shape)
            print("Shape of normals: ", np.zeros_like(self._xyz[indices]).shape)
            print("Shape of f_dc: ", self._features_dc[indices].shape)
            print("Shape of f_rest: ", self._features_rest[indices].shape)
            print("Shape of opacities: ", self._opacity[indices].shape)
            print("Shape of scale: ", self._scaling[indices].shape)
            print("Shape of rotation: ", self._rotation[indices].shape)

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
