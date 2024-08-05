import os
import numpy as np
from plyfile import PlyData, PlyElement
import torch
from torch import nn

class GaussianModel:
    def __init__(self):
        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._opacity = None
        self._scaling = None
        self._rotation = None

    def load_hier(self, path):
        # Custom loader function to read .hier file
        pos, shs, alphas, scales, rot, num_points = self.custom_loader(path)
        print("Positions loaded: ", pos.shape)
        print("SH coefficients loaded: ", shs.shape)
        print("Alphas loaded: ", alphas.shape)
        print("Scales loaded: ", scales.shape)
        print("Rotations loaded: ", rot.shape)
        self._xyz = torch.tensor(pos).float()
        shs_tensor = torch.tensor(shs).float()
        self._features_dc = shs_tensor[:, :1].reshape(num_points, -1)
        self._features_rest = shs_tensor[:, 1:16].reshape(num_points, -1)
        self._opacity = torch.tensor(alphas).float().reshape(-1, 1)
        self._scaling = torch.tensor(scales).float()
        self._rotation = torch.tensor(rot).float()

    def custom_loader(self, path):
        with open(path, 'rb') as f:
            P = int.from_bytes(f.read(4), 'little')
            if P < 0:
                raise ValueError("Invalid number of points in the .hier file")

            num_points = P
            pos = np.frombuffer(f.read(num_points * 12), dtype=np.float32).reshape(num_points, 3)
            rot = np.frombuffer(f.read(num_points * 16), dtype=np.float32).reshape(num_points, 4)
            scales = np.frombuffer(f.read(num_points * 12), dtype=np.float32).reshape(num_points, 3)
            alphas = np.frombuffer(f.read(num_points * 4), dtype=np.float32)
            shs = np.frombuffer(f.read(num_points * 192), dtype=np.float32).reshape(num_points, 48)

        return pos, shs, alphas, scales, rot, num_points

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1]):
            l.append(f'f_dc_{i}')
        for i in range(self._features_rest.shape[1]):
            l.append(f'f_rest_{i}')
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append(f'scale_{i}')
        for i in range(self._rotation.shape[1]):
            l.append(f'rot_{i}')
        return l

    def save_ply(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        xyz = self._xyz.numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.numpy()
        f_rest = self._features_rest.numpy()
        opacities = self._opacity.numpy()
        scale = self._scaling.numpy()
        rotation = self._rotation.numpy()

        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

def convert_hier_to_ply(base_path):
    model = GaussianModel()
    hier_path = os.path.join(base_path, "output", "merged.hier")
    if os.path.exists(hier_path):
        model.load_hier(hier_path)
        ply_path = os.path.join(base_path, "output", "output.ply")
        model.save_ply(ply_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert merged hier file to PLY file.")
    parser.add_argument("base_path", type=str, help="Base path containing the merged hier file.")
    args = parser.parse_args()

    convert_hier_to_ply(args.base_path)
