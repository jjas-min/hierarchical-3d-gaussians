import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement

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
    
    def load_pt(self, path):
        self._xyz = torch.load(os.path.join(path, "done_xyz.pt")).detach().cpu()
        self._features_dc = torch.load(os.path.join(path, "done_dc.pt")).detach().cpu()
        self._features_rest = torch.load(os.path.join(path, "done_rest.pt")).detach().cpu()
        self._opacity = torch.load(os.path.join(path, "done_opacity.pt")).detach().cpu()
        self._scaling = torch.load(os.path.join(path, "done_scaling.pt")).detach().cpu()
        self._rotation = torch.load(os.path.join(path, "done_rotation.pt")).detach().cpu()

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

        xyz = self._xyz.numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
        f_rest = self._features_rest.transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
        opacities = self._opacity.numpy()
        scale = self._scaling.numpy()
        rotation = self._rotation.numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

def convert_pt_to_ply(base_path):
    # GaussianModel 인스턴스 생성
    model = GaussianModel()
    # scaffold 폴더 내의 PT 파일을 로드하여 PLY 파일로 변환
    scaffold_path = os.path.join(base_path, "scaffold", "point_cloud", "iteration_30000")
    if os.path.exists(os.path.join(scaffold_path, "done_xyz.pt")):
        model.load_pt(scaffold_path)
        ply_file_path = os.path.join(base_path, "scaffold.ply")
        model.save_ply(ply_file_path)
    base_path = os.path.join(base_path, "output", "trained_chunks")
    # base_path 경로 내의 서브 폴더를 순회
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            pt_folder_path = os.path.join(folder_path, "point_cloud", "iteration_30000")
            if os.path.exists(pt_folder_path):
                # PT 파일을 로드하여 PLY 파일로 변환
                if os.path.exists(os.path.join(pt_folder_path, "done_xyz.pt")):
                    model.load_pt(pt_folder_path)
                    ply_file_path = os.path.join(base_path, f"{folder_name}.ply")
                    model.save_ply(ply_file_path)
                    print(f"Converted {folder_name} to PLY and saved as {ply_file_path}")

# 예시 사용법
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PT files to PLY files.")
    parser.add_argument("base_path", type=str, help="Base path containing folders with PT files.")
    args = parser.parse_args()

    convert_pt_to_ply(args.base_path)
