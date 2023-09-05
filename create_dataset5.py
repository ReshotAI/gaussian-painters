import os
import json
import shutil
import math
import numpy as np
from pathlib import Path
import argparse
import subprocess
from PIL import Image
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from PIL import Image
import math
import torchvision.transforms.functional as F
import kornia


def get_transform(projmatrix, coord_x, coord_y):
    X = torch.tensor([coord_x, coord_y, 0, 1], dtype=torch.float32)
    x, y, z = projmatrix @ X
    return [x / z, y / z]


def create_dataset(args):
    data_dir = Path(args.output_dir)
    data_dir.mkdir(exist_ok=True)

    images_dir = data_dir / "images"
    images_dir.mkdir(exist_ok=True)

    sparse_dir = data_dir / "sparse" / "0"
    sparse_dir.mkdir(exist_ok=True, parents=True)

    img_front = Image.open(args.img_path_front).convert("RGB")
    img_left = Image.open(args.img_path_left).convert("RGB")
    img_right = Image.open(args.img_path_right).convert("RGB")
    img_bottom = Image.open(args.img_path_bottom).convert("RGB")
    img_top = Image.open(args.img_path_top).convert("RGB")

    width, height = img_front.size
    focal = math.sqrt(width**2 + height**2)

    imgs = [img_right, img_front, img_left, img_top, img_bottom]

    I = torch.tensor([
        [focal, 0, width / 2],
        [0, focal, height / 2],
        [0, 0, 1]
    ])

    def process(f, angle, horizontal, img, idx):
        img_tensor = F.to_tensor(img)[None]

        arr = np.array([0, 1, 0]) if horizontal else np.array([1, 0, 0])

        R_ = R.from_rotvec(angle * arr).as_matrix()

        M = torch.concat([torch.tensor(R_).T, torch.tensor([0, 0, focal])[..., None]], dim=1).float()

        projmatrix = I @ M

        points_src = torch.FloatTensor([[
            [0, 0], [width - 1, 0],
            [width - 1, height - 1], [0, height - 1],
        ]])

        points_dst = torch.FloatTensor([[
            get_transform(projmatrix, -width / 2, -height / 2), get_transform(projmatrix, width / 2, -height / 2),
            get_transform(projmatrix, width / 2, height / 2), get_transform(projmatrix, -width / 2, height / 2),
        ]])
        
        M = kornia.geometry.transform.get_perspective_transform(points_src, points_dst)

        img_warp = kornia.geometry.transform.warp_perspective(img_tensor, M, dsize=(height, width))

        F.to_pil_image(img_warp[0]).save(images_dir / f"{idx:04d}.jpg")

        x, y, z, w = R.from_rotvec(-angle * arr).as_quat()
        f.write(f"{idx} {w} {x} {y} {z} 0 0 {focal} 1 {idx:04d}.jpg\n\n")

    with open(sparse_dir / "cameras.txt", "w") as f:
        f.write(f"1 PINHOLE {width} {height} {focal} {focal} {width/2} {height/2}\n")
    
    with open(sparse_dir / "images.txt", "w") as f:
        
        for idx, angle in enumerate([-np.pi / 4, 0, np.pi / 4, -np.pi / 4, np.pi / 4]):
            img = imgs[idx]
            horizontal = True if idx < 3 else False
            process(f, angle, horizontal, img, idx)
    
    with open(sparse_dir / "points3D.txt", "w") as f:
        BLOCKS = 256

        idx = 0

        for i in range(0, BLOCKS):
            for j in range(0, BLOCKS):
                idx += 1

                x = i / BLOCKS * width
                y = j / BLOCKS * height

                rgb = np.random.randint(0, 255, size=3)

                xyz = np.array([-width / 2, -height / 2, 0]) + np.array([x, y, 0])
                f.write(f"{idx} {' '.join(map(str, xyz.tolist()))} {' '.join(map(str, rgb))} 0\n")
    
    subprocess.run(["colmap", "model_converter", "--input_path", sparse_dir, "--output_path", sparse_dir, "--output_type", "BIN"])

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_path_front", type=str, required=True)
    parser.add_argument("--img_path_left", type=str, required=True)
    parser.add_argument("--img_path_right", type=str, required=True)
    parser.add_argument("--img_path_bottom", type=str, required=True)
    parser.add_argument("--img_path_top", type=str, required=True)

    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    create_dataset(args)
