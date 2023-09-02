import os
import ffmpeg
import json
import shutil
import math
import numpy as np
from pathlib import Path
import argparse
import subprocess
from PIL import Image


def create_dataset(args):
    data_dir = Path(args.output_dir)
    data_dir.mkdir(exist_ok=True)

    images_dir = data_dir / "images"
    images_dir.mkdir(exist_ok=True)

    sparse_dir = data_dir / "sparse" / "0"
    sparse_dir.mkdir(exist_ok=True, parents=True)

    img_path = Path(args.img_path)
    img_path2 = Path(args.img_path2)
    img_path3 = Path(args.img_path3)

    shutil.copy2(img_path, images_dir)
    shutil.copy2(img_path2, images_dir)
    shutil.copy2(img_path3, images_dir)

    img = Image.open(args.img_path)
    width, height = img.size
    focal = math.sqrt(width**2 + height**2)

    img2 = Image.open(args.img_path2)
    width2, height2 = img2.size
    focal2 = math.sqrt(width2**2 + height2**2)

    img3 = Image.open(args.img_path3)
    width3, height3 = img3.size
    focal3 = math.sqrt(width3**2 + height3**2)

    # create black image
    black_img = Image.new("RGB", (width, height), (0, 0, 0))
    black_img.save(images_dir / "black.jpg")


    with open(sparse_dir / "cameras.txt", "w") as f:
        f.write(f"1 PINHOLE {width} {height} {focal} {focal} {width/2} {height/2}\n")
        f.write(f"2 PINHOLE {width2} {height2} {focal2} {focal2} {width2/2} {height2/2}\n")
        f.write(f"3 PINHOLE {width3} {height3} {focal3} {focal3} {width3/2} {height3/2}\n")
    
    with open(sparse_dir / "images.txt", "w") as f:
        f.write(f"1 1 0 0 0 0 0 {focal} 1 {img_path.name}\n\n")
        f.write(f"2 {math.sqrt(2+math.sqrt(2))/2} 0 {-math.sqrt(2-math.sqrt(2))/2} 0 0 0 {focal2} 2 {img_path2.name}\n\n")
        f.write(f"3 {math.sqrt(2+math.sqrt(2))/2} 0 {math.sqrt(2-math.sqrt(2))/2} 0 0 0 {focal3} 3 {img_path3.name}\n\n")
        f.write(f"4 {math.sqrt(2)/2} 0 {math.sqrt(2)/2} 0 0 0 {focal} 1 black.jpg\n\n")
    
    with open(sparse_dir / "points3D.txt", "w") as f:
        BLOCKS = 10

        idx = 0

        for i in range(0, BLOCKS):
            for j in range(0, BLOCKS):
                idx += 1

                x = i / BLOCKS * width
                y = j / BLOCKS * height

                rgb = img.getpixel((x, y))

                xyz = np.array([-width / 2, -height / 2, 0]) + np.array([x, y, 0])
                f.write(f"{idx} {' '.join(map(str, xyz.tolist()))} {' '.join(map(str, rgb))} 0\n")
    
    subprocess.run(["colmap", "model_converter", "--input_path", sparse_dir, "--output_path", sparse_dir, "--output_type", "BIN"])

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--img_path2", type=str, required=True)
    parser.add_argument("--img_path3", type=str, required=True)

    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    create_dataset(args)
