"""
Convert single-channel MMSeg output PNG masks to 3-channel colorized JPG images.

Usage:
  Set input_folder and output_folder below, then run:
  python mmseg_output_to_jpg.py
"""
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# Color mapping: class ID -> RGB
color_mapping = {
    0: (0, 0, 0),    # background -> black
    1: (255, 0, 0),  # class 1 -> red
    2: (0, 0, 255),  # class 2 -> blue
}

input_folder  = 'work_dirs/output'
output_folder = 'work_dirs/output_jpg'

os.makedirs(output_folder, exist_ok=True)

for filename in tqdm(os.listdir(input_folder)):
    if not filename.endswith('.png'):
        continue

    input_path  = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.jpg')

    image = np.array(Image.open(input_path).convert('L'))

    rgb_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for value, rgb_value in color_mapping.items():
        rgb_image[image == value] = rgb_value

    Image.fromarray(rgb_image).save(output_path, 'JPEG')
