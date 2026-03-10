#!/usr/bin/env python3
"""
Convert AirSim satellite data to MMSegmentation format.
- Processes RGB images and semantic segmentation annotations
- Supports multi-process parallel conversion
- Dataset split follows the Semantic-KITTI convention

Usage:
  python airsim_to_mmseg.py --raw-data /path/to/raw_data --output ./mmseg_output
  python airsim_to_mmseg.py --raw-data /path/to/raw_data --serial
  python airsim_to_mmseg.py --raw-data /path/to/raw_data --workers 8
"""
import os
import shutil
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

RAW_DATA_ROOT = None
OUTPUT_ROOT = None

# Output format:
# spacesense_mmseg/
# ├── img_dir/
# │   ├── train/  ├── val/  └── test/
# └── ann_dir/
#     ├── train/  ├── val/  └── test/

# Dataset split (Semantic-KITTI convention):
# train: 117 satellites (seq 01-09, 11-129, excluding multiples of 10)
# val:   5 satellites (seq 131-135)
# test:  14 satellites (seq 00, 10, 20, ..., 130)

# RGB color -> class ID mapping
COLOR_TO_CLASS = {
    # main_body - class 1
    (156, 198, 23): 1, (68, 218, 116): 1, (11, 236, 9): 1, (0, 53, 65): 1,
    # solar_panel - class 2
    (146, 52, 70): 2, (194, 39, 7): 2, (211, 80, 208): 2, (189, 135, 188): 2,
    # dish_antenna - class 3
    (124, 21, 123): 3, (90, 162, 242): 3, (35, 196, 244): 3, (220, 163, 49): 3,
    # omni_antenna - class 4
    (86, 254, 214): 4, (125, 75, 48): 4, (85, 152, 34): 4, (173, 69, 31): 4,
    # payload - class 5
    (37, 128, 125): 5, (58, 19, 33): 5, (218, 124, 115): 5, (202, 97, 155): 5,
    # thruster - class 6
    (133, 244, 133): 6, (1, 222, 192): 6, (65, 54, 217): 6, (216, 78, 75): 6,
    # adapter_ring - class 7
    (158, 114, 88): 7, (181, 213, 93): 7,
}

CLASS_NAMES = ['background', 'main_body', 'solar_panel', 'dish_antenna',
               'omni_antenna', 'payload', 'thruster', 'adapter_ring']

# MMSeg palette: 8 classes (background + 7 parts), flattened RGB array
MMSEG_PALETTE = np.array([
    [0, 0, 0],       # 0: background
    [128, 0, 0],     # 1: main_body
    [0, 128, 0],     # 2: solar_panel
    [128, 128, 0],   # 3: dish_antenna
    [0, 0, 128],     # 4: omni_antenna
    [128, 0, 128],   # 5: payload
    [0, 128, 128],   # 6: thruster
    [128, 128, 128], # 7: adapter_ring
], dtype=np.uint8).flatten()


def extract_satellite_name(folder_name):
    """Extract satellite name from folder name (strips leading timestamp prefix)."""
    if '_' in folder_name:
        return folder_name.split('_', 1)[1]
    return folder_name


def convert_seg_to_mmseg(seg_image):
    """Convert an RGB segmentation image to a single-channel label map."""
    seg_rgb = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
    height, width = seg_rgb.shape[:2]
    label_map = np.zeros((height, width), dtype=np.uint8)
    for color, class_id in COLOR_TO_CLASS.items():
        mask = np.all(seg_rgb == color, axis=-1)
        label_map[mask] = class_id
    return label_map


def save_mmseg_annotation(label_map, output_path):
    """Save a label map as a palette-mode PNG (MMSeg format)."""
    img_p = Image.fromarray(label_map, mode='P')
    img_p.putpalette(MMSEG_PALETTE)
    img_p.save(output_path)


def convert_satellite_data(satellite_folder, output_images, output_labels, dataset_name):
    """Convert one satellite's data to MMSeg format.

    Returns:
        (satellite_name, frame_count, error_message)
    """
    satellite_folder = Path(satellite_folder)
    output_images = Path(output_images)
    output_labels = Path(output_labels)

    satellite_name = extract_satellite_name(satellite_folder.name)

    try:
        trajectory_dirs = sorted([
            d for d in satellite_folder.iterdir()
            if d.is_dir() and not d.name.startswith('trajectory')
        ])

        if not trajectory_dirs:
            return (satellite_name, 0, None)

        frame_count = 0

        for traj_dir in trajectory_dirs:
            image_dir = traj_dir / 'image'
            seg_dir = traj_dir / 'seg'

            if not image_dir.exists() or not seg_dir.exists():
                continue

            for img_file in sorted(image_dir.glob('*.png')):
                frame_id = img_file.stem
                seg_file = seg_dir / f'{frame_id}.png'

                if not seg_file.exists():
                    continue

                seg_image = cv2.imread(str(seg_file))
                if seg_image is None:
                    continue

                label_map = convert_seg_to_mmseg(seg_image)
                unique_name = f"{satellite_name}_{traj_dir.name}_{frame_id}"

                shutil.copy2(img_file, output_images / f"{unique_name}.png")
                save_mmseg_annotation(label_map, output_labels / f"{unique_name}.png")
                frame_count += 1

        return (satellite_name, frame_count, None)

    except Exception as e:
        return (satellite_name, 0, str(e))


# Test satellites: seq 00, 10, 20, ..., 130 (14 satellites)
TEST_SATELLITES = {
    'ACE', 'CALIPSO', 'Dawn', 'ExoMars_TGO', 'GRAIL', 'Integral', 'LADEE',
    'Lunar_Reconnaissance_Orbiter', 'Mercury_Magnetospheric_Orbiter',
    'OSIRIS_REX', 'Proba_2', 'SOHO', 'Suomi_NPP', 'Ulysses'
}

# Validation satellites: seq 131-135 (5 satellites)
VAL_SATELLITES = {
    'Van_Allen_Probe', 'Venus_Express', 'Voyager', 'WIND', 'XMM_newton'
}


def split_satellites(satellite_folders):
    """Split satellite folders into train / val / test sets."""
    train_folders, val_folders, test_folders = [], [], []
    for folder in satellite_folders:
        sat_name = extract_satellite_name(folder.name)
        if sat_name in TEST_SATELLITES:
            test_folders.append(folder)
        elif sat_name in VAL_SATELLITES:
            val_folders.append(folder)
        else:
            train_folders.append(folder)
    return train_folders, val_folders, test_folders


def process_single_satellite(args):
    satellite_folder, output_images, output_labels, dataset_name = args
    return convert_satellite_data(satellite_folder, output_images, output_labels, dataset_name)


def convert_parallel(satellite_folders, output_images, output_labels, dataset_name, max_workers=None):
    """Convert multiple satellites in parallel (or serial when max_workers==1)."""
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() * 2 // 3)

    tasks = [(folder, output_images, output_labels, dataset_name) for folder in satellite_folders]
    results = []
    total_frames = 0

    if max_workers == 1:
        for task in tqdm(tasks, desc=f"Converting {dataset_name}"):
            result = process_single_satellite(task)
            results.append(result)
            total_frames += result[1]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_sat = {
                executor.submit(process_single_satellite, task): extract_satellite_name(task[0].name)
                for task in tasks
            }
            with tqdm(total=len(tasks), desc=f"Converting {dataset_name}") as pbar:
                for future in as_completed(future_to_sat):
                    sat_name = future_to_sat[future]
                    try:
                        result = future.result()
                        results.append(result)
                        total_frames += result[1]
                        if result[2]:
                            tqdm.write(f"  [ERR] {result[0]}: {result[2]}")
                    except Exception as e:
                        tqdm.write(f"  [ERR] {sat_name}: {e}")
                        results.append((sat_name, 0, str(e)))
                    pbar.update(1)

    return total_frames, results


def main():
    parser = argparse.ArgumentParser(
        description='Convert AirSim raw data to MMSegmentation format (parallel supported)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python airsim_to_mmseg.py --raw-data /path/to/raw_data --output ./mmseg_output
  python airsim_to_mmseg.py --raw-data /path/to/raw_data --serial
  python airsim_to_mmseg.py --raw-data /path/to/raw_data --workers 8
        """
    )
    parser.add_argument('--raw-data', type=str, required=True,
                        help='Raw data root directory (contains satellite sub-folders)')
    parser.add_argument('--output', type=str, default='./spacesense_mmseg',
                        help='Output directory (default: ./spacesense_mmseg)')
    parser.add_argument('--serial', action='store_true',
                        help='Use serial processing (default: parallel)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: 2/3 of CPU cores)')

    args = parser.parse_args()

    global RAW_DATA_ROOT, OUTPUT_ROOT
    RAW_DATA_ROOT = Path(args.raw_data)
    OUTPUT_ROOT = Path(args.output)

    print("=" * 70)
    print("AirSim -> MMSegmentation Conversion Tool")
    print("=" * 70)

    if not RAW_DATA_ROOT.exists():
        print(f"[ERR] Data directory not found: {RAW_DATA_ROOT}")
        return

    train_images = OUTPUT_ROOT / "img_dir" / "train"
    train_labels = OUTPUT_ROOT / "ann_dir" / "train"
    val_images   = OUTPUT_ROOT / "img_dir" / "val"
    val_labels   = OUTPUT_ROOT / "ann_dir" / "val"
    test_images  = OUTPUT_ROOT / "img_dir" / "test"
    test_labels  = OUTPUT_ROOT / "ann_dir" / "test"

    for d in [train_images, train_labels, val_images, val_labels, test_images, test_labels]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[OK] Output directory: {OUTPUT_ROOT}")

    # Support two directory structures:
    #   1. raw_data/<satellite>/approach_xxx/   (flat)
    #   2. raw_data/<satellite>/<satellite>/approach_xxx/  (HuggingFace nested)
    raw_candidates = [d for d in RAW_DATA_ROOT.iterdir() if d.is_dir()]
    satellite_folders = []
    for d in raw_candidates:
        subdirs = [s for s in d.iterdir() if s.is_dir()]
        has_trajectory = any((s / 'image').exists() or (s / 'seg').exists() for s in subdirs)
        if has_trajectory:
            satellite_folders.append(d)
        else:
            satellite_folders.extend(subdirs)

    print(f"[OK] Found {len(satellite_folders)} satellite folders")

    train_folders, val_folders, test_folders = split_satellites(satellite_folders)

    print(f"\nDataset split (Semantic-KITTI convention):")
    print(f"  train: {len(train_folders)} satellites")
    print(f"  val:   {len(val_folders)} satellites (seq 131-135)")
    print(f"  test:  {len(test_folders)} satellites (seq 00, 10, ..., 130)")

    if args.serial:
        max_workers = 1
        print("\n[OK] Serial mode")
    elif args.workers is not None:
        max_workers = args.workers
        print(f"\n[OK] Using {max_workers} workers (user specified)")
    else:
        total = len(train_folders) + len(val_folders) + len(test_folders)
        max_workers = max(1, min(mp.cpu_count() * 2 // 3, total))
        print(f"\n[OK] CPU cores: {mp.cpu_count()}, using {max_workers} workers")

    print("\n" + "=" * 70)
    print("Converting train set...")
    print("=" * 70)
    train_count, train_results = convert_parallel(train_folders, train_images, train_labels, "train", max_workers)

    print("\n" + "=" * 70)
    print("Converting val set...")
    print("=" * 70)
    val_count, val_results = convert_parallel(val_folders, val_images, val_labels, "val", max_workers)

    print("\n" + "=" * 70)
    print("Converting test set...")
    print("=" * 70)
    test_count, test_results = convert_parallel(test_folders, test_images, test_labels, "test", max_workers)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    print(f"train: {train_count} frames ({len(train_folders)} satellites)")
    print(f"val:   {val_count} frames ({len(val_folders)} satellites)")
    print(f"test:  {test_count} frames ({len(test_folders)} satellites)")
    print(f"total: {train_count + val_count + test_count} frames")
    print(f"\nOutput: {OUTPUT_ROOT}")

    all_results = train_results + val_results + test_results
    failed = [r for r in all_results if r[2] is not None]
    if failed:
        print(f"\n[WARN] Failed satellites ({len(failed)}):")
        for sat_name, _, error in failed:
            print(f"  - {sat_name}: {error}")

    print("=" * 70)
    print("\nClass definitions:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}")
    print()


if __name__ == "__main__":
    mp.freeze_support()
    main()
