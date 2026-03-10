#!/usr/bin/env python3
"""
Convert AirSim satellite data to YOLO format.
Generates bounding boxes from semantic segmentation masks.
Supports multi-process parallel conversion.

Usage:
  python airsim_to_yolo.py --raw-data /path/to/raw_data --output ./yolo_output
  python airsim_to_yolo.py --raw-data /path/to/raw_data --serial
  python airsim_to_yolo.py --raw-data /path/to/raw_data --workers 8
"""
import os
import json
import shutil
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

RAW_DATA_ROOT = None
OUTPUT_ROOT = None

# RGB color -> class ID mapping (YOLO classes start from 0)
COLOR_TO_CLASS = {
    # main_body - class 0
    (156, 198, 23): 0, (68, 218, 116): 0, (11, 236, 9): 0, (0, 53, 65): 0,
    # solar_panel - class 1
    (146, 52, 70): 1, (194, 39, 7): 1, (211, 80, 208): 1, (189, 135, 188): 1,
    # dish_antenna - class 2
    (124, 21, 123): 2, (90, 162, 242): 2, (35, 196, 244): 2, (220, 163, 49): 2,
    # omni_antenna - class 3
    (86, 254, 214): 3, (125, 75, 48): 3, (85, 152, 34): 3, (173, 69, 31): 3,
    # payload - class 4
    (37, 128, 125): 4, (58, 19, 33): 4, (218, 124, 115): 4, (202, 97, 155): 4,
    # thruster - class 5
    (133, 244, 133): 5, (1, 222, 192): 5, (65, 54, 217): 5, (216, 78, 75): 5,
    # adapter_ring - class 6
    (158, 114, 88): 6, (181, 213, 93): 6,
}

CLASS_NAMES = ['main_body', 'solar_panel', 'dish_antenna', 'omni_antenna',
               'payload', 'thruster', 'adapter_ring']


def extract_satellite_name(folder_name):
    """Extract satellite name from folder name (strips leading timestamp prefix)."""
    if '_' in folder_name:
        return folder_name.split('_', 1)[1]
    return folder_name


def get_bounding_boxes_from_segmentation(seg_image):
    """Extract bounding boxes from a segmentation image.

    Strategy:
    - thruster (5) and payload (4): connected-component analysis (may have multiple instances)
    - other classes: single bounding box over all pixels of that color

    Returns:
        list of (class_id, x_center, y_center, width, height) normalized to [0, 1]
    """
    seg_rgb = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
    height, width = seg_rgb.shape[:2]

    CONNECTED_COMPONENT_CLASSES = {4, 5}
    bounding_boxes = []

    for color, class_id in COLOR_TO_CLASS.items():
        mask = np.all(seg_rgb == color, axis=-1).astype(np.uint8)

        if not np.any(mask):
            continue

        if class_id in CONNECTED_COMPONENT_CLASSES:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 10 or w < 3 or h < 3:
                    continue
                bounding_boxes.append((
                    class_id,
                    (x + w / 2) / width,
                    (y + h / 2) / height,
                    w / width,
                    h / height,
                ))
        else:
            rows, cols = np.where(mask)
            if len(rows) == 0:
                continue
            y_min, y_max = rows.min(), rows.max()
            x_min, x_max = cols.min(), cols.max()
            w = x_max - x_min + 1
            h = y_max - y_min + 1
            if w * h < 10:
                continue
            bounding_boxes.append((
                class_id,
                (x_min + w / 2) / width,
                (y_min + h / 2) / height,
                w / width,
                h / height,
            ))

    return bounding_boxes


def save_yolo_annotation(bboxes, output_path):
    """Save bounding boxes in YOLO label format."""
    with open(output_path, 'w') as f:
        for class_id, x_center, y_center, w, h in bboxes:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


def convert_satellite_data(satellite_folder, output_images, output_labels, dataset_name):
    """Convert one satellite's data to YOLO format.

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

                bboxes = get_bounding_boxes_from_segmentation(seg_image)
                if len(bboxes) == 0:
                    continue

                unique_name = f"{satellite_name}_{traj_dir.name}_{frame_id}"
                shutil.copy2(img_file, output_images / f"{unique_name}.png")
                save_yolo_annotation(bboxes, output_labels / f"{unique_name}.txt")
                frame_count += 1

        return (satellite_name, frame_count, None)

    except Exception as e:
        return (satellite_name, 0, str(e))


# Val (test) satellites: seq 00, 10, 20, ..., 130 (14 satellites)
VAL_SATELLITES = {
    'ACE', 'CALIPSO', 'Dawn', 'ExoMars_TGO', 'GRAIL', 'Integral', 'LADEE',
    'Lunar_Reconnaissance_Orbiter', 'Mercury_Magnetospheric_Orbiter',
    'OSIRIS_REX', 'Proba_2', 'SOHO', 'Suomi_NPP', 'Ulysses'
}

# Excluded satellites: seq 131-135 (reserved for future testing)
EXCLUDED_SATELLITES = {
    'Van_Allen_Probe', 'Venus_Express', 'Voyager', 'WIND', 'XMM_newton'
}


def split_satellites_train_val(satellite_folders):
    """Split satellite folders into train and val sets."""
    train_folders, val_folders = [], []
    for folder in satellite_folders:
        sat_name = extract_satellite_name(folder.name)
        if sat_name in EXCLUDED_SATELLITES:
            pass
        elif sat_name in VAL_SATELLITES:
            val_folders.append(folder)
        else:
            train_folders.append(folder)
    return train_folders, val_folders


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


def create_yaml_config(output_root):
    """Create YOLO dataset YAML config file."""
    yaml_content = f"""# SpaceSense-Bench YOLO Dataset Configuration
# Split follows Semantic-KITTI convention:
#   val:  seq 00, 10, 20, ..., 130 (14 satellites)
#   excluded: seq 131-135 (5 satellites, reserved)
#   train: remaining 117 satellites

path: {str(output_root.absolute()).replace(chr(92), '/')}
train: train/images
val: val/images

nc: 7
names: ['main_body', 'solar_panel', 'dish_antenna', 'omni_antenna', 'payload', 'thruster', 'adapter_ring']
"""
    yaml_path = output_root / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"[OK] YAML config saved: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert AirSim raw data to YOLO format (parallel supported)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python airsim_to_yolo.py --raw-data /path/to/raw_data --output ./yolo_output
  python airsim_to_yolo.py --raw-data /path/to/raw_data --serial
  python airsim_to_yolo.py --raw-data /path/to/raw_data --workers 8
        """
    )
    parser.add_argument('--raw-data', type=str, required=True,
                        help='Raw data root directory (contains satellite sub-folders)')
    parser.add_argument('--output', type=str, default='./spacesense_yolo',
                        help='Output directory (default: ./spacesense_yolo)')
    parser.add_argument('--serial', action='store_true',
                        help='Use serial processing (default: parallel)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: 2/3 of CPU cores)')

    args = parser.parse_args()

    global RAW_DATA_ROOT, OUTPUT_ROOT
    RAW_DATA_ROOT = Path(args.raw_data)
    OUTPUT_ROOT = Path(args.output)

    print("=" * 70)
    print("AirSim -> YOLO Conversion Tool")
    print("=" * 70)

    if not RAW_DATA_ROOT.exists():
        print(f"[ERR] Data directory not found: {RAW_DATA_ROOT}")
        return

    train_images = OUTPUT_ROOT / "train" / "images"
    train_labels = OUTPUT_ROOT / "train" / "labels"
    val_images   = OUTPUT_ROOT / "val" / "images"
    val_labels   = OUTPUT_ROOT / "val" / "labels"

    for d in [train_images, train_labels, val_images, val_labels]:
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

    train_folders, val_folders = split_satellites_train_val(satellite_folders)

    print(f"\nDataset split (Semantic-KITTI convention):")
    print(f"  train: {len(train_folders)} satellites")
    print(f"  val:   {len(val_folders)} satellites (seq 00, 10, ..., 130)")
    print(f"  excluded: seq 131-135 (not used)")

    if args.serial:
        max_workers = 1
        print("\n[OK] Serial mode")
    elif args.workers is not None:
        max_workers = args.workers
        print(f"\n[OK] Using {max_workers} workers (user specified)")
    else:
        total = len(train_folders) + len(val_folders)
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
    create_yaml_config(OUTPUT_ROOT)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    print(f"train: {train_count} frames ({len(train_folders)} satellites)")
    print(f"val:   {val_count} frames ({len(val_folders)} satellites)")
    print(f"total: {train_count + val_count} frames")
    print(f"\nOutput: {OUTPUT_ROOT}")

    all_results = train_results + val_results
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
    random.seed(42)
    mp.freeze_support()
    main()
