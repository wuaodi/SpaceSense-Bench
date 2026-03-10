#!/usr/bin/env python3
"""
Convert AirSim satellite data to Semantic-KITTI format.
Each satellite becomes one sequence; all trajectories are merged.
Parallel processing is used by default (2/3 of CPU cores).

Usage:
  python airsim_to_semantickitti.py --raw-data /path/to/raw_data --output ./converted_data
  python airsim_to_semantickitti.py --raw-data /path/to/raw_data --workers 4
  python airsim_to_semantickitti.py --raw-data /path/to/raw_data --serial
"""
import os
import sys

# Limit numpy thread count to avoid multi-process conflicts (must be set before importing numpy)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import csv
import json
import math
import shutil
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


def read_asc_pointcloud(filepath):
    """Read a .asc point cloud file (comma-separated x,y,z per line)."""
    points = []
    with open(filepath, 'r') as f:
        for line in f:
            values = [x.strip() for x in line.strip().split(',') if x.strip()]
            if len(values) >= 3:
                try:
                    x, y, z = map(float, values[:3])
                    points.append([x, y, z])
                except ValueError:
                    pass
    return np.array(points, dtype=np.float32)


def transform_lidar_to_camera_frame(point):
    """Transform a point from LiDAR frame to camera frame (identity here)."""
    return point + np.array([0, 0, 0])


def project_point_to_image(point, image_width, image_height):
    """Project a 3D point onto the image plane. Returns (u, v) or None."""
    fov_rad = math.radians(50)
    focal_length = (image_width / 2) / math.tan(fov_rad / 2)

    if point[0] <= 0:
        return None

    cx = image_width / 2
    cy = image_height / 2
    u = int(focal_length * point[1] / point[0] + cx)
    v = int(focal_length * point[2] / point[0] + cy)

    if 0 <= u < image_width and 0 <= v < image_height:
        return (u, v)
    return None


def get_label_from_segmentation(seg_image, u, v):
    """Look up semantic label at pixel (u, v) in the segmentation image."""
    pixel_value = seg_image[v, u]

    if len(seg_image.shape) == 3:
        pixel_rgb = (pixel_value[2], pixel_value[1], pixel_value[0])
        color_to_label = {
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
        return color_to_label.get(pixel_rgb, None)
    else:
        return int(pixel_value)


def convert_trajectory_to_kitti(pointcloud_dir, seg_dir, img_dir,
                                 output_bin_dir, output_label_dir, output_img_dir):
    """Convert one trajectory from AirSim format to Semantic-KITTI format."""
    os.makedirs(output_bin_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)

    pointcloud_files = sorted(glob(os.path.join(pointcloud_dir, "*.asc")))

    for pc_file in tqdm(pointcloud_files, desc="  Converting frames", leave=False):
        file_id = os.path.splitext(os.path.basename(pc_file))[0]
        seg_file = os.path.join(seg_dir, f"{file_id}.png")
        img_file = os.path.join(img_dir, f"{file_id}.png")

        if not os.path.exists(seg_file):
            continue

        shutil.copy(img_file, os.path.join(output_img_dir, f"{file_id}.png"))

        points = read_asc_pointcloud(pc_file)
        seg_image = cv2.imread(seg_file)
        height, width = seg_image.shape[:2]

        labels = np.zeros(len(points), dtype=np.uint32)
        valid_indices = []

        for i, point in enumerate(points):
            camera_point = transform_lidar_to_camera_frame(point)
            projection = project_point_to_image(camera_point, width, height)
            label = None
            if projection:
                u, v = projection
                label = get_label_from_segmentation(seg_image, u, v)
            if label is not None:
                labels[i] = label
                valid_indices.append(i)

        points = points[valid_indices]
        labels = labels[valid_indices]

        if len(points) == 0:
            continue

        xyz = np.zeros((len(points), 4), dtype=np.float32)
        xyz[:, :3] = points
        xyz[:, 3] = 0.0  # intensity placeholder

        xyz.tofile(os.path.join(output_bin_dir, f"{file_id}.bin"))
        labels.tofile(os.path.join(output_label_dir, f"{file_id}.label"))


def create_default_calib():
    """Return default calib.txt content for the SpaceSense camera."""
    return """P2: 1097.98754330 0 512.0 0 0 1097.98754330 512.0 0 0 0 1 0
Tr: 0 1 0 0 0 0 1 0 1 0 0 0
"""


def extract_satellite_name(folder_name):
    """Extract satellite name from folder name (strips leading timestamp prefix)."""
    if '_' in folder_name:
        return folder_name.split('_', 1)[1]
    return folder_name


def load_satellite_order(json_path):
    """Load satellite ordering from satellite_descriptions.json."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [sat['name'] for sat in data['satellites']]
    except Exception as e:
        print(f"Warning: cannot read {json_path}: {e}")
        return []


def sort_satellites_by_json_order(satellite_folders, json_path):
    """Sort satellite folders according to the order in satellite_descriptions.json."""
    json_order = load_satellite_order(json_path)

    if not json_order:
        print("Warning: JSON order not available, sorting by folder name")
        return sorted(satellite_folders)

    name_to_index = {name: idx for idx, name in enumerate(json_order)}

    def sort_key(folder):
        sat_name = extract_satellite_name(folder.name)
        return name_to_index.get(sat_name, len(json_order) + 1000)

    sorted_folders = sorted(satellite_folders, key=sort_key)

    print("\nSatellite processing order (from satellite_descriptions.json):")
    for idx, folder in enumerate(sorted_folders):
        sat_name = extract_satellite_name(folder.name)
        if sat_name in name_to_index:
            print(f"  {idx:3d}. {sat_name} (JSON index {name_to_index[sat_name]})")
        else:
            print(f"  {idx:3d}. {sat_name} (not in JSON)")

    return sorted_folders


def _get_satellite_folders(raw_data_root):
    """Detect satellite folders, supporting both flat and HuggingFace nested structures."""
    raw_candidates = [d for d in raw_data_root.iterdir() if d.is_dir()]
    satellite_folders = []
    for d in raw_candidates:
        subdirs = [s for s in d.iterdir() if s.is_dir()]
        has_trajectory = any((s / 'image').exists() or (s / 'lidar').exists() for s in subdirs)
        if has_trajectory:
            satellite_folders.append(d)
        else:
            satellite_folders.extend(subdirs)
    return satellite_folders


def process_single_satellite(satellite_folder, raw_data_root, sequences_dir, seq_id):
    """Process one satellite (used for parallel execution).

    Returns:
        (seq_id, satellite_name, frame_count, status)
    """
    try:
        satellite_name = extract_satellite_name(satellite_folder.name)

        seq_dir = sequences_dir / seq_id
        velodyne_dir = seq_dir / "velodyne"
        labels_dir   = seq_dir / "labels"
        image_dir    = seq_dir / "image_2"

        velodyne_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        with open(seq_dir / "calib.txt", 'w') as f:
            f.write(create_default_calib())

        trajectory_dirs = sorted([
            d for d in satellite_folder.iterdir()
            if d.is_dir() and not d.name.startswith('trajectory')
        ])

        if len(trajectory_dirs) == 0:
            return (seq_id, satellite_name, 0, "no trajectory folders found")

        temp_dir = sequences_dir.parent / f"temp_{seq_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        frame_counter = 0

        for traj_dir in trajectory_dirs:
            src_pointcloud = traj_dir / "lidar"
            src_seg        = traj_dir / "seg"
            src_img        = traj_dir / "image"

            if not all([src_pointcloud.exists(), src_seg.exists(), src_img.exists()]):
                continue

            temp_traj_dir = temp_dir / traj_dir.name
            temp_velodyne = temp_traj_dir / "velodyne"
            temp_labels   = temp_traj_dir / "labels"
            temp_images   = temp_traj_dir / "image_2"

            try:
                convert_trajectory_to_kitti(
                    str(src_pointcloud), str(src_seg), str(src_img),
                    str(temp_velodyne), str(temp_labels), str(temp_images)
                )

                for bin_file in sorted(temp_velodyne.glob("*.bin")):
                    new_name  = f"{frame_counter:06d}"
                    timestamp = bin_file.stem

                    shutil.copy2(bin_file, velodyne_dir / f"{new_name}.bin")

                    label_file = temp_labels / f"{timestamp}.label"
                    if label_file.exists():
                        shutil.copy2(label_file, labels_dir / f"{new_name}.label")

                    image_file = temp_images / f"{timestamp}.png"
                    if image_file.exists():
                        shutil.copy2(image_file, image_dir / f"{new_name}.png")

                    frame_counter += 1

            except Exception as e:
                print(f"  [Seq {seq_id}] error: {e}")
                continue

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        return (seq_id, satellite_name, frame_counter, "ok")

    except Exception as e:
        return (seq_id, extract_satellite_name(satellite_folder.name), 0, f"error: {str(e)}")


def convert_airsim_to_kitti_sequences(raw_data_root, output_root, json_path=None):
    """Serial conversion: one satellite per sequence."""
    raw_data_root = Path(raw_data_root)
    output_root   = Path(output_root)

    sequences_dir = output_root / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)

    satellite_folders = _get_satellite_folders(raw_data_root)

    if json_path and Path(json_path).exists():
        satellite_folders = sort_satellites_by_json_order(satellite_folders, json_path)
    else:
        print("Warning: no JSON file specified, sorting by folder name")
        satellite_folders = sorted(satellite_folders)

    print(f"\nFound {len(satellite_folders)} satellite folders")

    sequence_mapping = {}

    for seq_idx, satellite_folder in enumerate(satellite_folders):
        seq_id         = f"{seq_idx:02d}"
        satellite_name = extract_satellite_name(satellite_folder.name)
        sequence_mapping[seq_id] = satellite_name

        print(f"\n{'=' * 60}")
        print(f"Processing Sequence {seq_id}: {satellite_name}")
        print(f"{'=' * 60}")

        seq_dir      = sequences_dir / seq_id
        velodyne_dir = seq_dir / "velodyne"
        labels_dir   = seq_dir / "labels"
        image_dir    = seq_dir / "image_2"

        velodyne_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        with open(seq_dir / "calib.txt", 'w') as f:
            f.write(create_default_calib())

        trajectory_dirs = sorted([
            d for d in satellite_folder.iterdir()
            if d.is_dir() and not d.name.startswith('trajectory')
        ])

        if len(trajectory_dirs) == 0:
            print("  Warning: no trajectory folders, skipping")
            continue

        print(f"  Found {len(trajectory_dirs)} trajectories")

        temp_dir = output_root / "temp" / satellite_name
        temp_dir.mkdir(parents=True, exist_ok=True)

        frame_counter = 0

        for traj_dir in trajectory_dirs:
            print(f"\n  Processing trajectory: {traj_dir.name}")

            src_pointcloud = traj_dir / "lidar"
            src_seg        = traj_dir / "seg"
            src_img        = traj_dir / "image"

            if not all([src_pointcloud.exists(), src_seg.exists(), src_img.exists()]):
                print("    Skipping: missing required sub-folders")
                continue

            temp_traj_dir = temp_dir / traj_dir.name
            temp_velodyne = temp_traj_dir / "velodyne"
            temp_labels   = temp_traj_dir / "labels"
            temp_images   = temp_traj_dir / "image_2"

            try:
                convert_trajectory_to_kitti(
                    str(src_pointcloud), str(src_seg), str(src_img),
                    str(temp_velodyne), str(temp_labels), str(temp_images)
                )

                bin_files = sorted(temp_velodyne.glob("*.bin"))
                print(f"    Converted {len(bin_files)} frames")

                for bin_file in bin_files:
                    new_name  = f"{frame_counter:06d}"
                    timestamp = bin_file.stem

                    shutil.copy2(bin_file, velodyne_dir / f"{new_name}.bin")

                    label_file = temp_labels / f"{timestamp}.label"
                    if label_file.exists():
                        shutil.copy2(label_file, labels_dir / f"{new_name}.label")

                    image_file = temp_images / f"{timestamp}.png"
                    if image_file.exists():
                        shutil.copy2(image_file, image_dir / f"{new_name}.png")

                    frame_counter += 1

            except Exception as e:
                print(f"    Error: {e}")
                continue

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        print(f"\n  Done: Sequence {seq_id} - {frame_counter} frames")

    return sequence_mapping


def convert_airsim_to_kitti_sequences_parallel(raw_data_root, output_root, max_workers=None, json_path=None):
    """Parallel conversion: one satellite per sequence."""
    raw_data_root = Path(raw_data_root)
    output_root   = Path(output_root)

    sequences_dir = output_root / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)

    satellite_folders = _get_satellite_folders(raw_data_root)

    if json_path and Path(json_path).exists():
        satellite_folders = sort_satellites_by_json_order(satellite_folders, json_path)
    else:
        print("Warning: no JSON file specified, sorting by folder name")
        satellite_folders = sorted(satellite_folders)

    total_satellites = len(satellite_folders)
    print(f"\nFound {total_satellites} satellite folders")

    if max_workers is None:
        max_workers = max(1, min(mp.cpu_count() * 2 // 3, total_satellites))

    print(f"CPU cores: {mp.cpu_count()}, using {max_workers} workers")
    print("=" * 60)

    sequence_mapping = {}
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for seq_idx, satellite_folder in enumerate(satellite_folders):
            seq_id = f"{seq_idx:02d}"
            future = executor.submit(
                process_single_satellite,
                satellite_folder, raw_data_root, sequences_dir, seq_id
            )
            futures[future] = seq_id

        completed = 0
        for future in as_completed(futures):
            seq_id, satellite_name, frame_count, status = future.result()
            sequence_mapping[seq_id] = satellite_name
            results.append((seq_id, satellite_name, frame_count, status))
            completed += 1
            print(f"\n[{completed}/{total_satellites}] Seq {seq_id} ({satellite_name}): {frame_count} frames - {status}")

    print("\n" + "=" * 60)
    print("All satellites processed")

    results.sort(key=lambda x: x[0])
    return sequence_mapping, results


def save_sequence_mapping(sequence_mapping, output_path):
    """Save sequence ID -> satellite name mapping to CSV."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence_id', 'satellite_name'])
        for seq_id in sorted(sequence_mapping.keys()):
            writer.writerow([seq_id, sequence_mapping[seq_id]])
    print(f"\nSequence mapping saved to: {output_path}")


if __name__ == "__main__":
    mp.freeze_support()

    import argparse
    import time

    parser = argparse.ArgumentParser(description="Convert AirSim data to Semantic-KITTI format (parallel by default)")
    parser.add_argument("--raw-data", type=str, required=True,
                        help="Raw data root directory (contains satellite sub-folders)")
    parser.add_argument("--output", type=str, default="./converted_data",
                        help="Output directory (default: ./converted_data)")
    parser.add_argument("--satellite-json", type=str, default=None,
                        help="Path to satellite_descriptions.json (controls sequence ordering)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: 2/3 of CPU cores)")
    parser.add_argument("--serial", action="store_true",
                        help="Use serial processing (slower, but easier to debug)")
    args = parser.parse_args()

    raw_data_root = args.raw_data
    output_root   = args.output
    json_path     = args.satellite_json

    print("=" * 60)
    mode = "serial" if args.serial else "parallel"
    print(f"AirSim -> Semantic-KITTI Conversion Tool ({mode})")
    print("=" * 60)
    print(f"Source:    {raw_data_root}")
    print(f"Output:    {output_root}")
    print(f"JSON ref:  {json_path}")
    print("=" * 60)

    start_time = time.time()

    if args.serial:
        sequence_mapping = convert_airsim_to_kitti_sequences(raw_data_root, output_root, json_path)
        results = [(seq_id, name, 0, "ok") for seq_id, name in sequence_mapping.items()]
    else:
        sequence_mapping, results = convert_airsim_to_kitti_sequences_parallel(
            raw_data_root, output_root, max_workers=args.workers, json_path=json_path
        )

    if not args.serial:
        total_frames = sum(r[2] for r in results)
        successful   = sum(1 for r in results if r[3] == "ok")
        failed       = len(results) - successful

        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Total satellites: {len(results)}")
        print(f"  Succeeded: {successful}")
        print(f"  Failed:    {failed}")
        print(f"  Total frames: {total_frames}")

        if failed > 0:
            print("\nFailed satellites:")
            for seq_id, sat_name, _, status in results:
                if status != "ok":
                    print(f"  [{seq_id}] {sat_name}: {status}")

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    mapping_file = Path(output_root) / "sequence_mapping.csv"
    save_sequence_mapping(sequence_mapping, mapping_file)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Sequences created: {len(sequence_mapping)}")
    print(f"Output: {output_root}/sequences")
    print(f"Mapping: {mapping_file}")
    print(f"Elapsed: {minutes}m {seconds}s")
    print("=" * 60)
