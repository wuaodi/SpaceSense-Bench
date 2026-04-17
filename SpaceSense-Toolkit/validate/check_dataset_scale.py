import argparse
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

def check_single_sample(args_tuple, threshold):
    timestamp, traj_folder, pose_t = args_tuple
    
    depth_path = traj_folder / "depth" / f"{timestamp}.npz"
    mask_path = traj_folder / "seg" / f"{timestamp}.png"
    img_path = traj_folder / "image" / f"{timestamp}.png"
    
    if not depth_path.exists():
        # Fallback format checking
        depth_path = traj_folder / "depth" / f"{timestamp}.tiff"
        if not depth_path.exists():
            depth_path = traj_folder / "depth" / f"{timestamp}.png"
            if not depth_path.exists():
                return (str(img_path), False)
                
    gt_dist = np.linalg.norm(pose_t)
    
    try:
        # Load exact Depth map
        if depth_path.suffix == '.npz':
            depth_data = np.load(depth_path)
            depth_img = depth_data[depth_data.files[0]]
        elif depth_path.suffix in ['.tiff', '.tif']:
            import tifffile
            depth_img = tifffile.imread(depth_path)
        else:
            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            
        depth_img = depth_img.astype(np.float32) / 1000.0  # standard conversion
        
        # Mask the target to get true object distance
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.sum() > 0:
                valid_depths = depth_img[mask > 0]
            else:
                valid_depths = depth_img[(depth_img > 0.1) & (depth_img < 100.0)]
        else:
            valid_depths = depth_img[(depth_img > 0.1) & (depth_img < 100.0)]
            
        if len(valid_depths) == 0:
            return (str(img_path), True)
            
        median_depth = np.median(valid_depths)
        ratio = median_depth / (gt_dist + 1e-6)
        
        # Determine if scales fundamentally mismatch
        if ratio > threshold or ratio < 1.0 / threshold:
            return (str(img_path), True)
        else:
            return (str(img_path), False)
            
    except Exception as e:
        return (str(img_path), True)

def scan_dataset_native(root_dir, out_txt_path, threshold):
    print(f"Scanning native SpaceSense-Bench directory: {root_dir}")
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Error: {root_dir} does not exist.")
        return

    csv_files = list(root_path.rglob("pose_ground_truth.csv"))
    print(f"Found {len(csv_files)} trajectory CSV files.")

    all_samples = []
    
    for csv_file in csv_files:
        traj_folder = csv_file.parent
        df = pd.read_csv(csv_file, dtype={'timestamp': str})
        
        # SpaceSense-Bench standard column names
        req_cols = ['timestamp', 'target_spacecraft_in_camera_pos_x(m)', 'target_spacecraft_in_camera_pos_y(m)', 'target_spacecraft_in_camera_pos_z(m)']
        
        if all(col in df.columns for col in req_cols):
            for _, row in df.iterrows():
                ts = str(row['timestamp']).strip('"')
                t = np.array([
                    row['target_spacecraft_in_camera_pos_x(m)'],
                    row['target_spacecraft_in_camera_pos_y(m)'],
                    row['target_spacecraft_in_camera_pos_z(m)']
                ])
                all_samples.append((ts, traj_folder, t))
        else:
            print(f"[Warning] Skipping {csv_file}, missing required target coordinates.")

    print(f"Found {len(all_samples)} total images annotated in native CSVs.")
    print(f"Analyzing 3D physics scale matching using {cpu_count()} CPU cores...")

    corrupted_images = []
    worker_func = partial(check_single_sample, threshold=threshold)

    # Process all images in parallel
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(worker_func, all_samples), total=len(all_samples), desc="Scanning Images"))
        
    for img_path, is_corrupted in results:
        if is_corrupted:
            corrupted_images.append(img_path)

    print(f"\nDetection complete!")
    print(f"Clean Images Verified: {len(all_samples) - len(corrupted_images)}")
    print(f"Corrupted Images Caught: {len(corrupted_images)}")

    print(f"\nExporting the list of corrupted image paths to: {out_txt_path}")
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for t in corrupted_images:
            f.write(t + "\n")
            
    print(f"Done! {len(corrupted_images)} corrupt image paths logged in {out_txt_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Native Scanner for SpaceSense-Bench depth scale anomalies.")
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to the SpaceSense-Bench raw dataset root folder')
    parser.add_argument('--out_txt', type=str, default='corrupted_images.txt', help='Path to export the corrupted image checklist')
    parser.add_argument('--threshold', type=float, default=10.0, help='Threshold for scale detection')
    args = parser.parse_args()
    
    scan_dataset_native(args.dataset_root, args.out_txt, args.threshold)
