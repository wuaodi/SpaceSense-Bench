# SpaceSense-Bench Dataset Scale Validator

This directory contains `check_dataset_scale.py`, a diagnostic utility for identifying potential scale inconsistencies between raw depth maps and pose annotations in the SpaceSense-Bench dataset.

## Background

During multimodal pose estimation experiments, some trajectories appear to exhibit a significant mismatch between:

- the geometric scale implied by the raw `.npz` depth maps, and
- the translation magnitude recorded in `pose_ground_truth.csv`.

For example, in some cases, point clouds reconstructed from depth maps suggest a median Z-distance of about `79.0 m`, while the corresponding pose annotation indicates a translation norm of only about `7.7 m`.

This script is intended to help users automatically detect such suspicious frames for further inspection.

## Features

- **Recursive dataset traversal**  
  Scans the native SpaceSense-Bench directory structure directly by locating `pose_ground_truth.csv` files and matching them to corresponding depth frames.

- **Robust timestamp parsing**  
  Uses `dtype={'timestamp': str}` to avoid precision loss when parsing large timestamp identifiers.

- **Parallel scanning**  
  Supports multiprocessing for efficient large-scale dataset checking.

- **Suspicious frame logging**  
  Writes potentially anomalous frames to a text file for later filtering or inspection.

## Requirements

Install the required packages before running:

```bash
pip install numpy opencv-python pandas tqdm tifffile
```

## Usage
Run the validator from the terminal and point it to the extracted all_data directory:

```bash
python check_dataset_scale.py \
  --dataset_root /path/to/SpaceSenseData/raw/all_data \
  --out_txt corrupted_images.txt \
  --threshold 10.0
```

## Parameters
--dataset_root (required)

Path to the root directory of the extracted dataset (for example, .../raw/all_data).

--out_txt (optional, default: corrupted_images.txt)

Path to the output text file. Suspicious frame paths will be written to this file.

--threshold (optional, default: 10.0)

Threshold for flagging suspicious scale ratios between the depth-implied geometry and the ground-truth translation magnitude.

Larger values make the check more conservative. Smaller values make it more sensitive.

## Output

After scanning, the script writes suspicious frame paths to the specified text file. Users may then:

manually inspect these samples,
exclude them during training or evaluation,
or use the list in downstream preprocessing pipelines.

## Note

This tool is intended as a dataset validation utility. Flagged samples should be treated as candidates for further inspection rather than as automatically confirmed corrupted data.