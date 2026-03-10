#!/usr/bin/env python3
"""
Raw Data Web Visualizer
Visualize AirSim raw data: RGB, Segmentation, Depth Map, LiDAR Point Cloud.

Supports both flat and nested (HuggingFace) directory structures:
  - flat:   raw_data/<satellite>/approach_xxx/image/...
  - nested: raw_data/<satellite>/<satellite>/approach_xxx/image/...

Usage:
  python raw_data_web_visualizer.py --raw-data /path/to/raw_data
"""
import os
import csv
import json
import argparse
import numpy as np
import cv2
from flask import Flask, render_template, jsonify, request, send_file
from pathlib import Path
import io
from PIL import Image

app = Flask(__name__)

app.config['RAW_DATA_ROOT']  = None
app.config['SATELLITE_JSON'] = None

# Maps satellite display-id -> resolved folder path on disk
# Built once at startup by scan_satellites()
SATELLITE_MAP = {}

LABEL_COLORS = {
    # main_body - class 1
    (156, 198, 23): {'id': 1, 'name': 'main_body', 'color': '#1f77b4'},
    (68, 218, 116): {'id': 1, 'name': 'main_body', 'color': '#1f77b4'},
    (11, 236, 9):   {'id': 1, 'name': 'main_body', 'color': '#1f77b4'},
    (0, 53, 65):    {'id': 1, 'name': 'main_body', 'color': '#1f77b4'},
    # solar_panel - class 2
    (146, 52, 70):  {'id': 2, 'name': 'solar_panel', 'color': '#ff7f0e'},
    (194, 39, 7):   {'id': 2, 'name': 'solar_panel', 'color': '#ff7f0e'},
    (211, 80, 208): {'id': 2, 'name': 'solar_panel', 'color': '#ff7f0e'},
    (189, 135, 188):{'id': 2, 'name': 'solar_panel', 'color': '#ff7f0e'},
    # dish_antenna - class 3
    (124, 21, 123): {'id': 3, 'name': 'dish_antenna', 'color': '#2ca02c'},
    (90, 162, 242): {'id': 3, 'name': 'dish_antenna', 'color': '#2ca02c'},
    (35, 196, 244): {'id': 3, 'name': 'dish_antenna', 'color': '#2ca02c'},
    (220, 163, 49): {'id': 3, 'name': 'dish_antenna', 'color': '#2ca02c'},
    # omni_antenna - class 4
    (86, 254, 214): {'id': 4, 'name': 'omni_antenna', 'color': '#d62728'},
    (125, 75, 48):  {'id': 4, 'name': 'omni_antenna', 'color': '#d62728'},
    (85, 152, 34):  {'id': 4, 'name': 'omni_antenna', 'color': '#d62728'},
    (173, 69, 31):  {'id': 4, 'name': 'omni_antenna', 'color': '#d62728'},
    # payload - class 5
    (37, 128, 125): {'id': 5, 'name': 'payload', 'color': '#9467bd'},
    (58, 19, 33):   {'id': 5, 'name': 'payload', 'color': '#9467bd'},
    (218, 124, 115):{'id': 5, 'name': 'payload', 'color': '#9467bd'},
    (202, 97, 155): {'id': 5, 'name': 'payload', 'color': '#9467bd'},
    # thruster - class 6
    (133, 244, 133):{'id': 6, 'name': 'thruster', 'color': '#8c564b'},
    (1, 222, 192):  {'id': 6, 'name': 'thruster', 'color': '#8c564b'},
    (65, 54, 217):  {'id': 6, 'name': 'thruster', 'color': '#8c564b'},
    (216, 78, 75):  {'id': 6, 'name': 'thruster', 'color': '#8c564b'},
    # adapter_ring - class 7
    (158, 114, 88): {'id': 7, 'name': 'adapter_ring', 'color': '#e377c2'},
    (181, 213, 93): {'id': 7, 'name': 'adapter_ring', 'color': '#e377c2'},
}


def extract_satellite_name(folder_name):
    """Extract satellite name, stripping optional timestamp prefix like '20260114_ACE' -> 'ACE'."""
    if '_' in folder_name:
        return folder_name.split('_', 1)[1]
    return folder_name


def load_satellite_order():
    """Load satellite ordering from satellite_descriptions.json."""
    json_path = app.config['SATELLITE_JSON']
    if json_path is None:
        return []
    try:
        p = Path(json_path)
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [sat['name'] for sat in data['satellites']]
    except Exception as e:
        print(f"Warning: cannot read {json_path}: {e}")
    return []


def scan_satellites():
    """Scan RAW_DATA_ROOT and build SATELLITE_MAP.

    Handles two directory layouts:
      flat:   root/<sat>/approach_xxx/image/...
      nested: root/<sat>/<sat>/approach_xxx/image/...

    Each entry in SATELLITE_MAP: display_id -> Path to actual satellite folder
    (the folder that directly contains trajectory sub-dirs like approach_front/).
    """
    global SATELLITE_MAP
    SATELLITE_MAP = {}

    raw_root = app.config['RAW_DATA_ROOT']
    if raw_root is None or not Path(raw_root).exists():
        return

    root = Path(raw_root)
    json_order = load_satellite_order()
    name_to_index = {name: idx for idx, name in enumerate(json_order)}

    resolved_folders = []

    for top_dir in sorted(root.iterdir()):
        if not top_dir.is_dir():
            continue
        subdirs = [s for s in top_dir.iterdir() if s.is_dir()]
        has_trajectory = any(
            (s / 'image').exists() or (s / 'seg').exists() or (s / 'lidar').exists()
            for s in subdirs
        )
        if has_trajectory:
            resolved_folders.append(top_dir)
        else:
            for sub in subdirs:
                if sub.is_dir():
                    resolved_folders.append(sub)

    def sort_key(folder):
        sat_name = extract_satellite_name(folder.name)
        return name_to_index.get(sat_name, len(json_order) + 1000)

    resolved_folders.sort(key=sort_key)

    for folder in resolved_folders:
        sat_name = extract_satellite_name(folder.name)
        SATELLITE_MAP[sat_name] = folder


def _resolve_satellite(satellite_id):
    """Look up the actual folder Path for a satellite_id (display name)."""
    return SATELLITE_MAP.get(satellite_id)


def read_asc_pointcloud(filepath):
    """Read a .asc point cloud file."""
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
    return np.array(points, dtype=np.float32) if points else np.zeros((0, 3), dtype=np.float32)


def read_depth_image(filepath):
    """Read depth image (.npz, .tiff, .png)."""
    filepath = Path(filepath)
    if filepath.suffix == '.npz':
        data = np.load(str(filepath))
        return data['depth'].astype(np.float32) / 1000.0  # mm -> m
    else:
        depth = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        if depth is not None:
            return depth.astype(np.float32)
    return None


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('raw_data_viewer.html')


@app.route('/api/satellites')
def get_satellites():
    """Return all discovered satellites."""
    try:
        satellites = [{'id': name, 'name': name} for name in SATELLITE_MAP]
        return jsonify({'success': True, 'satellites': satellites})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/trajectories/<satellite_id>')
def get_trajectories(satellite_id):
    """Return all trajectory names for a satellite."""
    try:
        sat_folder = _resolve_satellite(satellite_id)
        if sat_folder is None:
            return jsonify({'success': False, 'error': f'Satellite {satellite_id} not found'})

        trajectory_dirs = sorted([
            d.name for d in sat_folder.iterdir()
            if d.is_dir() and not d.name.startswith('trajectory')
        ])

        return jsonify({'success': True, 'trajectories': trajectory_dirs, 'total': len(trajectory_dirs)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/frames/<satellite_id>/<trajectory_id>')
def get_frames(satellite_id, trajectory_id):
    """Return all frame IDs for a trajectory (based on image or lidar files)."""
    try:
        sat_folder = _resolve_satellite(satellite_id)
        if sat_folder is None:
            return jsonify({'success': False, 'error': 'Satellite not found'})

        traj_folder = sat_folder / trajectory_id

        # Try lidar first, then image folder
        frame_ids = []
        lidar_folder = traj_folder / 'lidar'
        image_folder = traj_folder / 'image'

        if lidar_folder.exists():
            frame_ids = sorted(f.stem for f in lidar_folder.glob('*.asc'))
        elif image_folder.exists():
            frame_ids = sorted(f.stem for f in image_folder.glob('*.png'))

        if not frame_ids:
            return jsonify({'success': False, 'error': 'No frames found'})

        return jsonify({'success': True, 'frames': frame_ids, 'total': len(frame_ids)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/image/<satellite_id>/<trajectory_id>/<frame_id>')
def get_image(satellite_id, trajectory_id, frame_id):
    try:
        sat_folder = _resolve_satellite(satellite_id)
        if sat_folder is None:
            return jsonify({'success': False, 'error': 'Satellite not found'}), 404
        img_path = sat_folder / trajectory_id / 'image' / f'{frame_id}.png'
        if img_path.exists():
            return send_file(str(img_path), mimetype='image/png')
        return jsonify({'success': False, 'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 404


@app.route('/api/segmentation/<satellite_id>/<trajectory_id>/<frame_id>')
def get_segmentation(satellite_id, trajectory_id, frame_id):
    try:
        sat_folder = _resolve_satellite(satellite_id)
        if sat_folder is None:
            return jsonify({'success': False, 'error': 'Satellite not found'}), 404
        seg_path = sat_folder / trajectory_id / 'seg' / f'{frame_id}.png'
        if seg_path.exists():
            return send_file(str(seg_path), mimetype='image/png')
        return jsonify({'success': False, 'error': 'Segmentation not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 404


@app.route('/api/depth/<satellite_id>/<trajectory_id>/<frame_id>')
def get_depth(satellite_id, trajectory_id, frame_id):
    """Return colorized depth map as PNG."""
    try:
        sat_folder = _resolve_satellite(satellite_id)
        if sat_folder is None:
            return jsonify({'success': False, 'error': 'Satellite not found'}), 404

        depth_dir = sat_folder / trajectory_id / 'depth'
        depth_path = None
        for ext in ['.npz', '.tiff', '.tif', '.png']:
            candidate = depth_dir / f'{frame_id}{ext}'
            if candidate.exists():
                depth_path = candidate
                break

        if depth_path is None:
            return jsonify({'success': False, 'error': 'Depth image not found'}), 404

        depth = read_depth_image(depth_path)
        if depth is None:
            return jsonify({'success': False, 'error': 'Failed to read depth'}), 404

        valid_mask = (depth > 0) & (depth < 5000.0)
        valid_depth = depth[valid_mask]

        if len(valid_depth) > 0:
            d_min, d_max = valid_depth.min(), valid_depth.max()
            depth_norm = np.zeros_like(depth, dtype=np.uint8)
            depth_norm[valid_mask] = ((depth[valid_mask] - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_JET)
            depth_colored[~valid_mask] = [0, 0, 0]
        else:
            depth_colored = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

        _, buffer = cv2.imencode('.png', depth_colored)
        return send_file(io.BytesIO(buffer), mimetype='image/png')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 404


@app.route('/api/pointcloud/<satellite_id>/<trajectory_id>/<frame_id>')
def get_pointcloud(satellite_id, trajectory_id, frame_id):
    try:
        sat_folder = _resolve_satellite(satellite_id)
        if sat_folder is None:
            return jsonify({'success': False, 'error': 'Satellite not found'})

        lidar_path = sat_folder / trajectory_id / 'lidar' / f'{frame_id}.asc'
        if not lidar_path.exists():
            return jsonify({'success': False, 'error': 'Point cloud not found'})

        points = read_asc_pointcloud(lidar_path)
        return jsonify({
            'success': True,
            'points': points[:, :3].tolist() if len(points) > 0 else [],
            'point_count': len(points)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/depth_stats/<satellite_id>/<trajectory_id>/<frame_id>')
def get_depth_stats(satellite_id, trajectory_id, frame_id):
    try:
        sat_folder = _resolve_satellite(satellite_id)
        if sat_folder is None:
            return jsonify({'success': False, 'error': 'Satellite not found'})

        depth_dir = sat_folder / trajectory_id / 'depth'
        depth_path = None
        for ext in ['.npz', '.tiff', '.tif', '.png']:
            candidate = depth_dir / f'{frame_id}{ext}'
            if candidate.exists():
                depth_path = candidate
                break

        if depth_path is None:
            return jsonify({'success': False, 'error': 'Depth image not found'})

        depth = read_depth_image(depth_path)
        if depth is None:
            return jsonify({'success': False, 'error': 'Failed to read depth'})

        valid_mask = (depth > 0) & (depth < 5000.0)
        valid_depth = depth[valid_mask]

        if len(valid_depth) > 0:
            stats = {
                'min': float(valid_depth.min()),
                'max': float(valid_depth.max()),
                'mean': float(valid_depth.mean()),
                'median': float(np.median(valid_depth)),
                'valid_pixels': int(len(valid_depth)),
                'total_pixels': int(depth.size),
                'valid_ratio': float(len(valid_depth) / depth.size)
            }
        else:
            stats = {
                'min': 0, 'max': 0, 'mean': 0, 'median': 0,
                'valid_pixels': 0,
                'total_pixels': int(depth.size),
                'valid_ratio': 0
            }

        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Raw data web visualizer for SpaceSense-Bench")
    parser.add_argument('--raw-data', type=str, required=True,
                        help='Raw data root directory (e.g. ./data or ./data_example)')
    parser.add_argument('--satellite-json', type=str, default=None,
                        help='Path to satellite_descriptions.json (optional, for ordering)')
    parser.add_argument('--port', type=int, default=5001, help='Port number (default: 5001)')
    args = parser.parse_args()

    app.config['RAW_DATA_ROOT']  = os.path.abspath(args.raw_data)
    app.config['SATELLITE_JSON'] = os.path.abspath(args.satellite_json) if args.satellite_json else None

    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)

    scan_satellites()

    print(f"\n=== Raw Data Web Visualizer ===")
    print(f"Data root: {args.raw_data}")
    print(f"Found {len(SATELLITE_MAP)} satellites")
    print(f"Open in browser: http://localhost:{args.port}")
    print("Press Ctrl+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=args.port)
