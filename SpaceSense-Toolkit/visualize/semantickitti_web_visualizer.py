import os
import csv
import numpy as np
from flask import Flask, render_template, jsonify, request, send_from_directory, send_file
from glob import glob
from pathlib import Path
import json

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))

app.config['KITTI_DATA_ROOT'] = None
app.config['SEQUENCES_DIR']   = None
app.config['MAPPING_FILE']    = None
app.config['SATELLITE_JSON']  = None

# Class definitions (labels 1-7; 0 = background, not used here)
LABEL_NAMES = {
    1: 'main_body',
    2: 'solar_panel',
    3: 'dish_antenna',
    4: 'omni_antenna',
    5: 'payload',
    6: 'thruster',
    7: 'adapter_ring'
}

# Colors per label (RGB, 0-255)
LABEL_COLORS = {
    1: [31, 119, 180],    # blue
    2: [255, 127, 14],    # orange
    3: [44, 160, 44],     # green
    4: [214, 39, 40],     # red
    5: [148, 103, 189],   # purple
    6: [140, 86, 75],     # brown
    7: [227, 119, 194]    # pink/magenta
}

def load_satellite_info():
    """Load satellite metadata from satellite_descriptions.json."""
    satellite_info = {}
    json_path = app.config['SATELLITE_JSON']
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for sat in data['satellites']:
                satellite_info[sat['name']] = {
                    'description': sat.get('description', ''),
                    'max_diameter_meters': sat.get('max_diameter_meters', None)
                }
        except Exception as e:
            print(f"Warning: cannot read satellite info: {e}")
    return satellite_info


def load_sequence_mapping():
    """Load sequence ID -> satellite name mapping with metadata."""
    mapping = {}
    satellite_info = load_satellite_info()
    mapping_file = app.config['MAPPING_FILE']
    if mapping_file and os.path.exists(mapping_file):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sat_name = row['satellite_name']
                mapping[row['sequence_id']] = {
                    'name': sat_name,
                    'info': satellite_info.get(sat_name, {})
                }
    return mapping

def read_bin_pointcloud(bin_path):
    """Read a .bin point cloud file (XYZI float32)."""
    points = np.fromfile(bin_path, dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(f"Point cloud size not divisible by 4: {points.size}")
    return points.reshape(-1, 4)

def read_label_file(label_path):
    """Read a .label file (uint32 per point)."""
    if not os.path.exists(label_path):
        return None
    return np.fromfile(label_path, dtype=np.uint32)

@app.route('/')
def index():
    return render_template('visualizer.html')

@app.route('/api/satellites')
def get_satellites():
    """Return all satellites with metadata (from sequence mapping)."""
    try:
        mapping = load_sequence_mapping()
        satellites = []
        for seq_id in sorted(mapping.keys(), key=lambda x: int(x)):
            sat_data = mapping[seq_id]
            satellites.append({
                'id': seq_id,
                'name': sat_data['name'],
                'max_diameter': sat_data['info'].get('max_diameter_meters'),
                'description': sat_data['info'].get('description', '')[:100] + '...' if len(sat_data['info'].get('description', '')) > 100 else sat_data['info'].get('description', '')
            })
        return jsonify({'success': True, 'satellites': satellites})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/frames/<sequence_id>')
def get_frames(sequence_id):
    """Return all frame IDs for a given sequence."""
    try:
        velodyne_dir = os.path.join(app.config['SEQUENCES_DIR'], sequence_id, 'velodyne')
        bin_files = sorted(glob(os.path.join(velodyne_dir, "*.bin")))
        frame_ids = [os.path.splitext(os.path.basename(f))[0] for f in bin_files]
        
        mapping = load_sequence_mapping()
        sat_data = mapping.get(sequence_id, {'name': f"Sequence {sequence_id}", 'info': {}})
        satellite_name = sat_data['name']
        
        return jsonify({
            'success': True, 
            'frames': frame_ids, 
            'total': len(frame_ids),
            'satellite_name': satellite_name,
            'max_diameter': sat_data['info'].get('max_diameter_meters')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/pointcloud/<sequence_id>/<frame_id>')
def get_pointcloud(sequence_id, frame_id):
    """Return XYZ points and per-point labels for a given frame."""
    try:
        bin_path   = os.path.join(app.config['SEQUENCES_DIR'], sequence_id, 'velodyne', f'{frame_id}.bin')
        label_path = os.path.join(app.config['SEQUENCES_DIR'], sequence_id, 'labels',   f'{frame_id}.label')

        points = read_bin_pointcloud(bin_path)
        labels = read_label_file(label_path)

        data = {
            'success': True,
            'points': points[:, :3].tolist(),
            'labels': labels.tolist() if labels is not None else None,
            'point_count': len(points)
        }

        if labels is not None:
            label_stats = {}
            for lbl in np.unique(labels):
                label_stats[int(lbl)] = {
                    'name':  LABEL_NAMES.get(int(lbl), 'unknown'),
                    'count': int(np.sum(labels == lbl)),
                    'color': LABEL_COLORS.get(int(lbl), [128, 128, 128])
                }
            data['label_stats'] = label_stats

        return jsonify(data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/label_info')
def get_label_info():
    """Return label name and color definitions."""
    label_info = {}
    for label_id, name in LABEL_NAMES.items():
        label_info[label_id] = {
            'name': name,
            'color': LABEL_COLORS[label_id]
        }
    return jsonify(label_info)

@app.route('/api/image/<sequence_id>/<frame_id>')
def get_image(sequence_id, frame_id):
    """Return the RGB image for a given frame."""
    image_path = os.path.join(app.config['SEQUENCES_DIR'], sequence_id, 'image_2', f'{frame_id}.png')
    if not os.path.exists(image_path):
        return jsonify({'success': False, 'error': f'Image not found: {image_path}'}), 404
    return send_file(image_path, mimetype='image/png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Semantic-KITTI point cloud web visualizer")
    parser.add_argument('--data-root', type=str, required=True,
                        help='Semantic-KITTI data root (contains sequences/)')
    parser.add_argument('--satellite-json', type=str, default=None,
                        help='Path to satellite_descriptions.json (optional)')
    parser.add_argument('--port', type=int, default=5000, help='Port number (default: 5000)')
    _args = parser.parse_args()

    data_root = os.path.abspath(_args.data_root)
    app.config['KITTI_DATA_ROOT'] = data_root
    app.config['SEQUENCES_DIR']   = os.path.join(data_root, 'sequences')
    app.config['MAPPING_FILE']    = os.path.join(data_root, 'sequence_mapping.csv')
    app.config['SATELLITE_JSON']  = os.path.abspath(_args.satellite_json) if _args.satellite_json else None

    print(f"\n=== Semantic-KITTI Web Visualizer ===")
    print(f"Data root: {_args.data_root}")
    print(f"Open in browser: http://localhost:{_args.port}")
    print("Press Ctrl+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=_args.port)

