#!/usr/bin/env python3
"""
YOLO format data web visualizer.
Visualizes converted YOLO annotations with bounding boxes overlaid on images.

Usage:
  python yolo_web_visualizer.py --data-root /path/to/yolo_data
"""
import os
import sys
import cv2
import numpy as np
from flask import Flask, render_template, jsonify, send_file
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))

app.config['YOLO_DATA_ROOT'] = None

# Class definitions
CLASS_NAMES = ['main_body', 'solar_panel', 'dish_antenna', 'omni_antenna',
               'payload', 'thruster', 'adapter_ring']

# Colors per class (BGR for OpenCV)
CLASS_COLORS = [
    (180, 119, 31),   # main_body - blue
    (14, 127, 255),   # solar_panel - orange
    (44, 160, 44),    # dish_antenna - green
    (40, 39, 214),    # omni_antenna - red
    (189, 103, 148),  # payload - purple
    (75, 86, 140),    # thruster - brown
    (194, 119, 227)   # adapter_ring - pink
]


def extract_satellite_name(filename):
    """Extract satellite name from filename (e.g. ACE_approach_front_xxx -> ACE)."""
    return filename.split('_')[0]


def get_dataset_info():
    """Collect per-split image counts grouped by satellite."""
    datasets = {}
    
    for split in ['train', 'val']:
        image_dir = app.config['YOLO_DATA_ROOT'] / split / 'images'
        label_dir = app.config['YOLO_DATA_ROOT'] / split / 'labels'
        
        if not image_dir.exists():
            continue
        
        image_files = sorted(image_dir.glob('*.png'))

        satellites = {}
        for img_file in image_files:
            sat_name = extract_satellite_name(img_file.stem)
            if sat_name not in satellites:
                satellites[sat_name] = []
            satellites[sat_name].append(img_file.stem)
        
        datasets[split] = {
            'total_images': len(image_files),
            'satellites': satellites,
            'satellite_count': len(satellites)
        }
    
    return datasets


def parse_yolo_label(label_path, img_width, img_height):
    """Parse a YOLO label file and return bounding boxes in pixel coordinates."""
    if not label_path.exists():
        return []

    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) != 5:
                continue

            class_id = int(values[0])
            cx, cy, w, h = map(float, values[1:])

            cx_px = cx * img_width
            cy_px = cy * img_height
            w_px  = w  * img_width
            h_px  = h  * img_height

            x1 = int(cx_px - w_px / 2)
            y1 = int(cy_px - h_px / 2)
            x2 = int(cx_px + w_px / 2)
            y2 = int(cy_px + h_px / 2)

            boxes.append({
                'class_id':   class_id,
                'class_name': CLASS_NAMES[class_id],
                'bbox':       [x1, y1, x2, y2],
                'center':     [int(cx_px), int(cy_px)],
                'size':       [int(w_px), int(h_px)],
                'normalized': [cx, cy, w, h]
            })

    return boxes


def draw_boxes_on_image(image_path, label_path):
    """Draw bounding boxes on an image and return the annotated PIL image."""
    img = Image.open(image_path)
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)

    boxes = parse_yolo_label(label_path, img_width, img_height)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for box in boxes:
        class_id   = box['class_id']
        class_name = box['class_name']
        x1, y1, x2, y2 = box['bbox']

        color = CLASS_COLORS[class_id]
        color_rgb = (color[2], color[1], color[0])  # BGR -> RGB

        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)

        text = class_name
        bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color_rgb)
        draw.text((x1, y1), text, fill=(255, 255, 255), font=font)

    return img, boxes


@app.route('/')
def index():
    return render_template('yolo_visualizer.html')


@app.route('/api/dataset_info')
def get_dataset_info_api():
    """Return dataset split statistics."""
    try:
        datasets = get_dataset_info()
        return jsonify({'success': True, 'datasets': datasets})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/satellites/<split>')
def get_satellites(split):
    """Return satellite list for a given split."""
    try:
        datasets = get_dataset_info()
        if split not in datasets:
            return jsonify({'success': False, 'error': f'Split {split} not found'})
        satellites = datasets[split]['satellites']
        satellite_list = [{'name': n, 'image_count': len(imgs)} for n, imgs in sorted(satellites.items())]
        return jsonify({'success': True, 'satellites': satellite_list, 'total': len(satellite_list)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/images/<split>/<satellite_name>')
def get_images(split, satellite_name):
    """Return all image IDs for a given satellite in a split."""
    try:
        datasets = get_dataset_info()
        if split not in datasets:
            return jsonify({'success': False, 'error': f'Split {split} not found'})
        satellites = datasets[split]['satellites']
        if satellite_name not in satellites:
            return jsonify({'success': False, 'error': f'Satellite {satellite_name} not found'})
        images = satellites[satellite_name]
        return jsonify({'success': True, 'images': images, 'total': len(images)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/image_with_boxes/<split>/<image_name>')
def get_image_with_boxes(split, image_name):
    """Return the image with bounding boxes drawn on it."""
    try:
        image_path = app.config['YOLO_DATA_ROOT'] / split / 'images' / f'{image_name}.png'
        label_path = app.config['YOLO_DATA_ROOT'] / split / 'labels' / f'{image_name}.txt'
        if not image_path.exists():
            return jsonify({'success': False, 'error': 'Image not found'}), 404
        img, _ = draw_boxes_on_image(image_path, label_path)
        img_io = BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/annotation/<split>/<image_name>')
def get_annotation(split, image_name):
    """Return bounding box annotations for a given image."""
    try:
        image_path = app.config['YOLO_DATA_ROOT'] / split / 'images' / f'{image_name}.png'
        label_path = app.config['YOLO_DATA_ROOT'] / split / 'labels' / f'{image_name}.txt'
        if not image_path.exists():
            return jsonify({'success': False, 'error': 'Image not found'})
        img = Image.open(image_path)
        img_width, img_height = img.size
        boxes = parse_yolo_label(label_path, img_width, img_height)
        class_stats = {}
        for box in boxes:
            cn = box['class_name']
            if cn not in class_stats:
                class_stats[cn] = {'count': 0, 'class_id': box['class_id'], 'color': CLASS_COLORS[box['class_id']]}
            class_stats[cn]['count'] += 1
        return jsonify({
            'success': True,
            'boxes': boxes,
            'total_boxes': len(boxes),
            'class_stats': class_stats,
            'image_size': [img_width, img_height]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/class_info')
def get_class_info():
    """Return class name and color definitions."""
    class_info = []
    for i, name in enumerate(CLASS_NAMES):
        color = CLASS_COLORS[i]
        class_info.append({
            'id': i,
            'name': name,
            'color': [color[2], color[1], color[0]]  # BGR -> RGB
        })
    return jsonify(class_info)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="YOLO annotation web visualizer")
    parser.add_argument('--data-root', type=str, required=True,
                        help='YOLO data root directory')
    parser.add_argument('--port', type=int, default=5001, help='Port number (default: 5001)')
    _args = parser.parse_args()

    app.config['YOLO_DATA_ROOT'] = Path(_args.data_root).resolve()

    print(f"\n=== YOLO Web Visualizer ===")
    print(f"Data root: {_args.data_root}")
    print(f"Open in browser: http://localhost:{_args.port}")
    print("Press Ctrl+C to stop\n")

    app.run(debug=True, host='0.0.0.0', port=_args.port)

