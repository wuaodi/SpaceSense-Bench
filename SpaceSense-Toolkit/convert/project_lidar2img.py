"""
Interactive LiDAR-to-image projection visualizer.
Projects AirSim .asc point clouds onto RGB images and displays them side-by-side.
Press any key to advance to the next frame.

Usage:
  python project_lidar2img.py --img-path /path/to/image --lidar-path /path/to/lidar
  python project_lidar2img.py --img-path /path/to/image --lidar-path /path/to/lidar --step 5
"""
import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse


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


class InteractiveVisualizer:
    """Interactive frame-by-frame LiDAR projection visualizer."""

    def __init__(self, img_path, lidar_path, step=10):
        self.img_path   = img_path
        self.lidar_path = lidar_path
        self.step       = step
        self.current_idx = 0

        asc_files = sorted([f for f in os.listdir(lidar_path) if f.endswith('.asc')])
        png_files = set(f for f in os.listdir(img_path) if f.endswith('.png'))

        self.file_pairs = []
        for i in range(0, len(asc_files), step):
            asc_file = asc_files[i]
            png_file = asc_file.replace('.asc', '.png')
            if png_file in png_files:
                self.file_pairs.append((asc_file, png_file))

        print(f"Found {len(asc_files)} files, sampling every {step} frames -> {len(self.file_pairs)} frames to display")

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7.5))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def visualize_current(self):
        if self.current_idx >= len(self.file_pairs):
            print("Reached last frame")
            return

        asc_file, png_file = self.file_pairs[self.current_idx]
        pointcloud_path = os.path.join(self.lidar_path, asc_file)
        image_path      = os.path.join(self.img_path, png_file)

        print(f"Frame {self.current_idx + 1}/{len(self.file_pairs)}: {png_file}")

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        points = read_asc_pointcloud(pointcloud_path)

        self.ax1.clear()
        self.ax2.clear()

        self.ax1.imshow(img)
        self.ax1.set_title('Original Image')

        projected_img  = img.copy()
        valid_points   = []

        for point in points:
            transformed = transform_lidar_to_camera_frame(point)
            uv = project_point_to_image(transformed, w, h)
            if uv:
                valid_points.append(uv)
                cv2.circle(projected_img, uv, radius=1, color=(255, 0, 0), thickness=2)

        self.ax2.imshow(projected_img)
        self.ax2.set_title(f'Projected Points ({len(valid_points)} valid) - Press any key for next')

        self.fig.canvas.draw()

    def on_key_press(self, event):
        self.current_idx += 1
        if self.current_idx < len(self.file_pairs):
            self.visualize_current()
        else:
            print("All frames displayed")
            plt.close(self.fig)

    def start(self):
        if len(self.file_pairs) == 0:
            print("No matching file pairs found")
            return
        self.visualize_current()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive LiDAR-to-image projection visualizer")
    parser.add_argument("--img-path",   type=str, required=True,
                        help="Image folder path (e.g. raw_data/ACE/approach_back/image)")
    parser.add_argument("--lidar-path", type=str, required=True,
                        help="LiDAR folder path (e.g. raw_data/ACE/approach_back/lidar)")
    parser.add_argument("--step", type=int, default=10,
                        help="Sample every N frames (default: 10)")
    args = parser.parse_args()

    visualizer = InteractiveVisualizer(args.img_path, args.lidar_path, args.step)
    visualizer.start()
