"""Shared LiDAR/camera geometry for the SpaceSense conversion tools."""

import json
import math
from pathlib import Path

import numpy as np


DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "legacy_lidar_camera_calibration_v1.json"
)

# AirSim camera axes (+X forward, +Y right, +Z down) to KITTI camera axes
# (+X right, +Y down, +Z forward).
AIRSIM_TO_KITTI_CAMERA = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _validate_rigid_transform(transform):
    """Validate and return a 4x4 rigid transform as float64."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(
            f"T_camera_from_lidar must have shape (4, 4), got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("T_camera_from_lidar contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("T_camera_from_lidar must have homogeneous last row [0, 0, 0, 1]")

    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("T_camera_from_lidar rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("T_camera_from_lidar rotation must have determinant +1")
    return matrix


def load_calibration(calibration_path=DEFAULT_CALIBRATION_PATH):
    """Load a versioned calibration document.

    Returns:
        (T_camera_from_lidar, metadata), where the transform is a float64 4x4
        matrix and metadata is the decoded JSON document.
    """
    path = Path(calibration_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as calibration_file:
        metadata = json.load(calibration_file)

    if metadata.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported calibration schema_version: {metadata.get('schema_version')!r}"
        )
    if metadata.get("units") != "metres":
        raise ValueError("Calibration translation units must be 'metres'")
    if "T_camera_from_lidar" not in metadata:
        raise ValueError("Calibration is missing T_camera_from_lidar")

    transform = _validate_rigid_transform(metadata["T_camera_from_lidar"])
    return transform, metadata


def transform_points(points, transform):
    """Apply a LiDAR-to-camera rigid transform to one point or an array."""
    matrix = _validate_rigid_transform(transform)
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.size == 0:
        return point_array.reshape(0, 3)
    if point_array.ndim == 0 or point_array.shape[-1] != 3:
        raise ValueError("points must have shape (3,) or (..., 3)")

    original_shape = point_array.shape
    flat_points = point_array.reshape(-1, 3)
    transformed = flat_points @ matrix[:3, :3].T + matrix[:3, 3]
    return transformed.reshape(original_shape)


def project_point_to_image(
    point,
    image_width,
    image_height,
    horizontal_fov_degrees=50.0,
):
    """Project an AirSim-camera-frame point and return integer (u, v)."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 < horizontal_fov_degrees < 180.0:
        raise ValueError("horizontal_fov_degrees must be in (0, 180)")

    camera_point = np.asarray(point, dtype=np.float64)
    if camera_point.shape != (3,):
        raise ValueError("point must have shape (3,)")
    if not np.isfinite(camera_point).all() or camera_point[0] <= 0.0:
        return None

    fov_radians = math.radians(horizontal_fov_degrees)
    focal_length = (image_width / 2.0) / math.tan(fov_radians / 2.0)
    center_x = image_width / 2.0
    center_y = image_height / 2.0
    pixel_x = int(focal_length * camera_point[1] / camera_point[0] + center_x)
    pixel_y = int(focal_length * camera_point[2] / camera_point[0] + center_y)

    if 0 <= pixel_x < image_width and 0 <= pixel_y < image_height:
        return pixel_x, pixel_y
    return None


def airsim_lidar_to_kitti_camera(transform):
    """Convert T_camera_from_lidar from AirSim axes to KITTI camera axes."""
    return AIRSIM_TO_KITTI_CAMERA @ _validate_rigid_transform(transform)


def _format_transform_value(value):
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.12g}"


def create_semantic_kitti_calibration(
    transform,
    image_width=1024,
    image_height=1024,
    horizontal_fov_degrees=50.0,
):
    """Create Semantic-KITTI calib.txt content from the same extrinsics."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 < horizontal_fov_degrees < 180.0:
        raise ValueError("horizontal_fov_degrees must be in (0, 180)")

    focal_length = (image_width / 2.0) / math.tan(
        math.radians(horizontal_fov_degrees) / 2.0
    )
    center_x = image_width / 2.0
    center_y = image_height / 2.0
    projection = (
        f"P2: {focal_length:.8f} 0 {center_x:.1f} 0 "
        f"0 {focal_length:.8f} {center_y:.1f} 0 0 0 1 0"
    )

    kitti_transform = airsim_lidar_to_kitti_camera(transform)[:3, :]
    transform_values = " ".join(
        _format_transform_value(value) for value in kitti_transform.reshape(-1)
    )
    return f"{projection}\nTr: {transform_values}\n"
