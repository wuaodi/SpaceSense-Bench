"""Unit tests for the shared SpaceSense sensor geometry."""

import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONVERT_DIRECTORY = REPOSITORY_ROOT / "SpaceSense-Toolkit" / "convert"
sys.path.insert(0, str(CONVERT_DIRECTORY))

from sensor_geometry import (  # noqa: E402
    DEFAULT_CALIBRATION_PATH,
    airsim_lidar_to_kitti_camera,
    create_semantic_kitti_calibration,
    load_calibration,
    project_point_to_image,
    transform_points,
)


def load_conversion_script(filename, module_name, cv2_module=None):
    """Import a conversion script without requiring optional UI dependencies."""
    fake_cv2 = cv2_module or types.ModuleType("cv2")
    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda iterable, **_kwargs: iterable
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_pyplot = types.ModuleType("matplotlib.pyplot")
    fake_matplotlib.pyplot = fake_pyplot

    script_path = CONVERT_DIRECTORY / filename
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "cv2": fake_cv2,
            "tqdm": fake_tqdm,
            "matplotlib": fake_matplotlib,
            "matplotlib.pyplot": fake_pyplot,
        },
    ):
        spec.loader.exec_module(module)
    return module


class SensorGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.transform, cls.metadata = load_calibration(DEFAULT_CALIBRATION_PATH)

    def test_legacy_profile_is_explicit_and_versioned(self):
        self.assertEqual(self.metadata["schema_version"], 1)
        self.assertEqual(
            self.metadata["calibration_id"], "spacesense_legacy_raw_v1"
        )
        np.testing.assert_allclose(self.transform[:3, 3], [0.9, 0.0, 0.0])

    def test_transform_single_point_and_batch(self):
        transformed_point = transform_points([10.0, 0.0, 0.0], self.transform)
        np.testing.assert_allclose(transformed_point, [10.9, 0.0, 0.0])

        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        transformed_points = transform_points(points, self.transform)
        np.testing.assert_allclose(
            transformed_points,
            [[1.9, 2.0, 3.0], [4.9, 5.0, 6.0]],
        )

    def test_projection_uses_transformed_forward_coordinate(self):
        raw_point = np.array([1.0, 0.1, 0.0])
        camera_point = transform_points(raw_point, self.transform)
        projected = project_point_to_image(camera_point, 1024, 1024)

        focal_length = 512.0 / math.tan(math.radians(50.0) / 2.0)
        expected_u = int(focal_length * 0.1 / 1.9 + 512.0)
        self.assertEqual(projected, (expected_u, 512))

        identity_projection = project_point_to_image(raw_point, 1024, 1024)
        self.assertNotEqual(projected, identity_projection)

    def test_point_behind_camera_is_rejected_after_transform(self):
        camera_point = transform_points([-1.0, 0.0, 0.0], self.transform)
        self.assertIsNone(project_point_to_image(camera_point, 1024, 1024))

    def test_semantic_kitti_transform_has_axis_mapping_and_translation(self):
        kitti_transform = airsim_lidar_to_kitti_camera(self.transform)
        expected = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.9],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(kitti_transform, expected)

        calibration_text = create_semantic_kitti_calibration(self.transform)
        self.assertIn("Tr: 0 1 0 0 0 0 1 0 1 0 0 0.9", calibration_text)

    def test_both_scripts_use_the_shared_transform(self):
        converter = load_conversion_script(
            "airsim_to_semantickitti.py", "test_airsim_to_semantickitti"
        )
        visualizer = load_conversion_script(
            "project_lidar2img.py", "test_project_lidar2img"
        )

        for script in (converter, visualizer):
            transformed = script.transform_lidar_to_camera_frame(
                [10.0, 0.0, 0.0], self.transform
            )
            np.testing.assert_allclose(transformed, [10.9, 0.0, 0.0])

        self.assertIn(
            "Tr: 0 1 0 0 0 0 1 0 1 0 0 0.9",
            converter.create_default_calib(self.transform),
        )

    def test_semantic_kitti_conversion_keeps_bin_in_lidar_frame(self):
        raw_point = np.array([1.0, 0.1, 0.0], dtype=np.float32)
        camera_point = transform_points(raw_point, self.transform)
        pixel_x, pixel_y = project_point_to_image(camera_point, 1024, 1024)

        segmentation = np.zeros((1024, 1024, 3), dtype=np.uint8)
        # OpenCV reads BGR; this is RGB (156, 198, 23), a class-1 color.
        segmentation[pixel_y, pixel_x] = [23, 198, 156]
        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.imread = lambda _path: segmentation
        converter = load_conversion_script(
            "airsim_to_semantickitti.py",
            "test_airsim_to_semantickitti_conversion",
            cv2_module=fake_cv2,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pointcloud_directory = root / "lidar"
            segmentation_directory = root / "seg"
            image_directory = root / "image"
            output_bin_directory = root / "velodyne"
            output_label_directory = root / "labels"
            output_image_directory = root / "image_2"
            for directory in (
                pointcloud_directory,
                segmentation_directory,
                image_directory,
            ):
                directory.mkdir()

            frame_id = "frame"
            (pointcloud_directory / f"{frame_id}.asc").write_text(
                "1.0,0.1,0.0\n", encoding="utf-8"
            )
            (segmentation_directory / f"{frame_id}.png").touch()
            (image_directory / f"{frame_id}.png").touch()

            converter.convert_trajectory_to_kitti(
                str(pointcloud_directory),
                str(segmentation_directory),
                str(image_directory),
                str(output_bin_directory),
                str(output_label_directory),
                str(output_image_directory),
                lidar_to_camera=self.transform,
            )

            stored_point = np.fromfile(
                output_bin_directory / f"{frame_id}.bin", dtype=np.float32
            ).reshape(-1, 4)
            stored_label = np.fromfile(
                output_label_directory / f"{frame_id}.label", dtype=np.uint32
            )
            np.testing.assert_allclose(stored_point[0, :3], raw_point)
            self.assertEqual(stored_point[0, 3], 0.0)
            np.testing.assert_array_equal(stored_label, [1])

    def test_invalid_non_rigid_transform_is_rejected(self):
        invalid_document = {
            "schema_version": 1,
            "units": "metres",
            "T_camera_from_lidar": [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            calibration_path = Path(temporary_directory) / "invalid.json"
            calibration_path.write_text(
                json.dumps(invalid_document), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "orthonormal"):
                load_calibration(calibration_path)


if __name__ == "__main__":
    unittest.main()
