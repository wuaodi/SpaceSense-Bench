# LiDAR-camera calibration

The raw SpaceSense tools use AirSim axes: `+X` forward, `+Y` right, and `+Z`
down. Calibration files store a homogeneous transform in this direction:

```text
p_camera = T_camera_from_lidar @ p_lidar
```

For the published legacy raw-data release discussed in
[issue #8](https://github.com/wuaodi/SpaceSense-Bench/issues/8), the checked-in
`legacy_lidar_camera_calibration_v1.json` profile applies a `+0.9 m`
translation along AirSim `+X`. This profile is explicit and versioned; it
should not be assumed to describe a different sensor layout or a future data
export.

Both conversion paths now use the same profile:

- `project_lidar2img.py` transforms each XYZ point before image projection.
- `airsim_to_semantickitti.py` uses the transformed point only to sample the
  pixel label. The output `.bin` remains in the original LiDAR frame, while
  `calib.txt` stores the corresponding LiDAR-to-KITTI-camera `Tr` matrix.

Use the default profile:

```bash
python SpaceSense-Toolkit/convert/project_lidar2img.py \
  --img-path /path/to/image --lidar-path /path/to/lidar

python SpaceSense-Toolkit/convert/airsim_to_semantickitti.py \
  --raw-data /path/to/raw_data --output /path/to/output
```

For another sensor layout, copy the JSON schema, provide its measured
`T_camera_from_lidar`, and pass it explicitly:

```bash
python SpaceSense-Toolkit/convert/project_lidar2img.py \
  --img-path /path/to/image --lidar-path /path/to/lidar \
  --calibration /path/to/calibration.json
```

The transform must be applied to XYZ coordinates before projection. Do not add
the translation directly to a scalar depth or LiDAR range.
