# SpaceSense-Bench

Official project page repository for the paper **"SpaceSense-Bench: A Large-Scale Multi-Modal Benchmark for Spacecraft Perception and Pose Estimation"**.

**Status:** Under review at **IROS 2026**.

[Aodi Wu](https://wuaodi.github.io/)<sup>1,2</sup>, Jianhong Zuo<sup>3</sup>, Zeyuan Zhao<sup>1,2</sup>, [Xubo Luo](https://luoxubo.github.io/)<sup>1,2</sup>, Ruisuo Wang<sup>2</sup>, [Xue Wan](https://people.ucas.edu.cn/~wanxue)<sup>2</sup>

<sup>1</sup> University of Chinese Academy of Sciences  
<sup>2</sup> Technology and Engineering Center for Space Utilization, Chinese Academy of Sciences  
<sup>3</sup> Nanjing University of Aeronautics and Astronautics

## Overview

SpaceSense-Bench is a large-scale multi-modal benchmark designed for spacecraft perception and pose estimation in autonomous space operations such as on-orbit servicing and active debris removal.

The benchmark contains:

- **136 satellite models**
- Approximately **70 GB** of data
- Time-synchronized **1024×1024 RGB images**
- **Millimeter-precision depth maps**
- **256-beam LiDAR point clouds**
- Dense **7-class part-level semantic labels** for both pixels and points
- Accurate **6-DoF pose ground truth**

The full dataset is generated with a high-fidelity space simulation environment built in **Unreal Engine 5**, together with an automated pipeline for acquisition, quality control, and data conversion.

## Abstract

Autonomous space operations such as on-orbit servicing and active debris removal demand robust part-level semantic understanding and precise relative navigation of target spacecraft, yet acquiring large-scale real data in orbit remains prohibitively expensive. Existing synthetic datasets, moreover, suffer from limited target diversity, single-modality sensing, and incomplete ground-truth annotations. To bridge these gaps, we present **SpaceSense-Bench**, a large-scale multi-modal benchmark for spacecraft perception encompassing 136 satellite models with approximately 70 GB of data. Each frame provides time-synchronized 1024×1024 RGB images, millimeter-precision depth maps, and 256-beam LiDAR point clouds, together with dense 7-class part-level semantic labels at both the pixel and point level as well as accurate 6-DoF pose ground truth. The dataset is generated through a high-fidelity space simulation built in Unreal Engine 5 and a fully automated pipeline covering data acquisition, multi-stage quality control, and conversion to mainstream formats. Comprehensive benchmarks on object detection, 2D semantic segmentation, RGB-LiDAR fusion 3D point cloud segmentation, monocular depth estimation, and orientation estimation reveal two key findings: (i) perceiving small-scale components such as thrusters and omni-antennas and generalizing to entirely unseen spacecraft in a zero-shot setting remain critical bottlenecks for current methods, and (ii) scaling up the number of training satellites yields substantial performance gains on novel targets, underscoring the value of large-scale, diverse datasets for space perception research.

## Benchmark Tasks

SpaceSense-Bench supports evaluation on multiple spacecraft perception tasks, including:

- Object detection
- 2D semantic segmentation
- RGB-LiDAR fusion 3D point cloud segmentation
- Monocular depth estimation
- Orientation estimation

## Key Findings

1. Small-scale components such as **thrusters** and **omni-antennas** remain difficult for current methods.
2. **Zero-shot generalization** to completely unseen spacecraft is still a major challenge.
3. Increasing the number and diversity of training satellites leads to clear gains on novel targets.

## Repository Structure

This repository currently contains the source code for the project webpage:

```text
.
├── index.html
├── README.md
└── static
		├── css
		├── images
		├── js
		├── pdfs
		└── videos
```

## Website

The homepage is implemented in [index.html](index.html) and uses the assets under [static/](static).

## Release Plan

The repository currently focuses on the project page and paper summary. Additional resources, including links to the paper, benchmark data, and other supporting materials, will be updated after the review process.

## Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{wu2026spacesensebench,
	title={SpaceSense-Bench: A Large-Scale Multi-Modal Benchmark for Spacecraft Perception and Pose Estimation},
	author={Wu, Aodi and Zuo, Jianhong and Zhao, Zeyuan and Luo, Xubo and Wang, Ruisuo and Wan, Xue},
	booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
	year={2026},
	note={Under review},
	url={https://github.com/wuaodi/SpaceSense-Bench}
}
```

## Acknowledgement

This website is based on the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template).
