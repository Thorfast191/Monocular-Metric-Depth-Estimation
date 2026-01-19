# Monocular Metric Depth Estimation on KITTI

This repository is a journal extension of our IEEE STI 2024 conference paper.
The work upgrades monocular depth estimation to **monocular metric depth estimation (MMDE)** using the KITTI dataset.

## Key Features
- Metric depth prediction (meters)
- KITTI Eigen protocol
- Deterministic 20% dataset subset
- Modular PyTorch implementation
- Full experiment logging (CSV, JSON, TXT)

## Dataset
KITTI Depth (Eigen split)

Depth scaling:
## Training
Open `main_train.ipynb` and run all cells.

## Logging
Each experiment logs:
- metrics.json
- log.csv
- description.txt

## Citation
If you use this work, please cite our IEEE STI 2024 paper.