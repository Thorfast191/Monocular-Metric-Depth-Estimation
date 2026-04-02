# 📏 MMDE: Monocular Metric Depth Estimation (NYU Depth V2)

![Status](https://img.shields.io/badge/status-research--prototype-orange)
![Framework](https://img.shields.io/badge/framework-PyTorch-blue)
![Model](https://img.shields.io/badge/model-DPT--Hybrid-purple)
![Task](https://img.shields.io/badge/task-Metric--Depth-green)
![Dataset](https://img.shields.io/badge/dataset-NYU--Depth--V2-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 🧾 Abstract

Monocular depth estimation traditionally predicts relative depth, lacking physical interpretability.  
This project explores Monocular Metric Depth Estimation (MMDE) using the NYU Depth V2 dataset.

---

## 🚀 Introduction

Recovering 3D structure from a single image is inherently ill-posed due to scale ambiguity.

---

## 🧠 Model

- DPT (ViT Hybrid)
- Multi-scale fusion
- Dense depth output

---

## 📊 Dataset

NYU Depth V2  
Indoor RGB-D dataset (0.5m – 10m)

---

## 🔧 Preprocessing

- Depth normalized to [0,1]
- Invalid masking
- ImageNet normalization

---

## ⚙️ Training

Run:
main_train.ipynb

---

## 📈 Metrics

- RMSE
- AbsRel
- δ1, δ2, δ3

---

## 📌 Status

Research Prototype

---

## 📄 License

MIT
