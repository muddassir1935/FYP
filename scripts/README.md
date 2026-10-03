# 🛠️ SENTINEL — Automation & Processing Scripts

This directory houses automated pipelines and standalone scripts for dataset curation, model training, and performance benchmarking.

## 📁 Subdirectories

* `data_preparation/`:
  * Scripts for downloading, cleaning, unzipping, filtering, and standardizing annotations into YOLO format.
  * Offline pose keypoint extraction scripts running YOLOv8m-Pose on video clips to output `(30, 51)` numpy sequences.
* `training/`:
  * Training and evaluation scripts for YOLOv8s weapon detection.
  * PyTorch LSTM training routines, cross-validation, and loss/accuracy plotting.
* `inference/`:
  * Standalone FPS benchmark runners.
  * Real-time webcam or video file verification testers.
