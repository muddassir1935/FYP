# 🧠 SENTINEL — Model Checkpoints & Weights

> **IMPORTANT**: Heavy deep learning weight files (`.pt`, `.pth`, `.onnx`, `.engine`) are **ignored by Git** via `.gitignore`. Store active weights locally inside `models/weights/`.

## 📁 Weights Directory
```
models/
├── README.md           # This guide
└── weights/            # Local model weights directory (gitignored)
    ├── yolov8s_weapon.pt       # Fine-tuned YOLOv8s weapon detection model
    ├── yolov8m-pose.pt         # Pretrained Ultralytics YOLOv8m pose checkpoint
    └── lstm_crime_classifier.pth # Trained 2-layer LSTM temporal action classifier
```

## ⚙️ Model Architecture Specifications

1. **Spatial Stream (Weapon Detection)**:
   * Architecture: `Ultralytics YOLOv8s`
   * Classes: `0: Gun`, `1: Knife`
   * Input Size: 640x640 RGB image

2. **Temporal Stream (Pose Extraction)**:
   * Architecture: `Ultralytics YOLOv8m-Pose`
   * Output: 17 Keypoints (51 coordinates: X, Y, Confidence score) per detected human

3. **Temporal Stream (Action Recognition)**:
   * Architecture: `2-Layer LSTM with Dropout`
   * Input Tensor Shape: `(batch_size, 30, 51)` — 30 temporal frames, 51 joint coordinates
   * Output: 4 Logits / Softmax probabilities (`Normal`, `Fighting`, `Robbery`, `Shooting`)
