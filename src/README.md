# 💻 SENTINEL — Production Source Code

This directory contains the production components of the SENTINEL system.

## 📁 Architecture Overview

```
src/
├── backend/            # FastAPI demo console server (app.py, runner.py, schemas.py)
│   ├── main.py         # [FYP-II] Application entry point
│   ├── database.py     # [FYP-II] SQLite + SQLAlchemy models & migrations
│   └── routers/        # [FYP-II] API endpoints (incidents, health, streaming)
│
├── frontend/           # Vanilla HTML/CSS/JS console (app.js, index.html, styles.css)
│   ├── src/            # [FYP-II] React + Vite: components, hooks, Canvas renderers
│   └── index.html      # SPA root with Canvas mount point
│
└── pipeline/           # Shared AI & Video processing engine
    ├── capture.py      # Threaded OpenCV RTSP / Webcam video capture
    ├── weapon_detector.py # YOLOv8s inference wrapper
    ├── pose_extractor.py  # [FYP-II] YOLOv8m-Pose feature extractor
    ├── action_classifier.py # [FYP-II] LSTM temporal classifier
    └── temporal_buffer.py # Deque-based IoU sliding window false-alarm filter
```
