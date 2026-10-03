# 💻 SENTINEL — Production Source Code

This directory contains the production components of the SENTINEL system.

## 📁 Architecture Overview

```
src/
├── backend/            # FastAPI asynchronous web server & WebSocket broadcaster
│   ├── main.py         # Application entry point
│   ├── database.py     # SQLite + SQLAlchemy models & migrations
│   └── routers/        # API endpoints (incidents, health, streaming)
│
├── frontend/           # React + Vite surveillance dashboard
│   ├── src/            # Components, hooks, Canvas renderers, state management
│   └── index.html      # SPA root with Canvas mount point
│
└── pipeline/           # Shared AI & Video processing engine
    ├── capture.py      # Threaded OpenCV RTSP / Webcam video capture
    ├── weapon_detector.py # YOLOv8s inference wrapper
    ├── pose_extractor.py  # YOLOv8m-Pose feature extractor
    ├── action_classifier.py # LSTM temporal classifier
    └── temporal_buffer.py # Deque-based IoU sliding window false-alarm filter
```
