# 🤖 AI Agent Guidelines & System Instructions — SENTINEL

This document serves as the direct operational guide for any AI agent (Cursor, Claude, Copilot, Antigravity, Roo Code, etc.) assisting developers in this repository.

---

## 1. 📖 First Mandatory Step
Before generating architecture changes, installing packages, or creating new components, **read [`PROJECT_CONTEXT.md`](./PROJECT_CONTEXT.md)**.
It contains the authorized decisions, model choices, dataset architectures, and scope boundaries for this project.

---

## 2. ⛔ Strict "Do NOT" Constraints

* ❌ **DO NOT implement Face Recognition / Person Re-ID**: Out of scope due to CCTV resolution and edge performance constraints.
* ❌ **DO NOT use 3D-CNNs (I3D, SlowFast, C3D)**: Action recognition uses **YOLOv8m-Pose + 2-layer LSTM** only.
* ❌ **DO NOT commit dataset files or model weights**: Raw videos (`.mp4`, `.avi`), images (`.jpg`, `.png`), extracted keypoints (`.npy`), and weight binaries (`.pt`, `.pth`, `.onnx`) must NEVER be committed to Git.
* ❌ **DO NOT use TailwindCSS**: The frontend dashboard uses **Vanilla CSS** with custom dark-mode styling and glassmorphism.
* ❌ **DO NOT block the FastAPI event loop**: Heavy inference loops must run in worker threads or dedicated async generator pipelines to preserve ≥25 FPS WebSocket streaming.

---

## 3. 🎯 Key Tech Stack & Conventions

* **Language (Backend / ML)**: Python 3.10+
* **ML / DL Framework**: PyTorch 2.x, Ultralytics (`ultralytics`), Scikit-learn
* **Backend**: FastAPI, Uvicorn, OpenCV (`opencv-python`), WebSockets
* **Database**: SQLite with SQLAlchemy ORM
* **Frontend**: React 18+ with Vite
* **Frontend Rendering**: HTML5 Canvas (`<canvas>`) for GPU-accelerated 2D bounding boxes and skeleton overlays.

---

## 4. 🗂️ Target Classes

* **Weapons (2 Classes)**:
  * `0: Gun` (Handguns, Pistols, Rifles)
  * `1: Knife` (Knives, Daggers, Blades)
* **Behaviors (4 Classes)**:
  * `0: Normal Activity`
  * `1: Fighting`
  * `2: Robbery`
  * `3: Shooting`

---

## 5. 📂 Directory Conventions

* Place dataset preparation scripts in [`scripts/data_preparation/`](./scripts/data_preparation/).
* Place model training scripts in [`scripts/training/`](./scripts/training/).
* Place backend code in [`src/backend/`](./src/backend/).
* Place frontend code in [`src/frontend/`](./src/frontend/).
* Place shared AI pipeline code (video ingestion, buffer, inference runner) in [`src/pipeline/`](./src/pipeline/).
* Detailed technical docs belong in [`docs/`](./docs/).

---

## 6. 💡 Code Quality & Git Directives

* Always include Python type annotations (`typing.List`, `typing.Dict`, `typing.Optional`, etc.).
* Keep functions modular, single-responsibility, and testable.
* Write informative git commit messages following Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`).
