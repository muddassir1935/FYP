# 🤖 AI Agent Guidelines & System Instructions — SENTINEL

This document serves as the direct operational guide for any AI coding assistant (Cursor, Claude, Copilot, Antigravity, Roo Code, etc.) working in this repository.

---

## 1. 🌿 Mandatory Git Branching Protocol ("NEVER Work on Main")

Every AI agent must follow this strict branch lifecycle before generating or modifying code:

1. **Check Current Branch First**: Run `git branch --show-current`.
2. **Never Commit to Main**: The `main` branch is reserved strictly for tested, integrated releases.
3. **Declare & Create Feature Branch**:
   * Before writing or editing code for a new feature, dataset script, model training, backend endpoint, or UI component, announce:
     > *"Creating and switching to branch `feature/<feature-name>` so that `main` remains protected."*
   * Run: `git checkout -b feature/<feature-name>` (or `fix/<bug-name>`, `docs/<doc-name>`).
4. **Test Before Merging**: Run tests or verification scripts locally.
5. **Merge Back Cleanly**: Once verified, merge the branch into `main` and push.

---

## 2. ✍️ Professional Conventional Commit Standard (Zero AI Fluff)

Commit messages must look like they were authored by a senior software engineer. **Generic, rambling, or robotic AI messages are strictly prohibited.**

### The Format:
```
<type>(<scope>): <imperative summary in lowercase, max 72 chars>

[optional body: concise bullet points explaining why and what technically]
```

### Types:
`feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`

### Allowed Scopes:
`(data)`, `(models)`, `(pipeline)`, `(backend)`, `(ui)`, `(buffer)`, `(docs)`, `(workflow)`

### Bad vs Good Examples:
* ❌ *"I have updated the backend files according to instructions to add websocket support"*
  👉 ✅ `feat(backend): add WebSocket live frame streaming endpoint`
* ❌ *"Fixed bug where bounding box was disappearing"*
  👉 ✅ `fix(canvas): prevent overlay clearing during frame transition`
* ❌ *"Updated project context and proposal synchronization as requested"*
  👉 ✅ `docs(spec): sync FR-01 through FR-10 with approved proposal`
* ❌ *"Added some dataset cleaning scripts"*
  👉 ✅ `feat(data): add negative background sample generator for weapon dataset`

---

## 3. 📖 Mandatory Context Sync: Approved FYP-I Proposal
Before recommending architecture changes or writing code, read [`PROJECT_CONTEXT.md`](./PROJECT_CONTEXT.md).  
All implementations must strictly match the approved **FYP-I Project Proposal** ([`FYP-I-Project-Proposal.docx`](./Literature%20Review/Documentation/Proposal/FYP-I-Project-Proposal.docx)).

### Key Requirements (Approved Proposal):
* **FR-01**: Ingest real-time video (OpenCV RTSP/Webcam).
* **FR-02**: Detect & localize Guns and Knives with bounding boxes and confidence scores (YOLOv8s).
* **FR-03**: Extract human skeletal keypoints (17 joints / 51-dim) for each person (YOLOv8m-Pose).
* **FR-04**: Classify behavior into 4 classes: `Fighting`, `Robbery`, `Shooting`, and `Normal Activity` (2-Layer LSTM).
* **FR-05**: Apply pure sliding-window temporal consistency filtering (deque buffer, IoU tracking).
* **FR-06**: Display live video with real-time bounding box & skeleton overlays on web dashboard (HTML5 Canvas).
* **FR-07**: Trigger audible alarm and visual alert banner upon confirmed threat.
* **FR-08**: Log confirmed incidents to database with timestamp, snapshot image, threat class, and confidence (SQLite).
* **FR-09**: Dispatch automated email alerts with incident snapshot attachments (Python smtplib).
* **FR-10**: Provide searchable and filterable incident history view.
* **NFR-01 & NFR-02**: Sustain **≥ 25 FPS** throughput on consumer-grade GPUs (e.g., RTX 3060).

---

## 4. ⛔ Strict "Do NOT" Constraints

* ❌ **DO NOT implement Face Recognition / Person Re-ID**: Out of scope due to CCTV resolution, privacy, and edge performance limits.
* ❌ **DO NOT use 3D-CNNs (I3D, SlowFast, C3D)**: Action recognition uses **YOLOv8m-Pose + 2-layer LSTM** only.
* ❌ **DO NOT commit dataset files or model weights**: Raw videos (`.mp4`, `.avi`), images (`.jpg`, `.png`), extracted keypoints (`.npy`), and weight binaries (`.pt`, `.pth`, `.onnx`) must NEVER be committed to Git.
* ❌ **DO NOT use TailwindCSS**: The frontend dashboard uses **Vanilla CSS** with custom dark-mode styling and glassmorphism.
* ❌ **DO NOT block the FastAPI event loop**: Heavy inference loops must run in worker threads (`run_in_executor`) to preserve ≥25 FPS WebSocket streaming.

---

## 5. 🛡️ Defensive Engineering Directives

* **No Hardcoded Absolute Paths**: Always use `pathlib.Path(__file__).resolve().parent` relative paths.
* **Device Portability**: Auto-detect `cuda` vs `cpu`:
  ```python
  import torch
  device = "cuda" if torch.cuda.is_available() else "cpu"
  ```
* **Robust Stream Reconnection**: Handle CCTV stream drops with automatic reconnection rather than crashing.
* **Type Annotations**: Always include Python type annotations (`typing.List`, `typing.Dict`, `typing.Optional`, etc.).
