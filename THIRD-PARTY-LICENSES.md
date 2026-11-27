# Third-Party Licenses

SENTINEL is built on open-source software. This file lists the main libraries
we use, their licenses, and what each one does in the project. All licenses
were checked against the installed packages (November 2026).

| Library | Version | License | Used for |
| --- | --- | --- | --- |
| Ultralytics YOLOv8 (`ultralytics`) | 8.4.208 | AGPL-3.0 | Weapon detection (YOLOv8s) and pose estimation (YOLOv8m-Pose) |
| PyTorch (`torch`, `torchvision`) | 2.15.0 | Apache-2.0 | Deep-learning framework, LSTM behaviour classifier training |
| OpenCV (`opencv-python`) | 5.0.0 | Apache-2.0 | Video capture, frame handling, image processing |
| OpenVINO (`openvino`) | 2026.4.1 | Apache-2.0 | Model inference acceleration on Intel hardware |
| NumPy (`numpy`) | 2.4.6 | BSD-3-Clause | Numerical arrays, frame and keypoint maths |
| FastAPI (`fastapi`) | 0.115.6 | MIT | Demo web UI backend (REST endpoints) |
| Uvicorn (`uvicorn`) | 0.54.0 | BSD-3-Clause | ASGI server for the demo web UI |
| python-multipart (`python-multipart`) | 0.0.34 | Apache-2.0 | Multipart form parsing for video uploads |
| HTTPX (`httpx`) | 0.28.1 | BSD-3-Clause | HTTP client for the UI smoke test |

## Note on YOLOv8 (AGPL-3.0)

YOLOv8 is the only copyleft dependency in the project. Under AGPL-3.0, anyone
who runs a modified version over a network must make the source code available.
For this academic project that is not a problem: our code is already public.
If the project were ever commercialised, we would either buy a commercial
licence from Ultralytics or replace YOLOv8 with a permissively licensed
detector.

The remaining libraries use permissive licences (Apache-2.0, BSD-3-Clause),
which allow commercial use as long as their copyright notices are kept.
