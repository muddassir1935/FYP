# Weapon Pipeline & Demo (FYP-I)

The Phase 3 pipeline turns the trained weapon model (FR-02) into a running
system: threaded capture (FR-01) -> per-frame detection (FR-02) -> temporal
consistency confirmation (FR-05) -> on-screen alert + snapshot.

## Components

| File | Role |
| --- | --- |
| `src/pipeline/capture.py` | Threaded latest-frame capture for webcam / video file / RTSP (`FrameSource`) |
| `src/pipeline/weapon_detector.py` | YOLOv8 wrapper returning `Detection` objects (`WeaponDetector`) |
| `src/pipeline/temporal_buffer.py` | Deque + IoU tracker; confirms a track seen in >= 6 of the last 10 frames |
| `scripts/inference/run_weapon_demo.py` | Standalone demo runner (display, HUD, snapshots) |
| `tests/test_temporal_buffer.py` | Unit tests for the buffer and IoU (no GPU/deps needed) |

## Setup

```bash
pip install ultralytics opencv-python
python -m unittest discover -s tests -v   # sanity check
```

Weights are never in Git. Download the current model (baseline v1,
`weapon_yolov8s_baseline_20261108_1837_best.pt`, SHA-256 in
`docs/training_run_log.md`) from Google Drive `SENTINEL_FYP/runs/` and save
it as:

```
models/weights/best.pt
```

**Upgrading the model is a file swap**: replace `models/weights/best.pt`
with the newer run's `best.pt` (or pass `--weights <path>`). Class names are
read from the model file itself, so no code changes are needed.

## Run

```bash
python scripts/inference/run_weapon_demo.py                      # webcam 0
python scripts/inference/run_weapon_demo.py --source demo.mp4    # video file
python scripts/inference/run_weapon_demo.py --source demo.mp4 --loop   # file loops instead of exiting at the end
python scripts/inference/run_weapon_demo.py --source rtsp://...  # RTSP stream
python scripts/inference/run_weapon_demo.py --conf 0.35 --hits 6 --window 10
```

Controls: `q` quit, `r` reset the temporal buffer.

- Unconfirmed detections draw as thin green boxes; a **confirmed** threat
  (seen in at least `--hits` of the last `--window` frames) draws as a thick
  red box labelled `CONFIRMED h/window`.
- On confirmation, an annotated snapshot is saved to `snapshots/`
  (5-second cooldown per tracked object) and an `[ALERT]` line is printed.
- `--save-video out.mp4` records the annotated stream; `--no-snapshots`
  disables snapshot saving.

## Faster inference on Intel machines (OpenVINO)

On machines without an NVIDIA GPU (e.g. an Intel Iris Xe laptop), PyTorch
runs the model on CPU only (~5 FPS for YOLOv8s at 640). OpenVINO — Intel's
inference toolkit — converts the *same* model into an Intel-optimised
format that runs on the Intel CPU/iGPU instead. The `.pt` stays the master
model; the OpenVINO folder is a derived copy for Intel hardware.

```bash
pip install openvino
python scripts/training/export_openvino.py     # -> models/weights/best_openvino_model/
python scripts/inference/run_weapon_demo.py --weights models/weights/best_openvino_model --source demo.mp4
```

The export script self-checks the exported model and prints an approximate
blank-frame FPS for the machine it ran on. On recent ultralytics versions,
adding `--device intel:gpu` to the demo selects the integrated GPU
explicitly. RTX/NVIDIA demo machines keep using the `.pt` file — no export
needed there.

## Tuning knobs

- `--conf` — detector confidence threshold. Lower = more sensitive, more
  false alarms for the buffer to filter.
- `--imgsz` — inference image size (default 640). Smaller values (416, 320)
  run much faster on CPU-only machines, at a small accuracy cost.
- `--hits` / `--window` — stricter (e.g. 8/12) = fewer false alerts, slower
  confirmation; looser (e.g. 4/10) = faster alerts, more false positives.

## Notes

- The capture thread always keeps the newest frame only, so a slow detector
  drops stale frames instead of building up lag on live streams.
- Display and inference run on separate threads: video files are paced to
  their native FPS and play smoothly even when detection is slower (e.g.
  CPU-only); the newest detection boxes are overlaid on the live picture.
  The HUD shows both rates (`FPS <display> (detect <inference>)`). The
  project's >= 25 FPS target refers to the *detection* rate and is
  validated on the demo machine (RTX 3060 class), not on CPU.
