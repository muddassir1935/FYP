# AMD Radeon RX 580 Hardware Optimization (DirectML)

## 1. Overview
The **AMD Radeon RX 580** (Polaris architecture) does not support NVIDIA CUDA, and native AMD ROCm on Windows is limited to RDNA architectures. To unlock the full acceleration capability of the RX 580 under Windows for SENTINEL, we leverage **DirectML** (DirectX 12 Compute) through **ONNX Runtime** (`onnxruntime-directml`).

## 2. Benchmark Results
Benchmark on the local system (**Intel Core i5-10600 + AMD Radeon RX 580 8GB**):

| Execution Backend | Hardware | Inference Speed | Latency | Real-time Target (NFR-01 $\ge$ 25 FPS) |
|---|---|---|---|---|
| PyTorch (`.pt`) | CPU (6 cores, 12 threads) | **9.72 FPS** | ~102.8 ms | ❌ FAILED (< 25 FPS) |
| ONNX Runtime CPU | CPU | **8.44 FPS** | ~118.5 ms | ❌ FAILED (< 25 FPS) |
| **DirectML (`.onnx`)** | **AMD Radeon RX 580 (DirectX 12)** | **80.29 FPS** (standalone) / **65.27 FPS** (end-to-end YOLO) | **~12.5 ms** | ✅ **PASSED** (~2.6x to 3.2x requirement, **~8x speedup over CPU**) |

## 3. How It Works
1. **Model Format**: The fine-tuned PyTorch model (`weapon_yolov8s_v3_20261120_2215_best.pt`) is converted into an ONNX representation via `scripts/training/export_directml.py`.
2. **Runtime Execution**: When an `.onnx` weight file is selected in SENTINEL:
   - `src/pipeline/weapon_detector.py` automatically injects `DmlExecutionProvider` as the primary execution provider.
   - Computations are offloaded directly to the AMD RX 580's GPU compute units via DirectML DirectX 12 drivers.
   - If DirectML is not available on a machine, it gracefully falls back to `CPUExecutionProvider`.
3. **OpenCV OpenCL**: OpenCV image preprocessing automatically utilizes OpenCL (`cv2.ocl.useOpenCL() == True`), ensuring color conversion and frame manipulation also leverage the GPU.
4. **Web UI Support**: The Demo Web Console automatically discovers `.onnx` weights and displays them as `(DirectML / AMD GPU)`.

## 4. Usage
Export any `.pt` checkpoint to DirectML ONNX:
```bash
python scripts/training/export_directml.py --weights models/weights/weapon_yolov8s_v3_20261120_2215_best.pt
```

Start the demo UI and select the DirectML weights from the dropdown:
```bash
python scripts/run_demo_ui.py
```
