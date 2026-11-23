# SENTINEL — Capability Matrix

**Comparison of Literature Review Papers vs SENTINEL's Goals**

The SENTINEL column is split in two so the table cannot be misread: **Target**
is what the approved proposal commits to; **Implemented (FYP-I)** is what
actually exists today (23 Nov 2026). ✅ in the Target column means "goal",
never "shipped" — check the Implemented column for the shipped state.
◐ = partially implemented (dataset or scaffold exists, model/feature does not).

| Capability | Bonde et al. (2025) | Shanthi & Manjula (2026) | Praveen & Sandeep (2025) | Elnady & Abdelmunim (2025) | Reddy et al. (2026) | SENTINEL Target | SENTINEL Implemented (FYP-I) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Weapon Detection** | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Face Recognition** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Optical Flow / Motion Vectors** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Pose-Based Skeleton Extraction** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ◐ |
| **Multi-Class Crime Classification** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| **Temporal Sequence Modeling (LSTM)** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| **False-Alarm Suppression** | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Multi-Channel Alerts (Email/Siren)** | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ◐ |
| **Real-Time Dashboard** | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ◐ |
| **Lightweight Model (<50 MB)** | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ |
| **Real-Time (≥25 FPS)** | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |

**Implemented-column notes (23 Nov 2026):** Weapon detection (YOLOv8s v3,
adopted) and the temporal-consistency buffer (FR-05) are live; ≥25 FPS is
measured on the demo laptop (Intel Iris Xe via OpenVINO, 39 FPS inference /
~30 FPS end-to-end) — see `docs/pipeline_weapon_demo.md`. Pose extraction
is ◐ because the pose-sequence dataset (3,579 sequences) exists but the
YOLOv8m-Pose + LSTM behavior stream (FR-03/FR-04) is not built — FYP-II
scope. Multi-channel alerts is ◐ because the demo UI has an audible siren
+ visual banner, but email alerts (FR-09) are not built — FYP-II scope.
Real-time dashboard is ◐ because the FYP-I demo is a vanilla single-page
console (`docs/demo_ui_plan.md`, shipped as PR #35), not the React + Vite
dashboard of the proposal — FYP-II scope.

---

### Legend & References

| Authors (Year) | Full Title | Core Focus |
|---|---|---|
| **Bonde et al. (2025)** | GuardVision: Enhanced Weapon Detection with Alert Mechanism | Weapon Detection + Alerts |
| **Shanthi & Manjula (2026)** | ConViDeTR: Hybrid CNN–Transformer for Weapon Detection & Face Recognition | Multi-task CNN-Transformer |
| **Praveen & Sandeep (2025)** | Anomaly Detection in Video Surveillance Using YOLOv8 | Optical Flow + Temporal Buffer |
| **Elnady & Abdelmunim (2025)** | A Novel YOLO LSTM Approach for Enhanced Human Action Recognition | Lightweight LSTM Sequence Modeling |
| **Reddy et al. (2026)** | Intelligent Weapon Detection System for Real Time Surveillance Using YOLOv8 | Combined Weapon & Anomaly (Closest to SENTINEL) |
