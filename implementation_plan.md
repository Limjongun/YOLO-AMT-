# AMT-YOLO: Adaptive Memory Trajectory YOLO
## Research Development Implementation Plan
## Status: APPROVED — 2026-06-27

---

## Keputusan yang Dikonfirmasi

| Keputusan | Pilihan |
|---|---|
| **Target Hardware** | RTX 4050 — 6GB VRAM (⚠️ perlu gradient checkpointing + konfigurasi khusus) |
| **Backbone YOLO** | **YOLOv8** (lebih banyak referensi komunitas) |
| **Dataset Awal** | COCO dulu → tambahkan MOT17 + VisDrone di Phase 3 |
| **Target Publikasi** | **CVPR** (ekspektasi) |
| **Temporal Memory** | **Benchmark keduanya**: ConvGRU + ConvLSTM |
| **Trajectory Horizon** | **5–10 frame** ke depan |

> ⚠️ RTX 4050 6GB VRAM Notes:
> - Gunakan Automatic Mixed Precision (AMP/FP16) wajib
> - Batch size kecil (4–8), gunakan gradient accumulation steps=4
> - Gradient checkpointing untuk training temporal sequence
> - Hindari resolusi 1024x1024 saat training (gunakan 640/768 max)
> - Pertimbangkan frozen backbone layers saat early training

---

## Project Structure

```
D:\YOLO next\
├── docs/
│   ├── AMT-YOLO_Architecture_and_Tech_Stack.txt  (existing)
│   ├── AMT-YOLO_Reasoning_Document.txt           (existing)
│   ├── implementation_plan.md                     (this file)
│   ├── paper_draft.md                             [Phase 7]
│   └── experiment_log.md                          [Phase 7]
│
├── amt_yolo/                      # Core package
│   ├── __init__.py
│   ├── models/
│   │   ├── amt_yolo.py            # Main AMT-YOLO model (base: YOLOv8)
│   │   ├── adaptive_resolution.py # Adaptive Resolution Router
│   │   ├── temporal_memory.py     # ConvGRU + ConvLSTM (benchmark both)
│   │   ├── trajectory_head.py     # Trajectory Head (5-10 frame horizon)
│   │   └── feature_fusion.py     # Adaptive Feature Fusion Neck
│   │
│   ├── tracking/
│   │   ├── tracker.py             # ByteTrack / BoT-SORT wrapper
│   │   └── embedding.py
│   │
│   ├── data/
│   │   ├── dataset.py             # COCO, MOT17, VisDrone loaders
│   │   ├── augmentation.py        # Temporal-aware augmentations
│   │   └── trajectory_labels.py   # Trajectory GT parser
│   │
│   ├── losses/
│   │   ├── detection_loss.py      # IoU + DFL + Classification
│   │   └── trajectory_loss.py     # Smooth L1 + MSE + ADE/FDE
│   │
│   └── utils/
│       ├── metrics.py             # mAP, HOTA, MOTA, IDF1, ADE, FDE
│       ├── visualizer.py
│       └── profiler.py
│
├── configs/
│   ├── amt_yolo_base.yaml
│   ├── amt_yolo_small.yaml        # Versi hemat VRAM untuk RTX 4050
│   └── ablation/
│       ├── no_memory.yaml
│       ├── no_trajectory.yaml
│       ├── fixed_resolution.yaml
│       ├── convgru_only.yaml
│       └── convlstm_only.yaml
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── demo.py
│   └── export.py
│
├── experiments/
│   └── logs/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_module_testing.ipynb
│   └── 03_ablation_analysis.ipynb
│
├── tests/
│   ├── test_adaptive_resolution.py
│   ├── test_temporal_memory.py
│   └── test_trajectory_head.py
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Phase 1 — Foundation & Environment Setup (1–2 minggu)

- [ ] Setup virtual environment (conda/venv)
- [ ] Install semua dependensi PyTorch + CUDA 12.x
- [ ] Verifikasi CUDA di RTX 4050
- [ ] Buat base package amt_yolo/__init__.py
- [ ] requirements.txt (dikonfigurasi untuk 6GB VRAM)
- [ ] setup.py
- [ ] README.md

---

## Phase 2 — Core Module Development (3–5 minggu)

### 2.1 Adaptive Resolution Router
- Complexity estimator: edge-density heuristic (lightweight)
- Output resolutions: 640 / 768 (training max untuk RTX 4050)

### 2.2 Temporal Memory Module (BENCHMARK BOTH)
- ConvGRU: lebih cepat, cocok untuk 6GB VRAM
- ConvLSTM: lebih expressive, lebih berat
- Gradient checkpointing enabled

### 2.3 Trajectory Prediction Head
- Horizon: 5–10 frame
- Architecture: GRU + MLP
- Loss: Smooth L1 + MSE + ADE/FDE

### 2.4 AMT-YOLO Main Model
- Base: YOLOv8n atau YOLOv8s (ringan untuk 6GB VRAM)

---

## Phase 3 — Data Pipeline (1–2 minggu)

- COCO Detection (pre-training)
- MOT17 / MOT20 (tracking + temporal)
- VisDrone (small object, aerial)
- BDD100K (opsional)

---

## Phase 4 — Loss Functions & Training (2–3 minggu)

### RTX 4050 6GB Training Config:
```yaml
batch_size: 4
accumulate_grad: 4     # effective batch = 16
amp: true              # FP16 wajib
gradient_checkpoint: true
max_resolution_train: 768
workers: 4
```

### Multi-stage Training:
1. Stage 1: Pre-train detection backbone pada COCO
2. Stage 2: Fine-tune dengan Temporal Memory pada MOT17
3. Stage 3: Full AMT-YOLO training

---

## Phase 5 — Evaluation & Ablation Study (2–3 minggu)

### Ablation Matrix:
| Config | Adaptive Res | Memory | Trajectory | Memory Type |
|---|---|---|---|---|
| Baseline YOLOv8 | ✗ | ✗ | ✗ | — |
| + Memory (ConvGRU) | ✗ | ✓ | ✗ | ConvGRU |
| + Memory (ConvLSTM) | ✗ | ✓ | ✗ | ConvLSTM |
| + Trajectory | ✗ | ✗ | ✓ | — |
| AMT-YOLO (no AdapRes) | ✗ | ✓ | ✓ | Best |
| AMT-YOLO Full | ✓ | ✓ | ✓ | Best |

### Metrics:
| Kategori | Metrik |
|---|---|
| Detection | mAP@50, mAP@50-95 |
| Tracking | HOTA, MOTA, IDF1 |
| Trajectory | ADE, FDE |
| Efficiency | FPS, Latency (ms), GFLOPs, Params |

---

## Phase 6 — Demo & Deployment (1–2 minggu)

- Real-time demo (webcam + video file)
- ONNX export
- TensorRT export
- Visualisasi trajectory overlay

---

## Phase 7 — Paper Writing / CVPR Target (3–4 minggu, paralel)

### CVPR Baselines:
- YOLOv8 (detection baseline)
- ByteTrack + YOLOv8 (tracking baseline)
- StrongSORT / BoT-SORT
- MotionBERT / Social-STGCNN (trajectory baseline)

---

## Tech Stack Summary

| Komponen | Teknologi |
|---|---|
| Language | Python 3.12+ |
| Deep Learning | PyTorch 2.x + CUDA 12.x |
| Base Model | YOLOv8 (Ultralytics) |
| Memory Module | ConvGRU + ConvLSTM (benchmark both) |
| Trajectory | GRU + MLP (horizon 5–10 frame) |
| Tracking | ByteTrack, BoT-SORT |
| Dataset | COCO, MOT17, MOT20, VisDrone |
| Logging | Weights & Biases (WandB) |
| Export | ONNX, TensorRT |
| Hardware | RTX 4050 6GB VRAM |
| Target | CVPR |

---

> Estimasi total: 3–4 bulan untuk full research cycle.
