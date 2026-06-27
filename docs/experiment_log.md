# AMT-YOLO Experiment Log
## Adaptive Memory Trajectory YOLO — Research Journal

---

## 2026-06-27 — Phase 1: Project Setup

### Actions Taken
- ✅ Created project folder structure under `D:\YOLO next\`
- ✅ Saved `implementation_plan.md` with confirmed decisions
- ✅ Created `requirements.txt` (RTX 4050 6GB optimized)
- ✅ Created `setup.py`
- ✅ Created `README.md`
- ✅ Scaffolded all core module files (Phase 2 head start)
- ✅ Created training + evaluation + demo + export scripts
- ✅ Created 6 ablation configuration YAML files
- ✅ Virtual environment `.venv` created
- ✅ PyTorch + CUDA install initiated

### Confirmed Decisions
| Decision | Value |
|---|---|
| GPU | RTX 4050 6GB |
| Base YOLO | YOLOv8 |
| Memory | ConvGRU + ConvLSTM (benchmark both) |
| Trajectory Horizon | 5–10 frames |
| Target | CVPR |

### Hardware Notes
- VRAM: 6GB → AMP required, batch=4, accumulate=4, max_train_res=768

---

## Upcoming: Phase 2 — Core Module Development
- Complete `temporal_memory.py` ConvGRU/ConvLSTM unit tests
- Complete `adaptive_resolution.py` integration tests
- Complete `trajectory_head.py` teacher forcing validation
- Integrate all modules into `amt_yolo.py` end-to-end pipeline
- Hook into YOLOv8 intermediate feature maps (P3, P4, P5)
