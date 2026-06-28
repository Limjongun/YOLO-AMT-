# AMT-YOLO: Implementation Plan — Phase 2 Lanjutan s/d Phase 4
## Status: DRAFT — 2026-06-27 (Option 2: MOT17 Neck Warmup)
## Dokumen ini melanjutkan dari `implementation_plan.md`

---

## Ringkasan Progres Saat Ini

| Phase | Status | Catatan |
|---|---|---|
| Phase 1: Setup Environment | ✅ Selesai | Python 3.12, CUDA, 7/8 checks passed |
| Phase 2: Core Modules | 🔶 95% | Semua modul ada, hook P3/P4/P5 diimplementasi |
| Phase 3: Data Pipeline | ⬜ Belum | Prioritas berikutnya |
| Phase 4: Training | ⬜ Belum | Blok utama riset |
| Phase 5: Evaluasi | 🔶 50% | Metrik & ablation config sudah ada |

---

## Phase 2 — Sisa Pekerjaan (Target: Selesai Hari Ini)

### 2A. Verifikasi Unit Tests
Status: 20/28 passed pada putaran pertama. Bug FP16/float32 mismatch di SEBlock sudah di-fix. Akan diverifikasi ulang.

### 2B. Verifikasi Hook Layer Indices YOLOv8
**PENTING:** Layer indices P3/P4/P5 pada Ultralytics YOLOv8 bergantung pada varian model.
**Langkah:** Inspeksi layer YOLOv8 di environment lokal lalu update `LAYER_INDICES` di `amt_yolo.py`.

### 2C. Integration Test — Sequence Forward Pass
Simulasi pipeline penuh: 5 frame berurutan melalui AMTYOLO, validasi:
- Memory state berubah antar frame (temporal continuity)
- Output P5 berubah (memory bukan pass-through)
- Trajectory output shape benar

---

## Phase 3 — Data Pipeline (MOT17 Only)

Karena kita menggunakan strategi **Neck Warmup**, kita **TIDAK PERLU** mengunduh COCO atau CrowdHuman. Kita langsung menggunakan dataset target.

### 3.1 Dataset yang Digunakan

| Dataset | Tujuan | Ukuran | Link |
|---|---|---|---|
| MOT17 | Stage 1 (Warmup) & Stage 2 (Temporal) | ~5.5GB | https://motchallenge.net |
| VisDrone2019 | Evaluasi objek kecil (Opsional) | ~10GB | https://github.com/VisDrone |

**Download Strategy untuk Storage Terbatas:**
- Hanya perlu mengunduh MOT17. Hemat 20GB+ storage di Drive D!

### 3.2 File yang Perlu Dibuat

#### [NEW] `amt_yolo/data/dataset.py`
```
MOT17SequenceDataset
    - __getitem__: sequence N frame berurutan (default: 5)
    - trajectory_labels: gt future positions per object
    - Mengembalikan: [frames_T, labels_per_frame, trajectory_gt]
    - Handles: sequence boundaries, ID continuity
```

#### [NEW] `amt_yolo/data/trajectory_labels.py`
```
TrajectoryLabelGenerator
    - Input: MOT17 annotation .txt file
    - Output: Per-object gt trajectory dict {track_id: [T, 4]}
    - Handles: occluded frames (vis < 0.25), missing IDs (interpolasi)
    - horizon: configurable (5–10 frames)
```

#### [NEW] `amt_yolo/data/augmentation.py`
```
TemporalAugmentation
    - Temporal-consistent: augmentasi yang SAMA untuk semua frame dalam 1 sequence
    - ColorJitter: diterapkan 1x, hasilnya direplikasi
    - RandomHorizontalFlip: flip konsisten + flip koordinat GT
```

---

## Phase 4 — Loss Functions & Training 

### 4.1 Loss Functions

Total Loss kita adalah kombinasi Ultralytics detection loss dan trajectory loss buatan kita:
```python
total_loss = (
    w_det  * detection_loss     +   # 1.0 (IoU, DFL, Class)
    w_traj * trajectory_loss        # 0.5 (SmoothL1, ADE, Confidence)
)
```

### 4.2 Multi-Stage Training Protocol (THE WARMUP TRICK) 🔥

#### Stage 1 — Neck Warmup (MOT17, ~10 epochs)
Tujuan: Mengajari `AdaptiveFeatureFusionNeck` (leher baru) agar tidak merusak beban YOLOv8 head yang sudah ahli, tanpa dataset eksternal.
```yaml
dataset: mot17
freeze_backbone: true          # Kunci Backbone
freeze_head: true              # Kunci YOLOv8 Detection Head
epochs: 10
lr: 0.001
batch_size: 4
accumulate: 4
amp: true
resolution: 640
memory: disabled               # Matikan memory dulu
trajectory: disabled           # Matikan trajectory dulu
```

#### Stage 2 — Full Temporal Training (MOT17, ~30 epochs)
Tujuan: Melatih Memory Module dan Trajectory Head bersama-sama dengan Head dan Neck.
```yaml
dataset: mot17
checkpoint: stage1_best.pt
freeze_backbone: false         # BUKA semua kunci
freeze_head: false             # BUKA semua kunci
epochs: 30
lr: 0.0001
batch_size: 2                  # Lebih kecil (2 batch x 5 frame = 10 image per step)
accumulate: 8                  # effective batch = 16
amp: true
gradient_checkpoint: true
resolution: 640
memory: convgru                # AKTIFKAN temporal memory
trajectory: true               # AKTIFKAN trajectory prediction
sequence_length: 5
```

### 4.3 VRAM Budget (RTX 4050 6GB)
Total estimasi VRAM untuk Stage 2 (paling berat): **~4.6 GB** ✅
Sangat aman untuk RTX 4050. Jika OOM, kita akan turunkan `sequence_length` dari 5 ke 3.

---

## Checklist Eksekusi (Berurutan)

```
[x] Phase 1 — Environment Setup ✅
[x] Phase 2 — Core Modules (skeleton) ✅
[x] Phase 2 — Hook P3/P4/P5 YOLOv8 ✅
[ ] Phase 2A — Verifikasi hook indices via layer inspection (Python runtime)
[ ] Phase 2B — Integration test forward_sequence (5 frames)
      ↓
[ ] Phase 3.1 — Download MOT17 ke Drive D (hemat 20GB karena skip COCO)
[ ] Phase 3.2 — Implementasi MOT17SequenceDataset
[ ] Phase 3.3 — Implementasi TrajectoryLabelGenerator
[ ] Phase 3.4 — Implementasi TemporalAugmentation
[ ] Phase 3.5 — Verifikasi dataloader (sample batch, shapes)
      ↓
[ ] Phase 4.1 — Implementasi AMTDetectionLoss & TrajectoryLoss
[ ] Phase 4.2 — Integrasi loss ke train.py
[ ] Phase 4.3 — Stage 1 test run (1 epoch WARMUP)
[ ] Phase 4.4 — Stage 2 test run (1 epoch FULL TEMPORAL)
[ ] Phase 4.5 — Full Training Run (Ditinggal tidur semalaman)
```
