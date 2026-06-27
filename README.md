

AMT-YOLO (Adaptive Memory Trajectory YOLO) adalah pengembangan arsitektur berbasis YOLO yang dirancang secara khusus untuk analisis video real-time. Proyek riset ini bertujuan untuk mengatasi keterbatasan deteksi berbasis frame tunggal pada YOLO konvensional dengan mengintegrasikan kontinuitas temporal, resolusi adaptif, dan kemampuan prediksi lintasan.

---



Arsitektur AMT-YOLO dibangun di atas backbone YOLOv8 dengan penambahan tiga modul utama:


   Modul pra-pemrosesan yang mengevaluasi kompleksitas *scene* secara dinamis (menggunakan estimasi kepadatan tepi atau model CNN ringan). Modul ini secara otomatis menyesuaikan resolusi input (misalnya 640x640 untuk *scene* sederhana, hingga 1024x1024 untuk *scene* kompleks). Pendekatan ini secara signifikan mengoptimalkan penggunaan komputasi (GFLOPs) tanpa mengorbankan akurasi deteksi pada objek kecil.


   Modul memori berbasis rekuren (mendukung implementasi ConvGRU dan ConvLSTM) yang mempertahankan status fitur spasial antar-frame. Modul ini mengatasi anomali temporal seperti *flickering* dan oklusi sementara, menghasilkan pelacakan (tracking) dan deteksi objek yang jauh lebih konsisten pada *sequence* video.


   Cabang prediksi tambahan yang memanfaatkan *object embedding* dan *memory embedding* untuk memprediksi posisi spasial objek 5 hingga 10 frame di masa depan. Decoder berbasis GRU digunakan secara autogresif bersama mekanisme *teacher forcing* selama fase pelatihan.

---



Proyek ini telah dioptimalkan untuk berjalan pada perangkat keras kelas menengah (NVIDIA RTX 4050 dengan VRAM 6GB). Berbagai teknik pengoptimalan memori telah diterapkan pada konfigurasi dasar:
- Automatic Mixed Precision (AMP/FP16)
- Gradient Accumulation (Effective batch size = 16)
- Gradient Checkpointing


- Python 3.12+
- PyTorch 2.x & CUDA 12.x
- Ultralytics YOLOv8 (sebagai ekstraktor fitur dasar)
- OpenCV dan NumPy

---



Untuk menginisialisasi env pengembangan secara lokal, jalankan perintah berikut:

```bash
# 1. Kloning repositori
git clone https://github.com/Limjongun/YOLO-AMT-.git
cd YOLO-AMT-

# 2. Inisialisasi virtual environment
python -m venv .venv
# Pada Windows:
.venv\Scripts\activate

# 3. Instalasi PyTorch dengan dukungan CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Instalasi dependensi tambahan
pip install -r requirements.txt

# 5. Instalasi paket secara editable
pip install -e .
```

Gunakan skrip diagnostik untuk memvalidasi instalasi Anda:
```bash
python verify_setup.py
```

---



Penelitian ini ditargetkan untuk publikasi ilmiah (target konferensi: CVPR). Evaluasi model dan *ablation study* akan dilakukan menggunakan dataset COCO (pra-pelatihan deteksi), MOT17 (pelatihan temporal dan tracking), serta VisDrone (evaluasi deteksi objek udara skala kecil). 

Kami menyambut kolaborasi, laporan kutu (bug report), dan kontribusi kode melalui platform GitHub.
