# 🚀 AMT-YOLO: Adaptive Memory Trajectory YOLO

Halo semuanya! Selamat datang di repositori **AMT-YOLO**! 👋

Proyek ini adalah hasil eksperimen dan riset kami untuk bikin model deteksi objek YOLO jadi jauh lebih pintar, khususnya buat memproses video. YOLO standar itu kan udah ngebut banget, tapi sayangnya dia cuma ngelihat gambar per frame doang. Artinya, dia nggak "ingat" apa yang terjadi di frame sebelumnya, dan kadang boros komputasi buat scene yang sebenarnya kosong atau gampang.

Nah, dari situlah ide **AMT-YOLO** lahir! Kami ngambil arsitektur YOLOv8 yang udah terbukti mantap, terus kami tambahin 3 "otak" baru biar modelnya nggak cuma bisa nebak *apa* objeknya, tapi juga *nginget* pergerakannya dan *mrediksi* mau ke mana arahnya. Keren kan? 😎

---

## 🧠 Tiga Senjata Utama AMT-YOLO

Kami nambahin 3 modul keren ke dalam arsitektur ini:

1. **Adaptive Resolution Router (Si Pintar Ngirit)** 📉
   Daripada maksa semua frame diproses pakai resolusi tinggi (misal 1024x1024) yang bikin GPU engap, modul ini bakal nebak dulu seberapa ribet scene-nya. Kalau jalanan lagi sepi, dia otomatis pakai resolusi rendah (640x640). Kalau lagi macet atau banyak objek kecil, baru deh resolusinya dinaikin. Hasilnya? FPS tetap kenceng tanpa ngorbanin akurasi!

2. **Temporal Memory Module (Si Paling Inget)** 🐘
   Ini nih obat buat ngatasin objek yang sering kedip-kedip (flickering) atau ketutupan tiang (occlusion). Kami pasangin memori (bisa pilih mau pakai **ConvGRU** yang enteng, atau **ConvLSTM** yang lebih kuat) biar YOLO bisa ingat "Oh, di frame kemarin di situ ada mobil merah". Jadi tracking-nya jauh lebih stabil!

3. **Trajectory Prediction Head (Si Peramal)** 🔮
   Bukan cuma deteksi, model ini bisa nebak 5 sampai 10 frame ke depan objek itu mau gerak ke mana! Sangat berguna buat ngurangin risiko kecelakaan di autonomous driving atau buat mantau pergerakan orang di CCTV.

---

## 💻 Tech Stack & Hardware

Karena kami ngedevelop ini pakai **RTX 4050 dengan VRAM 6GB** (iya, lumayan ngepas 😅), kami udah nge-tuning banyak hal biar model ini tetap bisa ditraining tanpa bikin VRAM jebol:
- **Automatic Mixed Precision (AMP)** wajib nyala!
- **Gradient Accumulation** biar seolah-olah batch size-nya gede.
- **Gradient Checkpointing** pas training temporal sequence.

**Teknologi yang dipakai:**
- Python 3.12+ 🐍
- PyTorch 2.x & CUDA 12.x 🔥
- Ultralytics YOLOv8 🚀
- OpenCV, Numpy, dll.

---

## 🛠️ Cara Setup (Buat yang Mau Nyoba)

Buat kalian yang mau nge-clone dan nyoba jalanin di lokal, ikutin langkah ini ya:

```bash
# 1. Clone dulu repo ini
git clone https://github.com/Limjongun/YOLO-AMT-.git
cd YOLO-AMT-

# 2. Bikin virtual environment biar rapi
python -m venv .venv
.venv\Scripts\activate  # Buat pengguna Windows

# 3. Install PyTorch (Penting! Sesuaikan sama CUDA kalian)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install requirement lainnya
pip install -r requirements.txt

# 5. Install package-nya
pip install -e .
```

Setelah itu, kalian bisa cek apakah setup-nya udah bener pakai script verifikasi kami:
```bash
python verify_setup.py
```

---

## 🎯 Target Kami (CVPR Here We Come!)

Tujuan utama dari repo ini adalah untuk riset dan publikasi paper (doa-in tembus **CVPR** ya! 🙏). Kami lagi nyiapin proses training multi-stage pakai dataset COCO, MOT17, dan VisDrone. 

Kalau kalian nemu bug atau punya ide buat ningkatin performanya, jangan sungkan buat buka Issue atau PR! Mari kita bikin sistem deteksi video yang lebih pinter bareng-bareng! 🚀

---
*Dibuat dengan ☕ dan semangat riset oleh Tim AMT-YOLO.*
