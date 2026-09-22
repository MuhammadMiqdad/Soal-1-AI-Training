# Soal 1 - AI Training (Fruit Object Detection)

Sistem *object detection* berbasis YOLO (Ultralytics) untuk mendeteksi dan
mengklasifikasikan 9 jenis buah: Apple, Banana, Grapes, Kiwi, Mango, Orange,
Pineapple, Sugerapple, Watermelon.

## Spesifikasi Sistem

- **OS:** Windows 11 Home Single Language 23H2 (build 22631.6199)
- **Spesifikasi Komputer:** Intel Core i5-1135G7 @ 2.40GHz, RAM 8 GB
- **GPU:** NVIDIA GeForce MX330, 2 GB VRAM
- **Python:** 3.11.8
- **Library utama:** Ultralytics 8.4.157, PyTorch 2.7.1+cu118

## Isi Repository

| File | Keterangan |
|---|---|
| `best.pt` | Bobot model hasil training (YOLO11n) |
| `train.py` | Script auto-labeling + training |
| `inference.py` | Script deteksi + pop-up window |
| `requirements.txt` | Daftar library Python |

Dataset tidak diikutsertakan (ukuran besar); tautan dataset ada di bagian
*Dataset* di bawah.

## Dataset dan Masalah Bounding Box

Dataset yang digunakan: [Fruits by YOLO - Fruits Detection](https://www.kaggle.com/datasets/kapturovalexander/fruits-by-yolo-fruits-detection).

Dataset asli **tidak berisi bounding box**, hanya gambar dan file `_classes.csv`
(one-hot label kelas per gambar, format klasifikasi, bukan deteksi). Nama file
`data.yaml` di dataset menyesatkan seolah-olah format YOLO deteksi sudah siap,
padahal isinya bukan.

### Solusi: Auto-labeling (Pseudo-labeling), 2 ronde

Karena jenis buah tiap gambar sudah diketahui dari `_classes.csv`, lokasi
buahnya dicari otomatis, dalam dua ronde, dijalankan otomatis oleh `train.py`:

**Ronde 1 - model open-vocabulary YOLO-World** (`yolov8m-worldv2.pt`), dua tahap:
1. **Pencarian dengan nama buah** (mis. prompt `"banana"`, `"sugar apple"`)
2. **Untuk gambar yang gagal di tahap 1:** dicoba lagi dengan prompt generik
   `"fruit"`, kotak yang ditemukan diberi label kelas dari `_classes.csv`

**Ronde 2 - self-training (`--refine`).** Setelah training pertama, `best.pt`
hasil training sendiri dipakai untuk melengkapi (bukan menimpa) gambar yang
masih gagal di ronde 1. Karena model ini sudah domain-specific dan mengenali
ke-9 kelas sekaligus, satu prediksi bisa menemukan beberapa buah dalam satu
gambar (mis. gambar campuran Banana+Mango), sesuatu yang lebih sulit
dilakukan YOLO-World yang generik. Model kemudian dilatih ulang dengan label
yang sudah lebih lengkap ini.

Hasil auto-labeling tiap ronde diperiksa manual lewat folder `qa_preview/`
(sampel acak + sampel khusus tiap tahap/ronde) sebelum training dijalankan.

**Cakupan auto-labeling:**
- Setelah ronde 1: 2.550 dari ±2.970 gambar (≈86%)
- Setelah ronde 2 (refine): 2.913 dari ±2.970 gambar (≈96%)

### Keterbatasan yang Diketahui

- **Bukan ground truth manusia.** Label adalah hasil prediksi model lain
  (YOLO-World pada ronde 1, `best.pt` sendiri pada ronde 2), bukan anotasi
  manusia. mAP yang dilaporkan mengukur kecocokan dengan label otomatis ini,
  bukan akurasi absolut.
- **Gambar dengan buah berkerumun** (tandan pisang, tumpukan anggur/srikaya)
  kadang hanya sebagian buah yang terdeteksi dan diberi kotak, bukan semuanya.
- **Kelas Sugerapple** paling banyak terbantu oleh ronde 2 (+121 gambar), dan
  akhirnya muncul di evaluasi split `valid` (mAP50 = 0,938, dari 5 instance),
  tapi tetap 0 instance di split `test`, sehingga performanya pada data yang
  benar-benar terpisah belum bisa dipastikan.

## Cara Menjalankan

### 1. Setup environment
```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # opsional, jika ada GPU NVIDIA
pip install -r requirements.txt
```

### 2. Training
Taruh folder dataset hasil ekstrak (berisi `train/`, `valid/`, `test/`) di
folder yang sama dengan `train.py`, lalu:
```bash
python train.py --prepare-only          # ronde 1: auto-labeling -> cek qa_preview/
python train.py                         # training (skip labeling jika sudah ada)
python train.py --prepare-only --refine # ronde 2: lengkapi label pakai best.pt -> cek qa_preview/
python train.py --refine                # lengkapi label + training ulang
```
`best.pt` akan otomatis disalin (dan ditimpa) ke folder proyek setiap kali
training selesai. Ronde 2 (`--refine`) bersifat opsional, dijalankan setelah
`best.pt` dari training pertama tersedia.

### 3. Inference
Path folder gambar diatur di dalam script (variabel `IMAGE_DIR` pada
`inference.py`), sesuai ketentuan soal.
```bash
python inference.py
```
Kontrol: tombol apa saja = gambar berikutnya, `a` = sebelumnya, `q`/`ESC` = keluar.

## Hasil Training

Konfigurasi: YOLO11n, 50 epoch, imgsz 640, batch 4 (dibatasi VRAM 2 GB).
Dilatih dua kali: training pertama (label ronde 1 saja) lalu training kedua
setelah label dilengkapi lewat `--refine` (ronde 2, ±6,1-6,3 jam per training).

| Split | Training 1 (mAP50 / mAP50-95) | Training 2, setelah refine (mAP50 / mAP50-95) |
|---|---|---|
| Validation | 0,821 / 0,733 | 0,831 / 0,750 |
| Test | 0,675 / 0,586 | **0,759 / 0,661** |

`best.pt` yang dikumpulkan adalah hasil training kedua (setelah refine).
mAP per kelas serta detail lengkap ada di output terminal `train.py` dan
folder `runs/fruits/`. Kelas dengan performa terlemah pada split test:
Kiwi dan Watermelon, kemungkinan besar karena jumlah sampel test yang kecil
(13 dan 6 gambar) sehingga sensitif terhadap satu-dua kesalahan.

`inference.py` diuji pada seluruh 85 gambar split test: 82 gambar
menghasilkan deteksi dengan confidence yang wajar (mayoritas di atas 0,7),
3 gambar tidak menghasilkan deteksi sama sekali.
