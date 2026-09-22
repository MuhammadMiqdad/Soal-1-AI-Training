"""
LANGKAH 3 - Inference: deteksi buah pada semua gambar di sebuah folder,
tampilkan hasilnya di pop-up window lengkap dengan bounding box + nama buah.

Cara pakai:
    python inference.py
Kontrol:
    tombol apa saja / d = gambar berikutnya
    a               = gambar sebelumnya
    q atau ESC      = keluar
"""
from pathlib import Path

import cv2
from ultralytics import YOLO

# ----------------------------- KONFIGURASI -----------------------------
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "best.pt"
IMAGE_DIR = Path(r"C:\codes\Soal 1 AI Training\dataset\images\test")
CONF = 0.40                 
MAX_W, MAX_H = 1200, 800   
# -----------------------------------------------------------------------

WIN = "Fruit Detection"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def fit_to_screen(img):
    h, w = img.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 2.0)
    if abs(scale - 1.0) < 1e-3:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model tidak ditemukan: {MODEL_PATH} (jalankan train.py dulu)")
    if not IMAGE_DIR.is_dir():
        raise SystemExit(f"Folder gambar tidak ditemukan: {IMAGE_DIR}")

    images = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in EXTS)
    if not images:
        raise SystemExit(f"Tidak ada gambar di {IMAGE_DIR}")

    model = YOLO(str(MODEL_PATH))
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    print(f"{len(images)} gambar. Tombol: (apa saja)=next, a=prev, q/ESC=keluar")

    i = 0
    while 0 <= i < len(images):
        path = images[i]
        result = model.predict(str(path), conf=CONF, verbose=False)[0]

        found = [f"{result.names[int(c)]} ({float(s):.2f})"
                 for c, s in zip(result.boxes.cls, result.boxes.conf)]
        print(f"[{i + 1}/{len(images)}] {path.name}: {', '.join(found) if found else 'tidak ada buah terdeteksi'}")

        annotated = result.plot()          # gambar + bounding box + label kelas (BGR)
        cv2.imshow(WIN, fit_to_screen(annotated))
        cv2.setWindowTitle(WIN, f"{WIN} [{i + 1}/{len(images)}] {path.name}")

        key = cv2.waitKey(0) & 0xFF
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:   
            break
        if key in (ord("q"), 27):
            break
        elif key == ord("a"):
            i = max(0, i - 1)
        else:
            i += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
