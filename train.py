"""
Training object detection buah dengan YOLO (Ultralytics).
 
Dataset Kaggle "Fruits by YOLO" cuma punya gambar + _classes.csv (tanpa bounding box),
jadi script ini auto-labeling dulu pakai YOLO-World sebelum training YOLO11.
 
    python train.py                       # auto-label (kalau perlu) + training
    python train.py --prepare-only        # cuma auto-label, cek qa_preview/
    python train.py --prepare-only --fill-missing
    python train.py --prepare-only --refine
    python train.py --refine              # lengkapi label pakai best.pt sendiri, lalu training ulang
 
Hapus folder dataset/ untuk mulai ulang dari nol. Taruh dataset (folder hasil ekstrak
atau archive.zip) di folder yang sama dengan script ini.
"""
import argparse
import csv
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

# ----------------------------- KONFIGURASI -----------------------------
BASE = Path(__file__).resolve().parent
ZIP_PATH = BASE / "archive.zip"      # dipakai hanya jika folder hasil ekstrak tidak ditemukan
RAW_DIR = BASE / "dataset_raw"       # tujuan ekstrak zip
OUT_DIR = BASE / "dataset"           # dataset format YOLO hasil auto-labeling
PREVIEW_DIR = BASE / "qa_preview"    # gambar + label boxing untuk dicek sebelum training
DATA_YAML = OUT_DIR / "dataset.yaml"

CLASSES = ["Apple", "Banana", "Grapes", "Kiwi", "Mango",
           "Orange", "Pineapple", "Sugerapple", "Watermelon"]

# (a) Kata (bahasa Inggris) untuk menunjuk buah ke YOLO-World
PROMPTS = {
    "Apple": "apple",
    "Banana": "banana",
    "Grapes": "grapes",
    "Kiwi": "kiwi fruit",
    "Mango": "mango",
    "Orange": "orange fruit",
    "Pineapple": "pineapple",
    "Sugerapple": "sugar apple",    
    "Watermelon": "watermelon",
}
# (b) Kata generik untuk gambar yang gagal di (a)
GENERIC_PROMPT = "fruit"

# --- auto-labeling ---
SPLITS = ["train", "valid", "test"]
WORLD_WEIGHTS = "yolov8s-worldv2.pt" 
LABEL_CONF = 0.05     
GENERIC_CONF = 0.03   
REL_CONF = 0.5        # buang box yang confidence-nya < 50%
MIN_AREA = 0.005      # buang box yang luasnya < 0.5% 
LABEL_BATCH = 8
LABEL_DEVICE = "cpu"  

# --- training ---
PRETRAINED = "yolo11n.pt"   
EPOCHS = 50
IMGSZ = 640
BATCH = 4                   
PATIENCE = 15               

# --- refine (relabel ronde kedua pakai best.pt sendiri) ---
REFINE_CONF = 0.25   
REFINE_BATCH = 8
# -----------------------------------------------------------------------


# ============================ TAHAP 1: AUTO-LABELING ============================
def _search_split_root():
    """Cari folder yang berisi train/_classes.csv (dataset yang sudah diekstrak)."""
    for where in (RAW_DIR, BASE / "Fruits by YOLO", BASE):
        if where.exists():
            for hit in where.rglob("train/_classes.csv"):
                if "venv" not in hit.parts and OUT_DIR not in hit.parents:
                    return hit.parent.parent
    return None


def find_split_root() -> Path:
    root = _search_split_root()
    if root is None and ZIP_PATH.exists():
        print("Ekstrak zip ...")
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(RAW_DIR)
        root = _search_split_root()
    if root is None:
        raise SystemExit(
            "Dataset tidak ditemukan. Taruh folder hasil ekstrak (berisi train/valid/test) "
            "atau archive.zip di folder yang sama dengan train.py."
        )
    return root


def read_labels(split_dir: Path):
    """Baca _classes.csv -> list (nama_file, (id_kelas, ...))."""
    items = []
    with open(split_dir / "_classes.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = [h.strip() for h in next(reader)]
        assert header[1:] == CLASSES, f"Urutan kelas di CSV berbeda: {header[1:]}"
        for row in reader:
            if not row:
                continue
            ids = tuple(i for i, v in enumerate(row[1:]) if v.strip() == "1")
            items.append((row[0].strip(), ids))
    return items


def _predict(model, paths, prompts, conf):
    """Jalankan YOLO-World pada daftar gambar dengan prompt tertentu -> (path, hasil)."""
    model.set_classes(prompts)
    model.predictor = None          
    for s in range(0, len(paths), LABEL_BATCH):
        chunk = paths[s:s + LABEL_BATCH]
        results = model.predict([str(p) for p in chunk], conf=conf,
                                device=LABEL_DEVICE, verbose=False)
        for p, r in zip(chunk, results):
            yield p, r


def _to_lines(r, ids, min_conf):
    """Hasil prediksi -> baris label YOLO. ids memetakan indeks prompt -> id kelas dataset."""
    b = r.boxes
    if b is None or len(b) == 0:
        return []
    conf = b.conf.cpu().numpy()
    cls = b.cls.cpu().numpy().astype(int)
    xywhn = b.xywhn.cpu().numpy()
    best = conf.max()
    lines = []
    for c, cf, (x, y, w, h) in zip(cls, conf, xywhn):
        if cf < max(min_conf, REL_CONF * best) or w * h < MIN_AREA:
            continue
        lines.append(f"{ids[c]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return lines


def _save(p: Path, lines, img_out: Path, lbl_out: Path):
    shutil.copy2(p, img_out / p.name)
    (lbl_out / (p.stem + ".txt")).write_text("\n".join(lines) + "\n")


def _progress(msg):
    print(msg + " " * 10, end="\r", flush=True)


def auto_label(fill_missing: bool = False):
    from ultralytics import YOLOWorld

    root = find_split_root()
    print("Dataset mentah:", root)
    model = YOLOWorld(WORLD_WEIGHTS)

    total = {"no_label": 0, "berhasil_baru": 0, "gagal": 0}
    per_class = defaultdict(lambda: {"nama": 0, "generik": 0, "gagal": 0})
    generic_files = []  

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            continue
        img_out = OUT_DIR / "images" / split
        lbl_out = OUT_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        # gambar yang belum punya label (gambar yang sudah berlabel dilewati / resume)
        pending = []
        for fname, ids in read_labels(split_dir):
            if not ids:
                total["no_label"] += 1
                continue
            p = split_dir / fname
            if p.exists() and not (lbl_out / (p.stem + ".txt")).exists():
                pending.append((p, ids))

        # ---- (a) pencarian dengan nama buah ----
        failed = []
        if fill_missing:
            failed = list(pending)
        else:
            groups = defaultdict(list)
            for p, ids in pending:
                groups[ids].append(p)
            done = 0
            for ids, paths in groups.items():
                key = "+".join(CLASSES[i] for i in ids)
                prompts = [PROMPTS[CLASSES[i]] for i in ids]
                for p, r in _predict(model, paths, prompts, LABEL_CONF):
                    lines = _to_lines(r, ids, LABEL_CONF)
                    if lines:
                        _save(p, lines, img_out, lbl_out)
                        per_class[key]["nama"] += 1
                        total["berhasil_baru"] += 1
                    else:
                        failed.append((p, ids))
                    done += 1
                    _progress(f"[{split}] (a) nama buah: {done}/{len(pending)}")
            print()

        # ---- (b) pencarian generik "fruit" untuk yang gagal (hanya gambar 1 kelas) ----
        single = [(p, ids) for p, ids in failed if len(ids) == 1]
        for p, ids in failed:
            if len(ids) != 1:
                per_class["+".join(CLASSES[i] for i in ids)]["gagal"] += 1
                total["gagal"] += 1
        ids_of = {p: ids for p, ids in single}
        if single:
            done = 0
            for p, r in _predict(model, [p for p, _ in single], [GENERIC_PROMPT], GENERIC_CONF):
                ids = ids_of[p]
                key = CLASSES[ids[0]]
                lines = _to_lines(r, ids, GENERIC_CONF)   
                if lines:
                    _save(p, lines, img_out, lbl_out)
                    per_class[key]["generik"] += 1
                    total["berhasil_baru"] += 1
                    generic_files.append((img_out / p.name, lbl_out / (p.stem + ".txt")))
                else:
                    per_class[key]["gagal"] += 1
                    total["gagal"] += 1
                done += 1
                _progress(f"[{split}] (b) generik 'fruit': {done}/{len(single)}")
            print()
        n_lbl = len(list(lbl_out.glob("*.txt")))
        print(f"[{split}] total gambar berlabel sekarang: {n_lbl}")

    yaml_text = (
        f'path: "{OUT_DIR.as_posix()}"\n'
        "train: images/train\nval: images/valid\ntest: images/test\n"
        f"nc: {len(CLASSES)}\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    )
    DATA_YAML.write_text(yaml_text, encoding="utf-8")

    print("\nRingkasan:", total)
    print("Rekap per kelas (nama buah / generik / gagal):")
    for k in sorted(per_class):
        v = per_class[k]
        print(f"  {k:<22} {v['nama']:>5} / {v['generik']:>5} / {v['gagal']:<5}")
    save_preview(generic_files)


def _draw(img_path: Path, lbl_path: Path, out_path: Path):
    import cv2

    img = cv2.imread(str(img_path))
    if img is None or not lbl_path.exists():
        return
    h, w = img.shape[:2]
    for line in lbl_path.read_text().splitlines():
        c, x, y, bw, bh = line.split()
        x, y, bw, bh = float(x), float(y), float(bw), float(bh)
        x1, y1 = int((x - bw / 2) * w), int((y - bh / 2) * h)
        x2, y2 = int((x + bw / 2) * w), int((y + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, CLASSES[int(c)], (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(out_path), img)


def save_preview(generic_files, n_random: int = 30, n_generic: int = 20):
    shutil.rmtree(PREVIEW_DIR, ignore_errors=True)
    PREVIEW_DIR.mkdir(exist_ok=True)
    random.seed(0)
    imgs = sorted((OUT_DIR / "images" / "train").glob("*.jpg"))
    for p in random.sample(imgs, min(n_random, len(imgs))):
        _draw(p, OUT_DIR / "labels" / "train" / (p.stem + ".txt"), PREVIEW_DIR / ("acak_" + p.name))
    for ip, lp in random.sample(generic_files, min(n_generic, len(generic_files))):
        _draw(ip, lp, PREVIEW_DIR / ("generik_" + ip.name))
    print(f"Preview tersimpan di: {PREVIEW_DIR}  (awalan 'generik_' = hasil pencarian 'fruit')")


# ============================ REFINE: RELABEL RONDE KEDUA ============================
def _refine_lines(r, allowed_ids, min_conf):
    """Hasil prediksi best.pt -> baris label YOLO, dibatasi hanya ke kelas yang memang
    ada di gambar itu menurut _classes.csv (allowed_ids). Kelas di best.pt sudah dalam
    urutan yang sama dengan CLASSES, jadi id prediksi dipakai langsung (tidak dipetakan)."""
    b = r.boxes
    if b is None or len(b) == 0:
        return []
    conf = b.conf.cpu().numpy()
    cls = b.cls.cpu().numpy().astype(int)
    xywhn = b.xywhn.cpu().numpy()
    lines = []
    for c, cf, (x, y, w, h) in zip(cls, conf, xywhn):
        if c not in allowed_ids or cf < min_conf or w * h < MIN_AREA:
            continue
        lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return lines


def refine_labels():
    """Pakai best.pt hasil training sendiri untuk melengkapi gambar yang gagal
    di auto-labeling tahap 1 (label yang sudah ada TIDAK diubah/ditimpa).
    Biasanya lebih baik dari YOLO-World karena sudah domain-specific dan bisa
    menemukan beberapa instance sekaligus (mis. buah bergerombol) dalam satu prediksi."""
    import torch
    from ultralytics import YOLO

    weight = BASE / "best.pt"
    if not weight.exists():
        weight = BASE / "runs" / "fruits" / "weights" / "best.pt"
    if not weight.exists():
        raise SystemExit("best.pt tidak ditemukan. Jalankan training biasa (python train.py) dulu.")
    if not DATA_YAML.exists():
        raise SystemExit("dataset/ belum ada. Jalankan 'python train.py --prepare-only' dulu.")

    root = find_split_root()
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Refine pakai model: {weight} (device={device})")
    model = YOLO(str(weight))

    total = {"dilengkapi": 0, "masih_gagal": 0}
    per_class = defaultdict(lambda: {"ok": 0, "gagal": 0})
    refined_files = []   

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            continue
        img_out = OUT_DIR / "images" / split
        lbl_out = OUT_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        # hanya gambar yang BELUM punya label (label yang sudah ada tidak disentuh)
        missing = []
        for fname, ids in read_labels(split_dir):
            if not ids:
                continue
            p = split_dir / fname
            if p.exists() and not (lbl_out / (p.stem + ".txt")).exists():
                missing.append((p, ids))

        done = 0
        for s in range(0, len(missing), REFINE_BATCH):
            chunk = missing[s:s + REFINE_BATCH]
            results = model.predict([str(p) for p, _ in chunk], conf=REFINE_CONF,
                                    device=device, verbose=False)
            for (p, ids), r in zip(chunk, results):
                key = "+".join(CLASSES[i] for i in ids)
                lines = _refine_lines(r, set(ids), REFINE_CONF)
                if lines:
                    _save(p, lines, img_out, lbl_out)
                    per_class[key]["ok"] += 1
                    total["dilengkapi"] += 1
                    refined_files.append((img_out / p.name, lbl_out / (p.stem + ".txt")))
                else:
                    per_class[key]["gagal"] += 1
                    total["masih_gagal"] += 1
            done += len(chunk)
            _progress(f"[{split}] refine: {done}/{len(missing)}")
        if missing:
            print()
        n_lbl = len(list(lbl_out.glob("*.txt")))
        print(f"[{split}] total gambar berlabel sekarang: {n_lbl}")

    print("\nRingkasan refine:", total)
    print("Rekap per kelas (dilengkapi / masih gagal):")
    for k in sorted(per_class):
        v = per_class[k]
        print(f"  {k:<22} {v['ok']:>5} / {v['gagal']:<5}")

    # preview khusus hasil refine, ditambahkan ke qa_preview yang sudah ada
    PREVIEW_DIR.mkdir(exist_ok=True)
    random.seed(1)
    for ip, lp in random.sample(refined_files, min(20, len(refined_files))):
        _draw(ip, lp, PREVIEW_DIR / ("refine_" + ip.name))
    print(f"Preview hasil refine (awalan 'refine_') ditambahkan ke: {PREVIEW_DIR}")


# ============================ TAHAP 2: TRAINING ============================
def train():
    import torch
    from ultralytics import YOLO

    use_gpu = torch.cuda.is_available()
    device = 0 if use_gpu else "cpu"
    print("Device:", f"GPU {torch.cuda.get_device_name(0)}" if use_gpu else "CPU (akan lambat)")

    model = YOLO(PRETRAINED)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        workers=2,
        patience=PATIENCE,
        project=str(BASE / "runs"),
        name="fruits",
        exist_ok=True,
    )

    best = Path(model.trainer.best)
    shutil.copy2(best, BASE / "best.pt")
    print("best.pt disalin ke:", BASE / "best.pt")

    metrics = YOLO(str(BASE / "best.pt")).val(data=str(DATA_YAML), split="test", imgsz=IMGSZ)
    print(f"Test mAP50: {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-only", action="store_true",
                    help="hanya buat/lengkapi label lalu berhenti (untuk dicek di qa_preview/)")
    ap.add_argument("--fill-missing", action="store_true",
                    help="dataset/ sudah ada: lengkapi hanya gambar yang belum berlabel (pakai YOLO-World)")
    ap.add_argument("--refine", action="store_true",
                    help="dataset/ dan best.pt sudah ada: lengkapi gambar yang masih belum "
                         "berlabel memakai best.pt hasil training sendiri (self-training), "
                         "lalu lanjut training ulang (kecuali dipakai bersama --prepare-only)")
    args = ap.parse_args()

    if args.refine:
        refine_labels()
    elif args.fill_missing and DATA_YAML.exists():
        auto_label(fill_missing=True)
    elif DATA_YAML.exists():
        print("dataset/ sudah ada -> lewati auto-labeling "
              "(hapus folder dataset/ untuk membuat ulang, atau pakai --fill-missing / --refine).")
    else:
        auto_label()

    if args.prepare_only:
        print("\nSelesai. Cek folder qa_preview/ sebelum menjalankan: python train.py")
        return
    train()

if __name__ == "__main__":
    main()