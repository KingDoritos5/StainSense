"""
StainSense - Data Pipeline Script
====================================
Mengunduh, mengekstrak, dan menormalisasi 3 dataset Kaggle ke dalam
satu folder master_dataset/ dan file metadata.csv yang terpadu.

Urutan eksekusi:
    python data_setup.py

Prasyarat:
    - File ~/.kaggle/kaggle.json sudah dikonfigurasi
    - Paket: kaggle, pandas, Pillow, tqdm
"""

import os
import sys
import shutil
import subprocess
import zipfile
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# ─── Konfigurasi ────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent
RAW_DIR        = BASE_DIR / "raw_datasets"
MASTER_DIR     = BASE_DIR / "master_dataset" / "images"
METADATA_PATH  = BASE_DIR / "master_dataset" / "metadata.csv"

DATASETS = [
    "varshinisandhi/fabric-dataset",
    "osman0/dirty-clothes",
    "priemshpathirana/fabric-stain-dataset",
]

# ─── Pemetaan Label ke Kelas Standar ────────────────────────────────────────
# Format: {keyword_dalam_nama_folder_asli : kelas_standar}
LABEL_MAP: dict[str, str] = {
    # Noda tinta & pulpen
    "ink":        "ink_stain",
    "pen":        "ink_stain",
    "ballpoint":  "ink_stain",
    "marker":     "ink_stain",

    # Noda minyak & lemak
    "oil":        "oil_stain",
    "grease":     "oil_stain",
    "butter":     "oil_stain",

    # Noda darah
    "blood":      "blood_stain",

    # Noda makanan
    "food":       "food_stain",
    "sauce":      "food_stain",
    "ketchup":    "food_stain",
    "curry":      "food_stain",
    "chocolate":  "food_stain",

    # Noda kopi & teh
    "coffee":     "coffee_tea_stain",
    "tea":        "coffee_tea_stain",

    # Noda lumpur & tanah
    "mud":        "mud_dirt_stain",
    "dirt":       "mud_dirt_stain",
    "soil":       "mud_dirt_stain",

    # Noda tumbuhan
    "grass":      "plant_stain",
    "leaf":       "plant_stain",
    "plant":      "plant_stain",

    # Kain bersih / tidak bernoda
    "clean":      "clean_fabric",
    "no_stain":   "clean_fabric",
    "unstained":  "clean_fabric",

    # Fallback
    "stain":      "unknown_stain",
    "dirty":      "unknown_stain",
}

TARGET_SIZE = (224, 224)  # Ukuran standar gambar
VALID_EXTS  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ─── Utilitas ────────────────────────────────────────────────────────────────

def run_cmd(cmd: list[str], step: str) -> None:
    """Jalankan perintah subprocess dengan penanganan error."""
    print(f"\n[{step}] Menjalankan: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠  STDERR: {result.stderr.strip()}")
        # Kaggle kadang mengembalikan kode non-0 meski sukses
        if "already exists" in result.stderr.lower():
            print("  ℹ  Dataset sudah ada, melewati unduhan.")
        else:
            raise RuntimeError(f"Perintah gagal: {' '.join(cmd)}\n{result.stderr}")
    else:
        print(f"  ✓ {step} selesai.")


def normalize_label(folder_name: str) -> str:
    """Petakan nama folder ke kelas standar berdasarkan LABEL_MAP."""
    name_lower = folder_name.lower()
    for keyword, standard_class in LABEL_MAP.items():
        if keyword in name_lower:
            return standard_class
    return "unknown_stain"


def sanitize_filename(name: str) -> str:
    """Bersihkan karakter tidak valid dari nama file."""
    return re.sub(r"[^a-zA-Z0-9_\-.]", "_", name)


def process_image(src: Path, dst: Path) -> bool:
    """
    Buka gambar, ubah ke RGB, resize ke TARGET_SIZE, simpan sebagai JPEG.
    Kembalikan True jika berhasil.
    """
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            img.save(dst, "JPEG", quality=90)
        return True
    except Exception as exc:
        print(f"    ✗ Gagal memproses {src.name}: {exc}")
        return False


# ─── Tahap 1: Unduh Dataset ──────────────────────────────────────────────────

def download_datasets() -> None:
    """Unduh semua dataset Kaggle ke RAW_DIR."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        slug = dataset.replace("/", "_")
        dest = RAW_DIR / slug

        if dest.exists():
            print(f"\n[Unduh] ⏩ {dataset} sudah ada, melewati.")
            continue

        dest.mkdir(parents=True, exist_ok=True)
        run_cmd(
            ["kaggle", "datasets", "download", "-d", dataset,
             "--path", str(dest), "--unzip"],
            step=f"Unduh {dataset}"
        )


# ─── Tahap 2: Ekstrak & Normalisasi ─────────────────────────────────────────

def collect_image_paths(root: Path) -> list[Path]:
    """Kumpulkan semua path gambar valid secara rekursif dari root."""
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ]


def normalize_datasets() -> pd.DataFrame:
    """
    Iterasi semua gambar dari RAW_DIR, petakan labelnya,
    resize, dan simpan ke MASTER_DIR. Kembalikan DataFrame metadata.
    """
    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    global_idx = 0

    for dataset_slug in [d.replace("/", "_") for d in DATASETS]:
        src_root = RAW_DIR / dataset_slug
        if not src_root.exists():
            print(f"\n⚠  Folder {src_root} tidak ditemukan, melewati.")
            continue

        images = collect_image_paths(src_root)
        print(f"\n[Normalisasi] {dataset_slug}: {len(images)} gambar ditemukan.")

        for img_path in tqdm(images, desc=f"  {dataset_slug}", unit="img"):
            # Label diambil dari nama folder induk langsung
            parent_folder = img_path.parent.name
            stain_label   = normalize_label(parent_folder)

            # Buat nama file unik
            new_name = f"{global_idx:06d}_{sanitize_filename(img_path.stem)}.jpg"
            dst_path = MASTER_DIR / new_name

            if process_image(img_path, dst_path):
                records.append({
                    "image_id":       global_idx,
                    "filename":       new_name,
                    "stain_label":    stain_label,
                    "original_label": parent_folder,
                    "source_dataset": dataset_slug,
                    "original_path":  str(img_path.relative_to(RAW_DIR)),
                    "width":          TARGET_SIZE[0],
                    "height":         TARGET_SIZE[1],
                })
                global_idx += 1

    df = pd.DataFrame(records)
    return df


# ─── Tahap 3: Simpan Metadata ────────────────────────────────────────────────

def save_metadata(df: pd.DataFrame) -> None:
    """Simpan metadata ke CSV dan cetak ringkasan statistik."""
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(METADATA_PATH, index=False)

    print("\n" + "═" * 55)
    print("  ✅  PIPELINE SELESAI")
    print("═" * 55)
    print(f"  Total gambar      : {len(df):,}")
    print(f"  Total kelas       : {df['stain_label'].nunique()}")
    print(f"  File metadata     : {METADATA_PATH}")
    print(f"  Folder master     : {MASTER_DIR}")
    print("\n  Distribusi kelas:")

    dist = df["stain_label"].value_counts()
    for label, count in dist.items():
        bar = "█" * min(count // max(dist.max() // 20, 1), 30)
        print(f"    {label:<25} {count:>5}  {bar}")

    print("═" * 55)


# ─── Tahap 4: Validasi Integritas ────────────────────────────────────────────

def validate_master(df: pd.DataFrame) -> None:
    """Periksa bahwa semua file yang terdaftar di metadata benar-benar ada."""
    print("\n[Validasi] Memeriksa integritas file...")
    missing = []
    for row in df.itertuples():
        fpath = MASTER_DIR / row.filename
        if not fpath.exists():
            missing.append(row.filename)

    if missing:
        print(f"  ⚠  {len(missing)} file hilang! Contoh: {missing[:3]}")
    else:
        print(f"  ✓ Semua {len(df)} file valid dan dapat diakses.")


# ─── Entri Utama ─────────────────────────────────────────────────────────────

def main() -> None:
    print("╔══════════════════════════════════════════════════╗")
    print("║        StainSense — Data Pipeline v1.0           ║")
    print("╚══════════════════════════════════════════════════╝\n")

    # Periksa konfigurasi Kaggle
    kaggle_cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_cfg.exists():
        print("❌ File kaggle.json tidak ditemukan!")
        print(f"   Letakkan di: {kaggle_cfg}")
        print("   Panduan: https://www.kaggle.com/docs/api#authentication")
        sys.exit(1)

    print(f"✓ kaggle.json ditemukan di {kaggle_cfg}")

    download_datasets()
    df = normalize_datasets()

    if df.empty:
        print("\n❌ Tidak ada gambar yang berhasil diproses. Periksa unduhan dataset.")
        sys.exit(1)

    save_metadata(df)
    validate_master(df)


if __name__ == "__main__":
    main()
