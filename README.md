# 🧹 StainSense — Panduan Instalasi & Penggunaan Lengkap

> **MVP v1.0** — Identifikasi noda kain berbasis AI: upload foto → deteksi noda → instruksi pembersihan step-by-step.

---

## Daftar Isi
1. [Prasyarat Sistem](#1-prasyarat-sistem)
2. [Setup Virtual Environment](#2-setup-virtual-environment-python)
3. [Konfigurasi Kaggle API](#3-konfigurasi-kaggle-api)
4. [Konfigurasi API Key LLM](#4-konfigurasi-api-key-llm)
5. [Instalasi Dependensi](#5-instalasi-dependensi)
6. [Menjalankan Data Pipeline](#6-menjalankan-data-pipeline-data_setuppy)
7. [Menjalankan Aplikasi](#7-menjalankan-aplikasi-apppy)
8. [Struktur Proyek](#8-struktur-proyek)
9. [Arsitektur & Alur Kerja](#9-arsitektur--alur-kerja)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prasyarat Sistem

| Komponen      | Versi Minimum | Catatan                              |
|---------------|---------------|--------------------------------------|
| Python        | 3.10+         | Cek: `python --version`              |
| pip           | 23+           | Cek: `pip --version`                 |
| Git           | 2.x           | Opsional, untuk clone repo           |
| Koneksi Internet | Stabil     | Untuk unduh dataset & API calls      |
| RAM           | 4 GB+         | Dataset bisa cukup besar             |
| Storage       | 5 GB+         | Untuk dataset mentah + master dataset |

---

## 2. Setup Virtual Environment Python

Selalu gunakan virtual environment untuk mengisolasi dependensi proyek.

### Linux / macOS
```bash
# Masuk ke folder proyek
cd stainsense/

# Buat virtual environment bernama 'venv'
python3 -m venv venv

# Aktifkan virtual environment
source venv/bin/activate

# Verifikasi: prompt akan berubah menjadi (venv) ...
which python  # Harus menunjuk ke ./venv/bin/python
```

### Windows (Command Prompt)
```cmd
cd stainsense\

python -m venv venv

venv\Scripts\activate.bat

where python
```

### Windows (PowerShell)
```powershell
cd stainsense\

python -m venv venv

.\venv\Scripts\Activate.ps1

# Jika muncul error "execution policy", jalankan dulu:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

> ⚠️ **Penting**: Pastikan `(venv)` selalu terlihat di prompt sebelum menjalankan perintah apapun.

---

## 3. Konfigurasi Kaggle API

Script `data_setup.py` menggunakan Kaggle CLI untuk mengunduh dataset. Kamu perlu file `kaggle.json` yang berisi kredensial API.

### Langkah 3.1 — Dapatkan `kaggle.json`
1. Login ke [kaggle.com](https://www.kaggle.com)
2. Klik foto profil → **Settings**
3. Scroll ke bagian **API**
4. Klik tombol **"Create New Token"**
5. File `kaggle.json` akan otomatis terunduh

Isi file `kaggle.json` akan terlihat seperti ini:
```json
{
    "username": "nama_pengguna_kaggle_kamu",
    "key": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
}
```

### Langkah 3.2 — Letakkan file di lokasi yang benar

**Linux / macOS:**
```bash
# Buat folder .kaggle jika belum ada
mkdir -p ~/.kaggle

# Pindahkan file kaggle.json
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json

# WAJIB: Set permission agar hanya owner yang bisa membaca
chmod 600 ~/.kaggle/kaggle.json

# Verifikasi
ls -la ~/.kaggle/kaggle.json
# Harus menampilkan: -rw------- 1 user user ... kaggle.json
```

**Windows:**
```cmd
# Buat folder C:\Users\<NamaUser>\.kaggle\ jika belum ada
mkdir %USERPROFILE%\.kaggle

# Pindahkan file (ganti path sesuai lokasi unduhan)
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

### Langkah 3.3 — Verifikasi koneksi Kaggle
```bash
# Install kaggle CLI (jika belum)
pip install kaggle

# Test koneksi
kaggle datasets list --search "fabric"
# Jika berhasil, akan muncul daftar dataset
```

---

## 4. Konfigurasi API Key LLM

StainSense mendukung dua provider AI. Pilih salah satu atau keduanya.

### Langkah 4.1 — Buat file `.env`
```bash
# Salin template
cp .env.example .env
```

### Langkah 4.2 — Option A: Google Gemini API (Recommended)

1. Buka [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Login dengan akun Google
3. Klik **"Create API Key"** → **"Create API key in new project"**
4. Salin API key yang tampil
5. Buka file `.env` dan isi:
   ```env
   GEMINI_API_KEY=AIzaSy_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

> 💡 **Tier Gratis Gemini**: 15 requests/menit, 1500 requests/hari. Cukup untuk MVP!

### Langkah 4.3 — Option B: OpenRouter API

1. Daftar/login di [openrouter.ai](https://openrouter.ai)
2. Buka menu **Keys** (pojok kanan atas)
3. Klik **"Create Key"** → beri nama "StainSense"
4. Salin API key
5. Buka file `.env` dan isi:
   ```env
   OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

> 💡 OpenRouter menyediakan akses ke 200+ model AI. Model gratis tersedia seperti `google/gemini-2.0-flash-thinking-exp:free`.

### Langkah 4.4 — Verifikasi file `.env`
```bash
cat .env
# Pastikan key sudah terisi, bukan "your_..._here"
```

---

## 5. Instalasi Dependensi

Pastikan virtual environment sudah aktif `(venv)`.

```bash
# Upgrade pip terlebih dahulu
pip install --upgrade pip

# Install semua dependensi
pip install -r requirements.txt

# Verifikasi instalasi kunci
python -c "import streamlit, cv2, PIL, pandas; print('✅ Semua dependensi OK')"
```

### Opsional: Install YOLOv8 (Deteksi Lebih Akurat)
```bash
pip install ultralytics

# Test
python -c "from ultralytics import YOLO; print('✅ YOLOv8 OK')"
```

---

## 6. Menjalankan Data Pipeline (`data_setup.py`)

> **Jalankan ini PERTAMA sebelum `app.py`**. Script ini mengunduh dan menyiapkan dataset.

```bash
# Pastikan (venv) aktif
python data_setup.py
```

Proses yang terjadi:
1. ✅ Memverifikasi `kaggle.json`
2. ⬇️ Mengunduh 3 dataset dari Kaggle (dapat memakan waktu 5-20 menit tergantung koneksi)
3. 🔄 Menormalisasi & merename gambar ke format standar
4. 📊 Membuat `master_dataset/metadata.csv`
5. ✔️ Validasi integritas file

Output yang diharapkan:
```
╔══════════════════════════════════════════════════╗
║        StainSense — Data Pipeline v1.0           ║
╚══════════════════════════════════════════════════╝

✓ kaggle.json ditemukan di /home/user/.kaggle/kaggle.json

[Unduh varshinisandhi/fabric-dataset] ...
[Unduh osman0/dirty-clothes] ...
[Unduh priemshpathirana/fabric-stain-dataset] ...

[Normalisasi] ... gambar diproses

═══════════════════════════════════════════════════════
  ✅  PIPELINE SELESAI
═══════════════════════════════════════════════════════
  Total gambar      : X,XXX
  Total kelas       : 10
  File metadata     : master_dataset/metadata.csv
  Folder master     : master_dataset/images/
```

---

## 7. Menjalankan Aplikasi (`app.py`)

```bash
# Pastikan (venv) aktif
streamlit run app.py
```

Streamlit akan otomatis membuka browser di `http://localhost:8501`.

### Menggunakan Aplikasi:
1. **Unggah Gambar** — Gunakan tab "Unggah Gambar" atau tab "Ambil Foto" dengan kamera
2. **Lihat Deteksi** — Bounding box akan muncul otomatis di panel kanan
3. **Klik Analisis** — Tekan tombol "Analisis & Dapatkan Instruksi Pembersihan"
4. **Baca Hasil** — Instruksi step-by-step muncul dalam kartu terstruktur

### Mode Demo (tanpa API key):
Aktifkan toggle **"Mode Demo"** di sidebar untuk testing UI tanpa API key.

---

## 8. Struktur Proyek

```
stainsense/
├── app.py                  # 🎯 Entry point Streamlit (frontend + backend)
├── ai_module.py            # 🤖 Vision (OpenCV/YOLO) + LLM integration
├── data_setup.py           # 📊 Data pipeline (unduh + normalisasi dataset)
├── requirements.txt        # 📦 Daftar dependensi Python
├── .env.example            # 🔑 Template variabel lingkungan
├── .env                    # 🔐 API keys (JANGAN di-commit ke Git!)
├── .gitignore              # 🚫 File yang dikecualikan dari Git
│
├── raw_datasets/           # 📁 Dataset mentah dari Kaggle (auto-generated)
│   ├── varshinisandhi_fabric-dataset/
│   ├── osman0_dirty-clothes/
│   └── priemshpathirana_fabric-stain-dataset/
│
└── master_dataset/         # 📁 Dataset terpadu (auto-generated)
    ├── metadata.csv        # Metadata semua gambar
    └── images/             # Semua gambar ternormalisasi (224x224 JPEG)
```

---

## 9. Arsitektur & Alur Kerja

```
Pengguna
   │
   ▼
[Streamlit UI - app.py]
   │  Upload/Capture foto
   ▼
[ai_module.py → detect_stains()]
   │  OpenCV LAB color analysis + Contour detection
   │  (Opsional: YOLOv8 zero-shot)
   ▼
[Bounding Box ditampilkan]
   │
   │  Klik "Analisis"
   ▼
[ai_module.py → analyze_stain()]
   │
   ├──[provider=gemini]──→ Google Gemini Vision API
   │                        model: gemini-2.0-flash
   │
   └──[provider=openrouter]→ OpenRouter API
                             model: google/gemini-2.0-flash-thinking-exp
   │
   ▼
[JSON Response Parsing]
   {jenis_noda, jenis_kain, peringatan_bahaya, langkah_pembersihan}
   │
   ▼
[Streamlit UI → Render Cards]
   ✅ Ditampilkan ke pengguna
```

---

## 10. Troubleshooting

### ❌ `kaggle: command not found`
```bash
pip install kaggle
# Jika masih error di Linux/Mac:
export PATH="$HOME/.local/bin:$PATH"
```

### ❌ `401 - Unauthorized` saat download Kaggle
```bash
# Periksa isi file kaggle.json
cat ~/.kaggle/kaggle.json
# Periksa permission
chmod 600 ~/.kaggle/kaggle.json
# Coba login manual
kaggle datasets list
```

### ❌ `GEMINI_API_KEY not set`
```bash
# Pastikan file .env ada dan terisi
cat .env
# Pastikan python-dotenv terinstal
pip install python-dotenv
```

### ❌ `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python-headless
```

### ❌ Streamlit error `DuplicateWidgetID`
Refresh browser (`Ctrl+R`) atau restart Streamlit (`Ctrl+C` lalu jalankan ulang).

### ❌ Kamera tidak berfungsi di browser
- Chrome/Edge: Pastikan akses kamera diizinkan (ikon kunci di address bar)
- Gunakan `http://localhost:8501` bukan `http://127.0.0.1:8501`
- Firefox mungkin memerlukan HTTPS untuk akses kamera

### ❌ Dataset unduhan lambat / timeout
```bash
# Coba per-dataset manual
kaggle datasets download -d varshinisandhi/fabric-dataset --path raw_datasets/varshinisandhi_fabric-dataset --unzip
```

---

## Rencana Pengembangan (Post-MVP)

- [ ] **Fine-tuning**: Latih model klasifikasi noda pada `master_dataset/`
- [ ] **Mobile App**: React Native / Flutter dengan kamera native
- [ ] **FastAPI Backend**: Ganti Streamlit dengan REST API yang terpisah
- [ ] **Real YOLO**: Anotasi dataset dan latih YOLOv8 khusus noda
- [ ] **Multi-bahasa**: Dukungan English, Bahasa Indonesia, dsb.
- [ ] **History**: Simpan riwayat analisis pengguna

---

*Dibuat dengan ❤️ menggunakan Streamlit, OpenCV, dan Google Gemini*
