# 🧹 StainSense — Panduan Lengkap

**AI-Powered Fabric Stain Identifier** — Upload foto noda → AI deteksi & berikan instruksi pembersihan step-by-step.

---

## Daftar Isi
1. [Tentang Aplikasi](#tentang-aplikasi)
2. [Cara Kerja](#cara-kerja)
3. [Fitur](#fitur)
4. [Setup Lokal](#setup-lokal)
5. [Konfigurasi API Key](#konfigurasi-api-key)
6. [Menjalankan Aplikasi](#menjalankan-aplikasi)
7. [Deploy ke Railway](#deploy-ke-railway)
8. [Troubleshooting](#troubleshooting)
9. [Struktur File](#struktur-file)

---

## Tentang Aplikasi

StainSense adalah aplikasi web berbasis Streamlit yang mengidentifikasi jenis noda pada kain dari foto, lalu memberikan instruksi pembersihan yang aman dan terperinci menggunakan AI multimodal (Google Gemini atau OpenRouter).

**Stack:**
- **Frontend & Backend**: Streamlit
- **Vision (Bounding Box)**: OpenCV — mendeteksi area noda secara visual
- **AI Analysis**: Google Gemini / OpenRouter — mengidentifikasi jenis noda, jenis kain, dan membuat instruksi
- **Tag Scanner**: AI membaca foto label/tag pakaian untuk mengenali komposisi kain otomatis

---

## Cara Kerja

```
1. User upload/foto gambar noda
         ↓
2. OpenCV → LAB color analysis → Bounding box ditampilkan
         ↓
3. (Opsional) User upload foto label/tag pakaian
         ↓
4. Gambar noda + gambar tag dikirim ke Gemini / OpenRouter API
         ↓
5. AI menganalisis: jenis noda, jenis kain, tingkat keparahan
         ↓
6. Output JSON → Ditampilkan sebagai kartu langkah-langkah pembersihan
```

**Mengapa ada dua komponen (OpenCV + LLM)?**
- **OpenCV**: Memberikan bounding box visual sebelum analisis — pengguna bisa melihat area yang terdeteksi sebelum tombol analisis diklik. Cepat (< 100ms), tidak butuh internet.
- **LLM**: Melakukan analisis semantik sesungguhnya — mengenali jenis noda, membaca label kain, dan menghasilkan instruksi berbahasa Indonesia.

---

## Fitur

| Fitur | Keterangan |
|---|---|
| Upload gambar | JPG, PNG, WebP hingga 20MB |
| Kamera langsung | Foto noda via kamera browser |
| Bounding box | Highlight area noda dengan OpenCV |
| Tag scanner | Upload foto label pakaian → AI baca komposisi kain otomatis |
| Dual provider | Pilih Google Gemini atau OpenRouter |
| Pilih model | Dropdown untuk memilih model spesifik tiap provider |
| Info tambahan | Field untuk keterangan tambahan ke AI |
| Dark mode UI | Antarmuka profesional dark mode |
| JSON viewer | Lihat raw response AI untuk debugging |

---

## Setup Lokal

### Prasyarat
- Python 3.10 atau lebih baru
- pip 23+

### Langkah 1 — Clone / Download Project

```bash
# Jika menggunakan Git
git clone https://github.com/username/stainsense.git
cd stainsense

# Atau: download ZIP → extract → cd ke folder
```

### Langkah 2 — Buat Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
# Prompt berubah menjadi (venv)
```

**Windows CMD:**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Windows PowerShell:**
```powershell
# Jika muncul error "execution policy":
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Langkah 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verifikasi:
```bash
python -c "import streamlit, cv2, PIL; print('✅ Semua OK')"
```

---

## Konfigurasi API Key

### Buat File `.env`

```bash
cp .env.example .env
```

Buka file `.env` dan isi dengan API key asli:

```env
GEMINI_API_KEY=AIzaSy_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> ⚠️ **PENTING:** Jangan pernah commit file `.env` ke Git. File `.gitignore` sudah mengecualikannya.

### Mendapatkan Gemini API Key

1. Buka [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Login dengan akun Google
3. Klik **"Create API Key"**
4. Salin key → paste ke `.env`

**Quota gratis Gemini 2.5 Flash:** 500 request/hari, 10 request/menit

### Mendapatkan OpenRouter API Key

1. Daftar/login di [https://openrouter.ai](https://openrouter.ai)
2. Buka menu **Keys** → klik **"Create Key"**
3. Salin key → paste ke `.env`

**Model gratis OpenRouter yang direkomendasikan:**
- `meta-llama/llama-3.2-11b-vision-instruct:free` — vision, gratis
- `qwen/qwen2-vl-7b-instruct:free` — vision, gratis
- `google/gemini-2.0-flash-exp:free` — vision, gratis (jika tersedia)

---

## Menjalankan Aplikasi

```bash
# Pastikan (venv) aktif dan .env sudah terisi
streamlit run app.py
```

Browser akan otomatis terbuka di `http://localhost:8501`.

### Cara Menggunakan

1. **Upload gambar noda** di tab "Unggah Gambar" atau gunakan tab "Ambil Foto"
2. *(Opsional)* **Upload foto label/tag pakaian** di sidebar kiri untuk deteksi kain otomatis
3. Pilih **Provider AI** (Gemini atau OpenRouter) dan model yang diinginkan
4. Tambahkan **keterangan tambahan** jika perlu (mis: "noda sudah 2 hari")
5. Klik **"🧹 Analisis & Dapatkan Instruksi Pembersihan"**
6. Hasil muncul: jenis noda, jenis kain, peringatan, dan langkah pembersihan step-by-step

---

## Deploy ke Railway

### Prasyarat
- Akun [Railway](https://railway.app)
- Repository di GitHub (push semua file kecuali `.env`)

### File yang Harus Ada di GitHub

```
stainsense/
├── app.py
├── ai_module.py
├── requirements.txt
├── Procfile
├── railway.toml
└── .gitignore
```

> ⚠️ **JANGAN push:** `.env`, `__pycache__/`, `venv/`

### Langkah Deploy

**1. Push ke GitHub:**
```bash
git init
git add app.py ai_module.py requirements.txt Procfile railway.toml .gitignore
git commit -m "feat: StainSense initial deploy"
git remote add origin https://github.com/username/stainsense.git
git push -u origin main
```

**2. Buat project di Railway:**
- Buka [railway.app](https://railway.app) → **New Project**
- Pilih **"Deploy from GitHub repo"**
- Pilih repo `stainsense`

**3. Set environment variables di Railway:**
- Klik project → tab **Variables**
- Tambahkan:
  ```
  GEMINI_API_KEY      = AIzaSy_xxx...
  OPENROUTER_API_KEY  = sk-or-v1-xxx...
  ```

**4. Deploy:**
Railway otomatis build dan deploy. Tunggu 2-3 menit → URL publik tersedia.

> ℹ️ **Port:** File `Procfile` dan `railway.toml` sudah dikonfigurasi untuk port `8080` sesuai Railway.

---

## Troubleshooting

### Error: `Gemini API error 429`
Quota harian/menit habis. Solusi:
- Tunggu 1 menit (quota per menit reset)
- Atau ganti provider ke OpenRouter di sidebar

### Error: `OpenRouter error 400 - model not valid`
Model ID tidak valid atau sudah tidak tersedia. Solusi:
- Pilih model lain di dropdown sidebar
- Cek model terbaru di [openrouter.ai/models](https://openrouter.ai/models?q=free)

### Error: `OpenRouter error 401 - User not found`
API key tidak valid. Solusi:
- Buka `.env` — pastikan tidak ada tanda kutip atau spasi
- Buat key baru di [openrouter.ai/keys](https://openrouter.ai/keys)
- Restart Streamlit setelah update `.env`

### Error: `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python-headless
```

### Aplikasi Railway: `Application failed to respond`
Pastikan `Procfile` berisi port `8080`:
```
web: streamlit run app.py --server.port=8080 --server.address=0.0.0.0 ...
```

### Hasil "gagal diparse"
JSON dari AI terpotong. Solusi:
- Coba lagi (koneksi tidak stabil)
- Ganti ke model lain
- Upload gambar yang lebih kecil/jelas

### Kamera tidak berfungsi
- Gunakan Chrome atau Edge
- Pastikan akses kamera diizinkan (ikon kunci di address bar)
- Gunakan `http://localhost:8501` (bukan `127.0.0.1`)

---

## Struktur File

```
stainsense/
├── app.py              → Frontend Streamlit + sidebar + UI rendering
├── ai_module.py        → OpenCV detection + Gemini/OpenRouter integration
├── requirements.txt    → Python dependencies
├── Procfile            → Railway start command
├── railway.toml        → Railway deployment config
├── .env.example        → Template API key (safe to commit)
├── .env                → API key asli (JANGAN commit!)
└── .gitignore          → File yang dikecualikan dari Git
```

---

## Catatan Teknis

### Mengapa OpenCV + LLM (bukan hanya LLM)?
OpenCV memberikan **bounding box visual** yang muncul instan (< 100ms) sebelum pengguna menekan tombol analisis. Ini memberi umpan balik visual bahwa sistem sudah mendeteksi sesuatu, dan meningkatkan UX secara signifikan. Identifikasi jenis noda tetap dilakukan oleh LLM.

### Mengapa JSON terkadang perlu di-repair?
LLM kadang menghentikan output di tengah JSON jika respons panjang melebihi `max_tokens`. `ai_module.py` memiliki fungsi `_repair_json()` yang secara otomatis menutup kurung yang belum tertutup dan mengisi field yang hilang dengan nilai default.

### Fitur Tag Scanner
Foto label pakaian dikirim sebagai **gambar kedua** ke LLM (Gemini dan OpenRouter mendukung multi-image). AI langsung membaca teks komposisi kain (mis: "95% Cotton, 5% Elastane") dari foto tanpa perlu library OCR terpisah.

---

*StainSense — Software Engineering Assessment of Learning*
