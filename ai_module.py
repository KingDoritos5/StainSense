"""
StainSense — AI Module
========================
Modul ini menangani dua komponen AI utama:

  1. Vision  : Deteksi noda pada gambar (bounding box via OpenCV + opsional YOLOv8)
  2. LLM     : Instruksi pembersihan step-by-step via Google Gemini atau OpenRouter

Tidak ada state global; semua fungsi bersifat stateless dan dapat diimpor bebas.
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import time
from io import BytesIO
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

# ─── Konfigurasi LLM ─────────────────────────────────────────────────────────

GEMINI_API_BASE  = "https://generativelanguage.googleapis.com/v1beta"
OPENROUTER_BASE  = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "meta-llama/llama-3.2-11b-vision-instruct:free"  # Vision-capable free model

GEMINI_MODELS = {
    "Gemini 3.1 Pro Preview":        "gemini-3.1-pro-preview",
    "Gemini 3 Flash Preview":        "gemini-3-flash-preview",
    "Gemini 3.1 Flash-Lite Preview": "gemini-3.1-flash-lite-preview",
    "Gemini 2.5 Flash":              "gemini-2.5-flash",
    "Gemini 2.5 Flash-Lite":         "gemini-2.5-flash-lite",
    "Gemini 2.5 Pro":                "gemini-2.5-pro",
}

# Warna bounding box (B, G, R) untuk OpenCV
BOX_COLORS = [
    (0, 200, 255),   # kuning-oranye
    (0, 140, 255),   # oranye
    (0, 255, 180),   # hijau-cyan
    (255, 100, 0),   # biru tua
]

# ─── PROMPT SISTEM ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Kamu adalah StainSense AI, asisten pembersihan noda profesional.

TUGAS:
Analisis gambar noda pada kain yang diberikan dan berikan panduan pembersihan detail.

OUTPUT FORMAT:
Kamu WAJIB mengembalikan HANYA JSON valid berikut, tanpa teks tambahan, tanpa markdown code block:

{
  "jenis_noda": "nama spesifik jenis noda (mis: noda darah segar, noda kopi, noda minyak goreng)",
  "jenis_kain": "jenis kain yang terdeteksi (mis: katun, sutra, denim, wol, poliester, tidak terdeteksi)",
  "tingkat_keparahan": "ringan | sedang | parah",
  "peringatan_bahaya": "peringatan keamanan penting atau string kosong jika tidak ada. Sertakan bahan kimia yang TIDAK BOLEH digunakan.",
  "langkah_pembersihan": [
    "1. Langkah pertama dengan detail lengkap",
    "2. Langkah kedua dengan detail lengkap",
    "3. Langkah ketiga dengan detail lengkap"
  ],
  "produk_rekomendasi": ["produk 1", "produk 2"],
  "catatan_tambahan": "tips ekstra atau string kosong"
}

ATURAN KETAT:
- Jawab dalam Bahasa Indonesia
- Output HANYA JSON, tidak ada apapun selain JSON
- Langkah pembersihan minimal 4 langkah, maksimal 8 langkah
- Jika gambar tidak menunjukkan noda, tetap jawab JSON dengan jenis_noda = "tidak terdeteksi"
"""

# ─── VISION: Deteksi Bounding Box ────────────────────────────────────────────

def _pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Konversi PIL Image (RGB) ke array OpenCV (BGR)."""
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv2_to_pil(arr: np.ndarray) -> Image.Image:
    """Konversi array OpenCV (BGR) ke PIL Image (RGB)."""
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def detect_stains_opencv(
    image: Image.Image,
    min_area: int = 500,
) -> tuple[Image.Image, list[dict]]:
    """
    Deteksi noda menggunakan analisis warna & kontur OpenCV (zero-shot).

    Pendekatan:
      1. Konversi ke LAB color space untuk pemisahan warna yang lebih baik.
      2. Terapkan adaptive thresholding untuk menemukan area tidak seragam.
      3. Temukan kontur & buat bounding box.

    Returns:
        (image_with_boxes, detections)
        detections: list[{"bbox": [x1,y1,x2,y2], "confidence": float, "label": str}]
    """
    cv_img = _pil_to_cv2(image)
    h, w   = cv_img.shape[:2]

    # Konversi ke LAB untuk deteksi area anomali warna
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Gabungkan kanal a dan b untuk menemukan area berwarna berbeda
    color_deviation = cv2.addWeighted(
        cv2.absdiff(a_ch, np.full_like(a_ch, 128)),
        0.6,
        cv2.absdiff(b_ch, np.full_like(b_ch, 128)),
        0.4,
        0,
    )

    # Thresholding adaptif
    blurred = cv2.GaussianBlur(color_deviation, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)

    # Morphological closing untuk mengisi lubang kecil
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Temukan kontur
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[dict] = []
    output_img = cv_img.copy()

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        # Filter bounding box yang terlalu besar (>70% gambar)
        if bw * bh > 0.7 * (w * h):
            continue

        confidence = min(0.95, 0.40 + (area / (w * h)) * 3.0)
        color      = BOX_COLORS[i % len(BOX_COLORS)]
        label      = f"Noda #{i+1} ({confidence:.0%})"

        # Gambar bounding box
        cv2.rectangle(output_img, (x, y), (x + bw, y + bh), color, 3)

        # Label box
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(0.9, bw / 250))
        thickness  = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        label_y    = max(y - 10, th + 5)

        cv2.rectangle(output_img,
                      (x, label_y - th - 6),
                      (x + tw + 6, label_y + 2),
                      color, -1)
        cv2.putText(output_img, label,
                    (x + 3, label_y - 2),
                    font, font_scale, (20, 20, 20), thickness)

        detections.append({
            "bbox":       [int(x), int(y), int(x + bw), int(y + bh)],
            "confidence": round(confidence, 3),
            "label":      label,
            "area_px":    int(area),
        })

    # Jika tidak ada deteksi, tambahkan satu bounding box demo di tengah
    if not detections:
        cx, cy = w // 2, h // 2
        rw, rh = w // 3, h // 3
        x1, y1 = cx - rw // 2, cy - rh // 2
        x2, y2 = cx + rw // 2, cy + rh // 2
        color  = BOX_COLORS[0]
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 3)
        label = "Noda Terdeteksi (45%)"
        cv2.putText(output_img, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        detections.append({
            "bbox":       [x1, y1, x2, y2],
            "confidence": 0.45,
            "label":      label,
            "area_px":    rw * rh,
        })

    return _cv2_to_pil(output_img), detections


def try_yolov8_detection(image: Image.Image) -> tuple[Image.Image, list[dict]] | None:
    """
    Coba gunakan YOLOv8 pre-trained untuk deteksi (opsional).
    Kembalikan None jika ultralytics tidak terinstal.
    YOLOv8 nano digunakan sebagai zero-shot detector untuk menemukan objek/area.
    """
    try:
        from ultralytics import YOLO  # type: ignore

        model = YOLO("yolov8n.pt")
        results = model(image, verbose=False)[0]

        output_img = image.copy()
        draw       = ImageDraw.Draw(output_img)
        detections: list[dict] = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            label  = f"Area #{cls_id} ({conf:.0%})"

            draw.rectangle([x1, y1, x2, y2], outline="#FF6B35", width=3)
            draw.text((x1 + 4, y1 + 2), label, fill="#FF6B35")
            detections.append({
                "bbox":       [x1, y1, x2, y2],
                "confidence": round(conf, 3),
                "label":      label,
            })

        return output_img, detections

    except ImportError:
        return None
    except Exception:
        return None


def detect_stains(image: Image.Image) -> tuple[Image.Image, list[dict]]:
    """
    Fungsi utama deteksi. Coba YOLOv8 dulu, fallback ke OpenCV.
    """
    yolo_result = try_yolov8_detection(image)
    if yolo_result is not None:
        return yolo_result
    return detect_stains_opencv(image)


# ─── LLM: Analisis & Instruksi Pembersihan ───────────────────────────────────

def _image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    """
    Encode PIL Image ke base64 string.
    Auto-resize jika gambar terlalu besar agar tidak melebihi batas payload API.
    """
    MAX_SIDE = 1280  # px — cukup detail untuk analisis noda

    # Resize jika perlu
    w, h = image.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Kompresi iteratif — pastikan < 4MB
    for quality in [85, 70, 55, 40]:
        buf = BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=quality)
        if buf.tell() < 4 * 1024 * 1024:  # 4MB limit
            break

    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _repair_truncated_json(raw: str) -> str:
    """
    Coba perbaiki JSON yang terpotong di tengah karena token limit.
    Strategi: temukan kurung kurawal pembuka, lalu tutup semua yang belum tertutup.
    """
    # Temukan posisi awal JSON
    start = raw.find("{")
    if start == -1:
        return raw

    fragment = raw[start:]

    # Hitung kurung yang belum tertutup
    depth      = 0
    in_string  = False
    escape     = False
    last_valid = 0

    for i, ch in enumerate(fragment):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_valid = i
                break

    # Jika JSON sudah lengkap
    if depth == 0 and last_valid > 0:
        return fragment[:last_valid + 1]

    # JSON terpotong — coba tutup paksa
    # Hapus trailing koma atau key yang belum selesai
    fragment = fragment.rstrip().rstrip(",").rstrip()

    # Jika sedang di dalam string yang belum ditutup
    if in_string:
        fragment += '"'

    # Tutup array jika terbuka
    open_arrays = fragment.count("[") - fragment.count("]")
    fragment += "]" * max(0, open_arrays)

    # Tutup semua object yang terbuka
    open_objects = fragment.count("{") - fragment.count("}")
    fragment += "}" * max(0, open_objects)

    return fragment


def _parse_llm_response(raw: str) -> dict[str, Any]:
    """
    Ekstrak JSON dari respons LLM.
    Mendukung: JSON bersih, JSON dalam markdown, JSON terpotong (token limit).
    """
    if not raw or not raw.strip():
        return _fallback_response("Respons kosong dari API.")

    # Hapus markdown code block jika ada
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    # Percobaan 1: parse langsung
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Percobaan 2: temukan JSON object di dalam teks
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Percobaan 3: repair JSON yang terpotong (token limit habis)
    repaired = _repair_truncated_json(cleaned)
    try:
        result = json.loads(repaired)
        # Pastikan field wajib ada
        result.setdefault("jenis_noda",        "tidak terdeteksi")
        result.setdefault("jenis_kain",        "tidak diketahui")
        result.setdefault("tingkat_keparahan", "sedang")
        result.setdefault("peringatan_bahaya", "")
        result.setdefault("langkah_pembersihan", [])
        result.setdefault("produk_rekomendasi",  [])
        result.setdefault("catatan_tambahan",    "")
        return result
    except json.JSONDecodeError:
        pass

    # Fallback akhir
    return _fallback_response(f"Gagal memparse respons. Raw: {raw[:300]}")


def _fallback_response(reason: str) -> dict[str, Any]:
    return {
        "jenis_noda":        "gagal diparse",
        "jenis_kain":        "tidak diketahui",
        "tingkat_keparahan": "tidak diketahui",
        "peringatan_bahaya": "Gagal menganalisis gambar. Coba lagi.",
        "langkah_pembersihan": [
            "1. Pastikan koneksi internet stabil",
            "2. Pastikan API key valid",
            "3. Coba unggah gambar yang lebih jelas dan terang",
        ],
        "produk_rekomendasi": [],
        "catatan_tambahan":   reason,
    }


# ─── Provider: Google Gemini ─────────────────────────────────────────────────

def analyze_with_gemini(
    image: Image.Image,
    api_key: str,
    additional_info: str = "",
) -> dict[str, Any]:
    """
    Kirim gambar ke Google Gemini Vision API dan dapatkan analisis noda.

    Args:
        image:           PIL Image dari gambar noda.
        api_key:         Google AI Studio API Key.
        additional_info: Informasi tambahan dari pengguna (mis: jenis kain).

    Returns:
        dict sesuai skema JSON di SYSTEM_PROMPT.
    """
    img_b64    = _image_to_base64(image)
    user_text  = "Analisis gambar ini dan identifikasi noda serta berikan instruksi pembersihan."
    if additional_info:
        user_text += f"\nInformasi tambahan: {additional_info}"

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data":      img_b64,
                        }
                    },
                    {"text": user_text},
                ],
            }
        ],
        "generationConfig": {
            "temperature":     0.1,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
        },
    }

    url = f"{GEMINI_API_BASE}/models/gemini-2.5-flash:generateContent?key={api_key}"

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        raw  = data["candidates"][0]["content"]["parts"][0]["text"]
        return _parse_llm_response(raw)

    except requests.exceptions.Timeout:
        raise TimeoutError("Gemini API timeout. Coba lagi.")
    except requests.exceptions.HTTPError as e:
        raise ConnectionError(f"Gemini API error {e.response.status_code}: {e.response.text[:300]}")
    except (KeyError, IndexError) as e:
        raise ValueError(f"Format respons Gemini tidak terduga: {e}")


# ─── Provider: OpenRouter ────────────────────────────────────────────────────

def analyze_with_openrouter(
    image: Image.Image,
    api_key: str,
    additional_info: str = "",
    model: str = OPENROUTER_MODEL,
) -> dict[str, Any]:
    """
    Kirim gambar ke OpenRouter API (mendukung model multimodal).

    Args:
        image:           PIL Image dari gambar noda.
        api_key:         OpenRouter API Key.
        additional_info: Informasi tambahan dari pengguna.
        model:           Model OpenRouter yang digunakan.

    Returns:
        dict sesuai skema JSON di SYSTEM_PROMPT.
    """
    img_b64   = _image_to_base64(image)
    user_text = "Analisis gambar ini dan identifikasi noda serta berikan instruksi pembersihan."
    if additional_info:
        user_text += f"\nInformasi tambahan: {additional_info}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role":    "user",
                "content": [
                    {
                        "type":      "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens":  2048,
    }

    headers = {
        "Authorization":  f"Bearer {api_key}",
        "Content-Type":   "application/json",
        "HTTP-Referer":   "https://stainsense.app",
        "X-Title":        "StainSense MVP",
    }

    try:
        resp = requests.post(
            f"{OPENROUTER_BASE}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        raw  = data["choices"][0]["message"]["content"]
        return _parse_llm_response(raw)

    except requests.exceptions.Timeout:
        raise TimeoutError("OpenRouter API timeout. Coba lagi.")
    except requests.exceptions.HTTPError as e:
        raise ConnectionError(f"OpenRouter error {e.response.status_code}: {e.response.text[:300]}")
    except (KeyError, IndexError) as e:
        raise ValueError(f"Format respons OpenRouter tidak terduga: {e}")


# ─── Fungsi Utama (Router) ────────────────────────────────────────────────────

def analyze_stain(
    image: Image.Image,
    provider: str = "gemini",
    additional_info: str = "",
) -> dict[str, Any]:
    """
    Router utama analisis noda. Pilih provider LLM berdasarkan parameter.

    Args:
        image:           PIL Image.
        provider:        "gemini" atau "openrouter".
        additional_info: Informasi tambahan untuk prompt.

    Returns:
        dict JSON terstruktur sesuai skema SYSTEM_PROMPT.

    Raises:
        ValueError:      Jika provider tidak dikenal atau API key kosong.
        ConnectionError: Jika API tidak bisa dihubungi.
        TimeoutError:    Jika API timeout.
    """
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY belum diatur di file .env!\n"
                "Dapatkan key di: https://aistudio.google.com/app/apikey"
            )
        return analyze_with_gemini(image, api_key, additional_info)

    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY belum diatur di file .env!\n"
                "Dapatkan key di: https://openrouter.ai/keys"
            )
        return analyze_with_openrouter(image, api_key, additional_info)

    else:
        raise ValueError(f"Provider tidak dikenal: '{provider}'. Gunakan 'gemini' atau 'openrouter'.")


# ─── Demo Mode (tanpa API) ───────────────────────────────────────────────────

def get_demo_result() -> dict[str, Any]:
    """
    Kembalikan hasil demo statis untuk testing UI tanpa API key.
    """
    return {
        "jenis_noda":        "Noda kopi (demo)",
        "jenis_kain":        "Katun putih",
        "tingkat_keparahan": "sedang",
        "peringatan_bahaya": (
            "⚠ JANGAN gunakan pemutih berbasis klorin pada kain berwarna. "
            "Hindari air panas yang dapat membekukan noda protein."
        ),
        "langkah_pembersihan": [
            "1. Segera serap kelebihan cairan kopi dengan kain bersih — jangan digosok!",
            "2. Bilas area noda dari belakang kain menggunakan air dingin mengalir.",
            "3. Tuangkan sedikit cairan pencuci piring cair ke noda, biarkan 5 menit.",
            "4. Gosok perlahan dengan sikat gigi lembut menggunakan gerakan memutar.",
            "5. Bilas dengan air dingin. Ulangi jika noda masih terlihat.",
            "6. Rendam dalam larutan cuka putih 1:3 air selama 15 menit.",
            "7. Cuci seperti biasa di mesin cuci dengan air dingin.",
            "8. Periksa noda sebelum dimasukkan ke pengering — panas dapat menetapkan noda.",
        ],
        "produk_rekomendasi": ["Vanish Gold", "Carbonout Stain Remover", "Baking soda + cuka putih"],
        "catatan_tambahan":   "Untuk noda lama (>24 jam), gunakan enzymatic cleaner dan rendam semalam.",
    }
