"""
StainSense — AI Module
========================
Handles two AI components:
  1. Vision  : OpenCV bounding box detection (visual highlight only)
  2. LLM     : Stain analysis via Google Gemini or OpenRouter

No TensorFlow. No demo mode. Stateless functions throughout.
"""

from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image

# ─── LLM Config ──────────────────────────────────────────────────────────────

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Reliable vision-capable models on OpenRouter (free tier)
OPENROUTER_MODELS = {
    "Llama 3.2 Vision 11B (Free)":   "meta-llama/llama-3.2-11b-vision-instruct:free",
    "Qwen2 VL 7B (Free)":            "qwen/qwen2-vl-7b-instruct:free",
    "Gemini 2.0 Flash Exp (Free)":   "google/gemini-2.0-flash-exp:free",
}
OPENROUTER_MODEL_DEFAULT = "meta-llama/llama-3.2-11b-vision-instruct:free"

# Gemini models (confirmed working)
GEMINI_MODELS = {
    "Gemini 2.5 Flash (Recommended)": "gemini-2.5-flash",
    "Gemini 2.5 Pro":                 "gemini-2.5-pro",
    "Gemini 2.0 Flash":               "gemini-2.0-flash",
}
GEMINI_MODEL_DEFAULT = "gemini-2.5-flash"

# OpenCV bounding box colors (B, G, R)
BOX_COLORS = [
    (0, 200, 255),
    (0, 140, 255),
    (0, 255, 180),
    (255, 100, 0),
]

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Kamu adalah StainSense AI, asisten pembersihan noda profesional.

TUGAS:
Analisis gambar noda pada kain. Jika ada gambar kedua (label/tag pakaian), gunakan
informasi komposisi kain dari tag tersebut untuk meningkatkan akurasi instruksi.

OUTPUT FORMAT:
Kembalikan HANYA JSON valid berikut — tanpa teks tambahan, tanpa markdown code block:

{
  "jenis_noda": "nama spesifik noda (mis: noda kopi, noda darah segar, noda minyak goreng)",
  "jenis_kain": "jenis kain (mis: katun, sutra, denim, wol, poliester, tidak terdeteksi)",
  "komposisi_kain": "jika label terbaca: mis '100% Cotton' — kosongkan jika tidak ada label",
  "tingkat_keparahan": "ringan | sedang | parah",
  "peringatan_bahaya": "peringatan WAJIB tentang bahan kimia yang tidak boleh digunakan, atau kosong jika tidak ada",
  "langkah_pembersihan": [
    "1. Langkah pertama lengkap",
    "2. Langkah kedua lengkap",
    "3. dst"
  ],
  "produk_rekomendasi": ["produk 1", "produk 2"],
  "catatan_tambahan": "tips ekstra atau string kosong"
}

ATURAN:
- Jawab dalam Bahasa Indonesia
- Output HANYA JSON, tidak ada teks lain
- Langkah pembersihan: minimal 4, maksimal 8 langkah
- Jika tidak ada noda: jenis_noda = "tidak terdeteksi"
- Jika ada label tag pakaian di gambar kedua: baca komposisi dan sesuaikan instruksi
"""

# ─── Image Utilities ──────────────────────────────────────────────────────────

def _compress_image(image: Image.Image, max_side: int = 1280) -> Image.Image:
    """
    Resize gambar agar sisi terpanjang <= max_side px.
    Menjaga aspect ratio.
    """
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


def _image_to_base64(image: Image.Image) -> str:
    """
    Encode PIL Image → base64 JPEG string.
    Kompresi iteratif untuk menjaga payload < 4MB.
    """
    image = _compress_image(image)
    for quality in [85, 72, 58, 42]:
        buf = BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=quality)
        if buf.tell() < 4 * 1024 * 1024:
            break
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── JSON Parsing & Repair ────────────────────────────────────────────────────

def _repair_json(raw: str) -> str:
    """
    Perbaiki JSON yang terpotong akibat token limit.
    Menutup kurung yang belum tertutup secara paksa.
    """
    start = raw.find("{")
    if start == -1:
        return raw
    frag = raw[start:]

    depth, in_str, escape, last_closed = 0, False, False, 0
    for i, ch in enumerate(frag):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_closed = i
                break

    if depth == 0 and last_closed:
        return frag[:last_closed + 1]

    # Paksa tutup
    frag = frag.rstrip().rstrip(",").rstrip()
    if in_str:
        frag += '"'
    frag += "]" * max(0, frag.count("[") - frag.count("]"))
    frag += "}" * max(0, frag.count("{") - frag.count("}"))
    return frag


def _parse_llm_response(raw: str) -> dict[str, Any]:
    """
    Parse JSON dari respons LLM.
    Handles: clean JSON, JSON in markdown, truncated JSON.
    """
    if not raw or not raw.strip():
        return _fallback("Respons kosong dari API.")

    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    # Try 1: direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try 2: extract JSON object from surrounding text
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try 3: repair truncated JSON
    try:
        repaired = _repair_json(cleaned)
        result   = json.loads(repaired)
        result.setdefault("jenis_noda",         "tidak terdeteksi")
        result.setdefault("jenis_kain",          "tidak diketahui")
        result.setdefault("komposisi_kain",       "")
        result.setdefault("tingkat_keparahan",    "sedang")
        result.setdefault("peringatan_bahaya",    "")
        result.setdefault("langkah_pembersihan",  [])
        result.setdefault("produk_rekomendasi",   [])
        result.setdefault("catatan_tambahan",     "")
        return result
    except json.JSONDecodeError:
        pass

    return _fallback(f"Gagal parse respons. Cuplikan: {raw[:200]}")


def _fallback(reason: str) -> dict[str, Any]:
    return {
        "jenis_noda":         "gagal diparse",
        "jenis_kain":         "tidak diketahui",
        "komposisi_kain":     "",
        "tingkat_keparahan":  "tidak diketahui",
        "peringatan_bahaya":  "Gagal menganalisis. Coba lagi.",
        "langkah_pembersihan": [
            "1. Pastikan koneksi internet stabil",
            "2. Pastikan API key valid dan belum habis kuota",
            "3. Coba unggah gambar yang lebih terang dan jelas",
            "4. Coba ganti provider AI di sidebar",
        ],
        "produk_rekomendasi": [],
        "catatan_tambahan":   reason,
    }


# ─── OpenCV: Bounding Box Detection ──────────────────────────────────────────

def _pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def _cv2_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def detect_stains(
    image: Image.Image,
    min_area: int = 500,
) -> tuple[Image.Image, list[dict]]:
    """
    Deteksi area noda via LAB color analysis + contour detection.
    Mengembalikan (gambar_dengan_bbox, list_deteksi).

    Catatan: ini adalah highlighting visual saja.
    Identifikasi jenis noda dilakukan oleh LLM, bukan OpenCV.
    """
    cv_img = _pil_to_cv2(image)
    h, w   = cv_img.shape[:2]

    lab   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    _, a, b = cv2.split(lab)

    deviation = cv2.addWeighted(
        cv2.absdiff(a, np.full_like(a, 128)), 0.6,
        cv2.absdiff(b, np.full_like(b, 128)), 0.4, 0,
    )
    blurred = cv2.GaussianBlur(deviation, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output      = cv_img.copy()
    detections: list[dict] = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh > 0.70 * w * h:
            continue

        conf  = min(0.95, 0.40 + (area / (w * h)) * 3.0)
        color = BOX_COLORS[i % len(BOX_COLORS)]
        label = f"Noda #{i+1} ({conf:.0%})"

        cv2.rectangle(output, (x, y), (x + bw, y + bh), color, 3)
        fs = max(0.5, min(0.9, bw / 250))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        ly = max(y - 10, th + 5)
        cv2.rectangle(output, (x, ly - th - 6), (x + tw + 6, ly + 2), color, -1)
        cv2.putText(output, label, (x + 3, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), 2)

        detections.append({
            "bbox":       [int(x), int(y), int(x + bw), int(y + bh)],
            "confidence": round(conf, 3),
            "label":      label,
        })

    if not detections:
        cx, cy = w // 2, h // 2
        rw, rh = w // 3, h // 3
        x1, y1 = cx - rw // 2, cy - rh // 2
        x2, y2 = cx + rw // 2, cy + rh // 2
        cv2.rectangle(output, (x1, y1), (x2, y2), BOX_COLORS[0], 3)
        cv2.putText(output, "Area Terdeteksi",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BOX_COLORS[0], 2)
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.40,
            "label": "Area Terdeteksi",
        })

    return _cv2_to_pil(output), detections


# ─── Gemini Provider ──────────────────────────────────────────────────────────

def analyze_with_gemini(
    image: Image.Image,
    api_key: str,
    additional_info: str = "",
    tag_image: Image.Image | None = None,
    model: str = GEMINI_MODEL_DEFAULT,
) -> dict[str, Any]:
    """
    Send stain image (+ optional clothing tag image) to Google Gemini.
    Tag image is sent as a second image so Gemini can read fabric composition directly.
    """
    stain_b64 = _image_to_base64(image)

    # Build parts: stain image always first
    parts: list[dict] = [
        {"inline_data": {"mime_type": "image/jpeg", "data": stain_b64}},
        {"text": "Gambar 1: Foto noda pada kain."},
    ]

    # Append tag image if provided
    if tag_image is not None:
        tag_b64 = _image_to_base64(tag_image)
        parts += [
            {"inline_data": {"mime_type": "image/jpeg", "data": tag_b64}},
            {"text": "Gambar 2: Label/tag pakaian. Baca komposisi kain dan gunakan untuk instruksi."},
        ]

    user_text = "Analisis noda dan berikan instruksi pembersihan lengkap."
    if additional_info:
        user_text += f"\nKonteks tambahan dari pengguna: {additional_info}"
    parts.append({"text": user_text})

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature":      0.1,
            "maxOutputTokens":  2048,
            "responseMimeType": "application/json",
        },
    }

    url = f"{GEMINI_API_BASE}/models/{model}:generateContent?key={api_key}"

    try:
        resp = requests.post(url, json=payload, timeout=90)
        resp.raise_for_status()
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return _parse_llm_response(raw)
    except requests.exceptions.Timeout:
        raise TimeoutError("Gemini API timeout. Coba lagi.")
    except requests.exceptions.HTTPError as e:
        raise ConnectionError(f"Gemini API error {e.response.status_code}: {e.response.text[:300]}")
    except (KeyError, IndexError) as e:
        raise ValueError(f"Format respons Gemini tidak terduga: {e}")


# ─── OpenRouter Provider ──────────────────────────────────────────────────────

def analyze_with_openrouter(
    image: Image.Image,
    api_key: str,
    additional_info: str = "",
    tag_image: Image.Image | None = None,
    model: str = OPENROUTER_MODEL_DEFAULT,
) -> dict[str, Any]:
    """
    Send stain image (+ optional tag image) to OpenRouter vision model.
    """
    stain_b64 = _image_to_base64(image)

    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{stain_b64}"}},
        {"type": "text",      "text": "Gambar 1: Foto noda pada kain."},
    ]

    if tag_image is not None:
        tag_b64 = _image_to_base64(tag_image)
        content += [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{tag_b64}"}},
            {"type": "text",      "text": "Gambar 2: Label/tag pakaian — baca komposisi kain."},
        ]

    user_text = "Analisis noda dan berikan instruksi pembersihan lengkap."
    if additional_info:
        user_text += f"\nKonteks tambahan: {additional_info}"
    content.append({"type": "text", "text": user_text})

    payload = {
        "model":    model,
        "messages": [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": content},
        ],
        "temperature": 0.1,
        "max_tokens":  2048,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://stainsense.app",
        "X-Title":       "StainSense",
    }

    try:
        resp = requests.post(
            f"{OPENROUTER_BASE}/chat/completions",
            json=payload, headers=headers, timeout=90,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        return _parse_llm_response(raw)
    except requests.exceptions.Timeout:
        raise TimeoutError("OpenRouter timeout. Coba lagi.")
    except requests.exceptions.HTTPError as e:
        raise ConnectionError(f"OpenRouter error {e.response.status_code}: {e.response.text[:300]}")
    except (KeyError, IndexError) as e:
        raise ValueError(f"Format respons OpenRouter tidak terduga: {e}")


# ─── Main Router ──────────────────────────────────────────────────────────────

def analyze_stain(
    image: Image.Image,
    provider: str = "gemini",
    additional_info: str = "",
    tag_image: Image.Image | None = None,
    gemini_model: str = GEMINI_MODEL_DEFAULT,
    openrouter_model: str = OPENROUTER_MODEL_DEFAULT,
) -> dict[str, Any]:
    """
    Route analysis to selected LLM provider.

    Args:
        image:            Stain photo (PIL Image)
        provider:         "gemini" or "openrouter"
        additional_info:  Extra context from user
        tag_image:        Optional clothing tag photo for fabric identification
        gemini_model:     Which Gemini model to use
        openrouter_model: Which OpenRouter model to use
    """
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "GEMINI_API_KEY belum diset di .env\n"
                "→ https://aistudio.google.com/app/apikey"
            )
        return analyze_with_gemini(image, key, additional_info, tag_image, gemini_model)

    elif provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "OPENROUTER_API_KEY belum diset di .env\n"
                "→ https://openrouter.ai/keys"
            )
        return analyze_with_openrouter(image, key, additional_info, tag_image, openrouter_model)

    else:
        raise ValueError(f"Provider tidak dikenal: '{provider}'. Pilih 'gemini' atau 'openrouter'.")
