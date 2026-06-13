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
    "Gemini 3.5 Flash (Recommended)": "gemini-3.5-flash",
    "Gemini 2.5 Flash":               "gemini-2.5-flash",
    "Gemini 2.5 Pro":                 "gemini-2.5-pro",
    "Gemini 2.0 Flash":               "gemini-2.0-flash",
}
GEMINI_MODEL_DEFAULT = "gemini-3.5-flash"

# OpenCV bounding box colors (B, G, R)
BOX_COLORS = [
    (0, 200, 255),
    (0, 140, 255),
    (0, 255, 180),
    (255, 100, 0),
]

# ─── JSON Schemas for Gemini Structured Output ───────────────────────────────

VERIFY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_stain": {"type": "BOOLEAN"},
        "confidence": {"type": "NUMBER"},
        "reason": {"type": "STRING"},
        "detected_object": {"type": "STRING"}
    },
    "required": ["is_stain", "confidence", "reason", "detected_object"]
}

SYSTEM_PROMPT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "noda": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "jenis_noda": {"type": "STRING"},
                    "lokasi_noda": {"type": "STRING"},
                    "jenis_kain": {"type": "STRING"},
                    "komposisi_kain": {"type": "STRING"},
                    "tingkat_keparahan": {"type": "STRING"},
                    "peringatan_bahaya": {"type": "STRING"},
                    "langkah_pembersihan": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    },
                    "produk_rekomendasi": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    },
                    "catatan_tambahan": {"type": "STRING"}
                },
                "required": [
                    "jenis_noda", "lokasi_noda", "jenis_kain",
                    "komposisi_kain", "tingkat_keparahan", "peringatan_bahaya",
                    "langkah_pembersihan", "produk_rekomendasi", "catatan_tambahan"
                ]
            }
        }
    },
    "required": ["noda"]
}

# ─── System Prompts ───────────────────────────────────────────────────────────

VERIFY_PROMPT = """Kamu adalah StainSense AI, validator noda kain.

TUGAS:
Periksa apakah gambar yang diberikan benar-benar menunjukkan NODA pada KAIN/PAKAIAN.
Analisis gambar dengan cermat.

OUTPUT FORMAT:
Kembalikan HANYA JSON valid berikut — tanpa teks tambahan, tanpa markdown code block:

{
  "is_stain": true atau false,
  "confidence": 0.0 sampai 1.0,
  "reason": "penjelasan singkat mengapa ini dianggap noda/bukan noda",
  "detected_object": "deskripsi singkat apa yang terlihat di gambar"
}

ATURAN:
- is_stain = true HANYA jika terlihat noda/kotoran pada permukaan kain/pakaian/tekstil
- is_stain = false jika: bukan kain, tidak ada noda, hanya pola/motif kain, gambar tidak jelas
- confidence: 0.8+ = sangat yakin, 0.5-0.8 = cukup yakin, <0.5 = tidak yakin
- Jawab dalam Bahasa Indonesia
- Output HANYA JSON, tidak ada teks lain
"""

SYSTEM_PROMPT = """Kamu adalah StainSense AI, asisten pembersihan noda profesional.

TUGAS:
Analisis gambar noda pada kain. Gambar mungkin mengandung SATU atau BEBERAPA noda berbeda.
Identifikasi SETIAP noda yang terlihat dan berikan instruksi pembersihan masing-masing.
Jika ada gambar kedua (label/tag pakaian), gunakan informasi komposisi kain dari tag
tersebut untuk meningkatkan akurasi instruksi.

OUTPUT FORMAT:
Kembalikan HANYA JSON valid berikut — tanpa teks tambahan, tanpa markdown code block.

Jika hanya ada SATU noda:
{
  "noda": [
    {
      "jenis_noda": "nama spesifik noda",
      "lokasi_noda": "deskripsi singkat posisi noda pada kain (mis: bagian depan tengah, lengan kanan, kerah)",
      "jenis_kain": "jenis kain",
      "komposisi_kain": "dari label jika ada, kosong jika tidak",
      "tingkat_keparahan": "ringan | sedang | parah",
      "peringatan_bahaya": "peringatan atau kosong",
      "langkah_pembersihan": ["1. ...", "2. ...", "3. ...", "4. ..."],
      "produk_rekomendasi": ["produk 1", "produk 2"],
      "catatan_tambahan": "tips ekstra atau kosong"
    }
  ]
}

Jika ada BEBERAPA noda berbeda, tambahkan objek ke dalam array "noda":
{
  "noda": [
    { ... noda pertama ... },
    { ... noda kedua ... },
    { ... dst ... }
  ]
}

ATURAN:
- Jawab dalam Bahasa Indonesia
- Output HANYA JSON, tidak ada teks lain
- Langkah pembersihan per noda: minimal 4, maksimal 8 langkah
- Jika tidak ada noda: kembalikan array kosong {"noda": []}
- Jika ada label tag pakaian di gambar kedua: baca komposisi dan sesuaikan instruksi
- Identifikasi SEMUA noda yang berbeda — setiap noda dengan jenis/posisi berbeda = entri terpisah
- Jika semua noda sejenis dan berdekatan, cukup satu entri
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
    Supports both multi-stain format ({"noda": [...]}) and legacy single-stain format.
    """
    if not raw or not raw.strip():
        return _fallback("Respons kosong dari API.")

    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    parsed = None

    # Try 1: direct parse
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try 2: extract JSON object from surrounding text
    if parsed is None:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Try 3: repair truncated JSON
    if parsed is None:
        try:
            repaired = _repair_json(cleaned)
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            pass

    if parsed is None:
        return _fallback(f"Gagal parse respons. Cuplikan: {raw[:200]}")

    # ── Normalize to multi-stain format ──
    return _normalize_result(parsed)


def _normalize_result(parsed: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize parsed JSON into consistent multi-stain format.
    Accepts both:
      - New format: {"noda": [{...}, {...}]}
      - Legacy format: {"jenis_noda": ..., "langkah_pembersihan": ...}
    Returns: {"noda": [list of stain objects with defaults filled]}
    """
    stain_defaults = {
        "jenis_noda":         "tidak terdeteksi",
        "lokasi_noda":        "",
        "jenis_kain":         "tidak diketahui",
        "komposisi_kain":     "",
        "tingkat_keparahan":  "sedang",
        "peringatan_bahaya":  "",
        "langkah_pembersihan": [],
        "produk_rekomendasi": [],
        "catatan_tambahan":   "",
    }

    # Key mapping from English/translated keys to standard Indonesian keys
    key_mapping = {
        "stain_type": "jenis_noda",
        "jenis_noda": "jenis_noda",
        "stain_location": "lokasi_noda",
        "lokasi_noda": "lokasi_noda",
        "fabric_type": "jenis_kain",
        "jenis_kain": "jenis_kain",
        "fabric_composition": "komposisi_kain",
        "komposisi_kain": "komposisi_kain",
        "severity": "tingkat_keparahan",
        "tingkat_keparahan": "tingkat_keparahan",
        "danger_warning": "peringatan_bahaya",
        "peringatan_bahaya": "peringatan_bahaya",
        "cleaning_steps": "langkah_pembersihan",
        "langkah_pembersihan": "langkah_pembersihan",
        "recommended_products": "produk_rekomendasi",
        "produk_rekomendasi": "produk_rekomendasi",
        "additional_notes": "catatan_tambahan",
        "catatan_tambahan": "catatan_tambahan",
    }

    def clean_stain_dict(stain: dict[str, Any]) -> dict[str, Any]:
        cleaned = {}
        for original_k, v in stain.items():
            matched = False
            for map_k, target_k in key_mapping.items():
                if original_k.lower().strip() == map_k:
                    cleaned[target_k] = v
                    matched = True
                    break
            if not matched:
                cleaned[original_k] = v
        
        for k, v in stain_defaults.items():
            cleaned.setdefault(k, v)
        return cleaned

    noda_key = None
    for k in ["noda", "stains", "stain_list"]:
        for pk in parsed:
            if pk.lower().strip() == k:
                noda_key = pk
                break
        if noda_key:
            break

    if noda_key and isinstance(parsed[noda_key], list):
        cleaned_stains = [clean_stain_dict(stain) for stain in parsed[noda_key] if isinstance(stain, dict)]
        return {"noda": cleaned_stains}

    if isinstance(parsed, list):
        cleaned_stains = [clean_stain_dict(stain) for stain in parsed if isinstance(stain, dict)]
        return {"noda": cleaned_stains}

    has_stain_keys = False
    for pk in parsed:
        if pk.lower().strip() in key_mapping:
            has_stain_keys = True
            break

    if has_stain_keys:
        return {"noda": [clean_stain_dict(parsed)]}

    for k, v in stain_defaults.items():
        parsed.setdefault(k, v)
    return {"noda": [parsed]}


def _parse_verify_response(raw: str) -> dict[str, Any]:
    """
    Parse verification response from LLM.
    Returns: {"is_stain": bool, "confidence": float, "reason": str, "detected_object": str}
    """
    defaults = {
        "is_stain": False,
        "confidence": 0.0,
        "reason": "Gagal memverifikasi gambar.",
        "detected_object": "tidak diketahui",
    }

    if not raw or not raw.strip():
        return defaults

    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    parsed = None
    for attempt in [cleaned, None]:
        try:
            if attempt is not None:
                parsed = json.loads(attempt)
            else:
                match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if not match:
                    break
                parsed = json.loads(match.group())
            break
        except json.JSONDecodeError:
            continue

    if parsed is None or not isinstance(parsed, dict):
        return defaults

    normalized = {}
    
    is_stain_keys = ["is_stain", "isstain", "apakah_noda", "noda", "is_stains", "stain", "apakah_ada_noda"]
    for k in is_stain_keys:
        for pk in parsed:
            if pk.lower().strip() == k:
                val = parsed[pk]
                if isinstance(val, str):
                    normalized["is_stain"] = val.lower().strip() in ["true", "ya", "yes", "noda", "1"]
                else:
                    normalized["is_stain"] = bool(val)
                break
        if "is_stain" in normalized:
            break
            
    conf_keys = ["confidence", "confidence_level", "tingkat_kepercayaan", "kepercayaan", "conf", "score", "akurasi"]
    for k in conf_keys:
        for pk in parsed:
            if pk.lower().strip() == k:
                try:
                    normalized["confidence"] = float(parsed[pk])
                except (ValueError, TypeError):
                    pass
                break
        if "confidence" in normalized:
            break

    reason_keys = ["reason", "alasan", "penjelasan", "explanation", "deskripsi", "keterangan"]
    for k in reason_keys:
        for pk in parsed:
            if pk.lower().strip() == k:
                normalized["reason"] = str(parsed[pk])
                break
        if "reason" in normalized:
            break

    obj_keys = ["detected_object", "detectedobject", "objek_terdeteksi", "objek", "object", "terdeteksi", "objek_noda"]
    for k in obj_keys:
        for pk in parsed:
            if pk.lower().strip() == k:
                normalized["detected_object"] = str(parsed[pk])
                break
        if "detected_object" in normalized:
            break

    for k, v in defaults.items():
        normalized.setdefault(k, v)

    return normalized


def _fallback(reason: str) -> dict[str, Any]:
    return {
        "noda": [{
            "jenis_noda":         "gagal diparse",
            "lokasi_noda":        "",
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
        }],
    }


# ─── OpenCV: Bounding Box Detection ──────────────────────────────────────────

def _pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def _cv2_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def detect_stains(
    image: Image.Image,
    min_area: int = 800,
) -> tuple[Image.Image, list[dict]]:
    """
    Deteksi area noda via LAB color analysis + contour detection.
    Mengembalikan (gambar_dengan_bbox, list_deteksi).

    Catatan: ini adalah highlighting visual saja.
    Identifikasi jenis noda dilakukan oleh LLM, bukan OpenCV.

    Strategi filtering untuk mengurangi false positive:
      1. Dynamic fabric baseline — menggunakan median warna kain
         sebagai referensi, bukan nilai tetap 128.
      2. Solidity filter — noda = blob padat, pola kain = berlubang.
      3. Aspect ratio filter — menolak bentuk sangat panjang/tipis.
      4. Local contrast check — memastikan deviasi area lebih tinggi
         dari sekitarnya.
      5. Edge density filter — noda meresap di kain (tepian lembut),
         objek solid (gelas, piring) punya tepian tajam → ditolak.
      6. Lightness variance filter — noda mengikuti tekstur kain
         (variance L moderat), objek 3D punya refleksi/bayangan
         (variance L sangat tinggi) → ditolak.
    """
    cv_img = _pil_to_cv2(image)
    h, w   = cv_img.shape[:2]

    # ── Step 1: LAB color deviation (DYNAMIC BASELINE) ──
    # Gunakan median warna kain sebagai referensi, bukan 128 tetap.
    # Ini memastikan kita mencari "apa yang beda dari kain" bukan
    # "apa yang beda dari abu-abu netral".
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Hitung median A dan B channel sebagai baseline warna kain
    a_median = int(np.median(a_ch))
    b_median = int(np.median(b_ch))

    deviation = cv2.addWeighted(
        cv2.absdiff(a_ch, np.full_like(a_ch, a_median)), 0.6,
        cv2.absdiff(b_ch, np.full_like(b_ch, b_median)), 0.4, 0,
    )

    # ── Step 2: Blur + threshold ──
    blurred = cv2.GaussianBlur(deviation, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)

    # ── Step 3: Morphology — close gaps then remove small specks ──
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # Precompute Canny edges for edge density filter
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output      = cv_img.copy()
    detections: list[dict] = []
    det_idx = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        bbox_area = bw * bh

        # ── Filter 1: Skip if bounding box covers > 85% of image ──
        if bbox_area > 0.85 * w * h:
            continue

        # ── Filter 2: Aspect ratio — reject extremely elongated shapes (like borders/seams) ──
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 10.0:
            continue

        # ── Filter 3: Solidity — noda = blob padat, pola kain = berlubang/tersebar ──
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1)
        if solidity < 0.20:
            continue

        # ── Filter 4: Local contrast check ──
        margin = 30
        inner_mean = np.mean(deviation[y:y+bh, x:x+bw])
        oy1 = max(0, y - margin)
        oy2 = min(h, y + bh + margin)
        ox1 = max(0, x - margin)
        ox2 = min(w, x + bw + margin)
        outer_region = deviation[oy1:oy2, ox1:ox2].copy()
        ry, rx = y - oy1, x - ox1
        outer_region[ry:ry+bh, rx:rx+bw] = 0
        outer_pixels = np.count_nonzero(outer_region > 0)
        outer_sum = np.sum(outer_region)
        outer_mean = outer_sum / max(outer_pixels, 1) if outer_pixels > 0 else 0
        if inner_mean < max(outer_mean * 1.2, 5):
            continue

        # ── Filter 5: Edge density — tolak objek solid dengan tepi sangat tegas ──
        # Noda meresap di kain → tepian lembut, RENDAH edge density.
        # Objek solid/tekstur kaku → TINGGI edge density.
        roi_edges = edges[y:y+bh, x:x+bw]
        edge_pixels = np.count_nonzero(roi_edges)
        edge_density = edge_pixels / max(bbox_area, 1)
        # Noda biasanya edge density < 0.15, objek kaku > 0.22
        if edge_density > 0.22:
            continue

        # ── Filter 6: Lightness variance — tolak objek 3D ──
        # Noda di kain mengikuti tekstur kain → variance L moderat.
        # Objek 3D dengan refleksi cahaya tajam → variance L sangat tinggi.
        roi_l = l_ch[y:y+bh, x:x+bw]
        l_std = float(np.std(roi_l))
        # Noda bisa mencapai L std tinggi pada kain putih (kontras tinggi). Kita batasi di 80.
        if l_std > 80:
            continue

        # ── Passed all filters — draw bounding box ──
        conf = min(0.95, 0.40 + (area / (w * h)) * 3.0)
        # Boost confidence for high-solidity, low-edge-density blobs (very stain-like)
        stain_quality = solidity * (1.0 - edge_density * 5)
        conf = min(0.97, conf + max(0, stain_quality) * 0.1)
        color = BOX_COLORS[det_idx % len(BOX_COLORS)]
        label = f"Noda #{det_idx+1} ({conf:.0%})"

        cv2.rectangle(output, (x, y), (x + bw, y + bh), color, 3)
        fs = max(0.5, min(0.9, bw / 250))
        (tw, th_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        ly = max(y - 10, th_text + 5)
        cv2.rectangle(output, (x, ly - th_text - 6), (x + tw + 6, ly + 2), color, -1)
        cv2.putText(output, label, (x + 3, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), 2)

        detections.append({
            "bbox":       [int(x), int(y), int(x + bw), int(y + bh)],
            "confidence": round(conf, 3),
            "label":      label,
        })
        det_idx += 1

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
            "responseSchema":   SYSTEM_PROMPT_SCHEMA,
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


# ─── Verification Functions ───────────────────────────────────────────────────

def _verify_with_gemini(image: Image.Image, api_key: str, model: str) -> dict[str, Any]:
    """Quick stain verification via Gemini before full analysis."""
    img_b64 = _image_to_base64(image)
    parts = [
        {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
        {"text": "Periksa apakah gambar ini menunjukkan noda pada kain/pakaian."},
    ]
    payload = {
        "system_instruction": {"parts": [{"text": VERIFY_PROMPT}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json",
            "responseSchema":   VERIFY_SCHEMA,
        },
    }
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent?key={api_key}"
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return _parse_verify_response(raw)
    except Exception:
        # If verification fails, assume it's a stain and let analysis decide
        return {"is_stain": True, "confidence": 0.5, "reason": "Verifikasi gagal, melanjutkan analisis.", "detected_object": "tidak diketahui"}


def _verify_with_openrouter(image: Image.Image, api_key: str, model: str) -> dict[str, Any]:
    """Quick stain verification via OpenRouter before full analysis."""
    img_b64 = _image_to_base64(image)
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        {"type": "text", "text": "Periksa apakah gambar ini menunjukkan noda pada kain/pakaian."},
    ]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VERIFY_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://stainsense.app",
        "X-Title": "StainSense",
    }
    try:
        resp = requests.post(
            f"{OPENROUTER_BASE}/chat/completions",
            json=payload, headers=headers, timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        return _parse_verify_response(raw)
    except Exception:
        return {"is_stain": True, "confidence": 0.5, "reason": "Verifikasi gagal, melanjutkan analisis.", "detected_object": "tidak diketahui"}


# ─── Main Routers ─────────────────────────────────────────────────────────────

def _get_api_key(provider: str) -> str:
    """Get and validate API key for the given provider."""
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "GEMINI_API_KEY belum diset di .env\n"
                "→ https://aistudio.google.com/app/apikey"
            )
        return key
    elif provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "OPENROUTER_API_KEY belum diset di .env\n"
                "→ https://openrouter.ai/keys"
            )
        return key
    else:
        raise ValueError(f"Provider tidak dikenal: '{provider}'. Pilih 'gemini' atau 'openrouter'.")


def verify_stain(
    image: Image.Image,
    provider: str = "gemini",
    gemini_model: str = GEMINI_MODEL_DEFAULT,
    openrouter_model: str = OPENROUTER_MODEL_DEFAULT,
) -> dict[str, Any]:
    """
    Verify whether the image actually contains a stain on fabric.
    Returns: {"is_stain": bool, "confidence": float, "reason": str, "detected_object": str}
    """
    key = _get_api_key(provider)
    model = gemini_model if provider == "gemini" else openrouter_model

    if provider == "gemini":
        return _verify_with_gemini(image, key, model)
    else:
        return _verify_with_openrouter(image, key, model)


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
    Returns multi-stain format: {"noda": [{...}, {...}, ...]}

    Args:
        image:            Stain photo (PIL Image)
        provider:         "gemini" or "openrouter"
        additional_info:  Extra context from user
        tag_image:        Optional clothing tag photo for fabric identification
        gemini_model:     Which Gemini model to use
        openrouter_model: Which OpenRouter model to use
    """
    key = _get_api_key(provider)

    if provider == "gemini":
        return analyze_with_gemini(image, key, additional_info, tag_image, gemini_model)
    else:
        return analyze_with_openrouter(image, key, additional_info, tag_image, openrouter_model)
