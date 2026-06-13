"""
StainSense — Streamlit App
============================
Dark mode UI. Gemini / OpenRouter. Clothing tag scanner.

Run: streamlit run app.py
"""

import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from ai_module import (
    analyze_stain, detect_stains, verify_stain,
    GEMINI_MODELS, OPENROUTER_MODELS,
    GEMINI_MODEL_DEFAULT, OPENROUTER_MODEL_DEFAULT,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StainSense",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "**StainSense** — AI Fabric Stain Identifier"},
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background-color: #121316 !important;
    color: #f3f4f6 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: #0d0e10 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}
.ss-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 28px 0 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    margin-bottom: 28px;
}
.ss-title {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.04em !important;
    color: #60a5fa !important;
    background: none !important;
    margin: 0 !important;
}
.ss-title a {
    display: none !important;
}
.ss-subtitle {
    font-size: 0.82rem;
    color: #9ca3af;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 6px 0 0;
    font-weight: 500;
}
.ss-section-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 12px;
}
.ss-result-card {
    background: #1c1d22;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.ss-result-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 0 15px rgba(99, 102, 241, 0.05);
}
.ss-result-card h4 {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9ca3af;
    margin: 0 0 12px;
}
.ss-result-card p {
    font-size: 1.05rem;
    font-weight: 500;
    color: #f3f4f6;
    margin: 0;
    line-height: 1.5;
}
.ss-danger-card {
    background: rgba(239, 68, 68, 0.05);
    border: 1px solid rgba(239, 68, 68, 0.15);
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
    transition: all 0.25s ease;
}
.ss-danger-card:hover {
    background: rgba(239, 68, 68, 0.07);
    border-color: rgba(239, 68, 68, 0.25);
}
.ss-danger-card h4 {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #f87171;
    margin: 0 0 8px;
}
.ss-danger-card p {
    font-size: 0.9rem;
    color: #fca5a5;
    margin: 0;
    line-height: 1.6;
}
.ss-warning-card {
    background: rgba(245, 158, 11, 0.04);
    border: 1px solid rgba(245, 158, 11, 0.15);
    border-left: 4px solid #f59e0b;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
    transition: all 0.25s ease;
}
.ss-warning-card:hover {
    background: rgba(245, 158, 11, 0.06);
    border-color: rgba(245, 158, 11, 0.25);
}
.ss-warning-card h4 {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #fbbf24;
    margin: 0 0 8px;
}
.ss-warning-card p {
    font-size: 0.9rem;
    color: #fde68a;
    margin: 0;
    line-height: 1.6;
}
.ss-tag-card {
    background: rgba(16, 185, 129, 0.05);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 12px;
    padding: 14px 18px;
    margin-top: 12px;
    transition: all 0.25s ease;
}
.ss-tag-card:hover {
    background: rgba(16, 185, 129, 0.08);
    border-color: rgba(16, 185, 129, 0.3);
}
.ss-tag-card h4 {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #34d399;
    margin: 0 0 6px;
}
.ss-tag-card p {
    font-size: 0.88rem;
    color: #a7f3d0;
    margin: 0;
    line-height: 1.5;
}
.ss-steps-container {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-top: 8px;
}
.ss-step {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    background: #1c1d22;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 16px 20px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.ss-step:hover {
    transform: translateX(4px);
    border-color: rgba(99, 102, 241, 0.4);
    background: #22232a;
}
.ss-step-num {
    flex-shrink: 0;
    width: 30px;
    height: 30px;
    background: linear-gradient(135deg, #312e81, #1e1b4b);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
    color: #a5b4fc;
    font-family: 'JetBrains Mono', monospace;
}
.ss-step-text {
    font-size: 0.94rem;
    color: #d1d5db;
    line-height: 1.6;
    padding-top: 2px;
}
.ss-detection-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.8rem;
    color: #a5b4fc;
    margin-top: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}
.ss-product-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.84rem;
    color: #d1d5db;
    margin: 4px 6px 4px 0;
    transition: all 0.2s ease;
}
.ss-product-chip:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(99, 102, 241, 0.3);
    color: #ffffff;
}
.severity-ringan {
    color: #34d399;
    background: rgba(52, 211, 153, 0.1);
    border: 1px solid rgba(52, 211, 153, 0.2);
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
}
.severity-sedang {
    color: #fbbf24;
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
}
.severity-parah {
    color: #f87171;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
}
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 14px 20px !important;
    height: auto !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    border-color: rgba(99, 102, 241, 0.6) !important;
    color: #ffffff !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4) !important;
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(1px);
    box-shadow: 0 2px 6px rgba(79, 70, 229, 0.2) !important;
}
.stButton > button:focus {
    outline: none !important;
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.4) !important;
}

/* Streamlit widget stylings for cohesive integration */
[data-testid="stFileUploader"] {
    background-color: #1c1d22 !important;
    border: 1px dashed rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 10px !important;
    transition: all 0.25s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99, 102, 241, 0.4) !important;
}
[data-testid="stFileUploader"] section {
    background-color: transparent !important;
}
[data-testid="stFileUploader"] section > div {
    color: #9ca3af !important;
}

div[data-baseweb="select"] > div {
    background-color: #1c1d22 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    color: #f3f4f6 !important;
    transition: all 0.25s ease !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: rgba(99, 102, 241, 0.3) !important;
}
div[data-baseweb="select"]:focus-within > div {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
}

.stTextArea textarea {
    background-color: #1c1d22 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    color: #f3f4f6 !important;
    padding: 12px !important;
    transition: all 0.25s ease !important;
}
.stTextArea textarea:hover {
    border-color: rgba(99, 102, 241, 0.3) !important;
}
.stTextArea textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    color: #ffffff !important;
}

[data-testid="stSidebar"] h3 {
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    color: #f3f4f6 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    margin-top: 24px !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.06) !important;
    margin: 16px 0 !important;
}

[data-testid="stTabs"] button {
    background-color: transparent !important;
    color: #9ca3af !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.25s ease !important;
}
[data-testid="stTabs"] button:hover {
    color: #f3f4f6 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #818cf8 !important;
    border-bottom: 2px solid #818cf8 !important;
    font-weight: 600 !important;
}

#MainMenu, footer, header { visibility:hidden; }
[data-testid="stToolbar"] { display:none; }

/* Verification cards */
.ss-verify-pass {
    background: rgba(52, 211, 153, 0.06);
    border: 1px solid rgba(52, 211, 153, 0.25);
    border-left: 4px solid #34d399;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
    transition: all 0.25s ease;
}
.ss-verify-pass h4 {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #34d399; margin: 0 0 8px;
}
.ss-verify-pass p { font-size: 0.9rem; color: #a7f3d0; margin: 0; line-height: 1.6; }
.ss-verify-fail {
    background: rgba(239, 68, 68, 0.06);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
    transition: all 0.25s ease;
}
.ss-verify-fail h4 {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #f87171; margin: 0 0 8px;
}
.ss-verify-fail p { font-size: 0.9rem; color: #fca5a5; margin: 0; line-height: 1.6; }
.ss-stain-header {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(99, 102, 241, 0.02));
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 14px 20px;
    margin: 20px 0 12px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.ss-stain-header .num {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #4f46e5, #3730a3);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; font-weight: 700; color: #e0e7ff;
    font-family: 'JetBrains Mono', monospace;
}
.ss-stain-header .info {
    font-size: 0.95rem; font-weight: 600; color: #c7d2fe;
}
.ss-stain-header .loc {
    font-size: 0.78rem; color: #6b7280; margin-left: auto;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def compress_image(image: Image.Image, max_mb: float = 10.0) -> Image.Image:
    """Resize + compress image before processing."""
    w, h = image.size
    if max(w, h) > 1920:
        scale = 1920 / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


def severity_badge(level: str) -> str:
    css = {"ringan": "severity-ringan", "sedang": "severity-sedang", "parah": "severity-parah"}
    em  = {"ringan": "🟢", "sedang": "🟡", "parah": "🔴"}
    lv  = level.lower()
    return f'<span class="{css.get(lv, "severity-sedang")}">{em.get(lv, "⚪")} {level.capitalize()}</span>'


def render_result_card(label: str, value: str, icon: str = "") -> None:
    st.markdown(f"""
    <div class="ss-result-card">
        <h4>{icon} {label}</h4>
        <p>{value}</p>
    </div>""", unsafe_allow_html=True)


def render_danger_card(text: str) -> None:
    if not text:
        return
    is_danger = any(k in text.lower() for k in ["jangan", "hindari", "bahaya", "tidak boleh", "wajib"])
    if is_danger:
        st.markdown(f"""
        <div class="ss-danger-card">
            <h4>⚠ Peringatan Bahaya</h4>
            <p>{text}</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ss-warning-card">
            <h4>💡 Catatan Penting</h4>
            <p>{text}</p>
        </div>""", unsafe_allow_html=True)


def render_cleaning_steps(steps: list[str]) -> None:
    st.markdown('<div class="ss-steps-container">', unsafe_allow_html=True)
    for i, step in enumerate(steps):
        clean = step.strip()
        num_match = clean[:3] if clean and clean[0].isdigit() else ""
        num  = num_match.rstrip(". )") if num_match else str(i + 1)
        text = clean[len(num_match):].strip() if num_match else clean
        st.markdown(f"""
        <div class="ss-step">
            <div class="ss-step-num">{num}</div>
            <div class="ss-step-text">{text}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_products(products: list[str]) -> None:
    if not products:
        return
    chips = "".join(f'<span class="ss-product-chip">🛒 {p}</span>' for p in products)
    st.markdown(f"""
    <div class="ss-result-card">
        <h4>🛍 Produk Rekomendasi</h4>
        <div style="margin-top:8px">{chips}</div>
    </div>""", unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────────────────

def render_header() -> None:
    st.markdown("""
    <div class="ss-header">
        <div style="font-size:2.6rem;line-height:1">🧹</div>
        <div>
            <h1 class="ss-title">StainSense</h1>
            <p class="ss-subtitle">AI-Powered Fabric Stain Analysis</p>
        </div>
    </div>""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    with st.sidebar:
        # ── Provider
        st.markdown("### ⚙ Pengaturan AI")
        st.markdown("---")

        provider = st.selectbox(
            "Provider AI",
            ["gemini", "openrouter"],
            format_func=lambda x: {"gemini": "🌟 Google Gemini", "openrouter": "🔀 OpenRouter"}[x],
        )

        gemini_model     = GEMINI_MODEL_DEFAULT
        openrouter_model = OPENROUTER_MODEL_DEFAULT

        if provider == "gemini":
            gemini_model = st.selectbox(
                "Model Gemini",
                list(GEMINI_MODELS.keys()),
                format_func=lambda x: x,
            )
            gemini_model = GEMINI_MODELS[gemini_model]
        else:
            or_choice = st.selectbox(
                "Model OpenRouter",
                list(OPENROUTER_MODELS.keys()),
                format_func=lambda x: x,
            )
            openrouter_model = OPENROUTER_MODELS[or_choice]

        # ── Status API
        st.markdown("---")
        st.markdown("### 📡 Status API")
        gkey = os.getenv("GEMINI_API_KEY", "").strip()
        okey = os.getenv("OPENROUTER_API_KEY", "").strip()
        col1, col2 = st.columns(2)
        with col1:
            if gkey:
                st.success("Gemini ✓")
            else:
                st.error("Gemini ✗")
        with col2:
            if okey:
                st.success("OpenRouter ✓")
            else:
                st.error("OpenRouter ✗")
        if not (gkey or okey):
            st.warning("Atur API key di file `.env`")

        # ── Clothing Tag Scanner
        st.markdown("---")
        st.markdown("### 🏷 Tag Pakaian (Opsional)")
        st.markdown(
            "<p style='font-size:0.78rem;color:#6B7280;margin-bottom:8px'>"
            "Upload foto label/tag pakaian — AI akan membaca komposisi kain "
            "secara otomatis untuk instruksi yang lebih akurat.</p>",
            unsafe_allow_html=True,
        )
        tag_file = st.file_uploader(
            "Foto tag pakaian",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            help="Foto bagian dalam pakaian yang menunjukkan komposisi kain (mis: 100% Cotton)",
        )
        tag_image = None
        if tag_file:
            tag_image = Image.open(tag_file).convert("RGB")
            st.image(tag_image, caption="Tag terdeteksi ✓", use_container_width=True)
            st.markdown("""
            <div class="ss-tag-card">
                <h4>🏷 Tag Diterima</h4>
                <p>AI akan membaca komposisi kain dari foto ini secara otomatis.</p>
            </div>""", unsafe_allow_html=True)

        # ── Additional Info
        st.markdown("---")
        st.markdown("### 📝 Keterangan Tambahan")
        additional_info = st.text_area(
            "Info tambahan (opsional)",
            placeholder="Contoh: Noda sudah 2 hari, baju putih, noda terkena panas...",
            height=80,
            label_visibility="collapsed",
        )

        # ── Footer
        st.markdown("---")
        st.markdown(
            "<p style='font-size:0.72rem;color:#374151;text-align:center'>"
            "StainSense v1.2<br>Gemini + OpenRouter + OpenCV</p>",
            unsafe_allow_html=True,
        )

    return {
        "provider":         provider,
        "gemini_model":     gemini_model,
        "openrouter_model": openrouter_model,
        "additional_info":  additional_info,
        "tag_image":        tag_image,
    }


# ─── Image Input ─────────────────────────────────────────────────────────────

def get_image_input() -> Image.Image | None:
    tab_upload, tab_camera = st.tabs(["📁  Unggah Gambar", "📷  Ambil Foto"])
    image = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Pilih gambar noda",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            size_mb = uploaded.size / (1024 * 1024)
            if size_mb > 20:
                st.error(f"File terlalu besar ({size_mb:.1f}MB). Maksimum 20MB.")
                return None
            raw = Image.open(uploaded).convert("RGB")
            image = compress_image(raw)
            st.caption(f"📐 {image.width}×{image.height}px · {size_mb:.2f}MB")

    with tab_camera:
        cam = st.camera_input("Arahkan kamera ke noda kain", label_visibility="collapsed")
        if cam:
            image = compress_image(Image.open(cam).convert("RGB"))

    return image


# ─── Result Display ───────────────────────────────────────────────────────────

def display_image_panels(original: Image.Image, detected: Image.Image, detections: list) -> None:
    col_orig, col_det = st.columns(2, gap="medium")
    with col_orig:
        st.markdown('<p class="ss-section-label">Gambar Asli</p>', unsafe_allow_html=True)
        st.image(original, use_container_width=True)
    with col_det:
        st.markdown('<p class="ss-section-label">Deteksi Area Noda</p>', unsafe_allow_html=True)
        st.image(detected, use_container_width=True)
        if detections:
            badges = "".join(
                f'<div class="ss-detection-badge">📍 {d["label"]} · conf {d["confidence"]:.2f}</div>'
                for d in detections
            )
            st.markdown(badges, unsafe_allow_html=True)


def render_verification_result(verify_result: dict) -> None:
    """Display the stain verification result as a card."""
    is_stain = verify_result.get("is_stain", False)
    confidence = verify_result.get("confidence", 0.0)
    reason = verify_result.get("reason", "")
    detected = verify_result.get("detected_object", "")
    conf_pct = f"{confidence:.0%}"

    if is_stain:
        st.markdown(f"""
        <div class="ss-verify-pass">
            <h4>✅ Verifikasi: Noda Terdeteksi ({conf_pct})</h4>
            <p><strong>Terdeteksi:</strong> {detected}<br>
            <strong>Alasan:</strong> {reason}</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ss-verify-fail">
            <h4>❌ Verifikasi: Bukan Noda ({conf_pct})</h4>
            <p><strong>Terdeteksi:</strong> {detected}<br>
            <strong>Alasan:</strong> {reason}</p>
        </div>""", unsafe_allow_html=True)


def display_single_stain(stain: dict, index: int, total: int) -> None:
    """Display analysis results for a single stain."""
    jenis = stain.get("jenis_noda", "–")
    lokasi = stain.get("lokasi_noda", "")

    # Header for this stain (only if multiple)
    if total > 1:
        loc_text = f'<span class="loc">📍 {lokasi}</span>' if lokasi else ""
        st.markdown(f"""
        <div class="ss-stain-header">
            <div class="num">{index}</div>
            <div class="info">🔍 {jenis}</div>
            {loc_text}
        </div>""", unsafe_allow_html=True)

    # Top info row
    cols = st.columns(3, gap="small")
    with cols[0]:
        noda_label = jenis
        if lokasi and total == 1:
            noda_label += f'<br><small style="color:#6B7280;font-size:0.78rem">📍 {lokasi}</small>'
        render_result_card("Jenis Noda", noda_label, "🔍")
    with cols[1]:
        kain_val = stain.get("jenis_kain", "–")
        komposisi = stain.get("komposisi_kain", "")
        if komposisi:
            kain_val += f'<br><small style="color:#6B7280;font-size:0.78rem">{komposisi}</small>'
        st.markdown(f"""
        <div class="ss-result-card">
            <h4>👕 Jenis Kain</h4>
            <p>{kain_val}</p>
        </div>""", unsafe_allow_html=True)
    with cols[2]:
        sev = stain.get("tingkat_keparahan", "sedang")
        badge = severity_badge(sev)
        st.markdown(f"""
        <div class="ss-result-card">
            <h4>📊 Tingkat Keparahan</h4>
            <p>{badge}</p>
        </div>""", unsafe_allow_html=True)

    # Danger
    render_danger_card(stain.get("peringatan_bahaya", ""))

    # Steps
    steps = stain.get("langkah_pembersihan", [])
    if steps:
        st.markdown(
            '<p class="ss-section-label" style="margin-top:20px">🧼 Langkah Pembersihan</p>',
            unsafe_allow_html=True,
        )
        render_cleaning_steps(steps)

    # Products
    render_products(stain.get("produk_rekomendasi", []))

    # Notes
    catatan = stain.get("catatan_tambahan", "")
    if catatan:
        st.markdown(f"""
        <div class="ss-result-card" style="border-color:#253049">
            <h4>💬 Catatan Tambahan</h4>
            <p style="font-size:0.88rem;color:#94A3B8">{catatan}</p>
        </div>""", unsafe_allow_html=True)


def display_analysis_results(result: dict) -> None:
    """Display analysis results — supports multi-stain format."""
    st.markdown("---")

    stains = result.get("noda", [])

    if not stains:
        st.markdown("""
        <div class="ss-warning-card">
            <h4>ℹ️ Tidak Ada Noda Terdeteksi</h4>
            <p>AI tidak menemukan noda pada gambar ini. Pastikan gambar menunjukkan
            noda pada permukaan kain dengan jelas.</p>
        </div>""", unsafe_allow_html=True)
        return

    total = len(stains)

    # Summary header
    if total == 1:
        st.markdown(
            '<p class="ss-section-label" style="font-size:0.9rem;margin-bottom:18px">'
            '🔬 Hasil Analisis AI</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<p class="ss-section-label" style="font-size:0.9rem;margin-bottom:18px">'
            f'🔬 Hasil Analisis AI — {total} Noda Terdeteksi</p>',
            unsafe_allow_html=True,
        )

    # Render each stain
    for i, stain in enumerate(stains):
        display_single_stain(stain, i + 1, total)
        if i < total - 1:
            st.markdown('<hr style="border-color:rgba(255,255,255,0.06);margin:24px 0">', unsafe_allow_html=True)

    # Raw JSON
    with st.expander("🔧 Respons JSON mentah"):
        st.json(result)


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    render_header()
    settings = render_sidebar()
    image    = get_image_input()

    if image is None:
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        features = [
            ("📸", "Upload atau Foto", "Unggah gambar noda atau ambil foto langsung dari kamera browser"),
            ("🏷", "Scan Tag Pakaian", "Upload foto label pakaian — AI membaca komposisi kain otomatis"),
            ("✅", "Verifikasi Noda", "AI memverifikasi apakah gambar benar menunjukkan noda pada kain"),
            ("🤖", "Analisis Multi-Noda", "Identifikasi setiap noda berbeda & panduan pembersihan masing-masing"),
        ]
        for col, (icon, title, desc) in zip([c1, c2, c3, c4], features):
            with col:
                st.markdown(f"""
                <div class="ss-result-card" style="text-align:center;padding:28px 20px">
                    <div style="font-size:2rem;margin-bottom:12px">{icon}</div>
                    <h4 style="font-size:0.88rem;color:#64B5F6;margin-bottom:10px">{title}</h4>
                    <p style="font-size:0.82rem;color:#6B7280;line-height:1.6">{desc}</p>
                </div>""", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align:center;color:#374151;margin-top:24px;font-size:0.85rem'>"
            "↑ Unggah gambar atau ambil foto di atas untuk memulai</p>",
            unsafe_allow_html=True,
        )
        return

    st.markdown("---")

    # Bounding box detection
    with st.spinner("🔍 Mendeteksi area noda..."):
        detected_image, detections = detect_stains(image)

    display_image_panels(image, detected_image, detections)

    # Show tag image preview if provided
    if settings["tag_image"] is not None:
        st.markdown(
            '<p class="ss-section-label" style="margin-top:16px">🏷 Tag Pakaian (akan dibaca AI)</p>',
            unsafe_allow_html=True,
        )
        col_tag, col_space = st.columns([1, 3])
        with col_tag:
            st.image(settings["tag_image"], use_container_width=True)

    # Analyze button
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        analyze_clicked = st.button("🧹 Analisis & Dapatkan Instruksi Pembersihan")

    if not analyze_clicked:
        return

    # ── Step 1: Verification ──
    with st.spinner("✅ Memverifikasi gambar..."):
        try:
            verify_result = verify_stain(
                image            = image,
                provider         = settings["provider"],
                gemini_model     = settings["gemini_model"],
                openrouter_model = settings["openrouter_model"],
            )
        except (ValueError, ConnectionError, TimeoutError) as err:
            st.error(f"❌ **Error verifikasi:** {err}")
            return
        except Exception:
            # If verification crashes, assume stain and continue
            verify_result = {
                "is_stain": True, "confidence": 0.5,
                "reason": "Verifikasi gagal, melanjutkan analisis.",
                "detected_object": "tidak diketahui",
            }

    render_verification_result(verify_result)

    # If verification says NOT a stain, warn but allow user to continue
    if not verify_result.get("is_stain", True):
        st.warning(
            "⚠️ AI mendeteksi bahwa gambar ini **mungkin bukan noda pada kain**. "
            "Hasil analisis mungkin tidak akurat."
        )
        col_continue, _ = st.columns([1, 2])
        with col_continue:
            if not st.button("🔄 Lanjutkan Analisis Tetap", key="force_analyze"):
                return

    # ── Step 2: Full Analysis (multi-stain) ──
    with st.spinner("🤖 AI sedang menganalisis semua noda..."):
        try:
            result = analyze_stain(
                image            = image,
                provider         = settings["provider"],
                additional_info  = settings["additional_info"],
                tag_image        = settings["tag_image"],
                gemini_model     = settings["gemini_model"],
                openrouter_model = settings["openrouter_model"],
            )
        except (ValueError, ConnectionError, TimeoutError) as err:
            st.error(f"❌ **Error:** {err}")
            return
        except Exception as err:
            st.error(f"❌ **Error tidak terduga:** {err}")
            return

    display_analysis_results(result)


if __name__ == "__main__":
    main()
