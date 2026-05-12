"""
StainSense — Aplikasi Streamlit MVP
=====================================
Antarmuka pengguna untuk identifikasi noda kain berbasis AI.

Jalankan:
    streamlit run app.py
"""

import os
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# ─── Muat variabel lingkungan ────────────────────────────────────────────────
load_dotenv()

# ─── Impor modul AI lokal ────────────────────────────────────────────────────
from ai_module import analyze_stain, detect_stains

# ─── Konfigurasi Halaman ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="StainSense",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":    "https://github.com/stainsense",
        "Report a bug": None,
        "About":       "**StainSense** — Identifikasi noda cerdas berbasis AI",
    },
)

# ─── INJEKSI CSS (Dark Mode Professional) ────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Import Font ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0D0F14 !important;
    color: #E4E8F0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: #13151C !important;
    border-right: 1px solid #1E2130 !important;
}

/* ── Header Utama ── */
.ss-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 28px 0 8px;
    border-bottom: 1px solid #1E2130;
    margin-bottom: 28px;
}
.ss-logo {
    font-size: 2.6rem;
    line-height: 1;
}
.ss-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #64B5F6 0%, #4DD0E1 50%, #80CBC4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1;
}
.ss-subtitle {
    font-size: 0.85rem;
    color: #6B7280;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 4px 0 0;
}

/* ── Section Labels ── */
.ss-section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4B5563;
    margin-bottom: 8px;
}

/* ── Image Panel ── */
.ss-image-panel {
    background: #13151C;
    border: 1px solid #1E2130;
    border-radius: 16px;
    padding: 16px;
    position: relative;
    min-height: 220px;
    display: flex;
    flex-direction: column;
}
.ss-panel-tag {
    position: absolute;
    top: 12px;
    left: 12px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    z-index: 2;
}
.tag-before { background: #1E2130; color: #6B7280; }
.tag-after  { background: #0D2137; color: #64B5F6; }

/* ── Detection Badge ── */
.ss-detection-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #0D2137;
    border: 1px solid #1A3F5C;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.78rem;
    color: #64B5F6;
    margin-top: 10px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Kartu Hasil ── */
.ss-result-card {
    background: #13151C;
    border: 1px solid #1E2130;
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 14px;
}
.ss-result-card h4 {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4B5563;
    margin: 0 0 10px;
}
.ss-result-card p {
    font-size: 1rem;
    font-weight: 500;
    color: #E4E8F0;
    margin: 0;
}

/* ── Peringatan Bahaya ── */
.ss-danger-card {
    background: #1A0A0A;
    border: 1px solid #7F1D1D;
    border-left: 4px solid #EF4444;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
}
.ss-danger-card .danger-icon {
    font-size: 1.1rem;
    margin-right: 8px;
}
.ss-danger-card h4 {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #EF4444;
    margin: 0 0 8px;
}
.ss-danger-card p {
    font-size: 0.9rem;
    color: #FCA5A5;
    margin: 0;
    line-height: 1.6;
}

/* ── Peringatan Ringan ── */
.ss-warning-card {
    background: #1A1200;
    border: 1px solid #78350F;
    border-left: 4px solid #F59E0B;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
}
.ss-warning-card h4 {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #F59E0B;
    margin: 0 0 8px;
}
.ss-warning-card p {
    font-size: 0.9rem;
    color: #FDE68A;
    margin: 0;
    line-height: 1.6;
}

/* ── Langkah Pembersihan ── */
.ss-steps-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 4px;
}
.ss-step {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    background: #13151C;
    border: 1px solid #1E2130;
    border-radius: 12px;
    padding: 14px 16px;
    transition: border-color 0.2s;
}
.ss-step:hover { border-color: #2D3748; }
.ss-step-num {
    flex-shrink: 0;
    width: 28px;
    height: 28px;
    background: linear-gradient(135deg, #1E3A5F, #0D2137);
    border: 1px solid #1A3F5C;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    color: #64B5F6;
    font-family: 'JetBrains Mono', monospace;
}
.ss-step-text {
    font-size: 0.92rem;
    color: #CBD5E1;
    line-height: 1.6;
    padding-top: 3px;
}

/* ── Produk Rekomendasi ── */
.ss-product-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1A2133;
    border: 1px solid #253049;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.82rem;
    color: #94A3B8;
    margin: 4px 4px 4px 0;
}

/* ── Severity Badge ── */
.severity-ringan  { color: #34D399; background: #052E16; border: 1px solid #064E3B; border-radius: 8px; padding: 3px 12px; font-size: 0.78rem; font-weight: 600; }
.severity-sedang  { color: #F59E0B; background: #1A1200; border: 1px solid #78350F; border-radius: 8px; padding: 3px 12px; font-size: 0.78rem; font-weight: 600; }
.severity-parah   { color: #EF4444; background: #1A0A0A; border: 1px solid #7F1D1D; border-radius: 8px; padding: 3px 12px; font-size: 0.78rem; font-weight: 600; }

/* ── Tombol Analisis ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #1E3A5F 0%, #0D2137 100%) !important;
    border: 1px solid #2563EB55 !important;
    border-radius: 12px !important;
    color: #93C5FD !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    padding: 14px 20px !important;
    height: auto !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563EB33 0%, #1E3A5F 100%) !important;
    border-color: #3B82F6 !important;
    color: #BFDBFE !important;
}

/* ── Sidebar Widget ── */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextArea label,
[data-testid="stSidebar"] .stRadio label {
    color: #9CA3AF !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em !important;
}

/* ── Divider ── */
hr { border-color: #1E2130 !important; }

/* ── Hide Streamlit default elements ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─── Helper HTML ──────────────────────────────────────────────────────────────

def render_header() -> None:
    st.markdown("""
    <div class="ss-header">
        <div class="ss-logo">🧹</div>
        <div>
            <h1 class="ss-title">StainSense</h1>
            <p class="ss-subtitle">AI-Powered Fabric Stain Analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def severity_badge(level: str) -> str:
    level_lower = level.lower()
    css_class   = {
        "ringan": "severity-ringan",
        "sedang": "severity-sedang",
        "parah":  "severity-parah",
    }.get(level_lower, "severity-sedang")
    emoji = {"ringan": "🟢", "sedang": "🟡", "parah": "🔴"}.get(level_lower, "⚪")
    return f'<span class="{css_class}">{emoji} {level.capitalize()}</span>'


def render_result_card(label: str, value: str, icon: str = "") -> None:
    st.markdown(f"""
    <div class="ss-result-card">
        <h4>{icon} {label}</h4>
        <p>{value}</p>
    </div>
    """, unsafe_allow_html=True)


def render_danger_card(text: str) -> None:
    if not text:
        return
    # Tentukan apakah bahaya atau sekadar peringatan
    is_danger = any(kw in text.lower() for kw in ["jangan", "hindari", "bahaya", "berbahaya", "tidak boleh"])
    if is_danger:
        st.markdown(f"""
        <div class="ss-danger-card">
            <h4>⚠ Peringatan Bahaya</h4>
            <p>{text}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ss-warning-card">
            <h4>💡 Catatan Penting</h4>
            <p>{text}</p>
        </div>
        """, unsafe_allow_html=True)


def render_cleaning_steps(steps: list[str]) -> None:
    st.markdown('<div class="ss-steps-container">', unsafe_allow_html=True)
    for step in steps:
        # Hapus awalan angka jika ada (mis: "1. " atau "1) ")
        clean = step.strip()
        num_match = clean[:3] if clean and clean[0].isdigit() else ""
        num_display = num_match.rstrip(". )") if num_match else str(steps.index(step) + 1)
        text = clean[len(num_match):].strip() if num_match else clean

        st.markdown(f"""
        <div class="ss-step">
            <div class="ss-step-num">{num_display}</div>
            <div class="ss-step-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_products(products: list[str]) -> None:
    if not products:
        return
    chips = "".join(
        f'<span class="ss-product-chip">🛒 {p}</span>'
        for p in products
    )
    st.markdown(f"""
    <div class="ss-result-card">
        <h4>🛍 Produk Rekomendasi</h4>
        <div style="margin-top: 8px">{chips}</div>
    </div>
    """, unsafe_allow_html=True)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("### ⚙ Pengaturan")
        st.markdown("---")

        provider = st.selectbox(
            "Provider AI",
            options=["gemini", "openrouter"],
            format_func=lambda x: {
                "gemini":     "🌟 Google Gemini",
                "openrouter": "🔀 OpenRouter",
            }[x],
            help="Pilih backend LLM. Pastikan API key sudah diatur di .env",
        )

        st.markdown("---")
        st.markdown("### 📝 Informasi Tambahan")
        additional_info = st.text_area(
            "Keterangan kain / noda (opsional)",
            placeholder="Contoh: Kain sutra putih, noda sudah 2 hari...",
            height=90,
            help="Berikan konteks tambahan untuk analisis yang lebih akurat",
        )

        st.markdown("---")
        st.markdown("### 📡 Status API")
        gemini_key     = os.getenv("GEMINI_API_KEY", "")
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

        col1, col2 = st.columns(2)
        with col1:
            if gemini_key:
                st.success("Gemini ✓")
            else:
                st.error("Gemini ✗")
        with col2:
            if openrouter_key:
                st.success("OpenRouter ✓")
            else:
                st.error("OpenRouter ✗")

        if not (gemini_key or openrouter_key):
            st.warning("Atur API key di file `.env` untuk menggunakan AI.")

        st.markdown("---")
        st.markdown(
            "<p style='font-size:0.72rem; color:#374151; text-align:center'>"
            "StainSense MVP v1.0<br>Built with Streamlit + Google Gemini</p>",
            unsafe_allow_html=True,
        )

    return {
        "provider":        provider,
        "additional_info": additional_info,
    }


# ─── INPUT GAMBAR ─────────────────────────────────────────────────────────────

def get_image_input() -> Image.Image | None:
    """
    Tampilkan pilihan input gambar: Upload file atau Kamera langsung.
    Kembalikan PIL Image atau None.
    """
    tab_upload, tab_camera = st.tabs(["📁  Unggah Gambar", "📷  Ambil Foto"])

    image: Image.Image | None = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Pilih gambar noda",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")

    with tab_camera:
        camera_img = st.camera_input(
            "Arahkan kamera ke noda kain",
            label_visibility="collapsed",
        )
        if camera_img:
            image = Image.open(camera_img).convert("RGB")

    return image


# ─── TAMPILKAN HASIL ──────────────────────────────────────────────────────────

def display_image_panels(original: Image.Image, detected: Image.Image, detections: list) -> None:
    """Tampilkan gambar sebelum dan sesudah bounding box secara berdampingan."""
    col_orig, col_det = st.columns(2, gap="medium")

    with col_orig:
        st.markdown('<p class="ss-section-label">Gambar Asli</p>', unsafe_allow_html=True)
        st.image(original, use_container_width=True)

    with col_det:
        st.markdown('<p class="ss-section-label">Deteksi Noda</p>', unsafe_allow_html=True)
        st.image(detected, use_container_width=True)

        if detections:
            badges_html = "".join(
                f'<div class="ss-detection-badge">📍 {d["label"]} — conf: {d["confidence"]:.2f}</div>'
                for d in detections
            )
            st.markdown(badges_html, unsafe_allow_html=True)


def display_analysis_results(result: dict) -> None:
    """Render hasil analisis LLM dalam format kartu elegan."""
    st.markdown("---")
    st.markdown(
        '<p class="ss-section-label" style="font-size:0.9rem; margin-bottom:18px">'
        '🔬 Hasil Analisis AI</p>',
        unsafe_allow_html=True,
    )

    # ── Baris info utama
    col_a, col_b, col_c = st.columns(3, gap="small")
    with col_a:
        render_result_card("Jenis Noda", result.get("jenis_noda", "–"), "🔍")
    with col_b:
        render_result_card("Jenis Kain", result.get("jenis_kain", "–"), "👕")
    with col_c:
        sev   = result.get("tingkat_keparahan", "sedang")
        badge = severity_badge(sev)
        st.markdown(f"""
        <div class="ss-result-card">
            <h4>📊 Tingkat Keparahan</h4>
            <p>{badge}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Peringatan
    danger_text = result.get("peringatan_bahaya", "")
    if danger_text:
        render_danger_card(danger_text)

    # ── Langkah Pembersihan
    steps = result.get("langkah_pembersihan", [])
    if steps:
        st.markdown(
            '<p class="ss-section-label" style="margin-top:20px">🧼 Langkah Pembersihan</p>',
            unsafe_allow_html=True,
        )
        render_cleaning_steps(steps)

    # ── Produk
    render_products(result.get("produk_rekomendasi", []))

    # ── Catatan tambahan
    catatan = result.get("catatan_tambahan", "")
    if catatan:
        st.markdown(f"""
        <div class="ss-result-card" style="border-color:#253049">
            <h4>💬 Catatan Tambahan</h4>
            <p style="font-size:0.88rem; color:#94A3B8">{catatan}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── JSON mentah (expandable)
    with st.expander("🔧 Lihat respons JSON mentah"):
        st.json(result)


# ─── HALAMAN UTAMA ─────────────────────────────────────────────────────────────

def main() -> None:
    render_header()

    settings = render_sidebar()
    image    = get_image_input()

    if image is None:
        # ── Halaman sambutan
        st.markdown("---")
        c1, c2, c3 = st.columns(3, gap="large")
        features = [
            ("🔍", "Deteksi Akurat", "Identifikasi jenis noda dari foto menggunakan computer vision"),
            ("🤖", "AI Analysis", "LLM canggih merumuskan instruksi pembersihan step-by-step"),
            ("⚡", "Real-time", "Analisis selesai dalam hitungan detik"),
        ]
        for col, (icon, title, desc) in zip([c1, c2, c3], features):
            with col:
                st.markdown(f"""
                <div class="ss-result-card" style="text-align:center; padding: 28px 20px">
                    <div style="font-size:2rem; margin-bottom:12px">{icon}</div>
                    <h4 style="font-size:0.88rem; color:#64B5F6; margin-bottom:10px">{title}</h4>
                    <p style="font-size:0.82rem; color:#6B7280; line-height:1.6">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align:center; color:#374151; margin-top:24px; font-size:0.85rem'>"
            "↑ Unggah gambar atau ambil foto di atas untuk memulai analisis</p>",
            unsafe_allow_html=True,
        )
        return

    # ── Tampilkan gambar yang dimasukkan
    st.markdown("---")

    # Deteksi bounding box
    with st.spinner("🔍 Mendeteksi area noda..."):
        detected_image, detections = detect_stains(image)

    display_image_panels(image, detected_image, detections)

    # ── Tombol Analisis
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, col_space = st.columns([1, 2])
    with col_btn:
        analyze_clicked = st.button(
            "🧹 Analisis & Dapatkan Instruksi Pembersihan",
            type="primary",
        )

    if not analyze_clicked:
        return

    # ── Jalankan analisis LLM
    with st.spinner("🤖 AI sedang menganalisis noda..."):
        try:
            result = analyze_stain(
                image           = image,
                provider        = settings["provider"],
                additional_info = settings["additional_info"],
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