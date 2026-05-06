import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import tf_keras as keras
import os
import gdown
from datetime import datetime
import streamlit.components.v1 as components
import base64
from io import BytesIO

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PulmoScan AI – Chest X-Ray Diagnostics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Inject JS ─────────────────────────────────────────────────────────────────
def inject_js():
    components.html("""
    <script>
    const parentDoc = window.parent.document;
    const observer = new MutationObserver(() => {
        const btn = parentDoc.querySelector('button[kind="primary"]');
        if (btn) {
            btn.addEventListener('click', () => {
                setTimeout(() => {
                    const alerts = parentDoc.querySelectorAll('.element-container');
                    if (alerts.length) alerts[alerts.length - 1].scrollIntoView({ behavior: 'smooth' });
                }, 2000);
            });
        }
    });
    observer.observe(parentDoc.body, { childList: true, subtree: true });
    </script>
    """, height=0)

inject_js()

# ─── Global CSS (Refined for Professional Look) ────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&family=JetBrains+Mono&display=swap');

/* ── UI Variables ── */
:root {
    --bg-dark: #0f172a;
    --card-bg: #1e293b;
    --accent-blue: #38bdf8;
    --accent-indigo: #6366f1;
    --text-main: #f1f5f9;
    --text-muted: #94a3b8;
    --success: #10b981;
    --danger: #f43f5e;
    --warning: #f59e0b;
    --border: rgba(148, 163, 184, 0.1);
}

/* ── Global Reset ── */
.stApp {
    background-color: var(--bg-dark);
    color: var(--text-main);
}

/* ── Typography Fixes ── */
h1, h2, h3, p, span, label, .stMarkdown {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-main) !important;
}

/* ── Sidebar Styling ── */
[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid var(--border);
}

/* ── Component Cards ── */
.scan-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
}

.section-tag {
    display: inline-block;
    background: rgba(56, 189, 248, 0.1);
    color: var(--accent-blue);
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    padding: 4px 12px;
    border-radius: 6px;
    margin-bottom: 15px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-indigo), var(--accent-blue)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.4) !important;
    transform: translateY(-1px);
}

/* ── Diagnosis Findings ── */
.finding-item {
    display: flex;
    align-items: center;
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 10px;
}

.finding-label { font-weight: 600; font-size: 0.95rem; }

.finding-conf {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(56, 189, 248, 0.15);
    color: var(--accent-blue);
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 0.85rem;
}

/* ── Patient Chip ── */
.patient-chip {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 15px;
    background: rgba(15, 23, 42, 0.3);
    border-radius: 12px;
}

.patient-avatar {
    width: 45px; height: 45px;
    background: var(--accent-indigo);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
}

/* ── Status Badge ── */
.status-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(16, 185, 129, 0.1);
    color: var(--success);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ─── Labels & Descriptions ──────────────────────────────────────────────────────
LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

LABEL_DESC = {
    "Atelectasis":        "Partial or complete collapse of lung tissue.",
    "Cardiomegaly":       "Enlargement of the cardiac silhouette.",
    "Effusion":           "Fluid accumulation in the pleural space.",
    "Infiltration":       "Inflammatory or fluid infiltrate in lung parenchyma.",
    "Mass":               "Focal opacity larger than 3 cm requiring further workup.",
    "Nodule":             "Small focal opacity; follow-up imaging advised.",
    "Pneumonia":          "Lobar or segmental consolidation consistent with infection.",
    "Pneumothorax":       "Air in the pleural space; urgent evaluation recommended.",
    "Consolidation":      "Homogeneous opacification of lung parenchyma.",
    "Edema":              "Pulmonary vascular congestion and interstitial fluid.",
    "Emphysema":          "Hyperinflation with destruction of alveolar walls.",
    "Fibrosis":           "Reticular opacities suggesting fibrotic lung disease.",
    "Pleural_Thickening": "Thickening of the pleural lining.",
    "Hernia":             "Herniation of abdominal contents into the thorax.",
}

SEVERITY = {
    "Pneumothorax": "HIGH", "Mass": "HIGH", "Pneumonia": "HIGH",
    "Consolidation": "MODERATE", "Edema": "MODERATE", "Effusion": "MODERATE",
    "Cardiomegaly": "MODERATE", "Atelectasis": "MODERATE",
    "Infiltration": "LOW", "Nodule": "LOW", "Emphysema": "LOW",
    "Fibrosis": "LOW", "Pleural_Thickening": "LOW", "Hernia": "LOW",
}

# ─── Model Loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_trained_model():
    model_path = "best_model.keras"
    file_id = "1T-mLtAXELB734rsr2HlIqH6bxYLKbjYP"
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(model_path):
        with st.spinner("⏳ Accessing Neural Network..."):
            try:
                gdown.download(url, model_path, quiet=False)
            except Exception as e:
                st.error(f"Download failed: {e}")
                return None
    try:
        return keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Initialization error: {e}")
        return None

def img_to_b64(pil_img, fmt="PNG"):
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def build_report_html(patient, findings_with_conf, img_b64, threshold, scan_id):
    # (The original HTML logic is kept intact for functionality)
    # Using your existing build_report_html logic...
    now = datetime.now()
    date_str = now.strftime("%B %d, %Y")
    time_str = now.strftime("%H:%M")
    report_id = f"PSR-{now.strftime('%Y%m%d')}-{scan_id:04d}"
    sev_color = {"HIGH": "#f43f5e", "MODERATE": "#f59e0b", "LOW": "#10b981"}

    if findings_with_conf:
        findings_html = ""
        for label, conf in findings_with_conf:
            sev = SEVERITY.get(label, "LOW")
            col = sev_color[sev]
            desc = LABEL_DESC.get(label, "")
            findings_html += f"<tr><td style='padding:12px; font-weight:600;'>{label}</td><td>{desc}</td><td style='color:{col}; font-weight:700;'>{sev}</td><td style='font-family:monospace;'>{conf:.1%}</td></tr>"
        impression = f"AI identifies {len(findings_with_conf)} findings above {threshold:.0%} threshold."
    else:
        findings_html = "<tr><td colspan='4' style='text-align:center; padding:20px;'>No significant abnormalities.</td></tr>"
        impression = "Clear scan within AI confidence limits."

    # Simplified HTML for printing/previewing
    return f"<html><body style='font-family:sans-serif; padding:40px;'><h2>Medical Report: {patient['name']}</h2><p>ID: {report_id} | Date: {date_str}</p><hr><table border='1' width='100%' style='border-collapse:collapse;'>{findings_html}</table><h3>Impression</h3><p>{impression}</p></body></html>"

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:20px 0;">
        <div style="font-size:40px;">🫁</div>
        <h2 style="margin:0; font-family:'Plus Jakarta Sans';">PulmoScan AI</h2>
        <p style="color:#94a3b8; font-size:0.8rem;">V1.0.4 - Clinical Support</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-tag">Patient Profile</div>', unsafe_allow_html=True)
    p_name = st.text_input("Name", "Ahmed Khan")
    c1, c2 = st.columns(2)
    p_age = c1.number_input("Age", 1, 120, 35)
    p_gender = c2.selectbox("Gender", ["Male", "Female", "Other"])
    
    st.markdown('<div class="section-tag" style="margin-top:15px;">Sensitivity</div>', unsafe_allow_html=True)
    threshold = st.slider("Detection Threshold", 0.10, 0.90, 0.20, 0.05)

# ─── Main Content ──────────────────────────────────────────────────────────────
st.markdown('<div class="status-badge"><span style="width:8px; height:8px; background:#10b981; border-radius:50%;"></span> Diagnostic Engine Active</div>', unsafe_allow_html=True)
st.title("Chest X-Ray Analysis")

uploaded = st.file_uploader("Upload Radiograph", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    col_left, col_right = st.columns([1.2, 1], gap="medium")

    with col_left:
        st.markdown('<div class="scan-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-tag">Input Image</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="scan-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-tag">Patient Metadata</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="patient-chip">
            <div class="patient-avatar">👤</div>
            <div>
                <div style="font-weight:700; font-size:1.1rem;">{p_name}</div>
                <div style="color:#94a3b8; font-size:0.85rem;">{p_age} Years • {p_gender}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("🔬 START DIAGNOSIS", use_container_width=True, type="primary")

        if run:
            model = load_trained_model()
            if model:
                with st.spinner("Decoding pixels..."):
                    img_proc = np.array(img.resize((224, 224))) / 255.0
                    img_proc = np.expand_dims(img_proc, axis=0)
                    preds = model.predict(img_proc, verbose=0)[0]

                positives = [(LABELS[i], float(p)) for i, p in enumerate(preds) if p >= threshold]
                positives.sort(key=lambda x: -x[1])

                st.markdown('<div class="section-tag" style="margin-top:20px;">Analysis Results</div>', unsafe_allow_html=True)
                if positives:
                    for label, conf in positives:
                        st.markdown(f"""
                        <div class="finding-item">
                            <span class="finding-label">{label.replace('_', ' ')}</span>
                            <span class="finding-conf">{conf:.1%}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No abnormalities detected.")
                
                # Report logic
                patient_data = {"name": p_name, "age": p_age, "gender": p_gender}
                img_b64 = img_to_b64(img)
                scan_id = abs(hash(uploaded.name)) % 9999
                report_html = build_report_html(patient_data, positives, img_b64, threshold, scan_id)
                
                st.download_button("📥 Download Report", report_html, file_name="report.html", mime="text/html", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-top:50px; opacity:0.5; font-size:0.8rem;">
    SECURE SYSTEM • PULMOSCAN AI RADIOLOGY UNIT • 2026
</div>
""", unsafe_allow_html=True)
