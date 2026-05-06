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

# ─── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,700;1,9..144,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg:       #f0f4f8;
    --surface:  #ffffff;
    --surface2: #f7fafc;
    --navy:     #0f2035;
    --navy2:    #1a3350;
    --cyan:     #00c2e0;
    --cyan2:    #0095b3;
    --cyan-soft:#e0f7fb;
    --amber:    #f59e0b;
    --danger:   #ef4444;
    --danger-soft: #fef2f2;
    --success:  #10b981;
    --success-soft: #ecfdf5;
    --text:     #0f2035;
    --text2:    #3d5068;
    --muted:    #7a90a4;
    --border:   #d4dfe8;
    --radius:   16px;
    --shadow:   0 4px 20px rgba(15,32,53,0.08);
    --shadow-lg:0 12px 40px rgba(15,32,53,0.14);
}

/* ── Base Reset ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
}

.stApp {
    background: var(--bg) !important;
    background-image:
        radial-gradient(ellipse at 0% 0%, rgba(0,194,224,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 100% 100%, rgba(15,32,53,0.04) 0%, transparent 50%);
}

/* ── Force ALL text dark on light bg ── */
.stApp p, .stApp span, .stApp div, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
.stMarkdown, .stMarkdown p, .stMarkdown span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"],
.stText { color: var(--text) !important; }

.stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

/* Streamlit native widget labels */
.stTextInput label, .stNumberInput label, .stSelectbox label,
.stSlider label, .stTextArea label, .stFileUploader label,
.stRadio label {
    color: var(--navy2) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    text-transform: uppercase !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--cyan2); border-radius: 4px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: 1px solid rgba(0,194,224,0.15) !important;
    box-shadow: 4px 0 24px rgba(15,32,53,0.15) !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #e8f0f7 !important; }

section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: rgba(232,240,247,0.55) !important;
}

section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stTextArea label {
    color: rgba(0,194,224,0.85) !important;
    font-size: 0.68rem !important;
    letter-spacing: 1.5px !important;
}

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(0,194,224,0.22) !important;
    border-radius: 10px !important;
    color: #e8f0f7 !important;
    font-family: 'Sora', sans-serif !important;
}

section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder {
    color: rgba(232,240,247,0.35) !important;
}

section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(0,194,224,0.22) !important;
    border-radius: 10px !important;
    color: #e8f0f7 !important;
}

/* ── Inputs ── */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: var(--cyan2) !important;
    box-shadow: 0 0 0 3px rgba(0,194,224,0.12) !important;
}
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── Slider ── */
.stSlider > div > div > div { background: var(--cyan) !important; }
.stSlider > div > div { background: var(--border) !important; }
[data-testid="stThumbValue"] { color: var(--navy) !important; font-weight: 600 !important; }

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: all 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--cyan) !important;
    background: var(--cyan-soft) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] div {
    color: var(--text2) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--navy), var(--navy2)) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.3px !important;
    padding: 12px 24px !important;
    transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    box-shadow: 0 4px 16px rgba(15,32,53,0.22) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 8px 28px rgba(15,32,53,0.3) !important;
    background: linear-gradient(135deg, var(--navy2), var(--cyan2)) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00a8c6, #00c2e0) !important;
    box-shadow: 0 4px 20px rgba(0,194,224,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0095b3, #00b4d0) !important;
    box-shadow: 0 8px 30px rgba(0,194,224,0.45) !important;
}

/* ── Download Button ── */
.stDownloadButton > button {
    background: var(--surface) !important;
    color: var(--navy) !important;
    border: 2px solid var(--border) !important;
    border-radius: 12px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    border-color: var(--cyan2) !important;
    color: var(--cyan2) !important;
    background: var(--cyan-soft) !important;
    transform: translateY(-2px) !important;
}

/* ── Alerts ── */
.stAlert { 
    border-radius: 12px !important; 
    border-left-width: 4px !important;
    color: var(--text) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 20px 0 !important; }

/* ── Metric ── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 14px 18px;
}
[data-testid="stMetric"] label { color: var(--muted) !important; font-size: 11px !important; font-weight: 600 !important; letter-spacing: 1px !important; text-transform: uppercase !important; }
[data-testid="stMetric"] div { color: var(--text) !important; }

/* ── Spinner ── */
.stSpinner { color: var(--cyan2) !important; }

/* ────────────────────────────────────────────
   CUSTOM COMPONENTS
   ──────────────────────────────────────────── */

/* Header */
.ps-header {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 28px 0 18px;
}
.ps-logo {
    width: 56px; height: 56px;
    background: linear-gradient(135deg, var(--navy), var(--navy2));
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    box-shadow: 0 8px 24px rgba(15,32,53,0.25), 0 0 0 1px rgba(0,194,224,0.15);
    flex-shrink: 0;
}
.ps-title {
    font-family: 'Fraunces', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--navy) !important;
    line-height: 1;
    letter-spacing: -1px;
}
.ps-subtitle {
    font-size: 0.7rem;
    color: var(--cyan2) !important;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 5px;
}

/* Status Badge */
.status-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--success-soft);
    border: 1.5px solid rgba(16,185,129,0.25);
    color: var(--success) !important;
    padding: 6px 14px;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.status-dot {
    width: 7px; height: 7px;
    background: var(--success);
    border-radius: 50%;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
    50% { opacity: 0.8; transform: scale(1.2); box-shadow: 0 0 0 5px rgba(16,185,129,0); }
}

/* Scan Card */
.scan-card {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    box-shadow: var(--shadow);
    height: 100%;
}

/* Section Tag */
.section-tag {
    display: inline-block;
    background: rgba(0,149,179,0.09);
    border: 1px solid rgba(0,149,179,0.2);
    color: var(--cyan2) !important;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 6px;
    margin-bottom: 12px;
}

/* Patient Chip */
.patient-chip {
    display: flex; align-items: center; gap: 12px;
    background: var(--surface2);
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 18px;
}
.patient-avatar {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, var(--navy), var(--navy2));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(15,32,53,0.15);
}
.patient-name { 
    font-weight: 700; 
    font-size: 1rem; 
    line-height: 1.3; 
    color: var(--text) !important;
}
.patient-meta { 
    font-size: 0.76rem; 
    color: var(--muted) !important; 
    margin-top: 2px;
}

/* Finding Item */
.finding-item {
    display: flex; align-items: center; gap: 12px;
    background: #fff8f8;
    border: 1.5px solid rgba(239,68,68,0.18);
    border-left: 4px solid var(--danger);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    animation: slide-in 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
    box-shadow: 0 2px 8px rgba(239,68,68,0.06);
}
@keyframes slide-in {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
}
.finding-label { 
    font-weight: 600; 
    font-size: 0.92rem; 
    color: var(--text) !important;
}
.finding-sev { 
    font-size: 0.7rem; 
    color: var(--text2) !important;
    margin-left: 4px;
}
.finding-conf {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--danger) !important;
    background: rgba(239,68,68,0.08);
    padding: 3px 10px;
    border-radius: 6px;
    flex-shrink: 0;
}

/* Normal Badge */
.normal-badge {
    display: flex; align-items: center; gap: 12px;
    background: var(--success-soft);
    border: 1.5px solid rgba(16,185,129,0.25);
    border-radius: 12px;
    padding: 16px 20px;
    animation: slide-in 0.35s ease forwards;
}
.normal-badge span { 
    font-weight: 600; 
    font-size: 0.95rem; 
    color: var(--success) !important; 
}

/* Sidebar Section Label */
.sidebar-section {
    display: inline-block;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: rgba(0,194,224,0.7) !important;
    margin: 14px 0 10px;
    border-bottom: 1px solid rgba(0,194,224,0.15);
    padding-bottom: 6px;
    width: 100%;
}

/* Upload hint */
.upload-hint {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 16px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
}
.upload-hint-icon {
    font-size: 24px;
    flex-shrink: 0;
    margin-top: 2px;
}
.upload-hint-text { 
    font-size: 0.85rem; 
    color: var(--text2) !important; 
    line-height: 1.6;
}
.upload-hint-title { 
    font-weight: 700; 
    font-size: 0.9rem; 
    color: var(--text) !important; 
    margin-bottom: 4px;
}

/* Print styles */
@media print {
    .stApp, .stSidebar, header, footer,
    [data-testid="stFileUploader"],
    [data-testid="stToolbar"],
    .stButton,
    .element-container:has(button),
    .stAlert { display: none !important; }
    #print-report { display: block !important; position: fixed !important; top: 0; left: 0; width: 100%; height: 100%; z-index: 99999; background: white !important; }
}
.report-print-wrapper { display: none; }
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
        with st.spinner("⏳ Downloading AI model weights…"):
            try:
                gdown.download(url, model_path, quiet=False)
            except Exception as e:
                st.error(f"Model download failed: {e}")
                return None
    try:
        return keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

# ─── Helper: image → base64 ─────────────────────────────────────────────────────
def img_to_b64(pil_img, fmt="PNG"):
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

# ─── Helper: build printable HTML report ───────────────────────────────────────
def build_report_html(patient, findings_with_conf, img_b64, threshold, scan_id):
    now = datetime.now()
    date_str = now.strftime("%B %d, %Y")
    time_str = now.strftime("%H:%M")
    report_id = f"PSR-{now.strftime('%Y%m%d')}-{scan_id:04d}"

    sev_color = {"HIGH": "#ef4444", "MODERATE": "#f59e0b", "LOW": "#10b981"}

    if findings_with_conf:
        findings_html = ""
        for label, conf in findings_with_conf:
            sev = SEVERITY.get(label, "LOW")
            col = sev_color[sev]
            desc = LABEL_DESC.get(label, "")
            findings_html += f"""
            <tr>
                <td style="padding:10px 14px; font-weight:600; color:#0f2035;">{label.replace('_', ' ')}</td>
                <td style="padding:10px 14px; color:#3d5068;">{desc}</td>
                <td style="padding:10px 14px; text-align:center;">
                    <span style="background:{col}18; color:{col}; border:1.5px solid {col}40;
                                 padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700;">
                        {sev}
                    </span>
                </td>
                <td style="padding:10px 14px; text-align:center; font-family:monospace; color:#0f2035; font-weight:600;">
                    {conf:.1%}
                </td>
            </tr>"""
        impression = f"Radiological analysis identifies <strong>{len(findings_with_conf)} finding(s)</strong> above the confidence threshold of {threshold:.0%}. Clinical correlation and specialist review are recommended."
        conclusion_color = "#ef4444"
        conclusion_icon = "⚠"
        conclusion_text = "FINDINGS DETECTED — Clinical correlation advised"
    else:
        findings_html = f"""
        <tr>
            <td colspan="4" style="padding:20px; text-align:center; color:#10b981; font-weight:600;">
                ✓ No significant radiological abnormalities detected above the {threshold:.0%} confidence threshold.
            </td>
        </tr>"""
        impression = f"No significant pathological findings identified above the {threshold:.0%} confidence threshold. Routine follow-up per clinical protocol."
        conclusion_color = "#10b981"
        conclusion_icon = "✓"
        conclusion_text = "NO SIGNIFICANT FINDINGS DETECTED"

    age_gender = f"{patient['age']} yrs"
    if patient.get("gender"):
        age_gender += f" / {patient['gender']}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Radiology Report – {patient['name']}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Fraunces:opsz,wght@9..144,700&family=JetBrains+Mono&display=swap');
  *, *::before, *::after {{ box-sizing: border-box; margin:0; padding:0; }}
  body {{ font-family:'Sora',sans-serif; background:#f0f4f8; color:#0f2035; font-size:13px; }}
  .page {{ max-width:820px; margin:0 auto; background:#fff; min-height:100vh; box-shadow:0 0 60px rgba(15,32,53,0.15); }}
  .rpt-header {{
    background: linear-gradient(135deg, #0f2035 0%, #1a3350 60%, #0095b3 100%);
    color:white; padding:30px 36px; position:relative; overflow:hidden;
  }}
  .rpt-header::before {{
    content:''; position:absolute; top:-60px; right:-60px;
    width:250px; height:250px;
    background: radial-gradient(circle, rgba(0,194,224,0.18) 0%, transparent 70%);
    pointer-events:none;
  }}
  .rpt-logo {{ display:flex; align-items:center; gap:14px; margin-bottom:20px; }}
  .rpt-logo-icon {{
    width:46px; height:46px; border-radius:12px;
    background:rgba(255,255,255,0.12);
    border:1px solid rgba(255,255,255,0.2);
    display:flex; align-items:center; justify-content:center;
    font-size:22px;
  }}
  .rpt-logo-text {{ font-family:'Fraunces',serif; font-size:1.7rem; color:white; letter-spacing:-0.5px; }}
  .rpt-logo-sub {{ font-size:0.62rem; letter-spacing:3px; text-transform:uppercase; color:rgba(0,194,224,0.85); margin-top:3px; }}
  .rpt-meta {{ display:flex; gap:28px; flex-wrap:wrap; }}
  .rpt-meta-item label {{ font-size:10px; text-transform:uppercase; letter-spacing:1.5px; color:rgba(255,255,255,0.45); display:block; margin-bottom:3px; }}
  .rpt-meta-item span {{ font-size:13px; font-weight:600; color:white; }}
  .rpt-patient {{ display:grid; grid-template-columns:1fr 1fr; gap:0; border-bottom:1px solid #e4ecf4; }}
  .rpt-patient-col {{ padding:18px 36px; }}
  .rpt-patient-col:first-child {{ border-right:1px solid #e4ecf4; }}
  .field-label {{ font-size:10px; text-transform:uppercase; letter-spacing:1.5px; color:#7a90a4; margin-bottom:5px; font-weight:600; }}
  .field-value {{ font-size:14px; font-weight:700; color:#0f2035; }}
  .field-value.large {{ font-size:16px; }}
  .rpt-section {{ padding:22px 36px; border-bottom:1px solid #e4ecf4; }}
  .rpt-section-title {{ font-size:10px; text-transform:uppercase; letter-spacing:2.5px; color:#0095b3; font-weight:700; margin-bottom:14px; padding-bottom:8px; border-bottom:2px solid #e4ecf4; }}
  .rpt-image-box {{ width:200px; height:200px; border:1.5px solid #e4ecf4; border-radius:10px; overflow:hidden; background:#000; float:right; margin:0 0 12px 20px; }}
  .rpt-image-box img {{ width:100%; height:100%; object-fit:contain; }}
  table.findings {{ width:100%; border-collapse:collapse; }}
  table.findings thead tr {{ background:#f0f4f8; }}
  table.findings thead th {{ padding:9px 14px; text-align:left; font-size:10px; text-transform:uppercase; letter-spacing:1.5px; color:#3d5068; font-weight:700; }}
  table.findings tbody tr {{ border-bottom:1px solid #f0f4f8; }}
  table.findings tbody tr:hover {{ background:#f8fafc; }}
  table.findings tbody tr:last-child {{ border-bottom:none; }}
  .rpt-conclusion {{ margin:0 36px 22px; background:{conclusion_color}0d; border:1.5px solid {conclusion_color}33; border-left:5px solid {conclusion_color}; border-radius:10px; padding:14px 20px; display:flex; align-items:center; gap:14px; }}
  .rpt-conclusion-icon {{ font-size:22px; color:{conclusion_color}; }}
  .rpt-conclusion-text {{ font-weight:700; color:{conclusion_color}; font-size:13px; letter-spacing:0.3px; }}
  .impression-text {{ background:#f8fafc; border-radius:10px; padding:14px 20px; font-size:13px; line-height:1.75; color:#3d5068; border-left:4px solid #0095b3; }}
  .rpt-disclaimer {{ background:#fffdf0; border-top:1px solid #f0e8c0; padding:14px 36px; font-size:11px; color:#7a6020; line-height:1.65; }}
  .rpt-footer {{ background:#f8fafc; border-top:1px solid #e4ecf4; padding:14px 36px; display:flex; justify-content:space-between; align-items:center; }}
  .rpt-footer-left {{ font-size:11px; color:#7a90a4; }}
  .rpt-footer-right {{ font-size:11px; color:#7a90a4; font-family:monospace; }}
  @media print {{ body {{ background:white; }} .page {{ max-width:none; }} .no-print {{ display:none; }} }}
</style>
</head>
<body>
<div class="page">
  <div class="rpt-header">
    <div class="rpt-logo">
      <div class="rpt-logo-icon">🫁</div>
      <div>
        <div class="rpt-logo-text">PulmoScan AI</div>
        <div class="rpt-logo-sub">AI-Assisted Radiology Report</div>
      </div>
    </div>
    <div class="rpt-meta">
      <div class="rpt-meta-item"><label>Report ID</label><span>{report_id}</span></div>
      <div class="rpt-meta-item"><label>Date</label><span>{date_str}</span></div>
      <div class="rpt-meta-item"><label>Time</label><span>{time_str}</span></div>
      <div class="rpt-meta-item"><label>Modality</label><span>Chest X-Ray (CXR)</span></div>
      <div class="rpt-meta-item"><label>AI Model</label><span>PulmoScan v1.0</span></div>
    </div>
  </div>
  <div class="rpt-patient">
    <div class="rpt-patient-col">
      <div class="field-label">Patient Name</div>
      <div class="field-value large">{patient['name']}</div>
    </div>
    <div class="rpt-patient-col">
      <div class="field-label">Age / Gender</div>
      <div class="field-value large">{age_gender}</div>
    </div>
    <div class="rpt-patient-col" style="border-top:1px solid #e4ecf4;">
      <div class="field-label">Referring Physician</div>
      <div class="field-value">{patient.get('ref_doc','—')}</div>
    </div>
    <div class="rpt-patient-col" style="border-top:1px solid #e4ecf4;">
      <div class="field-label">Clinical Notes</div>
      <div class="field-value">{patient.get('clinical_notes','—') or '—'}</div>
    </div>
  </div>
  <div class="rpt-section">
    <div class="rpt-section-title">Radiological Findings</div>
    <div>
      <div class="rpt-image-box"><img src="data:image/png;base64,{img_b64}" alt="X-Ray Scan"></div>
      <table class="findings">
        <thead><tr><th>Finding</th><th>Description</th><th style="text-align:center;">Severity</th><th style="text-align:center;">Confidence</th></tr></thead>
        <tbody>{findings_html}</tbody>
      </table>
      <div style="clear:both;"></div>
    </div>
  </div>
  <div class="rpt-conclusion">
    <div class="rpt-conclusion-icon">{conclusion_icon}</div>
    <div class="rpt-conclusion-text">{conclusion_text}</div>
  </div>
  <div class="rpt-section">
    <div class="rpt-section-title">Clinical Impression</div>
    <div class="impression-text">{impression}</div>
  </div>
  <div class="rpt-disclaimer">
    <strong>⚠ Important Disclaimer:</strong> This report is generated by an AI-assisted diagnostic tool and is intended to support, not replace, clinical judgment. All findings must be reviewed and confirmed by a qualified radiologist or treating physician. This system is not approved for standalone clinical diagnosis.
  </div>
  <div class="rpt-footer">
    <div class="rpt-footer-left">PulmoScan AI Diagnostics &nbsp;•&nbsp; Confidential Medical Document</div>
    <div class="rpt-footer-right">Report ID: {report_id}</div>
  </div>
</div>
</body>
</html>"""
    return html


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:16px 0 24px;">
      <div style="font-size:38px; margin-bottom:8px; filter:drop-shadow(0 4px 8px rgba(0,194,224,0.3));">🫁</div>
      <div style="font-family:'Fraunces',serif; font-size:1.35rem; color:#e8f0f7; font-weight:700; letter-spacing:-0.5px;">PulmoScan AI</div>
      <div style="font-size:0.6rem; letter-spacing:3px; color:rgba(0,194,224,0.8); text-transform:uppercase; margin-top:4px;">Chest X-Ray Diagnostics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Patient Information</div>', unsafe_allow_html=True)
    p_name   = st.text_input("Full Name", "Ahmed Khan", placeholder="Enter patient name")
    col_a, col_b = st.columns(2)
    with col_a:
        p_age = st.number_input("Age", 1, 120, 35)
    with col_b:
        p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    st.markdown('<div class="sidebar-section">Clinical Details</div>', unsafe_allow_html=True)
    p_ref_doc = st.text_input("Referring Physician", placeholder="Dr. Name")
    p_notes   = st.text_area("Clinical Notes", placeholder="Symptoms, history…", height=90)

    st.markdown("""<div style="height:1px; background:rgba(0,194,224,0.12); margin:18px 0;"></div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">AI Settings</div>', unsafe_allow_html=True)
    threshold = st.slider("Detection Threshold", 0.10, 0.90, 0.20, 0.05,
                          help="Lower = more sensitive, higher = more specific")
    st.caption(f"Threshold: **{threshold:.0%}** confidence")

    st.markdown("""
    <div style="margin-top:24px; padding:14px 16px; background:rgba(0,194,224,0.06); border:1px solid rgba(0,194,224,0.15); border-radius:10px;">
      <div style="font-size:0.68rem; color:rgba(0,194,224,0.7); font-weight:700; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:6px;">About</div>
      <div style="font-size:0.75rem; color:rgba(232,240,247,0.65); line-height:1.6;">Analyzes 14 pathological conditions from chest X-rays using deep learning.</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Layout ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ps-header">
  <div class="ps-logo">🫁</div>
  <div>
    <div class="ps-title">PulmoScan AI</div>
    <div class="ps-subtitle">AI-Powered Radiology Analysis</div>
  </div>
</div>
""", unsafe_allow_html=True)

col_status1, col_status2, col_status3 = st.columns([2, 3, 2])
with col_status1:
    st.markdown('<div class="status-badge"><div class="status-dot"></div>System Online</div>', unsafe_allow_html=True)
with col_status2:
    st.markdown(f'<div style="font-size:0.78rem; color:#7a90a4; padding-top:6px;">📅 {datetime.now().strftime("%d %b %Y  •  %H:%M")}</div>', unsafe_allow_html=True)

st.markdown("---")

# ─── Upload Area ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-tag">Scan Upload</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop or click to upload a Chest X-Ray (JPG / PNG)",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed"
)

if not uploaded:
    st.markdown("""
    <div class="upload-hint">
      <div class="upload-hint-icon">💡</div>
      <div>
        <div class="upload-hint-title">How to use PulmoScan AI</div>
        <div class="upload-hint-text">
          1. Fill in patient information in the left sidebar.<br>
          2. Upload a chest X-ray image (JPG or PNG format).<br>
          3. Click <strong>Run AI Diagnosis</strong> to analyze the scan.<br>
          4. Download or print the generated radiology report.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    col_img, col_ctrl = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="scan-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-tag">X-Ray Scan</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.caption(f"📁 {uploaded.name}  •  {img.size[0]} × {img.size[1]} px")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_ctrl:
        st.markdown('<div class="scan-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-tag">Patient</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="patient-chip">
          <div class="patient-avatar">👤</div>
          <div>
            <div class="patient-name">{p_name}</div>
            <div class="patient-meta">{p_age} yrs &nbsp;·&nbsp; {p_gender}{(' &nbsp;·&nbsp; ' + p_ref_doc) if p_ref_doc else ''}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        run = st.button("🔬  Run AI Diagnosis", type="primary", use_container_width=True)

        if run:
            model = load_trained_model()
            if model:
                with st.spinner("Analyzing radiograph…"):
                    img_proc = np.array(img.resize((224, 224))) / 255.0
                    img_proc = np.expand_dims(img_proc, axis=0)
                    preds = model.predict(img_proc, verbose=0)[0]

                positives = [(LABELS[i], float(p)) for i, p in enumerate(preds) if p >= threshold]
                positives.sort(key=lambda x: -x[1])

                st.markdown('<div class="section-tag" style="margin-top:20px;">AI Findings</div>', unsafe_allow_html=True)

                if positives:
                    for label, conf in positives:
                        sev = SEVERITY.get(label, "LOW")
                        sev_emoji = {"HIGH": "🔴", "MODERATE": "🟠", "LOW": "🟡"}[sev]
                        st.markdown(f"""
                        <div class="finding-item">
                          <div class="finding-label">{label.replace('_', ' ')}</div>
                          <div class="finding-sev">{sev_emoji} {sev}</div>
                          <div class="finding-conf">{conf:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="normal-badge">
                      <span style="font-size:20px;">✅</span>
                      <span>No abnormalities detected above threshold</span>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Generate Report ──────────────────────────────────────────
                patient_data = {
                    "name": p_name, "age": p_age, "gender": p_gender,
                    "ref_doc": p_ref_doc, "clinical_notes": p_notes
                }
                img_b64     = img_to_b64(img)
                scan_id     = abs(hash(uploaded.name + str(datetime.now().second))) % 9999
                report_html = build_report_html(patient_data, positives, img_b64, threshold, scan_id)
                report_b64  = base64.b64encode(report_html.encode()).decode()

                st.markdown("---")

                st.download_button(
                    label="📄  Download Report (HTML)",
                    data=report_html,
                    file_name=f"PulmoScan_Report_{p_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.html",
                    mime="text/html",
                    use_container_width=True,
                )

                st.markdown(f"""
                <div style="margin-top:10px;">
                  <a href="data:text/html;base64,{report_b64}"
                     target="_blank"
                     style="display:flex; align-items:center; justify-content:center; gap:8px;
                            width:100%; text-align:center;
                            background:#f0f4f8; border:1.5px solid #d4dfe8;
                            color:#0f2035; padding:11px 16px; border-radius:12px;
                            text-decoration:none; font-weight:600; font-size:14px;
                            font-family:'Sora',sans-serif;
                            transition:all 0.2s;">
                    🖨️ &nbsp;Open &amp; Print Report
                  </a>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.error("❌ Could not load AI model. Check model file / download.")

        st.markdown('</div>', unsafe_allow_html=True)

# ─── Info Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#7a90a4; font-size:0.74rem; padding:8px 0 24px;">
  PulmoScan AI &nbsp;•&nbsp; For research &amp; clinical support use only &nbsp;•&nbsp;
  Not a substitute for qualified radiologist review
</div>
""", unsafe_allow_html=True)
