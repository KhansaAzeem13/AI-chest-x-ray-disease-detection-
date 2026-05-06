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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:wght@600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --navy:    #0a1628;
    --navy2:   #112240;
    --navy3:   #1d3557;
    --teal:    #00b4d8;
    --teal2:   #0077b6;
    --amber:   #f4a261;
    --danger:  #e63946;
    --success: #2a9d8f;
    --text:    #e8edf5;
    --muted:   #8899aa;
    --card:    rgba(17, 34, 64, 0.85);
    --border:  rgba(0, 180, 216, 0.18);
    --radius:  14px;
    --shadow:  0 8px 32px rgba(0,0,0,0.35);
}

/* ── Base Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: #e8edf5 !important;
}
.stApp {
    background: #0a1628 !important;
    background-image:
        radial-gradient(ellipse at 15% 10%, rgba(0,119,182,0.15) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 80%, rgba(0,180,216,0.08) 0%, transparent 55%);
}
/* Force ALL streamlit text to be light */
.stApp p, .stApp span, .stApp label, .stApp div,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
.stMarkdown, .stMarkdown p, .stMarkdown span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"],
.stText, .stCaption { color: #e8edf5 !important; }
/* Streamlit native widget labels */
.stTextInput label, .stNumberInput label, .stSelectbox label,
.stSlider label, .stTextArea label, .stFileUploader label,
.stRadio label { color: #c8d4e3 !important; font-size: 0.85rem !important; font-weight: 500 !important; }
/* Caption text */
.stCaption, [data-testid="stCaptionContainer"] { color: #8899aa !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--navy); }
::-webkit-scrollbar-thumb { background: var(--teal2); border-radius: 3px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #112240 !important;
    border-right: 1px solid rgba(0,180,216,0.18);
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #e8edf5 !important; }
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select {
    background: #0a1628 !important;
    border: 1px solid rgba(0,180,216,0.25) !important;
    border-radius: 8px !important;
    color: #e8edf5 !important;
}
section[data-testid="stSidebar"] .stSelectbox > div { background: #0a1628 !important; }
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stTextArea label { color: #aabbcc !important; }

/* ── Inputs & Widgets ── */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: var(--navy2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stSelectbox > div > div {
    background: var(--navy2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
.stSlider > div > div > div { background: var(--teal) !important; }
.stSlider > div > div { background: rgba(0,180,216,0.2) !important; }

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: var(--navy2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--teal) !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--teal2), var(--teal)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    transition: all 0.25s !important;
    box-shadow: 0 4px 15px rgba(0,180,216,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0,180,216,0.45) !important;
}

/* ── Alerts ── */
.stAlert { border-radius: 10px !important; border-left-width: 4px !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Metric ── */
[data-testid="stMetric"] {
    background: var(--navy2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetric"] label { color: var(--muted) !important; font-size: 11px !important; }
[data-testid="stMetric"] div { color: var(--text) !important; }

/* ── Custom Components ── */
.pulsoscan-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 8px;
}
.pulsoscan-logo {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, var(--teal2), var(--teal));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    box-shadow: 0 4px 15px rgba(0,180,216,0.35);
}
.pulsoscan-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
    letter-spacing: -0.5px;
}
.pulsoscan-subtitle {
    font-size: 0.78rem;
    color: var(--teal);
    font-weight: 500;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-top: 3px;
}

.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(42,157,143,0.12);
    border: 1px solid rgba(42,157,143,0.3);
    color: #2a9d8f;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
}
.status-dot {
    width: 7px; height: 7px;
    background: #2a9d8f;
    border-radius: 50%;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.3); }
}

.scan-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    backdrop-filter: blur(8px);
}

.patient-chip {
    display: flex; align-items: center; gap: 10px;
    background: var(--navy);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 16px;
}
.patient-avatar {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, var(--teal2), var(--teal));
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}
.patient-name { font-weight: 600; font-size: 0.95rem; line-height: 1.2; }
.patient-meta { font-size: 0.75rem; color: var(--muted); }

.finding-item {
    display: flex; align-items: center; gap: 10px;
    background: rgba(230,57,70,0.08);
    border: 1px solid rgba(230,57,70,0.25);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    animation: slide-in 0.3s ease forwards;
}
@keyframes slide-in {
    from { opacity:0; transform: translateX(-8px); }
    to   { opacity:1; transform: translateX(0); }
}
.finding-dot {
    width: 8px; height: 8px;
    background: var(--danger);
    border-radius: 50%;
    flex-shrink: 0;
}
.finding-label { font-weight: 500; font-size: 0.9rem; }
.finding-conf {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--amber);
    background: rgba(244,162,97,0.12);
    padding: 2px 8px;
    border-radius: 4px;
}

.normal-badge {
    display: flex; align-items: center; gap: 10px;
    background: rgba(42,157,143,0.08);
    border: 1px solid rgba(42,157,143,0.3);
    border-radius: 10px;
    padding: 14px 18px;
    color: #2a9d8f;
    font-weight: 500;
}
.normal-icon { font-size: 20px; }

.section-tag {
    display: inline-block;
    background: rgba(0,180,216,0.12);
    border: 1px solid rgba(0,180,216,0.25);
    color: var(--teal);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 4px;
    margin-bottom: 10px;
}

/* ── Print Report Styles ── */
@media print {
    .stApp, .stSidebar, header, footer,
    [data-testid="stFileUploader"],
    [data-testid="stToolbar"],
    .stButton,
    .element-container:has(button),
    .stAlert { display: none !important; }
    
    #print-report {
        display: block !important;
        position: fixed !important;
        top: 0; left: 0;
        width: 100%; height: 100%;
        z-index: 99999;
        background: white !important;
    }
}

.report-print-wrapper {
    display: none;
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

    sev_color = {"HIGH": "#e63946", "MODERATE": "#f4a261", "LOW": "#2a9d8f"}

    if findings_with_conf:
        findings_html = ""
        for label, conf in findings_with_conf:
            sev = SEVERITY.get(label, "LOW")
            col = sev_color[sev]
            desc = LABEL_DESC.get(label, "")
            findings_html += f"""
            <tr>
                <td style="padding:10px 14px; font-weight:600; color:#1a1a2e;">{label.replace('_', ' ')}</td>
                <td style="padding:10px 14px; color:#555;">{desc}</td>
                <td style="padding:10px 14px; text-align:center;">
                    <span style="background:{col}22; color:{col}; border:1px solid {col}55;
                                 padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600;">
                        {sev}
                    </span>
                </td>
                <td style="padding:10px 14px; text-align:center; font-family:monospace; color:#333;">
                    {conf:.1%}
                </td>
            </tr>"""
        impression = f"Radiological analysis identifies <strong>{len(findings_with_conf)} finding(s)</strong> above the confidence threshold of {threshold:.0%}. Clinical correlation and specialist review are recommended."
        conclusion_color = "#e63946"
        conclusion_icon = "⚠"
        conclusion_text = "FINDINGS DETECTED — Clinical correlation advised"
    else:
        findings_html = f"""
        <tr>
            <td colspan="4" style="padding:20px; text-align:center; color:#2a9d8f; font-weight:600;">
                ✓ No significant radiological abnormalities detected above the {threshold:.0%} confidence threshold.
            </td>
        </tr>"""
        impression = f"No significant pathological findings identified above the {threshold:.0%} confidence threshold. Routine follow-up per clinical protocol."
        conclusion_color = "#2a9d8f"
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
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:wght@700&family=JetBrains+Mono&display=swap');
  *, *::before, *::after {{ box-sizing: border-box; margin:0; padding:0; }}
  body {{ font-family:'DM Sans',sans-serif; background:#f8f9fc; color:#1a1a2e; font-size:13px; }}
  
  .page {{ max-width:820px; margin:0 auto; background:#fff; min-height:100vh; position:relative; }}
  
  /* Header */
  .rpt-header {{
    background: linear-gradient(135deg, #0a1628 0%, #112240 60%, #1d3557 100%);
    color:white; padding:28px 36px; position:relative; overflow:hidden;
  }}
  .rpt-header::after {{
    content:''; position:absolute; top:-40px; right:-40px;
    width:200px; height:200px;
    background: radial-gradient(circle, rgba(0,180,216,0.15) 0%, transparent 70%);
    pointer-events:none;
  }}
  .rpt-logo {{ display:flex; align-items:center; gap:12px; margin-bottom:16px; }}
  .rpt-logo-icon {{
    width:44px; height:44px; border-radius:10px;
    background:linear-gradient(135deg,#0077b6,#00b4d8);
    display:flex; align-items:center; justify-content:center;
    font-size:20px; color:white;
  }}
  .rpt-logo-text {{ font-family:'Playfair Display',serif; font-size:1.6rem; color:white; }}
  .rpt-logo-sub {{ font-size:0.65rem; letter-spacing:2.5px; text-transform:uppercase; color:rgba(0,180,216,0.8); margin-top:2px; }}
  
  .rpt-meta {{ display:flex; gap:32px; flex-wrap:wrap; }}
  .rpt-meta-item label {{ font-size:10px; text-transform:uppercase; letter-spacing:1.5px; color:rgba(255,255,255,0.5); display:block; margin-bottom:3px; }}
  .rpt-meta-item span {{ font-size:13px; font-weight:500; color:white; }}
  
  /* Patient Card */
  .rpt-patient {{
    display:grid; grid-template-columns:1fr 1fr; gap:0;
    border-bottom:1px solid #e8ecf0;
  }}
  .rpt-patient-col {{ padding:20px 36px; }}
  .rpt-patient-col:first-child {{ border-right:1px solid #e8ecf0; }}
  .field-label {{ font-size:10px; text-transform:uppercase; letter-spacing:1.5px; color:#888; margin-bottom:4px; }}
  .field-value {{ font-size:14px; font-weight:600; color:#1a1a2e; }}
  .field-value.large {{ font-size:16px; }}
  
  /* Section */
  .rpt-section {{ padding:22px 36px; border-bottom:1px solid #e8ecf0; }}
  .rpt-section-title {{
    font-size:10px; text-transform:uppercase; letter-spacing:2px;
    color:#0077b6; font-weight:600; margin-bottom:14px;
    padding-bottom:6px; border-bottom:2px solid #e8ecf0;
  }}
  
  /* Scan Image */
  .rpt-image-box {{
    width:200px; height:200px; border:1px solid #e0e4ea; border-radius:8px;
    overflow:hidden; background:#000; float:right; margin:0 0 10px 20px;
  }}
  .rpt-image-box img {{ width:100%; height:100%; object-fit:contain; }}
  
  /* Findings Table */
  table.findings {{ width:100%; border-collapse:collapse; }}
  table.findings thead tr {{ background:#f0f4f8; }}
  table.findings thead th {{
    padding:9px 14px; text-align:left; font-size:10px;
    text-transform:uppercase; letter-spacing:1.5px; color:#555;
    font-weight:600;
  }}
  table.findings tbody tr {{ border-bottom:1px solid #f0f0f0; }}
  table.findings tbody tr:last-child {{ border-bottom:none; }}
  
  /* Conclusion Banner */
  .rpt-conclusion {{
    margin:0 36px 22px;
    background:{conclusion_color}0d;
    border:1px solid {conclusion_color}33;
    border-left:4px solid {conclusion_color};
    border-radius:8px;
    padding:14px 18px;
    display:flex; align-items:center; gap:12px;
  }}
  .rpt-conclusion-icon {{ font-size:20px; color:{conclusion_color}; }}
  .rpt-conclusion-text {{ font-weight:600; color:{conclusion_color}; font-size:13px; }}
  
  /* Impression */
  .impression-text {{
    background:#f8f9fc; border-radius:8px; padding:14px 18px;
    font-size:13px; line-height:1.7; color:#333;
    border-left:3px solid #0077b6;
  }}
  
  /* Disclaimer */
  .rpt-disclaimer {{
    background:#fffbf0; border-top:1px solid #f0e4c0;
    padding:14px 36px; font-size:11px; color:#8a7030; line-height:1.6;
  }}
  
  /* Footer */
  .rpt-footer {{
    background:#f8f9fc; border-top:1px solid #e8ecf0;
    padding:14px 36px; display:flex; justify-content:space-between; align-items:center;
  }}
  .rpt-footer-left {{ font-size:11px; color:#888; }}
  .rpt-footer-right {{ font-size:11px; color:#888; font-family:monospace; }}
  
  @media print {{
    body {{ background:white; }}
    .page {{ max-width:none; }}
    .no-print {{ display:none; }}
  }}
</style>
</head>
<body>
<div class="page">

  <!-- Header -->
  <div class="rpt-header">
    <div class="rpt-logo">
      <div class="rpt-logo-icon">🫁</div>
      <div>
        <div class="rpt-logo-text">PulmoScan AI</div>
        <div class="rpt-logo-sub">AI-Assisted Radiology Report</div>
      </div>
    </div>
    <div class="rpt-meta">
      <div class="rpt-meta-item">
        <label>Report ID</label>
        <span>{report_id}</span>
      </div>
      <div class="rpt-meta-item">
        <label>Date</label>
        <span>{date_str}</span>
      </div>
      <div class="rpt-meta-item">
        <label>Time</label>
        <span>{time_str}</span>
      </div>
      <div class="rpt-meta-item">
        <label>Modality</label>
        <span>Chest X-Ray (CXR)</span>
      </div>
      <div class="rpt-meta-item">
        <label>AI Model</label>
        <span>PulmoScan v1.0</span>
      </div>
    </div>
  </div>

  <!-- Patient Info -->
  <div class="rpt-patient">
    <div class="rpt-patient-col">
      <div class="field-label">Patient Name</div>
      <div class="field-value large">{patient['name']}</div>
    </div>
    <div class="rpt-patient-col">
      <div class="field-label">Age / Gender</div>
      <div class="field-value large">{age_gender}</div>
    </div>
    <div class="rpt-patient-col" style="border-top:1px solid #e8ecf0;">
      <div class="field-label">Referring Physician</div>
      <div class="field-value">{patient.get('ref_doc','—')}</div>
    </div>
    <div class="rpt-patient-col" style="border-top:1px solid #e8ecf0; border-right:none;">
      <div class="field-label">Clinical Notes</div>
      <div class="field-value">{patient.get('clinical_notes','—') or '—'}</div>
    </div>
  </div>

  <!-- Scan Image + Findings -->
  <div class="rpt-section">
    <div class="rpt-section-title">Radiological Findings</div>
    <div>
      <div class="rpt-image-box">
        <img src="data:image/png;base64,{img_b64}" alt="X-Ray Scan">
      </div>
      <table class="findings">
        <thead>
          <tr>
            <th>Finding</th>
            <th>Description</th>
            <th style="text-align:center;">Severity</th>
            <th style="text-align:center;">Confidence</th>
          </tr>
        </thead>
        <tbody>
          {findings_html}
        </tbody>
      </table>
      <div style="clear:both;"></div>
    </div>
  </div>

  <!-- Conclusion Banner -->
  <div class="rpt-conclusion">
    <div class="rpt-conclusion-icon">{conclusion_icon}</div>
    <div class="rpt-conclusion-text">{conclusion_text}</div>
  </div>

  <!-- Impression -->
  <div class="rpt-section">
    <div class="rpt-section-title">Clinical Impression</div>
    <div class="impression-text">{impression}</div>
  </div>

  <!-- Disclaimer -->
  <div class="rpt-disclaimer">
    <strong>⚠ Important Disclaimer:</strong> This report is generated by an AI-assisted diagnostic tool and is intended to support, not replace, clinical judgment. All findings must be reviewed and confirmed by a qualified radiologist or treating physician. This system is not approved for standalone clinical diagnosis.
  </div>

  <!-- Footer -->
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
    <div style="text-align:center; padding:10px 0 20px;">
      <div style="font-size:36px; margin-bottom:6px;">🫁</div>
      <div style="font-family:'Playfair Display',serif; font-size:1.2rem; color:var(--text,#e8edf5); font-weight:700;">PulmoScan AI</div>
      <div style="font-size:0.65rem; letter-spacing:2px; color:var(--teal,#00b4d8); text-transform:uppercase;">Chest X-Ray Diagnostics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-tag">Patient Information</div>', unsafe_allow_html=True)
    p_name   = st.text_input("Full Name", "Ahmed Khan", placeholder="Enter patient name")
    col_a, col_b = st.columns(2)
    with col_a:
        p_age = st.number_input("Age", 1, 120, 35)
    with col_b:
        p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    st.markdown('<div class="section-tag" style="margin-top:14px;">Clinical Details</div>', unsafe_allow_html=True)
    p_ref_doc = st.text_input("Referring Physician", placeholder="Dr. Name")
    p_notes   = st.text_area("Clinical Notes", placeholder="Symptoms, history…", height=80)

    st.divider()
    st.markdown('<div class="section-tag">AI Settings</div>', unsafe_allow_html=True)
    threshold = st.slider("Detection Threshold", 0.10, 0.90, 0.20, 0.05,
                          help="Lower = more sensitive, higher = more specific")
    st.caption(f"Current threshold: **{threshold:.0%}** confidence")


# ─── Main Layout ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pulsoscan-header">
  <div class="pulsoscan-logo">🫁</div>
  <div>
    <div class="pulsoscan-title">PulmoScan AI</div>
    <div class="pulsoscan-subtitle">AI-Powered Radiology Analysis</div>
  </div>
</div>
""", unsafe_allow_html=True)

col_status1, col_status2, col_status3 = st.columns([2, 2, 3])
with col_status1:
    st.markdown(f'<div class="status-badge"><div class="status-dot"></div>System Online</div>', unsafe_allow_html=True)
with col_status2:
    st.caption(f"📅 {datetime.now().strftime('%d %b %Y  %H:%M')}")

st.markdown("---")

# ─── Upload Area ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-tag">Scan Upload</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop or click to upload a Chest X-Ray (JPG / PNG)",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed"
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    col_img, col_ctrl = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="scan-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-tag">X-Ray Scan</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.caption(f"📁 {uploaded.name}  •  {img.size[0]}×{img.size[1]} px")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_ctrl:
        st.markdown('<div class="scan-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-tag">Patient</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="patient-chip">
          <div class="patient-avatar">👤</div>
          <div>
            <div class="patient-name">{p_name}</div>
            <div class="patient-meta">{p_age} yrs · {p_gender}{' · ' + p_ref_doc if p_ref_doc else ''}</div>
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

                st.markdown('<div class="section-tag" style="margin-top:18px;">Findings</div>', unsafe_allow_html=True)

                if positives:
                    for label, conf in positives:
                        sev = SEVERITY.get(label, "LOW")
                        sev_emoji = {"HIGH": "🔴", "MODERATE": "🟠", "LOW": "🟡"}[sev]
                        st.markdown(f"""
                        <div class="finding-item">
                          <div class="finding-dot"></div>
                          <div class="finding-label">{label.replace('_', ' ')}</div>
                          <div style="font-size:0.75rem;color:#8899aa;margin-left:4px;">{sev_emoji} {sev}</div>
                          <div class="finding-conf">{conf:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="normal-badge">
                      <span class="normal-icon">✅</span>
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

                # Download button
                st.download_button(
                    label="📄  Download Medical Report (HTML)",
                    data=report_html,
                    file_name=f"PulmoScan_Report_{p_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.html",
                    mime="text/html",
                    use_container_width=True,
                )

                # Open & print in new tab
                st.markdown(f"""
                <div style="margin-top:8px;">
                  <a href="data:text/html;base64,{report_b64}" 
                     target="_blank"
                     style="display:block; width:100%; text-align:center;
                            background:rgba(0,180,216,0.12); border:1px solid rgba(0,180,216,0.3);
                            color:#00b4d8; padding:10px; border-radius:8px;
                            text-decoration:none; font-weight:600; font-size:14px;">
                    🖨️  Open & Print Report
                  </a>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.error("❌ Could not load AI model. Check model file / download.")

        st.markdown('</div>', unsafe_allow_html=True)

# ─── Info Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.75rem; padding:10px 0 20px;">
  PulmoScan AI &nbsp;•&nbsp; For research & clinical support use only &nbsp;•&nbsp; 
  Not a substitute for qualified radiologist review
</div>
""", unsafe_allow_html=True)
