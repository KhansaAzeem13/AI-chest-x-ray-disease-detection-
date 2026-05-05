import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
import tf_keras as keras
from datetime import datetime
import io
import base64

st.set_page_config(
    page_title="AI Chest X-Ray Diagnostic System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.main { background: #f0f4f8; }

.report-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
    color: white;
    padding: 30px 40px;
    border-radius: 12px;
    margin-bottom: 25px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.hospital-name {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: 1px;
}

.hospital-sub {
    font-size: 13px;
    opacity: 0.8;
    margin-top: 4px;
}

.report-id {
    text-align: right;
    font-size: 13px;
    opacity: 0.9;
}

.section-card {
    background: white;
    border-radius: 10px;
    padding: 20px 25px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 4px solid #2d6a9f;
}

.section-title {
    color: #1a3a5c;
    font-size: 14px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 15px;
    padding-bottom: 8px;
    border-bottom: 1px solid #e8edf2;
}

.patient-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
}

.patient-field {
    background: #f8fafc;
    padding: 10px 14px;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
}

.field-label {
    font-size: 11px;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.field-value {
    font-size: 15px;
    color: #1e293b;
    font-weight: 500;
    margin-top: 3px;
}

.finding-positive {
    background: #fff5f5;
    border: 1px solid #feb2b2;
    border-left: 4px solid #e53e3e;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 6px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.finding-negative {
    background: #f0fff4;
    border: 1px solid #9ae6b4;
    border-left: 4px solid #38a169;
    padding: 10px 16px;
    border-radius: 6px;
    margin: 4px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.finding-name {
    font-weight: 600;
    font-size: 14px;
}

.finding-prob {
    font-size: 13px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
}

.prob-high { background: #fed7d7; color: #c53030; }
.prob-low  { background: #c6f6d5; color: #276749; }

.conclusion-box {
    background: linear-gradient(135deg, #fff8e1, #fff3cd);
    border: 1px solid #f6cc4f;
    border-left: 4px solid #d69e2e;
    padding: 16px 20px;
    border-radius: 8px;
    margin: 10px 0;
}

.disclaimer-box {
    background: #fff5f5;
    border: 1px solid #feb2b2;
    padding: 12px 18px;
    border-radius: 8px;
    font-size: 12px;
    color: #742a2a;
    text-align: center;
    margin-top: 15px;
}

.metric-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px 20px;
    border-radius: 10px;
    text-align: center;
}

.metric-number {
    font-size: 32px;
    font-weight: 700;
}

.metric-label {
    font-size: 12px;
    opacity: 0.85;
    margin-top: 2px;
}

.stButton > button {
    background: linear-gradient(135deg, #1a3a5c, #2d6a9f);
    color: white;
    border: none;
    padding: 12px 30px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 15px;
    width: 100%;
    cursor: pointer;
    transition: all 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(45,106,159,0.4);
}

.upload-zone {
    border: 2px dashed #2d6a9f;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    background: #f8fafc;
    margin: 10px 0;
}

@media print {
    .stSidebar, .stButton, header { display: none !important; }
    .main { background: white !important; }
}
</style>
""", unsafe_allow_html=True)

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

DISEASE_INFO = {
    "Atelectasis":       "Partial or complete collapse of lung",
    "Cardiomegaly":      "Enlargement of the heart",
    "Effusion":          "Fluid in the pleural space",
    "Infiltration":      "Substance denser than air in lungs",
    "Mass":              "Lesion > 3cm in diameter",
    "Nodule":            "Small round lesion < 3cm",
    "Pneumonia":         "Infection causing lung inflammation",
    "Pneumothorax":      "Air in the pleural space",
    "Consolidation":     "Lung tissue filled with fluid",
    "Edema":             "Fluid accumulation in lung tissue",
    "Emphysema":         "Damage to air sacs in lungs",
    "Fibrosis":          "Scarring of lung tissue",
    "Pleural_Thickening":"Thickening of pleural lining",
    "Hernia":            "Protrusion of organ through cavity"
}

@st.cache_resource
def load_model():
    return keras.models.load_model("model/best_model.keras")

def generate_gradcam(model, img_array, class_idx):
    last_conv = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv = layer
            break
    grad_model = keras.Model(
        inputs=[model.inputs],
        outputs=[last_conv.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, class_idx]
    grads   = tape.gradient(loss, conv_out)
    pooled  = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(img_array, heatmap, alpha=0.4):
    h = cv2.resize(heatmap, (224, 224))
    h = cv2.applyColorMap(np.uint8(255*h), cv2.COLORMAP_JET)
    h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    return (alpha * h + (1-alpha) * img_array).astype(np.uint8)

def img_to_base64(img_array):
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 AI Diagnostic System")
    st.markdown("---")
    st.markdown("#### 👤 Patient Information")

    patient_name   = st.text_input("Patient Full Name", placeholder="e.g. Ahmed Khan")
    col_a, col_b   = st.columns(2)
    with col_a:
        patient_age = st.number_input("Age", min_value=1, max_value=120, value=35)
    with col_b:
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    patient_id     = st.text_input("Patient ID", placeholder="e.g. PT-2024-001")
    referring_doc  = st.text_input("Referring Doctor", placeholder="e.g. Dr. Ali Raza")
    clinical_notes = st.text_area("Clinical Notes", placeholder="Symptoms, history...", height=80)

    st.markdown("---")
    st.markdown("#### ⚙️ Analysis Settings")
    threshold     = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)
    show_all      = st.checkbox("Show all diseases", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#64748b; text-align:center;'>
    🤖 Powered by DenseNet121<br>
    📊 NIH ChestX-ray14 Dataset<br>
    🔬 Grad-CAM Explainability
    </div>
    """, unsafe_allow_html=True)

# ── Main Header ───────────────────────────────────────────
report_date = datetime.now().strftime("%B %d, %Y")
report_time = datetime.now().strftime("%I:%M %p")
report_no   = f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"

st.markdown(f"""
<div class="report-header">
    <div>
        <div class="hospital-name">🏥 AI Chest X-Ray Diagnostic Center</div>
        <div class="hospital-sub">Advanced Deep Learning Radiology System | DenseNet121 + Grad-CAM</div>
        <div class="hospital-sub">📍 Digital Health Platform | 24/7 AI-Assisted Analysis</div>
    </div>
    <div class="report-id">
        <div style="font-size:18px; font-weight:700;">RADIOLOGY REPORT</div>
        <div>Report No: <b>{report_no}</b></div>
        <div>Date: {report_date} | {report_time}</div>
        <div>Type: Chest PA View</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Patient Info Card ─────────────────────────────────────
st.markdown(f"""
<div class="section-card">
    <div class="section-title">📋 Patient Information</div>
    <div class="patient-grid">
        <div class="patient-field">
            <div class="field-label">Patient Name</div>
            <div class="field-value">{patient_name if patient_name else '—'}</div>
        </div>
        <div class="patient-field">
            <div class="field-label">Age / Gender</div>
            <div class="field-value">{patient_age} Years / {patient_gender}</div>
        </div>
        <div class="patient-field">
            <div class="field-label">Patient ID</div>
            <div class="field-value">{patient_id if patient_id else '—'}</div>
        </div>
        <div class="patient-field">
            <div class="field-label">Referring Doctor</div>
            <div class="field-value">{referring_doc if referring_doc else '—'}</div>
        </div>
        <div class="patient-field">
            <div class="field-label">Study Date</div>
            <div class="field-value">{report_date}</div>
        </div>
        <div class="patient-field">
            <div class="field-label">Clinical Notes</div>
            <div class="field-value">{clinical_notes[:40]+'...' if len(clinical_notes)>40 else clinical_notes if clinical_notes else '—'}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Upload Section ────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">📤 Upload Chest X-Ray</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((224, 224))
    img_array = np.array(img)
    img_input = np.expand_dims(img_array/255.0, 0).astype(np.float32)

    col_prev, col_btn = st.columns([2,1])
    with col_prev:
        st.image(img, caption="Uploaded X-Ray", width=250)
    with col_btn:
        st.markdown("<br><br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔬 Generate Full Report", type="primary")

    if analyze_btn:
        with st.spinner("🧠 AI analyzing X-Ray... Generating report..."):
            model  = load_model()
            preds  = model.predict(img_input, verbose=0)[0]
            top_idx = int(np.argmax(preds))
            heatmap = generate_gradcam(model, img_input, top_idx)
            overlay = overlay_heatmap(img_array, heatmap)

        positives = [(LABELS[i], preds[i]) for i in range(14) if preds[i] >= threshold]
        positives.sort(key=lambda x: x[1], reverse=True)
        negatives = [(LABELS[i], preds[i]) for i in range(14) if preds[i] < threshold]

        st.markdown("---")

        # ── Summary Metrics ───────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg,#e53e3e,#c53030)">
                <div class="metric-number">{len(positives)}</div>
                <div class="metric-label">Findings Detected</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg,#38a169,#276749)">
                <div class="metric-number">{14-len(positives)}</div>
                <div class="metric-label">Normal Findings</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg,#d69e2e,#b7791f)">
                <div class="metric-number">{preds[top_idx]*100:.0f}%</div>
                <div class="metric-label">Top Confidence</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg,#667eea,#764ba2)">
                <div class="metric-number">AI</div>
                <div class="metric-label">DenseNet121</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Images ────────────────────────────────────────
        st.markdown('<div class="section-card"><div class="section-title">🖼️ Radiological Images</div>', unsafe_allow_html=True)
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.image(img_array, caption="Original X-Ray (PA View)", use_column_width=True)
        with ic2:
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * cv2.resize(heatmap, (224,224))), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            st.image(heatmap_rgb, caption="Grad-CAM Attention Map", use_column_width=True)
        with ic3:
            st.image(overlay, caption="AI Overlay (Abnormality Regions)", use_column_width=True)
        st.markdown("""
        <p style='font-size:11px; color:#64748b; text-align:center; margin-top:8px;'>
        🔴 Red/Yellow = High AI attention regions | 🔵 Blue = Low attention regions
        </p></div>""", unsafe_allow_html=True)

        # ── Findings ──────────────────────────────────────
        fc1, fc2 = st.columns(2)

        with fc1:
            st.markdown('<div class="section-card"><div class="section-title">⚠️ Positive Findings</div>', unsafe_allow_html=True)
            if positives:
                for name, prob in positives:
                    st.markdown(f"""
                    <div class="finding-positive">
                        <div>
                            <div class="finding-name">🔴 {name}</div>
                            <div style="font-size:11px; color:#718096;">{DISEASE_INFO[name]}</div>
                        </div>
                        <span class="finding-prob prob-high">{prob*100:.1f}%</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='color:#38a169; font-weight:600; padding:10px;'>
                ✅ No significant findings above threshold
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with fc2:
            st.markdown('<div class="section-card"><div class="section-title">✅ Negative Findings</div>', unsafe_allow_html=True)
            for name, prob in negatives[:7]:
                st.markdown(f"""
                <div class="finding-negative">
                    <div class="finding-name" style="color:#276749;">✓ {name}</div>
                    <span class="finding-prob prob-low">{prob*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Probability Chart ─────────────────────────────
        st.markdown('<div class="section-card"><div class="section-title">📊 Disease Probability Analysis</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(14, 5))
        colors  = ['#e53e3e' if p >= threshold else '#38a169' for p in preds]
        bars    = ax.bar(LABELS, preds*100, color=colors, width=0.6, edgecolor='white', linewidth=0.5)
        ax.axhline(y=threshold*100, color='#d69e2e', linestyle='--',
                   linewidth=1.5, label=f'Threshold ({threshold*100:.0f}%)')
        for bar, prob in zip(bars, preds):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                   f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='600')
        ax.set_ylabel('Probability (%)', fontsize=11)
        ax.set_ylim(0, 110)
        ax.set_title('AI Disease Probability Scores — DenseNet121', fontsize=13, fontweight='bold', pad=15)
        plt.xticks(rotation=35, ha='right', fontsize=9)
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#f8fafc')
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Conclusion ────────────────────────────────────
        st.markdown('<div class="section-card"><div class="section-title">📝 AI Conclusion & Impression</div>', unsafe_allow_html=True)

        if positives:
            primary = positives[0][0]
            conclusion_text = f"""
            The AI analysis of the chest radiograph reveals <b>{len(positives)} significant finding(s)</b>.
            The primary finding is <b>{primary}</b> with a confidence of <b>{positives[0][1]*100:.1f}%</b>.
            {f'Additional findings include: {", ".join([p[0] for p in positives[1:]])}.' if len(positives) > 1 else ''}
            The Grad-CAM visualization highlights the regions of interest in the radiograph.
            """
        else:
            conclusion_text = """
            The AI analysis of the chest radiograph reveals <b>no significant pathological findings</b>
            above the detection threshold. The lung fields appear within normal limits as assessed by
            the DenseNet121 model.
            """

        st.markdown(f"""
        <div class="conclusion-box">
            <b>IMPRESSION:</b><br>
            {conclusion_text}
        </div>
        <div style='margin-top:15px; padding:12px; background:#f8fafc; border-radius:6px;'>
            <b>Technique:</b> AI-assisted analysis using DenseNet121 Transfer Learning model
            trained on NIH ChestX-ray14 dataset (112,120 images, 14 pathology classes).<br>
            <b>Explainability:</b> Grad-CAM gradient visualization applied to highlight
            model attention regions.
        </div>
        </div>""", unsafe_allow_html=True)

        # ── Signature ─────────────────────────────────────
        st.markdown(f"""
        <div class="section-card">
            <div class="section-title">✍️ Report Authorization</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:10px;">
                <div style="text-align:center; padding:20px; border:1px dashed #cbd5e0; border-radius:8px;">
                    <div style="font-size:12px; color:#64748b;">AI System</div>
                    <div style="font-size:16px; font-weight:700; color:#1a3a5c; margin:8px 0;">
                    🤖 DenseNet121 AI</div>
                    <div style="font-size:11px; color:#64748b;">Automated Analysis Engine</div>
                    <div style="font-size:11px; color:#64748b;">{report_date} | {report_time}</div>
                </div>
                <div style="text-align:center; padding:20px; border:1px dashed #cbd5e0; border-radius:8px;">
                    <div style="font-size:12px; color:#64748b;">Referring Physician</div>
                    <div style="font-size:16px; font-weight:700; color:#1a3a5c; margin:8px 0;">
                    👨‍⚕️ {referring_doc if referring_doc else 'Dr. ___________'}</div>
                    <div style="font-size:11px; color:#64748b;">MBBS, FCPS Radiology</div>
                    <div style="font-size:11px; color:#64748b;">Signature: ___________</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Disclaimer ────────────────────────────────────
        st.markdown("""
        <div class="disclaimer-box">
            ⚠️ <b>IMPORTANT DISCLAIMER:</b> This report is generated by an AI system for
            educational and research purposes only. It should NOT be used as a substitute
            for professional medical diagnosis. Always consult a qualified radiologist or
            physician for clinical decisions. The AI model was trained on a subset dataset
            and may not reflect clinical accuracy.
        </div>""", unsafe_allow_html=True)

        # ── Print Button ──────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center;">
            <button onclick="window.print()"
            style="background:linear-gradient(135deg,#1a3a5c,#2d6a9f);
            color:white; border:none; padding:14px 40px; border-radius:8px;
            font-size:16px; font-weight:600; cursor:pointer;
            box-shadow:0 4px 15px rgba(26,58,92,0.3);">
            🖨️ Print / Download Report as PDF
            </button>
        </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-zone">
        <div style="font-size:50px;">🫁</div>
        <div style="font-size:20px; font-weight:600; color:#1a3a5c; margin:10px 0;">
        Upload Chest X-Ray to Begin Analysis</div>
        <div style="color:#64748b; font-size:14px;">
        Supported formats: PNG, JPG, JPEG<br>
        Fill patient information in the sidebar before uploading
        </div>
    </div>""", unsafe_allow_html=True)
