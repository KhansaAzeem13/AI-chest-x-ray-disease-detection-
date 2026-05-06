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

# --- Page Config ---
st.set_page_config(
    page_title="AI Chest X-Ray Diagnostic System",
    page_icon="🫁",
    layout="wide"
)

# --- JavaScript for User Friendliness ---
def inject_js():
    js_code = """
    <script>
    // 1. Smooth Scroll to Results
    const scrollToResults = () => {
        window.parent.document.querySelectorAll('.stAlert').forEach(el => {
            el.scrollIntoView({behavior: 'smooth'});
        });
    }

    // 2. Custom UI tweaks
    const parentDoc = window.parent.document;
    const scrollObserver = new MutationObserver(() => {
        const btn = parentDoc.querySelector('button[kind="primary"]');
        if (btn) {
            btn.addEventListener('click', () => {
                setTimeout(scrollToResults, 1500);
            });
        }
    });
    scrollObserver.observe(parentDoc.body, {childList: true, subtree: true});
    </script>
    """
    components.html(js_code, height=0)

# --- Custom CSS (Print Support & Medical Theme) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
    
    .report-card { 
        background: white; padding: 30px; border-radius: 15px; 
        box-shadow: 0 10px 25px rgba(0,0,0,0.1); border-top: 8px solid #1a3a5c;
    }
    
    /* Hide elements during Print */
    @media print {
        .stSidebar, .stFileUpload, button, header, .stMarkdown:first-child { display: none !important; }
        .report-card { box-shadow: none !important; border: 1px solid #eee !important; width: 100% !important; }
        .main { background: white !important; }
    }
</style>
""", unsafe_allow_html=True)

# Call JS
inject_js()

# --- Model Loading Logic ---
@st.cache_resource
def load_trained_model():
    model_path = "best_model.keras"
    file_id = "1T-mLtAXELB734rsr2HlIqH6bxYLKbjYP"
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_path):
        with st.spinner("🧠 AI Model is being prepared... please wait."):
            try:
                gdown.download(url, model_path, quiet=False)
            except: return None
    return keras.models.load_model(model_path)

# --- Labels ---
LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2868/2868233.png", width=100)
    st.title("Patient Portal")
    p_name = st.text_input("Full Name", "Ahmed Khan")
    p_age = st.number_input("Age", 1, 100, 35)
    st.divider()
    threshold = st.slider("Sensitivity Threshold", 0.1, 0.9, 0.5)

# --- Main App ---
st.title("🏥 AI Radiology Analysis")
st.write(f"System Status: **Ready** | Date: {datetime.now().strftime('%d-%m-%Y')}")

# Report Container
st.markdown('<div class="report-card">', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload Chest X-Ray (DICOM/JPG/PNG conversion)", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(img, caption="Patient X-Ray Scan", use_container_width=True)
    
    with col2:
        st.info(f"**Patient:** {p_name} | **Age:** {p_age}")
        if st.button("🔬 RUN AI DIAGNOSIS", type="primary"):
            model = load_trained_model()
            if model:
                # Preprocessing
                img_proc = np.array(img.resize((224, 224))) / 255.0
                img_proc = np.expand_dims(img_proc, axis=0)
                
                # Predict
                preds = model.predict(img_proc)[0]
                
                st.subheader("Findings:")
                positives = [LABELS[i] for i, p in enumerate(preds) if p >= threshold]
                
                if positives:
                    for disease in positives:
                        st.error(f"● {disease} Detected")
                else:
                    st.success("✅ No abnormalities detected above threshold.")

                # JS Print Button
                st.markdown("""
                    <hr>
                    <div style="display: flex; gap: 10px;">
                        <button onclick="window.print()" style="padding: 10px 20px; background: #1a3a5c; color: white; border: none; border-radius: 5px; cursor: pointer; flex: 1;">
                            🖨️ Print Medical Report
                        </button>
                        <button onclick="window.location.reload()" style="padding: 10px 20px; background: #f0f2f6; color: #333; border: 1px solid #ccc; border-radius: 5px; cursor: pointer; flex: 1;">
                            🔄 New Scan
                        </button>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Error connecting to AI Server. Please check model ID.")
st.markdown('</div>', unsafe_allow_html=True)
