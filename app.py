app_code = '''
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

st.set_page_config(
    page_title="Chest X-Ray AI",
    page_icon="🫁",
    layout="wide"
)

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

@st.cache_resource
def load_model():
    import tf_keras as keras
    return keras.models.load_model("model/best_model.keras")

def generate_gradcam(model, img_array, class_idx):
    import tf_keras
    last_conv = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv = layer
            break
    grad_model = tf_keras.Model(
        inputs=[model.inputs],
        outputs=[last_conv.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(img_array, heatmap, alpha=0.4):
    h = cv2.resize(heatmap, (224, 224))
    h = cv2.applyColorMap(np.uint8(255*h), cv2.COLORMAP_JET)
    h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    return (alpha * h + (1-alpha) * img_array).astype(np.uint8)

# ── Header ───────────────────────────────────────────────
st.title("🫁 Chest X-Ray AI Diagnostic System")
st.markdown("""
**Deep Learning model** that detects **14 thoracic diseases** from chest X-rays  
using **DenseNet121 Transfer Learning** + **Grad-CAM explainability**
""")
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    **Model:** DenseNet121  
    **Dataset:** NIH ChestX-ray14  
    **Diseases:** 14  
    **Explainability:** Grad-CAM  
    **Framework:** TensorFlow/Keras
    """)
    st.warning("⚠️ For educational purposes only. Not for clinical use.")

# ── Main ─────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 Upload Chest X-Ray")
    uploaded = st.file_uploader(
        "Supported: PNG, JPG, JPEG",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded:
        img = Image.open(uploaded).convert("RGB").resize((224, 224))
        st.image(img, caption="Uploaded X-Ray", use_column_width=True)
        analyze = st.button("🔍 Analyze X-Ray", type="primary")

if uploaded and analyze:
    model = load_model()
    img_array = np.array(img)
    img_input = np.expand_dims(img_array/255.0, axis=0).astype(np.float32)

    with st.spinner("🧠 AI is analyzing..."):
        preds = model.predict(img_input, verbose=0)[0]
        top_idx = np.argmax(preds)
        heatmap = generate_gradcam(model, img_input, top_idx)
        overlay = overlay_heatmap(img_array, heatmap)

    with col2:
        st.subheader("🔬 Diagnosis Results")

        # Top finding
        confidence = preds[top_idx] * 100
        if confidence > 50:
            st.error(f"🔴 Primary Finding: **{LABELS[top_idx]}** ({confidence:.1f}%)")
        else:
            st.success(f"🟢 Primary Finding: **{LABELS[top_idx]}** ({confidence:.1f}%)")

        # Bar chart
        st.subheader("📊 Disease Probability Scores")
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["crimson" if p > 0.5 else "steelblue" for p in preds]
        bars = ax.barh(LABELS, preds*100, color=colors)
        ax.axvline(x=50, color="red", linestyle="--", alpha=0.7, label="50% threshold")
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, 100)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        # Grad-CAM
        st.subheader("🔥 Grad-CAM — Where AI is Looking")
        c1, c2 = st.columns(2)
        with c1:
            st.image(img_array, caption="Original X-Ray", use_column_width=True)
        with c2:
            st.image(overlay, caption="AI Attention Heatmap", use_column_width=True)

        st.caption("🔴 Red/Yellow areas = high AI attention | 🔵 Blue = low attention")
'''

with open(f'{project_dir}/app.py', 'w') as f:
    f.write(app_code)

print("✅ app.py created!")
