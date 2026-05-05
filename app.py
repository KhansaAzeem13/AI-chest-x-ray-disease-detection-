import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

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
    import tensorflow as tf
    import tf_keras as keras
    return keras.models.load_model("model/best_model.keras")

def generate_gradcam(model, img_array, class_idx):
    import tensorflow as tf
    import tf_keras as keras
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

# ── UI ───────────────────────────────────────────────────
st.title("🫁 Chest X-Ray AI Diagnostic System")
st.markdown("**DenseNet121 Transfer Learning** + **Grad-CAM Explainability** | NIH ChestX-ray14")
st.markdown("---")

with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    **Model:** DenseNet121
    **Dataset:** NIH ChestX-ray14
    **Diseases Detected:** 14
    **Explainability:** Grad-CAM
    **Framework:** TensorFlow + tf-keras
    """)
    st.warning("⚠️ Educational use only. Not for clinical diagnosis.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 Upload Chest X-Ray")
    uploaded = st.file_uploader("PNG / JPG / JPEG", type=["png","jpg","jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB").resize((224,224))
        st.image(img, caption="Uploaded X-Ray", use_column_width=True)
        analyze = st.button("🔍 Analyze", type="primary")

if uploaded and analyze:
    model = load_model()
    img_array = np.array(img)
    img_input = np.expand_dims(img_array/255.0, 0).astype(np.float32)

    with st.spinner("🧠 Analyzing..."):
        preds = model.predict(img_input, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        heatmap  = generate_gradcam(model, img_input, top_idx)
        overlay  = overlay_heatmap(img_array, heatmap)

    with col2:
        st.subheader("🔬 Results")
        confidence = float(preds[top_idx]) * 100
        if confidence > 50:
            st.error(f"🔴 **{LABELS[top_idx]}** detected ({confidence:.1f}%)")
        else:
            st.success(f"🟢 **{LABELS[top_idx]}** ({confidence:.1f}% confidence)")

        st.subheader("📊 All Disease Probabilities")
        fig, ax = plt.subplots(figsize=(10,5))
        colors = ["crimson" if p > 0.5 else "steelblue" for p in preds]
        ax.barh(LABELS, preds*100, color=colors)
        ax.axvline(x=50, color="red", linestyle="--", alpha=0.7)
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, 100)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("🔥 Grad-CAM Heatmap")
        c1, c2 = st.columns(2)
        with c1:
            st.image(img_array, caption="Original", use_column_width=True)
        with c2:
            st.image(overlay, caption="AI Attention", use_column_width=True)
        st.caption("🔴 Red = high attention | 🔵 Blue = low attention")
