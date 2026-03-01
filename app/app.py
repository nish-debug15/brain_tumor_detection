import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pandas as pd

try:
    from tensorflow import keras
    import tensorflow as tf
except ImportError:
    st.error("TensorFlow not installed. Run: pip install tensorflow")
    st.stop()

st.set_page_config(
    page_title="Brain Tumor Classifier · Grad-CAM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background: #0a0a0f; color: #e8e8f0; }
  section[data-testid="stSidebar"] { background: #0f0f1a; border-right: 1px solid #1e1e2e; }
  h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.02em; }
  h1 { color: #00e5ff; font-size: 2rem; }
  h2 { color: #b0b8d8; font-size: 1.1rem; font-weight: 400; }
  h3 { color: #00e5ff; font-size: 0.9rem; }
  div[data-testid="metric-container"] { background: #12121f; border: 1px solid #1e1e35; border-radius: 8px; padding: 1rem; }
  div[data-testid="metric-container"] label { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #6070a0; text-transform: uppercase; letter-spacing: 0.1em; }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; color: #00e5ff; font-size: 1.8rem; }
  [data-testid="stFileUploaderDropzone"] { background: #0f0f1a !important; border: 1px dashed #2a2a4a !important; border-radius: 8px !important; }
  .conf-bar-wrap { background: #12121f; border-radius: 4px; overflow: hidden; height: 8px; margin-top: 4px; }
  .conf-bar-fill { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #0061ff, #00e5ff); }
  .pred-badge { display: inline-block; background: rgba(0,229,255,0.12); border: 1px solid #00e5ff44; color: #00e5ff; font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; padding: 2px 10px; border-radius: 20px; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "final_model.keras")
IMG_SIZE    = (224, 224)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH)

def preprocess(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
            return layer.name
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
                    return sub.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    try:
        grad_model = keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )
    except ValueError:
        return None
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(original_img: Image.Image, heatmap: np.ndarray, alpha=0.45):
    img = np.array(original_img.convert("RGB").resize(IMG_SIZE))
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    colormap        = cm.get_cmap("inferno")
    heatmap_color   = colormap(heatmap_uint8 / 255.0)[:, :, :3]
    heatmap_color   = np.uint8(255 * heatmap_color)
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(superimposed)

with st.sidebar:
    st.markdown("## 🧠 Brain Tumor Classifier")
    st.markdown("---")
    st.markdown("**Model**")
    st.code(MODEL_PATH, language="")
    st.markdown("**Classes**")
    for c in CLASS_NAMES:
        st.markdown(f"- `{c}`")
    st.markdown("---")
    alpha_slider     = st.slider("Grad-CAM overlay strength", 0.1, 0.9, 0.45, 0.05)
    show_raw_heatmap = st.checkbox("Show raw heatmap", value=False)

st.markdown("# 🧠 Brain Tumor Classifier · Grad-CAM")
st.markdown("## Upload an MRI scan — get a prediction and saliency map")
st.markdown("---")

model = load_model()
if model is None:
    st.error(f"Model not found at `{MODEL_PATH}`. Make sure you run streamlit from your project root directory.")
    st.stop()

last_conv = find_last_conv_layer(model)

uploaded = st.file_uploader("Drop an MRI image here (JPG / PNG / WEBP)", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.info("👆  Upload a brain MRI image to get started.")
    st.stop()

original_img = Image.open(uploaded)

with st.spinner("Running inference…"):
    img_array   = preprocess(original_img)
    preds       = model.predict(img_array, verbose=0)[0]
    pred_idx    = int(np.argmax(preds))
    confidence  = float(preds[pred_idx])
    pred_label  = CLASS_NAMES[pred_idx]
    heatmap     = None
    gradcam_img = None
    if last_conv is not None:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv, pred_idx)
        if heatmap is not None:
            gradcam_img = overlay_gradcam(original_img, heatmap, alpha=alpha_slider)

col_img, col_info = st.columns([1.1, 1], gap="large")

with col_img:
    st.markdown("### Original")
    st.image(original_img, use_container_width=True)

with col_info:
    st.markdown("### Prediction")
    m1, m2 = st.columns(2)
    m1.metric("Prediction", pred_label.upper())
    m2.metric("Confidence", f"{confidence * 100:.1f}%")
    bar_pct = int(confidence * 100)
    st.markdown(
        f'<div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{bar_pct}%"></div></div>'
        f'<div class="pred-badge">{pred_label}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### All class probabilities")
    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#12121f")
    colors = ["#00e5ff" if i == pred_idx else "#2a3a5a" for i in range(len(CLASS_NAMES))]
    ax.barh(CLASS_NAMES, preds * 100, color=colors, height=0.55)
    ax.set_xlabel("Confidence (%)", color="#6070a0", fontsize=8)
    ax.tick_params(colors="#b0b8d8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e1e35")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.markdown("---")
st.markdown("### 🔥 Grad-CAM Saliency")

if gradcam_img is None:
    st.warning("Could not generate Grad-CAM — no Conv2D layer found in model.")
else:
    gcol1, gcol2 = st.columns(2, gap="medium")
    with gcol1:
        st.markdown("**Overlay** — model attention on input")
        st.image(gradcam_img, use_container_width=True)
    with gcol2:
        if show_raw_heatmap:
            st.markdown("**Raw heatmap**")
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            fig2.patch.set_facecolor("#0a0a0f")
            im = ax2.imshow(heatmap, cmap="inferno", interpolation="bilinear")
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            ax2.axis("off")
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
        else:
            st.markdown("**Side-by-side**")
            fig3, axes = plt.subplots(1, 2, figsize=(7, 3.5))
            fig3.patch.set_facecolor("#0a0a0f")
            for ax in axes:
                ax.set_facecolor("#0a0a0f")
                ax.axis("off")
            axes[0].imshow(original_img.convert("RGB").resize(IMG_SIZE))
            axes[0].set_title("Original", color="#b0b8d8", fontsize=9)
            axes[1].imshow(gradcam_img)
            axes[1].set_title("Grad-CAM", color="#00e5ff", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)
    if last_conv:
        st.caption(f"Grad-CAM target layer: `{last_conv}`")

with st.expander("📊 Full probability breakdown"):
    df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Score": [f"{p*100:.2f}%" for p in preds],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)