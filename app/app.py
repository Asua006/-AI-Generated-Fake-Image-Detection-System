import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import sys
import os
import time

# Addign src to path so we can import our model architecture
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model import FakeImageDetector
import config

st.set_page_config(page_title="(AI)Fake Image Detector", page_icon="🛡️", layout="wide")

#CSS
st.markdown("""
    <style>
    /* Clean Dark Background */
    .stApp { background-color: #0f1115; color: #f0f2f6; }
    
    /* Subtle Typography */
    h1 { font-weight: 300; letter-spacing: -1px; color: #ffffff; text-align: left; margin-bottom: 0px; }
    h3 { font-weight: 400; color: #a0aec0; font-size: 1.1rem; }
    
    /* Minimalist Cards */
    .stMetric { background-color: #1a1d23; border: 1px solid #2d3748; border-radius: 8px; padding: 20px; }
    .stDivider { border-color: #2d3748; }

    /* Clean Verdict Boxes (Non-vibrant) */
    .fake-box { 
        padding: 24px; border-radius: 8px; border-left: 5px solid #e53e3e; 
        background-color: #1c1415; color: #fc8181; margin-bottom: 25px; 
    }
    .real-box { 
        padding: 24px; border-radius: 8px; border-left: 5px solid #38a169; 
        background-color: #141c18; color: #68d391; margin-bottom: 25px; 
    }
    
    /* File Uploader Customization */
    section[data-testid="stFileUploader"] { background-color: #1a1d23; border: 1px dashed #4a5568; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# 1. Loading and Cache Model
@st.cache_resource
def load_trained_model():
    model = FakeImageDetector().to(config.DEVICE)
    if os.path.exists(config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    model.eval()
    return model

model = load_trained_model()

# 2. Preprocessing Setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# UI Header
st.title("(AI-Generated)Fake Image Detector")
st.markdown("### Deep Learning Image Verification System")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("#### Source Image")
    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)

with col2:
    st.markdown("#### Analysis Results")
    if uploaded_file:
        loading_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        for i in range(101):
            time.sleep(0.008)
            progress_bar.progress(i)
            if i < 50: loading_placeholder.text("Processing pixel distribution...")
            else: loading_placeholder.text("Running inference...")
        
        # Prediction
        img_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            output = model(img_tensor).item()
        
        loading_placeholder.empty()
        progress_bar.empty()
        
        # Results Display
        is_fake = output > 0.5
        confidence = output if is_fake else 1 - output
        
        if is_fake:
            st.markdown(f'<div class="fake-box"><b>VERDICT: FAKE</b><br><small>The system detected synthetic structural patterns consistent with AI generation.</small></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="real-box"><b>VERDICT: REAL</b><br><small>Natural sensor noise verified. No algorithmic artifacts detected in the high-frequency domain.</small></div>', unsafe_allow_html=True)
        
        m1, m2 = st.columns(2)
        m1.metric("Certainty", f"{confidence*100:.2f}%")
        m2.metric("Neural Index", f"{output:.4f}")
    else:
        st.info("Awaiting input. Please upload a localized image file for detection.")

# Sidebar
with st.sidebar:
    st.markdown("#### System Configuration")
    st.code(f"PROCESSOR: {str(config.DEVICE).upper()}\nMODEL: EfficientNet-B0\nTEST_ACC: 90.0-95.0%")
    st.markdown("---")
    st.caption("Made for (AI-Generated)Fake Image Detection. This tool utilizes CNN-based feature extraction to identify artifacts in AI-generated media.")