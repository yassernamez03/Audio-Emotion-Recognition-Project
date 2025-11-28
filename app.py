import streamlit as st
import numpy as np
import pickle
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from model_utils import extract_features, StandardScalerCustom, KNNClassifier, LogisticRegression, SVM, PolynomialFeatureGenerator

# ------------------------------------------------------------------------------
# Configuration & Setup
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Audio Emotion Recognition",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design with dark/light mode support
st.markdown("""
    <style>
    /* Root variables for theming */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --danger-color: #ef4444;
        --warning-color: #f59e0b;
    }
    
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Dark mode specific */
    [data-testid="stAppViewContainer"][data-theme="dark"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stAppViewContainer"][data-theme="dark"] .stMarkdown h1 {
        color: #f8f9fa;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stAppViewContainer"][data-theme="dark"] .stMarkdown h3 {
        color: #e2e8f0;
    }
    
    /* Light mode specific */
    [data-testid="stAppViewContainer"][data-theme="light"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%);
    }
    
    [data-testid="stAppViewContainer"][data-theme="light"] .stMarkdown h1 {
        color: #1e293b;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stAppViewContainer"][data-theme="light"] .stMarkdown h3 {
        color: #334155;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }
    
    /* Prediction box - Dark mode */
    [data-theme="dark"] .prediction-box {
        background: rgba(30, 30, 46, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    /* Prediction box - Light mode */
    [data-theme="light"] .prediction-box {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .prediction-box:hover {
        transform: scale(1.02);
    }
    
    /* Emotion text styling */
    .emotion-text {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 2px;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* Sidebar styling - Dark mode */
    [data-theme="dark"] [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2e 0%, #2d2d44 100%);
    }
    
    /* Sidebar styling - Light mode */
    [data-theme="light"] [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
    }
    
    /* Info box styling - Dark mode */
    [data-theme="dark"] .stAlert {
        background: rgba(99, 102, 241, 0.1);
        border-left: 4px solid var(--primary-color);
        border-radius: 8px;
    }
    
    /* Info box styling - Light mode */
    [data-theme="light"] .stAlert {
        background: rgba(99, 102, 241, 0.05);
        border-left: 4px solid var(--primary-color);
        border-radius: 8px;
    }
    
    /* Audio player styling */
    audio {
        width: 100%;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    /* Chart/Plot container */
    [data-testid="stImage"], [data-testid="stPlotlyChart"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--primary-color) !important;
    }
    
    /* Success message */
    .element-container .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
        border-left: 4px solid var(--success-color);
    }
    </style>
    """, unsafe_allow_html=True)

MODEL_PATH = "model_parameters.pkl"
SCALER_PATH = "scaler.pkl"

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

def plot_waveform(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='blue', alpha=0.6)
    ax.set_title("Waveform", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

def plot_probabilities(probs, classes):
    # If model supports probability (Logistic Regression)
    if probs is None:
        return
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=classes, y=probs, ax=ax, palette="viridis")
    ax.set_title("Emotion Probabilities", fontsize=14)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# ------------------------------------------------------------------------------
# Main App
# ------------------------------------------------------------------------------

def main():
    # Header
    st.title("Audio Emotion Recognition System")
    st.markdown("### Advanced ML-based emotion detection from audio files")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("About This Application")    

    # Load Model
    model_artifact, scaler = load_artifacts()
    
    if model_artifact is None:
        st.error("Model artifacts not found! Please run the training script/notebook first.")
        return

    # Handle new artifact structure (dict) or old (model object)
    if isinstance(model_artifact, dict):
        model = model_artifact.get('model')
        poly_generator = model_artifact.get('poly_generator')
        model_name = model_artifact.get('model_name', 'Unknown')
    else:
        # Fallback for old artifacts
        model = model_artifact
        poly_generator = None
        model_name = "Custom Model"

    st.sidebar.success(f"Loaded Model: {model_name}")

    # File Uploader
    st.markdown("### Upload Audio File")
    uploaded_file = st.file_uploader("Choose a WAV file for emotion analysis", type=["wav"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        # Save temp file
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.markdown("---")
        
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Audio Playback")
            st.audio("temp.wav", format='audio/wav')
            
            st.markdown("#### Analysis Results")
            # Extract features
            with st.spinner("Analyzing audio features..."):
                features = extract_features("temp.wav")
                
            if features is not None:
                # Scale
                features_scaled = scaler.transform([features])
                
                # Apply Polynomial Features if needed
                if poly_generator is not None:
                    features_final = poly_generator.transform(features_scaled)
                else:
                    features_final = features_scaled
                
                # Predict
                prediction = model.predict(features_final)[0]
                
                st.markdown("#### Detected Emotion")
                st.markdown(f"<div class='prediction-box'><span class='emotion-text'>{prediction.upper()}</span></div>", unsafe_allow_html=True)
                
                # Show feature count
                st.info(f"Analyzed {len(features)} handcrafted audio features")
                
                # Probabilities (if available)
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(features_final)[0]
                    # Assuming model.classes is available or we know the order
                    # For our custom LR, classes are stored in model.classes
                    if hasattr(model, 'classes'):
                        st.markdown("#### Confidence Distribution")
                        # Create a dataframe for better visualization
                        probs_dict = {cls.capitalize(): prob for cls, prob in zip(model.classes, probs)}
                        st.bar_chart(probs_dict, height=300)
            else:
                st.error("Could not extract features from the audio file. Please ensure it's a valid WAV file.")

        with col2:
            st.markdown("#### Audio Visualizations")
            
            # Waveform
            plot_waveform("temp.wav")
            
            # Show feature stats
            if features is not None:
                st.markdown("#### Feature Analysis")
                
                # MFCC visualization
                st.markdown("**MFCC Coefficients (Mean)**")
                fig_mfcc, ax_mfcc = plt.subplots(figsize=(10, 3))
                ax_mfcc.plot(features[:40], color='#6366f1', linewidth=2)
                ax_mfcc.set_title("MFCC Mean Values", fontsize=12, fontweight='bold')
                ax_mfcc.set_xlabel("Coefficient Index")
                ax_mfcc.set_ylabel("Value")
                ax_mfcc.grid(True, alpha=0.3)
                ax_mfcc.set_facecolor('#f8f9fa')
                fig_mfcc.tight_layout()
                st.pyplot(fig_mfcc)
                
                # Feature statistics
                st.markdown("**Feature Statistics**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("MFCC Features", "80")
                with col_b:
                    st.metric("Chroma Features", "24")
                with col_c:
                    st.metric("Spectral Features", "8")

if __name__ == "__main__":
    main()
