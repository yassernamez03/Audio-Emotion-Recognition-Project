# Streamlit Cloud Deployment Guide

## Prerequisites
1. A GitHub account
2. Your trained model files (`model_parameters.pkl` and `scaler.pkl`)

## Deployment Steps

### 1. Prepare Your Repository
Since the Data folder and model files are large, you'll need to handle them separately:

**Option A: Use Git LFS (Large File Storage)**
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add model_parameters.pkl scaler.pkl
git commit -m "Add model files with LFS"
```

**Option B: Upload models to cloud storage**
- Upload `model_parameters.pkl` and `scaler.pkl` to Google Drive, Dropbox, or AWS S3
- Modify `app.py` to download these files on startup
- Example for Google Drive:
```python
import gdown
# Download model
gdown.download('https://drive.google.com/uc?id=YOUR_FILE_ID', 'model_parameters.pkl', quiet=False)
```

### 2. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit - Audio Emotion Recognition App"
git branch -M main
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

### 3. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path: `app.py`
6. Click "Deploy"

### 4. Important Notes
- **Model Files**: The `.pkl` files are required for the app to work. Make sure they're accessible.
- **Data Folder**: Not needed for deployment (only for training)
- **Memory Limits**: Streamlit Cloud has memory limits. The polynomial model might be large.
- **Upload Size**: Max file upload is set to 200MB in config.toml

## Files Required for Deployment
- ✅ `app.py` - Main application
- ✅ `model_utils.py` - Model classes and utilities
- ✅ `requirements.txt` - Python dependencies
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `model_parameters.pkl` - Trained model (handle separately if large)
- ✅ `scaler.pkl` - Feature scaler

## Files NOT Needed for Deployment
- ❌ `Data/` - Training data (excluded in .gitignore)
- ❌ `train_models.py` - Training script
- ❌ `emotion_recognition.ipynb` - Training notebook
- ❌ `generate_notebook.py` - Notebook generator
- ❌ `__pycache__/` - Python cache

## Troubleshooting
- **App crashes on startup**: Check if model files are accessible
- **Memory errors**: Consider using a simpler model or reducing polynomial degree
- **Slow loading**: Model files might be too large, consider model compression
