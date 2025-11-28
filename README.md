# Audio Emotion Recognition Project

This project implements an audio emotion recognition system using the RAVDESS dataset.
It includes a Jupyter Notebook for training models from scratch and a Streamlit application for real-time prediction.

## Live Demo
🚀 [Try the app on Streamlit Cloud](#) *(Add your deployment URL here)*

## Project Structure
- `app.py`: Streamlit web application
- `model_utils.py`: Shared utility functions and model classes
- `emotion_recognition.ipynb`: Jupyter Notebook with full implementation
- `train_models.py`: Script to train models and generate artifacts
- `model_parameters.pkl`: Trained model weights
- `scaler.pkl`: Scaler for feature normalization
- `requirements.txt`: Python dependencies
- `.streamlit/config.toml`: Streamlit configuration
- `Data/`: RAVDESS dataset (not included in deployment)

## Quick Start

### Local Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Training Models (Optional)
```bash
# Using Python script
python train_models.py

# Or use the Jupyter Notebook
jupyter notebook emotion_recognition.ipynb
```

## Deployment
See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions on deploying to Streamlit Cloud.

## Models Implemented (From Scratch)
- **KNN**: K-Nearest Neighbors with Euclidean distance
- **Logistic Regression**: Binary Cross-Entropy, Gradient Descent, One-vs-Rest
- **SVM**: Soft-Margin SVM with Hinge Loss and Gradient Descent
- **Polynomial Logistic Regression**: Manual polynomial feature expansion

## Features Extracted
- **MFCCs** (80 features): Mel-Frequency Cepstral Coefficients
- **Chroma** (24 features): Pitch class profiles
- **Spectral** (8 features): Centroid, Rolloff, Bandwidth, Zero-Crossing Rate
- **Total**: 112 handcrafted features per audio sample

## Supported Emotions
Neutral • Calm • Happy • Sad • Angry • Fearful • Disgust • Surprised

## Technologies
- Python 3.13
- NumPy, Pandas, SciPy
- Librosa (audio processing)
- Streamlit (web interface)
- Matplotlib, Seaborn (visualization)
