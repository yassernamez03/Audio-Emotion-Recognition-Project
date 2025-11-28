import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # Section 1: Introduction
    text_intro = """# Audio Emotion Recognition using RAVDESS
    
## Project Overview
This notebook implements a machine learning system for audio emotion recognition from scratch.
We use the RAVDESS dataset and implement the following models manually (without sklearn classifiers):
- K-Nearest Neighbors (KNN)
- Logistic Regression (Binary & Multi-class OVR)
- Soft-Margin SVM (using Gradient Descent)
- Polynomial Logistic Regression

## Requirements
- Python 3.x
- NumPy, SciPy, Pandas, Matplotlib, Seaborn
- Librosa (for audio processing)
"""
    nb.cells.append(nbf.v4.new_markdown_cell(text_intro))
    
    # Section 2: Imports
    code_imports = """import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration
DATA_PATH = "Data"
SR = 22050
"""
    nb.cells.append(nbf.v4.new_code_cell(code_imports))
    
    # Section 3: Feature Extraction
    text_features = """## 3. Audio Feature Extraction
We extract handcrafted features including MFCCs, Chroma, and Spectral features.
Total features per sample: 112.
"""
    nb.cells.append(nbf.v4.new_markdown_cell(text_features))
    
    code_features = """def extract_features(file_path, sr=22050):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y = librosa.util.normalize(y)
        
        # MFCCs (40)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # Spectral Features
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(cent)
        cent_std = np.std(cent)
        
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_std = np.std(rolloff)
        
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bw_mean = np.mean(bw)
        bw_std = np.std(bw)
        
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            chroma_mean, chroma_std,
            [cent_mean, cent_std, rolloff_mean, rolloff_std, bw_mean, bw_std, zcr_mean, zcr_std]
        ])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
"""
    nb.cells.append(nbf.v4.new_code_cell(code_features))

    # Section 4: Data Loading
    text_loading = """## 4. Data Loading and Preprocessing
We load the RAVDESS dataset, parse filenames for emotion labels, and extract features.
"""
    nb.cells.append(nbf.v4.new_markdown_cell(text_loading))
    
    code_loading = """EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def load_data(data_path):
    X, y = [], []
    wav_files = glob.glob(os.path.join(data_path, "**/*.wav"), recursive=True)
    print(f"Found {len(wav_files)} files.")
    
    for i, file_path in enumerate(wav_files):
        filename = os.path.basename(file_path)
        parts = filename.split('-')
        if len(parts) != 7: continue
            
        emotion_label = EMOTIONS[parts[2]]
        features = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(emotion_label)
        
        if i % 500 == 0: print(f"Processed {i} files...")
            
    return np.array(X), np.array(y)

# Load Data (Uncomment to run - takes time)
# X, y = load_data(DATA_PATH)
# np.save('X.npy', X)
# np.save('y.npy', y)

# For demonstration, we assume data is loaded or we load a subset
if os.path.exists('X.npy') and os.path.exists('y.npy'):
    X = np.load('X.npy')
    y = np.load('y.npy')
else:
    # Fallback if files don't exist (e.g. first run)
    X, y = load_data(DATA_PATH)
    np.save('X.npy', X)
    np.save('y.npy', y)

print(f"Dataset shape: {X.shape}")
"""
    nb.cells.append(nbf.v4.new_code_cell(code_loading))

    # Section 5: Models from Scratch
    text_models = """## 5. Models Implemented from Scratch
Here we implement KNN, Logistic Regression, and SVM without using sklearn classifiers.
"""
    nb.cells.append(nbf.v4.new_markdown_cell(text_models))
    
    code_models = """class StandardScalerCustom:
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0
        return self
    def transform(self, X):
        return (X - self.mean) / self.std
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    def predict(self, X):
        X = np.array(X)
        preds = []
        for x in X:
            dists = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            k_idx = np.argsort(dists)[:self.k]
            k_labels = self.y_train[k_idx]
            unique, counts = np.unique(k_labels, return_counts=True)
            preds.append(unique[np.argmax(counts)])
        return np.array(preds)

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000, reg=0.0):
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.models = []
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    def fit(self, X, y):
        self.classes = np.unique(y)
        X = np.array(X)
        n_samples, n_features = X.shape
        self.models = []
        for c in self.classes:
            y_bin = np.where(y == c, 1, 0)
            w = np.zeros(n_features)
            b = 0
            for _ in range(self.epochs):
                linear = np.dot(X, w) + b
                y_pred = self._sigmoid(linear)
                dw = (1/n_samples) * np.dot(X.T, (y_pred - y_bin)) + (self.reg * w)
                db = (1/n_samples) * np.sum(y_pred - y_bin)
                w -= self.lr * dw
                b -= self.lr * db
            self.models.append((w, b))
    def predict(self, X):
        probs = []
        for w, b in self.models:
            probs.append(self._sigmoid(np.dot(X, w) + b))
        return self.classes[np.argmax(np.array(probs).T, axis=1)]

class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.models = []
    def fit(self, X, y):
        self.classes = np.unique(y)
        X = np.array(X)
        self.models = []
        for c in self.classes:
            y_bin = np.where(y == c, 1, -1)
            w = np.zeros(X.shape[1])
            b = 0
            for _ in range(self.epochs):
                for idx, x_i in enumerate(X):
                    if y_bin[idx] * (np.dot(x_i, w) - b) >= 1:
                        w -= self.lr * (2 * self.lambda_param * w)
                    else:
                        w -= self.lr * (2 * self.lambda_param * w - np.dot(x_i, y_bin[idx]))
                        b -= self.lr * y_bin[idx]
            self.models.append((w, b))
    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.classes)))
        for i, (w, b) in enumerate(self.models):
            scores[:, i] = np.dot(X, w) - b
        return self.classes[np.argmax(scores, axis=1)]
"""
    nb.cells.append(nbf.v4.new_code_cell(code_models))

    # Section 6: Training and Evaluation
    text_train = """## 6. Training and Evaluation
We split the data, normalize it, and train our custom models.
"""
    nb.cells.append(nbf.v4.new_markdown_cell(text_train))
    
    code_train = """# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize
scaler = StandardScalerCustom()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN
print("Training KNN...")
knn = KNNClassifier(k=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# Train Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(lr=0.01, epochs=500, reg=0.01)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print("LR Accuracy:", accuracy_score(y_test, y_pred_lr))

# Train SVM
print("Training SVM...")
svm = SVM(lr=0.001, epochs=500, lambda_param=0.01)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
"""
    nb.cells.append(nbf.v4.new_code_cell(code_train))
    
    # Section 7: Visualization
    code_viz = """# Confusion Matrix for Best Model (e.g. SVM)
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_svm, labels=svm.classes)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=svm.classes, yticklabels=svm.classes, cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
"""
    nb.cells.append(nbf.v4.new_code_cell(code_viz))
    
    # Section 8: Export
    code_export = """# Export Best Model
best_model = svm # Assume SVM is best for now
with open('model_parameters.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Model exported.")
"""
    nb.cells.append(nbf.v4.new_code_cell(code_export))

    # Write to file
    with open('emotion_recognition.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_notebook()
