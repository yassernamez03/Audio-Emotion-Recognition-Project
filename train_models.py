import os
import glob
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model_utils import extract_features, StandardScalerCustom, KNNClassifier, LogisticRegression, SVM, PolynomialFeatureGenerator

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DATA_PATH = "Data"
MODEL_SAVE_PATH = "model_parameters.pkl"
SCALER_SAVE_PATH = "scaler.pkl"

# Emotion mapping (RAVDESS)
# 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def load_data(data_path):
    print("Loading data...")
    X = []
    y = []
    
    # Find all wav files
    wav_files = glob.glob(os.path.join(data_path, "**/*.wav"), recursive=True)
    
    # Limit for testing if needed, but we want full dataset
    # wav_files = wav_files[:100] 
    
    count = 0
    total = len(wav_files)
    
    for file_path in wav_files:
        filename = os.path.basename(file_path)
        parts = filename.split('-')
        
        if len(parts) != 7:
            continue
            
        emotion_code = parts[2]
        emotion_label = EMOTIONS[emotion_code]
        
        features = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(emotion_label)
            
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{total} files")
            
    return np.array(X), np.array(y)

def main():
    # 1. Load Data
    X, y = load_data(DATA_PATH)
    print(f"Data loaded: {X.shape} samples")
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Scale Data
    scaler = StandardScalerCustom()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    results = {}
    
    # 4. Train Models
    
    # KNN
    print("\nTraining KNN...")
    knn = KNNClassifier(k=5)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    results['KNN'] = acc_knn
    print(f"KNN Accuracy: {acc_knn:.4f}")
    
    # Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(learning_rate=0.01, epochs=500, regularization=0.01)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    results['LogisticRegression'] = acc_lr
    print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
    
    # SVM
    print("\nTraining SVM...")
    svm = SVM(learning_rate=0.001, epochs=500, lambda_param=0.01)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    results['SVM'] = acc_svm
    print(f"SVM Accuracy: {acc_svm:.4f}")
    
    # Polynomial Logistic Regression
    # Note: Full polynomial expansion might be slow, so we'll skip or do a simplified version if needed.
    # For now, let's try it.
    print("\nTraining Polynomial Logistic Regression (Degree 2)...")
    poly = PolynomialFeatureGenerator(degree=2)
    # Warning: This might explode memory if X is large. 112 features -> ~6000 features.
    # Should be fine for ~1400 samples.
    try:
        X_train_poly = poly.transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)
        
        lr_poly = LogisticRegression(learning_rate=0.005, epochs=500, regularization=0.01)
        lr_poly.fit(X_train_poly, y_train)
        y_pred_poly = lr_poly.predict(X_test_poly)
        acc_poly = accuracy_score(y_test, y_pred_poly)
        results['PolynomialLR'] = acc_poly
        print(f"Polynomial LR Accuracy: {acc_poly:.4f}")
    except Exception as e:
        print(f"Skipping Polynomial LR due to error: {e}")
        acc_poly = 0

    # 5. Save Best Model
    best_model_name = max(results, key=results.get)
    print(f"\nBest Model: {best_model_name} with accuracy {results[best_model_name]:.4f}")
    
    best_model = None
    poly_generator = None
    
    if best_model_name == 'KNN':
        best_model = knn
    elif best_model_name == 'LogisticRegression':
        best_model = lr
    elif best_model_name == 'SVM':
        best_model = svm
    elif best_model_name == 'PolynomialLR':
        best_model = lr_poly
        poly_generator = poly
    
    # Save model artifact as a dictionary
    model_artifact = {
        'model': best_model,
        'poly_generator': poly_generator,
        'model_name': best_model_name
    }
    
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model_artifact, f)
        
    print("Training complete. Model artifact and scaler saved.")

if __name__ == "__main__":
    main()
