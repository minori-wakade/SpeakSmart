import librosa
import numpy as np
import joblib
import os

xgb = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel, contrast, tonnetz])

file_path = r"D:\Minori Wakade\3rd sem\IPD\ml-public-speaking-evaluator\ml_training\Fear.wav"

if not os.path.exists(file_path):
    print("⚠️ File not found!")
    exit()

features = extract_features(file_path)
features_scaled = scaler.transform(features.reshape(1, -1))

pred = xgb.predict(features_scaled)
label = le.inverse_transform(pred)[0]

print(f"\n🔹 XGBoost Prediction: {label}")
