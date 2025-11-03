# # emotion_logic.py
# import numpy as np
# import librosa
# import soundfile as sf
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import LabelEncoder
# import streamlit as st
# import tempfile
# import pygame
# import os

# # --------------------------
# # Load Model and LabelEncoder
# # --------------------------
# def load_emotion_model():
#     base_path = os.path.dirname(os.path.abspath(__file__))
#     model_path = os.path.join(base_path, "emotion_recognition_model.h5")
#     label_path = os.path.join(base_path, "label_encoder_classes.npy")

#     model = load_model(model_path)
#     le = LabelEncoder()
#     le.classes_ = np.load(label_path, allow_pickle=True)
#     return model, le



# # --------------------------
# # Feature Extraction
# # --------------------------
# def extract_features(audio_data, sr, n_mfcc=40, max_pad_len=174):
#     try:
#         mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
#         delta = librosa.feature.delta(mfcc)
#         delta2 = librosa.feature.delta(mfcc, order=2)
#         stacked = np.vstack([mfcc, delta, delta2])

#         if stacked.shape[1] < max_pad_len:
#             pad_width = max_pad_len - stacked.shape[1]
#             stacked = np.pad(stacked, pad_width=((0, 0), (0, pad_width)), mode='constant')
#         else:
#             stacked = stacked[:, :max_pad_len]
#         return stacked
#     except Exception as e:
#         st.error(f"Error extracting features: {e}")
#         return None


# # --------------------------
# # Emotion Category Groups
# # --------------------------
# positive_emotions = {"happy", "calm", "neutral", "surprised"}
# negative_emotions = {"angry", "disgust", "sad", "fearful"}


# # --------------------------
# # Predict Emotion (Group + Threshold Averaging)
# # --------------------------
# def predict_emotion(model, le, features, threshold=0.2):
#     features = features[np.newaxis, ..., np.newaxis]
#     prediction = model.predict(features)[0]

#     positive_indices = [i for i, e in enumerate(le.classes_) if e.lower() in positive_emotions]
#     negative_indices = [i for i, e in enumerate(le.classes_) if e.lower() in negative_emotions]

#     pos_probs = [prediction[i] for i in positive_indices if prediction[i] > threshold]
#     neg_probs = [prediction[i] for i in negative_indices if prediction[i] > threshold]

#     pos_conf = np.mean(pos_probs) if pos_probs else 0
#     neg_conf = np.mean(neg_probs) if neg_probs else 0

#     predicted_class = np.argmax(prediction)
#     emotion_label = le.inverse_transform([predicted_class])[0]

#     if pos_conf > neg_conf:
#         overall_category = "😊 Positive"
#         overall_confidence = pos_conf
#     elif neg_conf > pos_conf:
#         overall_category = "😟 Negative"
#         overall_confidence = neg_conf
#     else:
#         overall_category = "😐 Neutral/Uncertain"
#         overall_confidence = 0.0

#     return emotion_label, overall_confidence, overall_category


# # --------------------------
# # Optional: Audio Playback (for local testing)
# # --------------------------
# def play_audio(audio_path):
#     try:
#         pygame.mixer.init()
#         pygame.mixer.music.load(audio_path)
#         pygame.mixer.music.play()
#     except Exception:
#         pass


# # --------------------------
# # Streamlit App
# # --------------------------
# def run_streamlit_app():
#     st.title("🎭 Emotion Recognition from Audio")

#     model, le = load_emotion_model()

#     uploaded_file = st.file_uploader("Upload Audio File (.wav)", type=["wav"])
#     if uploaded_file:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#             tmp.write(uploaded_file.read())
#             tmp_path = tmp.name

#         audio_data, sr = librosa.load(tmp_path, sr=None)
#         st.audio(tmp_path)

#         features = extract_features(audio_data, sr)
#         if features is not None:
#             emotion, conf, category = predict_emotion(model, le, features)
#             st.success(f"Predicted Emotion: **{emotion}**")
#             st.info(f"Overall Category: {category} ({conf:.2f})")

# if __name__ == "__main__":
#     run_streamlit_app()

import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

# --------------------------
# Load Model and LabelEncoder
# --------------------------
def load_emotion_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "emotion_recognition_model.h5")
    label_path = os.path.join(base_path, "label_encoder_classes.npy")

    model = load_model(model_path)
    le = LabelEncoder()
    le.classes_ = np.load(label_path, allow_pickle=True)
    return model, le


# --------------------------
# Feature Extraction
# --------------------------
def extract_features(audio_data, sr, n_mfcc=40, max_pad_len=174):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    stacked = np.vstack([mfcc, delta, delta2])

    if stacked.shape[1] < max_pad_len:
        pad_width = max_pad_len - stacked.shape[1]
        stacked = np.pad(stacked, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        stacked = stacked[:, :max_pad_len]
    return stacked


# --------------------------
# Emotion Category Groups
# --------------------------
positive_emotions = {"happy", "calm", "neutral", "surprised"}
negative_emotions = {"angry", "disgust", "sad", "fearful"}


# --------------------------
# Predict Emotion
# --------------------------
def predict_emotion(model, le, features, threshold=0.2):
    features = features[np.newaxis, ..., np.newaxis]
    prediction = model.predict(features)[0]

    positive_indices = [i for i, e in enumerate(le.classes_) if e.lower() in positive_emotions]
    negative_indices = [i for i, e in enumerate(le.classes_) if e.lower() in negative_emotions]

    pos_probs = [prediction[i] for i in positive_indices if prediction[i] > threshold]
    neg_probs = [prediction[i] for i in negative_indices if prediction[i] > threshold]

    pos_conf = np.mean(pos_probs) if pos_probs else 0
    neg_conf = np.mean(neg_probs) if neg_probs else 0

    predicted_class = np.argmax(prediction)
    emotion_label = le.inverse_transform([predicted_class])[0]

    if pos_conf > neg_conf:
        overall_category = "😊 Positive"
        overall_confidence = pos_conf
    elif neg_conf > pos_conf:
        overall_category = "😟 Negative"
        overall_confidence = neg_conf
    else:
        overall_category = "😐 Neutral/Uncertain"
        overall_confidence = 0.0

    return emotion_label, overall_confidence, overall_category


# --------------------------
# Main callable function
# --------------------------
def run_emotion_analysis(audio_path):
    """Runs full emotion recognition pipeline on an audio file."""
    try:
        model, le = load_emotion_model()
        audio_data, sr = librosa.load(audio_path, sr=None)
        features = extract_features(audio_data, sr)
        if features is None:
            return {"confidence": 0.0, "category": "⚠️ Feature extraction failed"}
        emotion, conf, category = predict_emotion(model, le, features)
        return {
            "confidence": conf,
            "category": category
        }
    except Exception as e:
        return {"confidence": 0.0, "category": f"⚠️ Error: {e}"}
