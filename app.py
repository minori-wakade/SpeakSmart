import streamlit as st
import librosa
import numpy as np
import joblib
import speech_recognition as sr_module
import os
from utils import video_to_wav
from Eye_Contact.eyecontact import process_eye_contact
from Speech_Speed.speech_speed import extract_wpm, categorize_wpm
from Posture.posture_utils import run_posture_analysis

st.set_page_config(page_title="SpeakSmart", layout="wide")
st.title("🎯 SpeakSmart: AI Public Speaking Analyzer")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your speaking video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded file
    file_path = "input" + os.path.splitext(uploaded_file.name)[1]
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Convert video to audio if needed
    if file_path.lower().endswith((".mp4", ".mov", ".avi")):
        # --- Display uploaded video with reduced height ---
        st.markdown(
            """
            <style>
            video {
                max-height: 240px;  /* 👈 Adjust this height */
                width: auto;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.video(file_path)

        audio_path = video_to_wav(file_path)
    else:
        audio_path = file_path

    st.audio(audio_path)
    st.info(f"✅ Audio extracted successfully from `{uploaded_file.name}`")

    # --- Initialize models ---
    xgb = joblib.load("Sentiment Analysis/xgboost_model.pkl")
    scaler = joblib.load("Sentiment Analysis/scaler.pkl")
    le = joblib.load("Sentiment Analysis/label_encoder.pkl")

    # --- Feature Extraction Functions ---
    def extract_features(file_path):
        y, sr = librosa.load(file_path, sr=48000)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
        return np.hstack([mfccs, chroma, mel, contrast, tonnetz])

    # --- Combined Analysis ---
    st.header("🧠 Processing your video...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(current, total):
        progress_bar.progress(current / total)
        status_text.text(f"Processing frame {current}/{total}")

    # Eye Contact Analysis
    processed_video_path = "processed_output.mp4"
    eye_contact_percentage, eye_feedback, output_path = process_eye_contact(file_path, processed_video_path, update_progress)

    # Speech Analysis
    wpm = extract_wpm(audio_path)
    category = categorize_wpm(wpm)

    features = extract_features(audio_path)
    features_scaled = scaler.transform(features.reshape(1, -1))
    pred = xgb.predict(features_scaled)
    pred_sentiment = le.inverse_transform(pred)[0]

    feedback_messages = {
        "positive": "Great job! You sound confident and engaging. 🚀",
        "negative": "Try to stay calm and positive — it will improve your delivery. 🌱"
    }

    st.success("✅ Analysis Complete!")

    posture_result = run_posture_analysis(file_path)

    # --- Display Results ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📊 Final Report")

    # Refined CSS
    st.markdown("""
        <style>
        .report-card {
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 25px;
            box-shadow: 0 0 20px rgba(255, 255, 255, 0.05);
            transition: all 0.3s ease;
            backdrop-filter: blur(6px);
        }
        .report-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 30px rgba(255, 255, 255, 0.15);
        }
        .report-title {
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 12px;
            text-align: center;
        }
        .report-metric {
            font-size: 18px;
            margin: 6px 0;
        }

        /* Soft tinted cards */
        .green-card {
            border: 2px solid #80ED99;
            background: rgba(128, 237, 153, 0.07);
        }
        .yellow-card {
            border: 2px solid #FFD166;
            background: rgba(255, 209, 102, 0.07);
        }
        .blue-card {
            border: 2px solid #4CC9F0;
            background: rgba(76, 201, 240, 0.07);
        }
        .gray-card {
            border: 2px solid #ADB5BD;
            background: rgba(173, 181, 189, 0.07);
        }

        .dim-text { color: #aaa; font-size: 15px; }
        </style>
    """, unsafe_allow_html=True)

    # --- ROW 1 ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
            <div class="report-card green-card">
                <div class="report-title">🎵 Speech Speed</div>
                <div class="report-metric"><b>Words Per Minute:</b> {round(wpm)}</div>
                <div class="report-metric"><b>Category:</b> {category}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="report-card yellow-card">
                <div class="report-title">😊 Sentiment Analysis</div>
                <div class="report-metric"><b>Detected Sentiment:</b> <code>{pred_sentiment}</code></div>
                <div class="report-metric"><b>Feedback:</b> {feedback_messages.get(pred_sentiment, 'No feedback available')}</div>
            </div>
        """, unsafe_allow_html=True)

    # --- ROW 2 ---
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"""
            <div class="report-card blue-card">
                <div class="report-title">👁️ Eye Contact</div>
                <div class="report-metric"><b>Eye Contact:</b> {eye_contact_percentage:.2f}%</div>
                <div class="report-metric"><b>Feedback:</b> {eye_feedback}</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="report-card gray-card">
                <div class="report-title">🧍‍♂️ Posture Analysis</div>
                <div class="report-metric"><b>Good Posture:</b> {posture_result.get('good_percent', 0):.2f}%</div>
                <div class="report-metric"><b>Summary:</b> {posture_result.get('summary', 'No summary available')}</div>
        """, unsafe_allow_html=True)


