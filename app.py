import streamlit as st
import librosa
import numpy as np
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from utils import video_to_wav
from Eye_Contact.eyecontact import process_eye_contact
from Speech_Speed.speech_speed import extract_wpm, categorize_wpm
from Posture.posture_utils import run_posture_analysis

# -----------------------------------------
# 🧠 FRONTEND + BACKEND MERGED SpeakSmart APP
# -----------------------------------------
st.set_page_config(page_title="SpeakSmart", layout="wide")

# --- Global CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #f5f5f5;
}
img {
    margin-bottom: -40px;  /* pull title closer */
}
.main-title {font-size: 2.4rem; font-weight: 700; text-align: center; margin-bottom: 0.4em;}
.subtitle {text-align: center; color: #ddd; margin-bottom: 2em;}
.report-card {
    border-radius: 16px; padding: 24px; margin-bottom: 25px;
    box-shadow: 0 0 25px rgba(255,255,255,0.08); transition: all 0.3s ease;
    backdrop-filter: blur(10px);
    height: 180px;
    overflow: hidden; /* prevent overflow */
}
.report-card:hover {transform: translateY(-3px); box-shadow: 0 4px 25px rgba(255,255,255,0.15);}
.green-card {border: 2px solid #80ED99; background: rgba(128,237,153,0.08);}
.yellow-card {border: 2px solid #FFD166; background: rgba(255,209,102,0.08);}
.blue-card {border: 2px solid #4CC9F0; background: rgba(76,201,240,0.08);}
.gray-card {border: 2px solid #ADB5BD; background: rgba(173,181,189,0.08);}
.report-title {font-size: 1.2rem; font-weight: 600; margin-bottom: 10px; text-align:center;}
.report-metric {font-size: 1rem; margin: 4px 0;}
.centered {text-align: center;}
video {
    max-height: 240px;
    width: auto;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# --- Login State ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 0.4, 1])
    with col2:
        st.image("logo.png", width=210)
    st.markdown('<div class="main-title">SpeakSmart</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your AI-powered Public Speaking Coach</div>', unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")
        login = st.form_submit_button("Login")
        if login:
            st.session_state.logged_in = True
            st.rerun()

else:
    # --- Sidebar Navigation ---
    st.sidebar.markdown("<h2 style='margin-bottom:20px; text-align:center'>📋 Hey User!</h2>", unsafe_allow_html=True)
    nav_items = ["🎥 Analyze Video", "📚 Learning Modules", "📈 My Progress", "🗂️ Library"]

    if "page" not in st.session_state:
        st.session_state.page = "🎥 Analyze Video"

    for item in nav_items:
        if st.sidebar.button(item, use_container_width=True):
            st.session_state.page = item

    if st.sidebar.button("🔒 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

    page = st.session_state.page

    # ------------------------
    # 🎥 Analyze Video Page
    # ------------------------
    if page == "🎥 Analyze Video":
        st.markdown('<div class="main-title">🎥 Analyze Your Speech</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your speaking video", type=["mp4", "mov", "avi"])

        if uploaded_file is not None:
            file_path = "input" + os.path.splitext(uploaded_file.name)[1]
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.video(file_path)
            st.info("⚙️ Running AI Analysis...")

            # --- Extract Audio ---
            audio_path = video_to_wav(file_path)
            st.audio(audio_path)

            # --- Load Models ---
            xgb = joblib.load("Sentiment Analysis/xgboost_model.pkl")
            scaler = joblib.load("Sentiment Analysis/scaler.pkl")
            le = joblib.load("Sentiment Analysis/label_encoder.pkl")

            # --- Extract Audio Features ---
            def extract_features(file_path):
                y, sr = librosa.load(file_path, sr=48000)
                mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
                chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
                mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
                contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
                tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
                return np.hstack([mfccs, chroma, mel, contrast, tonnetz])

            # --- Progress ---
            st.header("🧠 Processing your video...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current, total):
                progress_bar.progress(current / total)
                status_text.text(f"Processing frame {current}/{total}")

            # --- AI Analyses ---
            eye_contact_percentage, eye_feedback, output_path = process_eye_contact(file_path, "processed_output.mp4", update_progress)
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
            posture_result = run_posture_analysis(file_path)

            # --- Display Results ---
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("📊 AI Feedback Report")

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
                        <div class="report-title">😊 Sentiment</div>
                        <div class="report-metric"><b>Detected:</b> {pred_sentiment}</div>
                        <div class="report-metric"><b>Feedback:</b> {feedback_messages.get(pred_sentiment, 'No feedback')}</div>
                    </div>
                """, unsafe_allow_html=True)

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
                        <div class="report-title">🧍 Posture</div>
                        <div class="report-metric"><b>Good Posture:</b> {posture_result.get('good_percent', 0):.2f}%</div>
                        <div class="report-metric"><b>Summary:</b> {posture_result.get('summary', 'Not available')}</div>
                    </div>
                """, unsafe_allow_html=True)

    # ------------------------
    # 📚 Learning Modules
    # ------------------------
    elif page == "📚 Learning Modules":
        st.markdown('<div class="main-title">📚 Learning Modules</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Learn, practice, and master your speaking skills</div>', unsafe_allow_html=True)
    
        modules = [
            {
                "title": "Module 1 - Posture & Presence",
                "outcome": "Learners will understand how posture influences audience perception and develop confident, balanced body language that projects credibility during speeches.",
                "subs": [
                    ("1.1", "Importance of Posture in Public Speaking", "https://youtu.be/Ks-_Mh1QhMc?si=bh4_4B-nnpS1puhX"),
                    ("1.2", "Common Posture Mistakes and How to Fix Them", "https://youtu.be/KP7XdGmtX5I"),
                    ("1.3", "Exercises to Improve Body Alignment and Balance", "https://youtu.be/mS_VU71kS24?si=1AxzFrrZPXRKILTh"),
                    ("1.4", "Applying Confident Posture During Real Speeches", "https://youtu.be/U1O3UFeCEeU?si=YS777AE_CYr81LtR"),
                ],
            },
            {
                "title": "Module 2 - Eye Contact Mastery",
                "outcome": "Learners will build natural, controlled eye-contact habits that strengthen audience connection and trust across in-person and virtual settings.",
                "subs": [
                    ("2.1", "Why Eye Contact Builds Audience Trust?", "https://youtu.be/MGhwnXV80ZQ?si=T8eyO8iyZqnWaHqS"),
                    ("2.2", "Maintaining Natural Eye Movement Without Staring", "https://youtu.be/Ior3iIXjMTU?si=eJTq2FvxEOBJNmnk"),
                    ("2.3", "Techniques for Handling Large or Virtual Audiences", "https://youtu.be/JtTl8-TsZpk?si=lW-OgwIpEl7nYeUK"),
                    ("2.4", "Practicing Eye Contact with a Mirror or Camera", "https://youtu.be/8OGDhlUvSK4?si=4qvqnGRgtXlkpZzI"),
                ],
            },
            {
                "title": "Module 3 - Tone & Confidence",
                "outcome": "Learners will learn to use vocal tone, variety, and breathing techniques to convey confidence, reduce anxiety, and emotionally engage listeners.",
                "subs": [
                    ("3.1", "How Tone Reflects Your Confidence and Emotion", "https://youtu.be/hPQyHXc1ksA?si=EGevfjyA8lkZs5V1"),
                    ("3.2", "Avoiding Monotone Delivery — Adding Vocal Variety", "https://youtu.be/ikuNOsPU5E0?si=VjquxImV5-J5E9Jv"),
                    ("3.3", "Voice Warm-Up & Breathing Exercises for Power", "https://youtu.be/7eDcHZZn7hU?si=Hz0dFLhz5gDsx71a"),
                    ("3.4", "Managing Stage Anxiety and Nervousness", "https://youtu.be/VEStYVONy-0?si=M_Q4C4AIfVmpGTbz"),
                ],
            },
            {
                "title": "Module 4 - Speech Speed & Pauses",
                "outcome": "Learners will master pacing, rhythm, and pauses to deliver clear, impactful, and well-timed speeches that maintain audience attention.",
                "subs": [
                    ("4.1", "Finding Your Ideal Speaking Speed", "https://youtu.be/032Hum9KNjw?si=QNP8UZzErO90lLJG"),
                    ("4.2", "Using Strategic Pauses for Impact", "https://youtu.be/kXsd25D25HI?si=oucAnC9kM9FkKvwr"),
                    ("4.3", "Exercises to Control Speed and Breathing", "https://youtu.be/QxpR2_gwUEY?si=rXj91M0YWZ42Px4-"),
                    ("4.4", "Improving Clarity and Rhythm While Speaking", "https://youtu.be/5YtbvUSdt5Q?si=ck6OfVIeJfUix3FE"),
                ],
            },
        ]

        for module in modules:
            st.markdown(f"<div class='report-card blue-card'style='margin-top: 60px;'><div class='report-title'>{module['title']}</div>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:15px; color:#aaa;'>{module['outcome']}</p>", unsafe_allow_html=True)

            # First submodule visible
            first_sub = module["subs"][0]
            st.markdown(f"**{first_sub[0]} {first_sub[1]}**", unsafe_allow_html=True)
            st.markdown(f"""
                <a href="{first_sub[2]}" target="_blank">
                    <button style="
                        background-color:grey;
                        color:white;
                        border:none;
                        padding:6px 16px;
                        border-radius:6px;
                        cursor:pointer;
                        margin-top:5px;
                        margin-bottom:10px;">
                        Learn ▶
                    </button>
                </a>
            """, unsafe_allow_html=True)

            # Remaining submodules in an expander
            with st.expander("Show more submodules"):
                for sub in module["subs"][1:]:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"- {sub[0]} {sub[1]}")
                    with col2:
                        st.markdown(f"""
                            <a href="{sub[2]}" target="_blank">
                                <button style="
                                    background-color:grey;
                                    color:white;
                                    border:none;
                                    padding:6px 16px;
                                    border-radius:6px;
                                    cursor:pointer;
                                    margin-top:5px;
                                    margin-bottom:10px;">
                                    Learn ▶
                                </button>
                            </a>
            """, unsafe_allow_html=True)

    # ------------------------
    # 📈 My Progress
    # ------------------------
    elif page == "📈 My Progress":
        st.markdown('<div class="main-title">📈 My Progress</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Track your improvements and milestones</div>', unsafe_allow_html=True)
        # ... [use same progress graph section as code1] ...
        # ---- Summary Stats ----
        st.markdown("### 📊 Usage Summary")
        cols = st.columns(4)
        with cols[0]:
            st.metric("Videos Analyzed", "12")

        with cols[2]:
            st.metric("Feedback Reports Viewed", "8")

        # ---- Progress Graph ----
        st.markdown("<br>### 💫 Monthly Progress", unsafe_allow_html=True)
            

        data = {
            "Month": ["June", "July", "Aug", "Sept", "Oct"],
            "Videos": [2, 3, 5, 4, 6]
        }
        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(5, 3))

        # Gradient bars
        colors = plt.cm.cool(np.linspace(0.3, 1, len(df)))
        bars = ax.bar(df["Month"], df["Videos"], color=colors, width=0.25, edgecolor="none", zorder=3)

            # Rounded bar tops
        for bar in bars:
            bar.set_linewidth(0)
            x, y = bar.get_xy()
            w, h = bar.get_width(), bar.get_height()
            ax.add_patch(plt.Rectangle((x, y + h - 0.15), w, 0.15, color=bar.get_facecolor(), ec="none", lw=0))

            # Add cute markers
        ax.plot(df["Month"], df["Videos"], 'o--', color="red", markersize=4, linewidth=0.5, zorder=4)

            # Clean layout
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#f8f9fa")            
        ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_ylabel("Videos", fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title("Video Analysis Trend", fontsize=10, weight='bold', pad=10)

        st.pyplot(fig)
        # ---- View Feedback Button ----
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="text-align:center;">
                <button style="
                    background-color:#4A90E2;
                    color:white;
                    border:none;
                    padding:10px 25px;
                    border-radius:8px;
                    cursor:pointer;
                    font-size:14px;
                ">
                    View Feedback
                </button>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ------------------------
    # 🗂️ Library
    # ------------------------
    elif page == "🗂️ Library":
        st.markdown('<div class="main-title">🗂️ Library</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Access your saved analyses and feedback</div>', unsafe_allow_html=True)

        # ---- Uploaded Videos ----
        st.markdown("### 🎞️ Uploaded Videos")
        cols = st.columns(6)
        for i, date in enumerate(["Oct 10", "Oct 12", "Oct 15"]):
            with cols[i]:
                st.markdown(f"""
                    <div class="report-card yellow-card" 
                        style="padding:8px;text-align:center;border-radius:10px;font-size:13px;">
                        📹<br>{date}, 2025
                    </div>
                """, unsafe_allow_html=True)
        
        # ---- Feedback Videos ----
        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        st.markdown("### 🧠 Feedback Videos")
        cols = st.columns(6)
        for i, date in enumerate(["Oct 10", "Oct 12", "Oct 15"]):
            with cols[i]:
                st.markdown(f"""
                    <div class="report-card blue-card" 
                        style="padding:8px;text-align:center;border-radius:10px;font-size:13px;">
                        🎧<br>{date}, 2025
                    </div>
                """, unsafe_allow_html=True)
