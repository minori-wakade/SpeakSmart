import streamlit as st
import os
import json
from datetime import datetime
import numpy as np
from utils import video_to_wav
from Eye_Contact.eyecontact import process_eye_contact
from Speech_Speed.speech_speed import extract_wpm, categorize_wpm
from Posture.posture_utils import run_posture_analysis
from CNN_Sentiment_Analysis.test_emotion import run_emotion_analysis
from Deepface_Expressions.video_processor import process_video  
from concurrent.futures import ThreadPoolExecutor, as_completed
from learning_modules import show_learning_modules
from library import show_library

# -----------------------------------------
# 🧠 SpeakSmart App Setup
# -----------------------------------------
st.set_page_config(page_title="SpeakSmart", layout="wide")

# Create uploads directory
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)
ANNOTATED_DIR = "Annotated_eye_contact"
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# --- Global Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    color: #f5f5f5;
}

/* 🔹 Custom Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(40, 40, 40, 0.85); /* lighter black shade */
    border: none !important;
    box-shadow: 0 0 20px rgba(0,0,0,0.3);
    backdrop-filter: blur(10px);
}
[data-testid="stSidebarNav"] {
    display: none !important;
}
.sidebar-title {
    text-align: center;
    font-size: 1.8rem;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 30px;
    color: #f5f5f5;
}

/* Sidebar Buttons */
.stButton>button {
    background: rgba(255, 255, 255, 0.08);
    color: #f5f5f5;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 10px;
    width: 100%;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background: rgba(255, 255, 255, 0.25);
    transform: translateY(-2px);
}

/* General Cards and Layout */
.main-title {
    font-size: 2.4rem;
    font-weight: 700;
    text-align: center;
    color: #f5f5f5;
    letter-spacing: 1px;
    margin-bottom: 0.4em;
}
.subtitle {
    text-align: center;
    color: #ddd;
    margin-bottom: 2em;
}
.report-card {
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 25px;
    box-shadow: 0 0 25px rgba(255,255,255,0.08);
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
    height: 200px;
}
.report-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 25px rgba(255,255,255,0.15);
}
.green-card {border: 2px solid #80ED99; background: rgba(128,237,153,0.08); height: 160px}
.yellow-card {border: 2px solid #FFD166; background: rgba(255,209,102,0.08); height: 160px}
.blue-card {border: 2px solid #4CC9F0; background: rgba(76,201,240,0.08);height: 160px}
.gray-card {border: 2px solid #ADB5BD; background: rgba(173,181,189,0.08);height: 160px}
.purple-card {border: 2px solid #C77DFF; background: rgba(199,125,255,0.08); height: 200px}
.magenta-card {border: 2px solid #E75480; background: rgba(231, 84, 128, 0.08);height: 220px;}

.report-title {font-size: 1.2rem; font-weight: 600; margin-bottom: 10px; text-align:center;}
.report-metric {font-size: 1rem; margin: 4px 0;}
            
/* Badge Styles */
.status-badge {
    position: absolute;
    top: 10px;
    left: 10px;
    padding: 4px 10px;
    border-radius: 12px;
    color: #fff;
    font-weight: 600;
    font-size: 0.8rem;
    z-index: 10;
}

.status-good { background-color: #28a745; }   /* Green */
.status-bad { background-color: #dc3545; }    /* Red */
.status-ok { background-color: #ffc107; }     /* Yellow/Orange */

video {
    max-height: 240px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

def get_score_color(score):
    if score < 4:
        return "#dc3545"   
    elif score < 7:
        return "#ffc107"  
    else:
        return "#28a745"   

# -----------------------------------------
# 🧭 Custom Sidebar Navigation
# -----------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>SpeakSmart</div>", unsafe_allow_html=True)
    nav_items = [
        ("🎥 Analyze Video", "🎥 Analyze Video"),
        ("📚 Learning Modules", "📚 Learning Modules"),
        ("🗂️ Library", "🗂️ Library"),
    ]
    if "page" not in st.session_state:
        st.session_state.page = "🎥 Analyze Video"

    for label, value in nav_items:
        if st.button(label, use_container_width=True):
            st.session_state.page = value

page = st.session_state.page

# ==========================================================
# 🎥 ANALYZE VIDEO PAGE
# ==========================================================
def run_audio_pipeline(audio_path):
    """Runs all audio-related analysis"""
    audio_emotion = run_emotion_analysis(audio_path)

    wpm = extract_wpm(audio_path)
    category, speed_status = categorize_wpm(wpm)

    return {
        "audio_emotion": audio_emotion,
        "wpm": wpm,
        "category": category,
        "speed_status": speed_status
    }


def run_video_pipeline(video_path, timestamp):
    """Runs all video-related analysis"""

    annotated_output_path = os.path.join(
        ANNOTATED_DIR, f"{timestamp}_eye_contact.mp4"
    )

    eye = process_eye_contact(
        video_path, annotated_output_path, None
    )

    posture = run_posture_analysis(video_path)

    face = process_video(video_path)

    return {
        "eye": eye,
        "posture": posture,
        "face": face
    }

if page == "🎥 Analyze Video":
    st.markdown('<div class="main-title">🎥 Analyze Your Speech</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your speaking video", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_folder = os.path.join(UPLOADS_DIR, timestamp)
        os.makedirs(upload_folder, exist_ok=True)

        video_name = f"{timestamp}_{uploaded_file.name}"
        file_path = os.path.join(upload_folder, video_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(file_path)

        # Audio extraction
        audio_path = video_to_wav(file_path)
        # st.audio(audio_path)
        # Run analysis
        # -----------------------------------------
    # 🚀 Parallel Analysis
    # -----------------------------------------

        with st.spinner("Analyzing your video..."):

            results = {}

            with ThreadPoolExecutor(max_workers=2) as executor:

                futures = {
                    executor.submit(
                        run_audio_pipeline, audio_path
                    ): "audio",

                    executor.submit(
                        run_video_pipeline, file_path, timestamp
                    ): "video"
                }

                for future in as_completed(futures):

                    key = futures[future]

                    try:
                        results[key] = future.result()

                    except Exception as e:
                        st.error(f"{key} analysis failed: {e}")
                        st.stop()


        # -----------------------------------------
        # ✅ Extract Results
        # -----------------------------------------

        audio_result = results["audio"]
        video_result = results["video"]

        # Audio
        audio_emotion_result = audio_result["audio_emotion"]
        wpm = audio_result["wpm"]
        category = audio_result["category"]
        speed_status = audio_result["speed_status"]

        # Video
        eye_contact_percentage, eye_status, eye_feedback, _ = video_result["eye"]
        posture_result = video_result["posture"]
        deepface_result = video_result["face"]

        st.success("🎉 All Analysis Complete!")

        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(i) for i in obj]
            elif isinstance(obj, (np.float32, np.float64, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64, np.int16)):
                return int(obj)
            else:
                return obj

        results = {
            "video_name": video_name,
            "timestamp": timestamp,
            "wpm": round(wpm),
            "category": category,
            "audio_emotion": audio_emotion_result,
            "eye_contact_percentage": eye_contact_percentage,
            "eye_feedback": eye_feedback,
            "posture": posture_result,
            "deepface": deepface_result,
        }
        status_to_class = {"GOOD": "status-good", "BAD": "status-bad", "OK": "status-ok"}

        # ✅ Convert NumPy/tensor values to native types before saving
        results_native = convert_to_native(results)
        category, speed_status = categorize_wpm(wpm) 
        audio_status = audio_emotion_result.get("status", "OK").upper()
        eye_contact_percentage, eye_status, eye_feedback, _ = video_result["eye"]
        posture_status = posture_result.get("status", "OK").upper()
        facial_status = deepface_result.get("status", "BAD").upper()

        # -------------------------------
        # Overall Score (Simple Average)
        # -------------------------------
        STATUS_SCORE_MAP = {
            "GOOD": 10,
            "OK": 6,
            "BAD": 2
        }

        scores = [
            STATUS_SCORE_MAP.get(speed_status, 6),
            STATUS_SCORE_MAP.get(audio_status, 6),
            STATUS_SCORE_MAP.get(eye_status, 6),
            STATUS_SCORE_MAP.get(posture_status, 6),
            STATUS_SCORE_MAP.get(facial_status, 6),
        ]

        overall_score = round(sum(scores) / len(scores), 1)

        # Add status for each metric
        results_native.update({
            "speech_status": speed_status,               
            "sentiment_status": audio_status,           
            "eye_status": eye_status,                    
            "posture_status": posture_status,            
            "facial_status": facial_status,
            "overall_score": overall_score             
        })

        with open(os.path.join(upload_folder, "results.json"), "w") as f:
            json.dump(results_native, f, indent=4)
        
        # ---- Feedback Report ----
       
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("📊 Feedback Report")

        score_color = get_score_color(overall_score)
        progress_angle = int((overall_score / 10) * 360)

        st.markdown(f"""
        <div class="report-card magenta-card" style="position: relative; height: 220px; text-align:center;">
            <div class="report-title">🏆 Overall Score</div>
            <div style="
                width:120px;
                height:120px;
                border-radius:50%;
                background: conic-gradient(rgba(255,255,255,0.1) 0deg {360 - progress_angle}deg, {score_color} {360 - progress_angle}deg 360deg);
                display:flex;
                align-items:center;
                justify-content:center;
                font-weight:700;
                color:#fff;
                margin:auto;
            ">
                <div style="
                    text-align:center;
                ">
                    {overall_score} / 10
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


        col1, col2 = st.columns(2)
        with col1:
            badge_class = status_to_class.get(speed_status, "status-ok")
            st.markdown(f"""
        <div class="report-card green-card" style="position: relative;">
            <div class="status-badge {badge_class}">{speed_status}</div>
            <div class="report-title">🎵 Speech Speed</div>
            <div class="report-metric"><b>Words Per Minute:</b> {round(wpm)}</div>
            <div class="report-metric"><b>Category:</b> {category}</div>
        </div>
        """, unsafe_allow_html=True)

        with col2:
            badge_class = status_to_class.get(audio_status, "status-ok")
            st.markdown(f"""
            <div class="report-card yellow-card" style="position: relative;">
                <div class="status-badge {badge_class}">{audio_status}</div>
                <div class="report-title">😊 Sentiment</div>
                <div class="report-metric"><b>Category:</b> {audio_emotion_result['category']}</div>
                <div class="report-metric"><b>Confidence:</b> {audio_emotion_result['confidence'] * 100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)


        col3, col4 = st.columns(2)
        with col3:
            badge_class = status_to_class.get(eye_status, "status-ok")

            st.markdown(f"""
            <div class="report-card blue-card" style="position: relative;">
                <div class="status-badge {badge_class}">{eye_status}</div>
                <div class="report-title">👁️ Eye Contact</div>
                <div class="report-metric"><b>Eye Contact:</b> {eye_contact_percentage:.2f}%</div>
                <div class="report-metric"><b>Feedback:</b> {eye_feedback}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            badge_class = status_to_class.get(posture_status, "status-ok")

            st.markdown(f"""
            <div class="report-card gray-card" style="position: relative;">
                <div class="status-badge {badge_class}">{posture_status}</div>
                <div class="report-title">🧍 Posture</div>
                <div class="report-metric"><b>Good Posture:</b> {posture_result.get('good_percent', 0):.2f}%</div>
                <div class="report-metric"><b>Summary:</b> {posture_result.get('summary', 'Not available')}</div>
            </div>
            """, unsafe_allow_html=True)


        badge_class = status_to_class.get(facial_status, "status-bad")

        st.markdown(f"""
        <div class="report-card purple-card" style="position: relative;">
            <div class="status-badge {badge_class}">{facial_status}</div>
            <div class="report-title">🎭 Facial Expressions</div>
            <div class="report-metric"><b>Dominant Emotion:</b> {deepface_result.get('dominant', 'N/A')}</div>
            <div class="report-metric"><b>Positive:</b> {deepface_result.get('positive', 0):.1f}%</div>
            <div class="report-metric"><b>Neutral:</b> {deepface_result.get('neutral', 0):.1f}%</div>
            <div class="report-metric"><b>Negative:</b> {deepface_result.get('negative', 0):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

# LEARNING MODULES PAGE
elif page == "📚 Learning Modules":
    show_learning_modules()

# LIBRARY PAGE
elif page == "🗂️ Library":
    show_library()
    