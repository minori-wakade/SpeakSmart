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
        
        with st.spinner("Analyzing your video, please wait..."):
            audio_emotion_result = run_emotion_analysis(audio_path)
            annotated_output_path = os.path.join(ANNOTATED_DIR, f"{timestamp}_eye_contact.mp4")
            eye_contact_percentage, status, eye_feedback, _ = process_eye_contact(file_path, annotated_output_path, None)
            wpm = extract_wpm(audio_path)
            category = categorize_wpm(wpm)
            posture_result = run_posture_analysis(file_path)
            deepface_result = process_video(file_path)

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
        eye_contact_percentage, eye_status, eye_feedback, _ = process_eye_contact(file_path, annotated_output_path, None)
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


# ==========================================================
# 📚 LEARNING MODULES PAGE
# ==========================================================
elif page == "📚 Learning Modules":
    st.markdown("""
        <style>
            .module-card {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 16px;
                padding: 30px;
                margin-top: 40px;
                margin-bottom: 30px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.1);
                transition: 0.3s ease-in-out;
            }
            .module-card:hover {
                box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            }
            .module-title {
                font-size: 22px;
                font-weight: bold;
                color: #3A7CA5;
                margin-bottom: 10px;
            }
            .module-outcome {
                font-size: 15px;
                color: #444;
                margin-bottom: 20px;
                line-height: 1.6;
            }
            .learn-btn {
                background-color: #3A7CA5;
                color: white;
                border: none;
                padding: 7px 18px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                transition: 0.3s;
                margin-bottom: 1rem;
            }
            .learn-btn:hover {
                background-color: #245d7a;
            }
        </style>
    """, unsafe_allow_html=True)

    # ✅ Page headers
    st.markdown('<div class="main-title">📚 Learning Modules</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Learn, practice, and master your speaking skills</div>', unsafe_allow_html=True)

    # ✅ Module data
    modules = [
        {
            "title": "Module 1 - Posture & Presence",
            "outcome": "Understand how posture influences audience perception and develop confident, balanced body language that projects credibility during speeches.",
            "subs": [
                ("1.1", "Importance of Posture in Public Speaking", "https://youtu.be/Ks-_Mh1QhMc?si=bh4_4B-nnpS1puhX"),
                ("1.2", "Common Posture Mistakes and How to Fix Them", "https://youtu.be/KP7XdGmtX5I"),
                ("1.3", "Exercises to Improve Body Alignment and Balance", "https://youtu.be/mS_VU71kS24?si=1AxzFrrZPXRKILTh"),
                ("1.4", "Applying Confident Posture During Real Speeches", "https://youtu.be/U1O3UFeCEeU?si=YS777AE_CYr81LtR"),
            ],
        },
        {
            "title": "Module 2 - Eye Contact Mastery",
            "outcome": "Build natural, controlled eye-contact habits that strengthen audience connection and trust in both in-person and virtual settings.",
            "subs": [
                ("2.1", "Why Eye Contact Builds Audience Trust?", "https://youtu.be/MGhwnXV80ZQ?si=T8eyO8iyZqnWaHqS"),
                ("2.2", "Maintaining Natural Eye Movement Without Staring", "https://youtu.be/Ior3iIXjMTU?si=eJTq2FvxEOBJNmnk"),
                ("2.3", "Techniques for Handling Large or Virtual Audiences", "https://youtu.be/JtTl8-TsZpk?si=lW-OgwIpEl7nYeUK"),
                ("2.4", "Practicing Eye Contact with a Mirror or Camera", "https://youtu.be/8OGDhlUvSK4?si=4qvqnGRgtXlkpZzI"),
            ],
        },
        {
            "title": "Module 3 - Tone & Confidence",
            "outcome": "Use vocal tone, variety, and breathing techniques to convey confidence, reduce anxiety, and emotionally engage listeners.",
            "subs": [
                ("3.1", "How Tone Reflects Your Confidence and Emotion", "https://youtu.be/hPQyHXc1ksA?si=EGevfjyA8lkZs5V1"),
                ("3.2", "Avoiding Monotone Delivery — Adding Vocal Variety", "https://youtu.be/ikuNOsPU5E0?si=VjquxImV5-J5E9Jv"),
                ("3.3", "Voice Warm-Up & Breathing Exercises for Power", "https://youtu.be/7eDcHZZn7hU?si=Hz0dFLhz5gDsx71a"),
                ("3.4", "Managing Stage Anxiety and Nervousness", "https://youtu.be/VEStYVONy-0?si=M_Q4C4AIfVmpGTbz"),
            ],
        },
        {
            "title": "Module 4 - Speech Speed & Pauses",
            "outcome": "Master pacing, rhythm, and pauses to deliver impactful speeches that maintain audience attention.",
            "subs": [
                ("4.1", "Finding Your Ideal Speaking Speed", "https://youtu.be/032Hum9KNjw?si=QNP8UZzErO90lLJG"),
                ("4.2", "Using Strategic Pauses for Impact", "https://youtu.be/kXsd25D25HI?si=oucAnC9kM9FkKvwr"),
                ("4.3", "Exercises to Control Speed and Breathing", "https://youtu.be/QxpR2_gwUEY?si=rXj91M0YWZ42Px4-"),
                ("4.4", "Improving Clarity and Rhythm While Speaking", "https://youtu.be/5YtbvUSdt5Q?si=ck6OfVIeJfUix3FE"),
            ],
        },
    ]

    for module in modules:
        st.markdown(f"<div class='module-card'><div class='module-title'>{module['title']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='module-outcome'>{module['outcome']}</div>", unsafe_allow_html=True)

        first = module["subs"][0]
        st.markdown(f"**{first[0]} {first[1]}**", unsafe_allow_html=True)
        st.markdown(f"""
        <a href="{first[2]}" target="_blank">
            <button class="learn-btn">Learn ▶</button>
        </a>
        """, unsafe_allow_html=True)

        with st.expander("Show more submodules"):
            for sub in module["subs"][1:]:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"- {sub[0]} {sub[1]}")
                with col2:
                    st.markdown(f"""
                    <a href="{sub[2]}" target="_blank">
                        <button class="learn-btn">Learn ▶</button>
                    </a>
                    """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# 🗂️ LIBRARY PAGE
# ==========================================================
elif page == "🗂️ Library":
    st.markdown('<div class="main-title">🗂️ Library</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Access your saved analysis and feedback</div>', unsafe_allow_html=True)

    
    folders = sorted(os.listdir(UPLOADS_DIR), reverse=True)

        # Create pairs of folders for 2-column layout
    for i in range(0, len(folders), 2):
            cols = st.columns(2)

            for j in range(2):
                if i + j >= len(folders):
                    break  # handle odd number of folders
                folder = folders[i + j]
                folder_path = os.path.join(UPLOADS_DIR, folder)
                results_file = os.path.join(folder_path, "results.json")

                if not os.path.exists(results_file):
                    continue

                try:
                    with open(results_file, "r") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, Exception):
                    continue
                # Format timestamp for display
                raw_timestamp = data.get("timestamp", "")
                try:
                    dt = datetime.fromisoformat(raw_timestamp)
                except Exception:
                    try:
                        dt = datetime.strptime(raw_timestamp, "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        dt = None

                if dt:
                    formatted_time = dt.strftime("%B %d, %Y — %I:%M %p")
                else:
                    formatted_time = raw_timestamp 

                audio_emotion = data.get("audio_emotion", {})
                emotion_category = audio_emotion.get("category", "N/A")
                emotion_confidence = audio_emotion.get("confidence", 0)

                with cols[j]:
                    # Emoji map
                    status_emoji_map = {
                        "GOOD": "🟢",
                        "OK": "🟡",
                        "BAD": "🔴"
                    }

                    # Get statuses from JSON
                    speech_status = data.get("speech_status", "OK").upper()
                    sentiment_status = data.get("sentiment_status", "OK").upper()
                    eye_status = data.get("eye_status", "OK").upper()
                    posture_status = data.get("posture_status", "OK").upper()
                    facial_status = data.get("facial_status", "BAD").upper()
                    overall_score = data.get("overall_score", 0)
                    st.markdown(f"""
                        <div class="report-card" style="background:rgba(255,255,255,0.05);margin-top:20px; height: 220px">
                            <b>Overall Score: {overall_score} / 10</b><br>
                            {status_emoji_map.get(speech_status,"🟡")} <b>Speech Speed:</b> {data['wpm']} WPM ({data['category'][0] if isinstance(data['category'], list) else data['category']})<br>
                            {status_emoji_map.get(sentiment_status,"🟡")} <b>Sentiment:</b> {emotion_category} ({emotion_confidence * 100:.1f}% confidence)<br>
                            {status_emoji_map.get(eye_status,"🟡")} <b>Eye Contact:</b> {data['eye_contact_percentage']:.1f}%<br>
                            {status_emoji_map.get(posture_status,"🟡")} <b>Good Posture:</b> {data['posture'].get('good_percent', 0):.1f}% <br>
                            {status_emoji_map.get(facial_status,"🔴")} <b>Facial Expressions:</b> {data['deepface'].get('dominant', 'N/A')}<br>
                            <b>Uploaded:</b> {formatted_time}<br><br>
                        </div>
                    """, unsafe_allow_html=True)

                    # Video player below each card
                    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.mov', '.avi'))]
                    if video_files:
                        video_path = os.path.join(folder_path, video_files[0])
                        st.video(video_path)
