import streamlit as st
import os
import json
from datetime import datetime


UPLOADS_DIR = "uploads"


def show_library():

    # Page Header
    st.markdown('<div class="main-title">🗂️ Library</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Access your saved analysis and feedback</div>', unsafe_allow_html=True)

    # Check folder
    if not os.path.exists(UPLOADS_DIR):
        st.info("No reports available yet.")
        return


    folders = sorted(os.listdir(UPLOADS_DIR), reverse=True)


    # Create 2-column layout
    for i in range(0, len(folders), 2):

        cols = st.columns(2)

        for j in range(2):

            if i + j >= len(folders):
                break


            folder = folders[i + j]
            folder_path = os.path.join(UPLOADS_DIR, folder)
            results_file = os.path.join(folder_path, "results.json")

            if not os.path.exists(results_file):
                continue


            # Load JSON
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
            except Exception:
                continue


            # Format timestamp
            raw_timestamp = data.get("timestamp", "")

            dt = None

            try:
                dt = datetime.fromisoformat(raw_timestamp)
            except Exception:
                try:
                    dt = datetime.strptime(raw_timestamp, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass


            if dt:
                formatted_time = dt.strftime("%B %d, %Y — %I:%M %p")
            else:
                formatted_time = raw_timestamp


            # Audio emotion
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


                # Status values
                speech_status = data.get("speech_status", "OK").upper()
                sentiment_status = data.get("sentiment_status", "OK").upper()
                eye_status = data.get("eye_status", "OK").upper()
                posture_status = data.get("posture_status", "OK").upper()
                facial_status = data.get("facial_status", "BAD").upper()

                overall_score = data.get("overall_score", 0)

                st.markdown(f"""
                    <div class="report-card" style="
                        background: rgba(255,255,255,0.05);
                        margin-top: 20px;
                        padding: 14px;
                        border-radius: 14px;
                        height: 220px
                    ">

                    **🏆 Overall Score:** {overall_score} / 10  
                    {status_emoji_map.get(speech_status,"🟡")} Speech Speed: {data.get('wpm',0)} WPM ({data.get('category','N/A')})  
                    {status_emoji_map.get(sentiment_status,"🟡")} **Sentiment:** {emotion_category} ({emotion_confidence * 100:.1f}%)  
                    {status_emoji_map.get(eye_status,"🟡")} **Eye Contact:** {data.get('eye_contact_percentage',0):.1f}%  
                    {status_emoji_map.get(posture_status,"🟡")} **Good Posture:** {data.get('posture',{}).get('good_percent',0):.1f}%  
                    {status_emoji_map.get(facial_status,"🔴")} **Facial Expressions:** {data.get('deepface',{}).get('dominant','N/A')}  
                    🕒 **Uploaded:** {formatted_time}

                    </div>
                    """, unsafe_allow_html=True)

                # Video
                video_files = [
                    f for f in os.listdir(folder_path)
                    if f.endswith(('.mp4', '.mov', '.avi'))
                ]

                if video_files:

                    video_path = os.path.join(folder_path, video_files[0])
                    st.video(video_path)
