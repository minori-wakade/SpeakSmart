import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import numpy as np
import joblib
import os
from tqdm import tqdm

# Load model, scaler, and encoder
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "posture_svm.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def save_uploaded_video(uploaded_file):
    """Temporarily saves uploaded file and returns its path."""
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    return tfile.name

def process_video(video_path):
    """Processes video, applies posture model, and saves annotated output."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the absolute path to the project root (one level above Posture)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    ANNOTATED_DIR = os.path.join(PROJECT_ROOT, "Annotated_Posture")
    os.makedirs(ANNOTATED_DIR, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(ANNOTATED_DIR, f"{video_name}_annotated.mp4")

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    predictions = []
    frame_no = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1) as pose:
        #progress_bar = st.progress(0)
        for _ in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_no % 15 == 0:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    row = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten().reshape(1, -1)

                    X_scaled = scaler.transform(row)
                    pred = model.predict(X_scaled)
                    label = encoder.inverse_transform(pred)[0]
                    predictions.append(label)

                    color = (0, 255, 0) if label == "Good" else (0, 0, 255)
                    cv2.putText(frame, f"Posture: {label}", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            out.write(frame)
            frame_no += 1
            #progress_bar.progress(min(frame_no / total_frames, 1.0))

    cap.release()
    out.release()
    return output_path, predictions


def analyze_posture(predictions):
    """Analyzes good/bad frame ratio and returns result summary."""
    good_count = sum(1 for p in predictions if p.lower() == "good")
    bad_count = sum(1 for p in predictions if p.lower() == "bad")
    total = len(predictions)

    if total == 0:
        return {"summary": "⚠️ No posture detected — ensure full body is visible."}

    good_percent = (good_count / total) * 100
    bad_percent = (bad_count / total) * 100

    if good_count / total > 0.75:
        summary = "✅ Good posture overall!"
    elif bad_count / total > 0.6:
        summary = "⚠️ Poor posture overall!"
    else:
        summary = "Mixed posture — needs improvement."

    return {
        "good": good_count,
        "bad": bad_count,
        "total": total,
        "good_percent": good_percent,
        "bad_percent": bad_percent,
        "summary": summary
    }


def run_posture_analysis(video_path):
    """Processes posture and returns summarized results (no Streamlit output)."""
    output_path, predictions = process_video(video_path)
    result = analyze_posture(predictions)
    return result

