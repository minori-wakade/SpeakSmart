import cv2
import os
from pathlib import Path

# -------------------------
# CONFIG — change here
# -------------------------

# Full path of the video you want to process
VIDEO_PATH = r"C:\Users\manji\OneDrive\Desktop\Posture ds\DATA\videos\bad6.mp4"

# Label / folder name to save frames
LABEL_TYPE = "mix"   # or "good"

# Folder where frames will be saved
OUTPUT_FOLDER = r"C:\Users\manji\OneDrive\Desktop\Posture ds\DATA"  # base folder
OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, LABEL_TYPE)

# Number of frames to extract
NUM_FRAMES = 20

# -------------------------
# SCRIPT
# -------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Cannot open video:", VIDEO_PATH)
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
interval = max(total_frames // NUM_FRAMES, 1)

count = 0
saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if count % interval == 0:
        frame_name = f"{Path(VIDEO_PATH).stem}_frame_{saved:04d}.jpg"
        save_path = os.path.join(OUTPUT_FOLDER, frame_name)
        cv2.imwrite(save_path, frame)
        saved += 1
    count += 1

cap.release()
print(f"✅ Saved {saved} frames to '{OUTPUT_FOLDER}'")
