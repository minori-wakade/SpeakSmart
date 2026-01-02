import cv2
import mediapipe as mp
import os
import csv

# Initialize mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Paths
base_dir = r"C:\Users\manji\OneDrive\Desktop\Posture ds\Data"  # change if needed
categories = ["good", "bad"]

# Output CSV
csv_file = "pose_data.csv"

# Header
header = ["label"]
for i in range(33):
    header += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]

# Write header
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for category in categories:
        folder = os.path.join(base_dir, category)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                row = [category]
                for lm in results.pose_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])
                writer.writerow(row)

print(f"✅ Pose data extracted successfully and saved to {csv_file}")
