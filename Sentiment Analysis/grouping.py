import pandas as pd

# Load your CSV
df = pd.read_csv("initial.csv")

# Define mapping
emotion_map = {
    "happy": "positive",
    "surprised": "positive",
    "neutral": "positive",
    "calm": "positive",
    "sad": "negative",
    "angry": "negative",
    "fearful": "negative",
    "disgust": "negative"
}

df["sentiment"] = df["emotion"].map(emotion_map)

df.to_csv("grouped.csv", index=False)

print("✅ File saved as grouped.csv")