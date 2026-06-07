# 🎤 SpeakSmart – AI Based Public Speaking Assistant

**SpeakSmart** is a Streamlit-based application that analyzes recorded speeches using computer vision and audio processing to deliver actionable feedback on communication skills.

---

## 🚀 Key Features
- Video-based speech analysis  
- Eye contact detection (MediaPipe)  
- Posture evaluation (SVM Model trained on custom dataset)  
- Speech speed analysis (WPM)  
- Audio emotion recognition (CNN model trained on RAVDESS dataset)  
- Facial expression analysis (DeepFace)  
- Performance report with overall score  
- History tracking (Library view) 

---

## Output Files

- `uploads/` — Automatically created folder containing timestamped analysis reports.
- `results.json` — Saved JSON report for each upload.
- `Annotated_eye_contact/` — Annotated eye contact output videos.
- `Annotated_Deepface/` — Annotated facial emotion videos.
- `Annotated_Posture/` — Annotated posture videos.

## Recommended Workflow

1. Upload a speech video
2. System analyzes video + audio
3. Get feedback (posture, eye contact, speed, speech emotions and facial expressions)
4. Track progress via saved reports on library page

## 🎥 Demo Video

[Watch Demo](https://drive.google.com/drive/folders/1LIf-tnulzbCy6ArfuDCAGhYgoFk2pFGT?usp=sharing)
   
## ⚙️ Run Locally

```bash
git clone https://github.com/minori-wakade/SpeakSmart.git
cd SpeakSmart

pip install -r requirements.txt
streamlit run app.py
