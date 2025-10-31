# speech_speed.py
import librosa
import speech_recognition as sr_module

def extract_wpm(file_path):
    """Extracts words per minute (WPM) from an audio file."""
    try:
        y, sample_rate = librosa.load(file_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sample_rate)

        recognizer = sr_module.Recognizer()
        with sr_module.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            word_count = len(text.split())
            wpm = (word_count / duration) * 60
    except Exception:
        wpm = 0  # If recognition fails, return 0

    return wpm


def categorize_wpm(wpm):
    """Categorizes speech speed based on WPM."""
    if wpm < 130:
        return "Slow"
    elif 130 <= wpm <= 165:
        return "Normal"
    else:
        return "Fast"
