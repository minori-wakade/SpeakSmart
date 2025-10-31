from moviepy.editor import VideoFileClip

def video_to_wav(video_path, wav_path="temp.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(wav_path, fps=16000)
    clip.close()
    return wav_path
