# video_processor.py
import cv2
import time
import os
from deepface import DeepFace
import numpy as np

# Parameters
ANALYZE_EVERY_N_FRAMES = 15  # reduce CPU — analyze once every N frames
OUTPUT_FOLDER = "Annotated_Deepface"  # Output folder for processed videos

# Emotion grouping
EMOTION_GROUPS = {
    'positive': ['happy', 'surprise'],
    'negative': ['sad', 'angry', 'fear', 'disgust'],
    'neutral': ['neutral']
}

def get_emotion_group(emotion):
    """Map individual emotion to emotion group (positive/negative/neutral)"""
    if not emotion:
        return None
    
    emotion_lower = emotion.lower()
    for group, emotions in EMOTION_GROUPS.items():
        if emotion_lower in emotions:
            return group
    return 'neutral'  # Default to neutral if emotion not found


def analyze_emotion_pytorch(face_img):
    """Try PyTorch-based detectors for emotion analysis"""
    # Resize to smaller resolution
    small_face = cv2.resize(face_img, (112, 112))  # Much faster
    
    pytorch_detectors = ['opencv']  # Use only one detector for speed
    
    for detector in pytorch_detectors:
        try:
            # analyze expects BGR or path; enforce_detection=False avoids exception
            analysis = DeepFace.analyze(
                small_face,
                actions=['emotion'], 
                detector_backend=detector, 
                enforce_detection=False,
                
            )
            
            # Handle case where result might be a list or dict
            if isinstance(analysis, list):
                if len(analysis) > 0:
                    analysis = analysis[0]  # Take first face if multiple detected
                else:
                    continue  # Try next detector
            
            if analysis and 'dominant_emotion' in analysis:
                dominant = analysis.get('dominant_emotion', None)
                score = None
                if dominant:
                    emotion_scores = analysis.get('emotion', {})
                    if isinstance(emotion_scores, dict) and dominant in emotion_scores:
                        score = emotion_scores[dominant]
                return (dominant, score)
                
        except Exception as e:
            # Try next detector if this one fails
            print(f"Error with {detector}: {e}")
            continue
    
    return None  # All detectors failed

def process_video(input_path, output_path=None):
    """Process a video file and save with emotion analysis"""
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input video '{input_path}' not found!")
        return False
    
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Generate output path if not provided
    if output_path is None:
        input_filename = os.path.basename(input_path)
        name, ext = os.path.splitext(input_filename)
        output_path = os.path.join(OUTPUT_FOLDER, f"{name}_emotion_analyzed{ext}")
    
    # Initialize emotion counters
    emotion_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open input video '{input_path}'")
        return False

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {width}x{height}, {fps} FPS, {total_frames} frames")
    print(f"Output will be saved to: {output_path}")
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Cannot create output video '{output_path}'")
        cap.release()
        return False

    # Load OpenCV's Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    frame_count = 0
    last_result = None  # store last analyze result for display
    start_time = time.time()

    print("Starting video emotion analysis...")
    print("Processing frames...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using OpenCV
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )

            if len(faces) > 0:
                # Process only the first face
                x, y, bw, bh = faces[0]
                x2 = x + bw
                y2 = y + bh

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

                # Every N frames, run DeepFace analyze on cropped face
                if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
                    # expand bbox a little
                    pad = int(0.15 * max(bw, bh))
                    xa = max(0, x - pad)
                    ya = max(0, y - pad)
                    xb = min(w, x2 + pad)
                    yb = min(h, y2 + pad)

                    face_img = frame[ya:yb, xa:xb]
                    if face_img.size != 0:
                        result = analyze_emotion_pytorch(face_img)
                        if result:
                            # Group the emotion and update counters
                            dominant_emotion, score = result
                            emotion_group = get_emotion_group(dominant_emotion)
                            if emotion_group:
                                emotion_counts[emotion_group] += 1
                                last_result = (emotion_group, score)
                            else:
                                last_result = result
                        # If result is None, keep the last_result for display
                    else:
                        last_result = None

                # Draw last_result text in top-left of box
                if last_result and last_result[0] is not None:
                    dominant, score = last_result
                    # Format score; if None, display '?'
                    score_text = f"{score:.2f}%" if (score is not None) else "?"
                    text = f"{dominant} {score_text}"
                    # Put text with background for readability
                    (tx, ty), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    # background rectangle
                    cv2.rectangle(frame, (x, y - ty - baseline - 6), (x + tx + 6, y), (0, 255, 0), -1)
                    cv2.putText(frame, text, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            else:
                # Show "No face" text
                cv2.putText(frame, "No face", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # Add progress indicator
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            cv2.putText(frame, f"Progress: {progress:.1f}%", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count + 1}/{total_frames}", (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Write frame to output video
            out.write(frame)
            frame_count += 1

            # Print progress every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed if elapsed > 0 else 0
                remaining_frames = total_frames - frame_count
                eta_seconds = remaining_frames / fps_processing if fps_processing > 0 else 0
                eta_minutes = eta_seconds / 60
                print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%) - Processing FPS: {fps_processing:.1f} - ETA: {eta_minutes:.1f} minutes")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Clean up
        cap.release()
        out.release()
        
        total_time = time.time() - start_time
        print(f"\nVideo processing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {total_time:.1f} seconds or {total_time/60:.1f} minutes")
        print(f"Average processing FPS: {frame_count / total_time:.1f}")
        print(f"Output saved to: {output_path}")
        
        # Calculate and display final dominant emotion
                # Calculate and return final dominant emotion results
        total_detections = sum(emotion_counts.values())

        if total_detections > 0:
            # Calculate emotion percentages
            emotion_percentages = {
                group: round((count / total_detections) * 100, 1)
                for group, count in emotion_counts.items()
            }

            # Determine dominant group
            dominant_emotion_group = max(emotion_counts, key=emotion_counts.get)
            dominant_percentage = emotion_percentages[dominant_emotion_group]

            # Print (for debugging)
            print("\nEmotion Analysis Summary:")
            for g, pct in emotion_percentages.items():
                print(f"  {g.capitalize()}: {pct}%")
            print(f"Dominant Emotion: {dominant_emotion_group.upper()} ({dominant_percentage}%)")

            if dominant_emotion_group in ['positive', 'neutral']:
                status = "GOOD"
            else:
                status = "BAD"

            # ✅ Return structured data for Streamlit
            return {
                "dominant": dominant_emotion_group.capitalize(),
                "positive": emotion_percentages.get("positive", 0),
                "neutral": emotion_percentages.get("neutral", 0),
                "negative": emotion_percentages.get("negative", 0),
                "status": status,
                "video_path": output_path
            }
        else:
            print("No emotions detected during video processing.")
            return {
                "dominant": "None",
                "positive": 0,
                "neutral": 0,
                "negative": 0,
                "status": "BAD",
                "video_path": output_path
            }


def main():
    """Main function for command line usage"""
    import sys
    
    # CHANGE THIS PATH TO YOUR VIDEO FILE
    input_path = "./uploads/test.mp4"
        
    # Optional: uncomment below for command line usage instead
    # if len(sys.argv) < 2:
    #     print("Usage: python deepface_video_processor.py <input_video_path> [output_video_path]")
    #     print("Example: python deepface_video_processor.py my_video.mp4")
    #     print("Example: python deepface_video_processor.py my_video.mp4 custom_output.mp4")
    #     return
    # 
    # input_path = sys.argv[1]
    output_path = None  # Will auto-generate output filename
    
    success = process_video(input_path, output_path)
    if success:
        print("Video processing completed successfully!")
    else:
        print("Video processing failed!")

if __name__ == "__main__":
    main()
