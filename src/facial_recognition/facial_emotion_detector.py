import cv2
import numpy as np
from deepface import DeepFace
import warnings
from tqdm import tqdm

class VideoEmotionAnalyzer:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def analyze_frame(self, frame):
        """
        Analyze emotions in a single frame
        Returns: dict with emotion probabilities or None if no face detected
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                
                if result:
                    return result[0]['emotion']
                return None
                
        except Exception as e:
            return None

    def format_emotion_dict(self, emotions_dict, dominant_emotion=None):
        """Format emotions dictionary for pretty printing"""
        formatted_emotions = [
            f"'{emotion}': {value:.2f}%{' # Dominant' if emotion == dominant_emotion else ''}"
            for emotion, value in emotions_dict.items()
        ]
        return "{\n    " + ",\n    ".join(formatted_emotions) + "\n}"

    def analyze_video(self, video_path, sample_rate=1):
        """
        Analyze emotions in video file
        :param video_path: Path to video file
        :param sample_rate: Analyze every Nth frame
        :return: List of emotions over time
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return None

        frame_count = 0
        emotions_timeline = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Analyzing video emotions... Total frames: {total_frames}")
        
        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sample_rate == 0:
                    emotions = self.analyze_frame(frame)
                    if emotions:
                        timestamp = frame_count / fps
                        emotions_timeline.append({
                            'timestamp': timestamp,
                            'emotions': emotions
                        })

                frame_count += 1
                pbar.update(1)

        cap.release()
        
        if emotions_timeline:
            avg_emotions = {emotion: 0 for emotion in self.emotions}
            for entry in emotions_timeline:
                for emotion, value in entry['emotions'].items():
                    avg_emotions[emotion] += value
            
            for emotion in avg_emotions:
                avg_emotions[emotion] /= len(emotions_timeline)
            
            dominant_emotion = max(avg_emotions.items(), key=lambda x: x[1])[0]
                
            return {
                'timeline': emotions_timeline,
                'average': avg_emotions,
                'dominant_emotion': dominant_emotion
            }
        
        return None

def main():
    analyzer = VideoEmotionAnalyzer()
    video_path = "data/raw/emotion_recording_20241130_131455.mp4"  # Замените на реальный путь
    
    results = analyzer.analyze_video(video_path)
    if results:
        print("\nAverage emotions:")
        print(analyzer.format_emotion_dict(results['average'], results['dominant_emotion']))
        
        print("\nFirst 3 entries in timeline:")
        for entry in results['timeline'][:3]:
            print(f"\nTime: {entry['timestamp']:.2f}s")
            print(analyzer.format_emotion_dict(entry['emotions']))

if __name__ == "__main__":
    main()