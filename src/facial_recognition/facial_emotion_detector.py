import cv2
import numpy as np
from deepface import DeepFace
import warnings
from tqdm import tqdm
import logging
import os

# Настройка логирования
logger = logging.getLogger(__name__)

class VideoEmotionAnalyzer:
    def __init__(self):
        """Инициализация анализатора видео"""
        try:
            self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            logger.info("VideoEmotionAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing VideoEmotionAnalyzer: {e}")
            raise

    def analyze_frame(self, frame):
        """
        Анализ эмоций на одном кадре
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
            logger.error(f"Error analyzing frame: {e}")
            return None

    def analyze_video(self, video_path, sample_rate=1):
        """
        Анализ эмоций в видео файле
        :param video_path: Path to video file
        :param sample_rate: Analyze every Nth frame
        :return: Dict with analysis results
        """
        try:
            logger.info(f"Starting video analysis: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error("Could not open video file")
                return None

            frame_count = 0
            emotions_timeline = []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"Video info: {total_frames} frames, {fps} FPS")
            
            with tqdm(total=total_frames, desc="Analyzing frames") as pbar:
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
                # Вычисляем средние значения эмоций
                avg_emotions = {emotion: 0 for emotion in self.emotions}
                for entry in emotions_timeline:
                    for emotion, value in entry['emotions'].items():
                        avg_emotions[emotion] += value
                
                for emotion in avg_emotions:
                    avg_emotions[emotion] /= len(emotions_timeline)
                
                dominant_emotion = max(avg_emotions.items(), key=lambda x: x[1])[0]
                
                results = {
                    'timeline': emotions_timeline,
                    'average': avg_emotions,
                    'dominant_emotion': dominant_emotion,
                    'frames_processed': frame_count,
                    'frames_with_emotions': len(emotions_timeline)
                }
                
                logger.info(f"Analysis completed. Dominant emotion: {dominant_emotion}")
                return results
            
            logger.warning("No emotions detected in video")
            return None
            
        except Exception as e:
            logger.error(f"Error during video analysis: {e}")
            return None

    def format_emotion_dict(self, emotions_dict, dominant_emotion=None):
        """Format emotions dictionary for pretty printing"""
        try:
            formatted_emotions = [
                f"'{emotion}': {value:.2f}%{' # Dominant' if emotion == dominant_emotion else ''}"
                for emotion, value in emotions_dict.items()
            ]
            return "{\n    " + ",\n    ".join(formatted_emotions) + "\n}"
        except Exception as e:
            logger.error(f"Error formatting emotions dict: {e}")
            return str(emotions_dict)

def test_analyzer():
    """Функция для тестирования анализатора"""
    try:
        analyzer = VideoEmotionAnalyzer()
        test_path = os.path.join('app', 'static', 'temp', 'test_video.webm')
        
        if os.path.exists(test_path):
            results = analyzer.analyze_video(test_path)
            if results:
                logger.info("\nTest Results:")
                logger.info(f"Frames processed: {results['frames_processed']}")
                logger.info(f"Frames with emotions: {results['frames_with_emotions']}")
                logger.info(f"Dominant emotion: {results['dominant_emotion']}")
                return True
        
        logger.warning("Test file not found")
        return False
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_analyzer()