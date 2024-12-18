import os
import numpy as np
from scipy.io import wavfile
from transformers import pipeline
import warnings
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from visualizer.audio_visualizer import AudioVisualizer

logger = logging.getLogger(__name__)

class SpeechEmotionAnalyzer:
    def __init__(self):
        """Инициализация анализатора речи"""
        try:
            # Подавляем предупреждения при загрузке модели
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.emotion_classifier = pipeline(
                    "audio-classification",
                    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                )
            
            self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            self.visualizer = AudioVisualizer()
            logger.info("SpeechEmotionAnalyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing SpeechEmotionAnalyzer: {e}")
            raise

    def read_and_normalize_audio(self, audio_path):
        """Чтение и нормализация аудио файла"""
        try:
            logger.info(f"Reading audio file: {audio_path}")
            sr, audio = wavfile.read(audio_path)
            
            # Нормализация аудио
            audio = audio.astype(np.float32)
            if audio.max() > 1.0:
                audio = audio / 32768.0
                
            logger.info(f"Audio loaded: {len(audio)} samples, {sr}Hz, duration: {len(audio)/sr:.2f}s")
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error reading audio file: {e}")
            return None, None

    def analyze_emotion(self, audio_path):
        """
        Анализ эмоций в аудио файле
        :param audio_path: путь к аудио файлу
        :return: результаты анализа
        """
        try:
            logger.info(f"Starting audio analysis: {audio_path}")
            
            # Загружаем аудио
            audio, sr = self.read_and_normalize_audio(audio_path)
            if audio is None:
                return None
                
            duration = len(audio) / sr
            
            # Анализируем эмоции
            result = self.emotion_classifier(audio_path)
            
            # Преобразуем результаты
            emotions = {emotion: 0.0 for emotion in self.emotions}
            for pred in result:
                label = pred['label'].lower()
                if label in self.emotions:
                    emotions[label] = pred['score'] * 100
            
            # Определяем доминирующую эмоцию
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            # Создаем визуализацию
            visualization_path = os.path.join(
                'app', 'static', 'temp',
                f'audio_visualization_{os.path.basename(audio_path)}.png'
            )
            
            self.visualizer.create_visualization(
                audio_data=audio,
                sr=sr,
                emotions=emotions,
                save_path=visualization_path
            )
            
            results = {
                'average': emotions,
                'dominant_emotion': dominant_emotion,
                'duration': duration,
                'visualization_path': visualization_path
            }
            
            logger.info(f"Audio analysis completed. Dominant emotion: {dominant_emotion}")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing speech emotion: {e}")
            return None

def test_analyzer():
    """Тестирование анализатора"""
    try:
        analyzer = SpeechEmotionAnalyzer()
        test_path = os.path.join('app', 'static', 'temp', 'test_video.webm')
        
        if os.path.exists(test_path):
            results = analyzer.analyze_emotion(test_path)
            if results:
                logger.info("\nTest Results:")
                logger.info(f"Duration: {results['duration']:.2f}s")
                logger.info(f"Dominant emotion: {results['dominant_emotion']}")
                return True
                
        logger.warning("Test file not found")
        return False
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_analyzer()