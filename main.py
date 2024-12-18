from src.facial_recognition.facial_emotion_detector import VideoEmotionAnalyzer
from src.speech_recognition.speech_emotion import SpeechEmotionAnalyzer
from src.text_analysis.sentiment_analyzer import TextEmotionAnalyzer
from src.fusion.emotion_fusion import EmotionFusion
from src.visualizer.visualizer import EmotionVisualizer
import os
import logging
from typing import Dict, Optional

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Базовые пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'app', 'static', 'temp')
VISUALIZATION_DIR = os.path.join(TEMP_DIR, 'visualizations')

# Создаем необходимые директории
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

class EmotionAnalysisSystem:
    def __init__(self):
        """Инициализация всех компонентов системы"""
        try:
            logger.info("Initializing EmotionAnalysisSystem...")
            self.video_analyzer = VideoEmotionAnalyzer()
            self.speech_analyzer = SpeechEmotionAnalyzer()
            self.text_analyzer = TextEmotionAnalyzer()
            self.fusion = EmotionFusion()
            self.visualizer = EmotionVisualizer()
            logger.info("EmotionAnalysisSystem initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing EmotionAnalysisSystem: {e}")
            raise

    def analyze_session(self, data: Dict) -> Optional[Dict]:
        """
        Анализ записанной сессии
        :param data: словарь с путями к видео и аудио файлам
        :return: результаты анализа или None в случае ошибки
        """
        try:
            logger.info(f"Starting analysis for session with data: {data}")
            
            # 1. Анализ видео
            logger.info("Analyzing video emotions...")
            video_emotions = self.video_analyzer.analyze_video(data['video_path'])
            if not video_emotions:
                logger.error("Video analysis failed")
                return None

            # 2. Анализ аудио
            logger.info("Analyzing speech emotions...")
            speech_emotions = self.speech_analyzer.analyze_emotion(data['audio_path'])
            if not speech_emotions:
                logger.error("Speech analysis failed")
                return None

            # 3. Анализ текста
            logger.info("Analyzing text emotions...")
            text_results = self.text_analyzer.process_audio(data['audio_path'])
            if not text_results:
                logger.error("Text analysis failed")
                return None

            # 4. Объединение результатов
            logger.info("Fusing emotion results...")
            fusion_results = self.fusion.fuse_emotions(
                video_emotions['average'],
                speech_emotions['average'],
                text_results['emotions']
            )

            # 5. Создание визуализации
            logger.info("Creating visualization...")
            visualization_path = os.path.join(
                VISUALIZATION_DIR,
                f'visualization_{os.path.basename(data["video_path"])}.png'
            )
            
            self.visualizer.create_visualization(
                video_emotions,
                speech_emotions,
                text_results,
                fusion_results,
                visualization_path
            )

            results = {
                'video_emotions': video_emotions,
                'speech_emotions': speech_emotions,
                'text_emotions': text_results,
                'fusion_results': fusion_results,
                'visualization_path': visualization_path
            }

            logger.info("Analysis completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return None

    def cleanup(self):
        """Очистка временных файлов"""
        try:
            for dir_path in [TEMP_DIR, VISUALIZATION_DIR]:
                if os.path.exists(dir_path):
                    for file_name in os.listdir(dir_path):
                        file_path = os.path.join(dir_path, file_name)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            logger.error(f"Error deleting {file_path}: {e}")
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def test_system():
    """Тестирование системы"""
    try:
        system = EmotionAnalysisSystem()
        test_data = {
            'video_path': os.path.join(TEMP_DIR, 'test_video.webm'),
            'audio_path': os.path.join(TEMP_DIR, 'test_video.webm')
        }
        results = system.analyze_session(test_data)
        if results:
            logger.info("Test successful")
            return True
        logger.error("Test failed")
        return False
    except Exception as e:
        logger.error(f"Test error: {e}")
        return False

if __name__ == "__main__":
    test_system()