from src.facial_recognition.facial_emotion_detector import VideoEmotionAnalyzer
from src.speech_recognition.speech_emotion import SpeechEmotionAnalyzer
from src.text_analysis.sentiment_analyzer import TextEmotionAnalyzer
from src.fusion.emotion_fusion import EmotionFusion
import os
from typing import Dict, Optional

class EmotionAnalysisSystem:
    def __init__(self, base_dir: str = 'data'):
        """
        :param base_dir: Базовая директория для сохранения данных
        """
        self.base_dir = base_dir
        
        # Инициализация анализаторов
        self.video_analyzer = VideoEmotionAnalyzer()
        self.speech_analyzer = SpeechEmotionAnalyzer()
        self.text_analyzer = TextEmotionAnalyzer()
        
        # Инициализация fusion с настроенными весами
        self.fusion = EmotionFusion(weights={
            'video': 0.4,
            'audio': 0.3,
            'text': 0.3
        })

    def analyze_session(self, session_data: Dict) -> Optional[Dict]:
        """
        Анализ записанной сессии
        """
        try:
            # Анализ видео
            print("\nAnalyzing video emotions...")
            video_results = self.video_analyzer.analyze_video(
                session_data['video_path']
            )
            if video_results:
                video_emotions = video_results['average']
            else:
                print("Video analysis failed")
                return None

            # Анализ аудио
            print("\nAnalyzing speech emotions...")
            speech_results = self.speech_analyzer.analyze_emotion(
                session_data['audio_path']
            )
            if speech_results:
                speech_emotions = speech_results['average']
            else:
                print("Speech analysis failed")
                return None

            # Анализ текста
            print("\nAnalyzing text emotions...")
            text_results = self.text_analyzer.process_audio(
                session_data['audio_path']
            )
            if text_results:
                text_emotions = text_results['emotions']
            else:
                print("Text analysis failed")
                return None

            # Объединение результатов
            print("\nFusing emotion results...")
            fusion_results = self.fusion.fuse_emotions(
                video_emotions,
                speech_emotions,
                text_emotions
            )

            return {
                'session_id': session_data['session_id'],
                'video_emotions': video_emotions,
                'speech_emotions': speech_emotions,
                'text_emotions': text_emotions,
                'fusion_results': fusion_results
            }

        except Exception as e:
            print(f"Error during analysis: {e}")
            return None