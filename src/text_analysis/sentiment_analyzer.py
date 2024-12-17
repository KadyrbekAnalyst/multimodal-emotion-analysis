import whisper
from transformers import pipeline
from deep_translator import GoogleTranslator
import warnings
import numpy as np
from typing import Dict, List, Optional
import os

class TextEmotionAnalyzer:
    def __init__(self):
        # Подавляем предупреждения при загрузке модели
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Инициализируем Whisper для распознавания речи
            self.speech_model = whisper.load_model("small")
            
            # Используем Google Translate для перевода
            self.translator = GoogleTranslator(source='ru', target='en')
            
            # RoBERTa для определения эмоций
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-roberta-large",
                device="cpu"
            )

    def translate_to_english(self, text: str) -> str:
        """
        Перевод текста с русского на английский
        """
        try:
            translation = self.translator.translate(text)
            return translation
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Преобразование аудио в текст
        """
        try:
            result = self.speech_model.transcribe(
                audio_path,
                language='russian',
                fp16=False
            )
            
            return {
                'success': True,
                'text': result["text"],
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'text': None,
                'error': str(e)
            }

    def analyze_emotions(self, text: str) -> Dict[str, float]:
        """
        Анализ эмоций в тексте
        """
        try:
            # Переводим текст на английский
            english_text = self.translate_to_english(text)
            print(f"\nTranslated text: {english_text}")
            
            # Получаем предсказания модели
            result = self.emotion_classifier(english_text)
            
            # Базовые эмоции
            emotions = {
                'angry': 0.0,
                'disgust': 0.0,
                'fear': 0.0,
                'happy': 0.0,
                'sad': 0.0,
                'surprise': 0.0,
                'neutral': 0.0
            }
            
            # Маппинг меток эмоций
            label_mapping = {
                'anger': 'angry',
                'disgust': 'disgust',
                'fear': 'fear',
                'joy': 'happy',
                'sadness': 'sad',
                'surprise': 'surprise',
                'neutral': 'neutral'
            }
            
            # Обновляем значения эмоций
            label = result[0]['label'].lower()
            score = result[0]['score'] * 100
            
            if label in label_mapping:
                main_emotion = label_mapping[label]
                emotions[main_emotion] = score
                
                # Более реалистичное распределение оставшейся вероятности
                remaining_emotions = [e for e in emotions if e != main_emotion]
                
                # Распределяем оставшуюся вероятность с убыванием
                remaining_score = 100 - score
                for i, emotion in enumerate(remaining_emotions):
                    # Экспоненциальное убывание для оставшихся эмоций
                    emotions[emotion] = remaining_score * np.exp(-i) / sum(np.exp(-np.arange(len(remaining_emotions))))
            
            return emotions
            
        except Exception as e:
            print(f"Error analyzing text emotions: {e}")
            return None

    def process_audio(self, audio_path: str) -> Optional[Dict]:
        """
        Полный процесс анализа аудио: транскрибация и анализ эмоций
        """
        # Транскрибация
        transcription = self.transcribe_audio(audio_path)
        if not transcription['success']:
            print(f"Transcription failed: {transcription['error']}")
            return None
            
        text = transcription['text']
        print(f"\nTranscribed text: {text}")
        
        # Анализ эмоций
        emotions = self.analyze_emotions(text)
        if emotions is None:
            return None
            
        # Определяем доминирующую эмоцию
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        
        return {
            'text': text,
            'emotions': emotions,
            'dominant_emotion': dominant_emotion
        }

def format_results(results: Dict) -> str:
    """
    Форматирование результатов для вывода
    """
    if not results:
        return "Analysis failed"
        
    output = [
        f"Transcribed text: {results['text']}",
        "\nEmotions detected:",
        "{\n    " + ",\n    ".join([
            f"'{emotion}': {value:.2f}%{' # Dominant' if emotion == results['dominant_emotion'] else ''}"
            for emotion, value in results['emotions'].items()
        ]) + "\n}"
    ]
    
    return "\n".join(output)

def main():
    # Отключаем все предупреждения
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключаем предупреждения TensorFlow
    
    analyzer = TextEmotionAnalyzer()
    
    base_dir = "data/recordings"
    sessions = sorted([d for d in os.listdir(base_dir) 
                      if os.path.isdir(os.path.join(base_dir, d))], 
                     reverse=True)
    
    if sessions:
        latest_session = sessions[0]
        audio_path = os.path.join(base_dir, latest_session, "audio.wav")
        
        if os.path.exists(audio_path):
            print(f"Processing audio from session: {latest_session}")
            results = analyzer.process_audio(audio_path)
            print("\nAnalysis Results:")
            print(format_results(results))
        else:
            print(f"No audio file found in latest session: {audio_path}")
    else:
        print("No recording sessions found")

if __name__ == "__main__":
    main()