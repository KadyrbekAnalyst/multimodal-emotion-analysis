import whisper
from transformers import pipeline
from deep_translator import GoogleTranslator
import warnings
import logging
import os

logger = logging.getLogger(__name__)

class TextEmotionAnalyzer:
    def __init__(self):
        """Инициализация анализатора текста"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                logger.info("Loading Whisper model...")
                self.speech_model = whisper.load_model("small")
                
                logger.info("Initializing translator...")
                self.translator = GoogleTranslator(source='ru', target='en')
                
                logger.info("Loading emotion classifier...")
                self.emotion_classifier = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-roberta-large",
                    device="cpu"
                )
                
                self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                logger.info("TextEmotionAnalyzer initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing TextEmotionAnalyzer: {e}")
            raise

    def translate_to_english(self, text: str) -> str:
        """Перевод текста с русского на английский"""
        try:
            logger.info("Translating text to English")
            translation = self.translator.translate(text)
            logger.info("Translation completed")
            return translation
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text

    def transcribe_audio(self, audio_path: str) -> dict:
        """Преобразование аудио в текст"""
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            result = self.speech_model.transcribe(
                audio_path,
                language='russian',
                fp16=False
            )
            
            logger.info("Transcription completed")
            return {
                'success': True,
                'text': result["text"],
                'error': None
            }
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                'success': False,
                'text': None,
                'error': str(e)
            }

    def analyze_emotions(self, text: str) -> dict:
        """Анализ эмоций в тексте"""
        try:
            logger.info("Starting emotion analysis")
            # Переводим текст на английский
            english_text = self.translate_to_english(text)
            logger.info(f"Translated text: {english_text}")
            
            # Получаем предсказания модели
            result = self.emotion_classifier(english_text)
            
            # Инициализируем базовые эмоции
            emotions = {emotion: 0.0 for emotion in self.emotions}
            
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
                
                # Распределяем оставшуюся вероятность
                remaining_emotions = [e for e in emotions if e != main_emotion]
                remaining_score = 100 - score
                for i, emotion in enumerate(remaining_emotions):
                    emotions[emotion] = remaining_score / len(remaining_emotions)
            
            logger.info("Emotion analysis completed")
            return emotions
            
        except Exception as e:
            logger.error(f"Error analyzing text emotions: {e}")
            return None

    def process_audio(self, audio_path: str) -> dict:
        """Полный процесс анализа: транскрибация и анализ эмоций"""
        try:
            logger.info(f"Starting audio processing: {audio_path}")
            
            # Транскрибация
            transcription = self.transcribe_audio(audio_path)
            if not transcription['success']:
                logger.error(f"Transcription failed: {transcription['error']}")
                return None
                
            text = transcription['text']
            logger.info(f"Transcribed text: {text}")
            
            # Анализ эмоций
            emotions = self.analyze_emotions(text)
            if emotions is None:
                return None
                
            # Определяем доминирующую эмоцию
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            results = {
                'text': text,
                'emotions': emotions,
                'dominant_emotion': dominant_emotion
            }
            
            logger.info(f"Processing completed. Dominant emotion: {dominant_emotion}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return None

def test_analyzer():
    """Тестирование анализатора"""
    try:
        analyzer = TextEmotionAnalyzer()
        test_path = os.path.join('app', 'static', 'temp', 'test_video.webm')
        
        if os.path.exists(test_path):
            results = analyzer.process_audio(test_path)
            if results:
                logger.info("\nTest Results:")
                logger.info(f"Text: {results['text']}")
                logger.info(f"Dominant emotion: {results['dominant_emotion']}")
                return True
                
        logger.warning("Test file not found")
        return False
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_analyzer()