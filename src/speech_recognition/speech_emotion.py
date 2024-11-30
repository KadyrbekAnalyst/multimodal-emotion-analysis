import os
import numpy as np
from scipy.io import wavfile
from transformers import pipeline
import warnings
import shutil
from datetime import datetime
from tqdm import tqdm
from visualizer import AudioVisualizer, generate_summary_visualization

# Отключаем предупреждения TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class SpeechEmotionAnalyzer:
    def __init__(self, visualization_config=None):
        """
        :param visualization_config: Настройки визуализации
            {
                'show_waveform': True/False,
                'show_spectrogram': True/False,
                'show_emotions': True/False
            }
        """
        self.visualization_config = visualization_config or {
            'show_waveform': True,
            'show_spectrogram': True,
            'show_emotions': True
        }
        self.visualizer = AudioVisualizer(**self.visualization_config)
        
        # Подавляем предупреждения при загрузке модели
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.emotion_classifier = pipeline(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            )
        
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Создаем структуру директорий
        self.base_dir = 'data'
        self.temp_dir = os.path.join(self.base_dir, 'temp')
        self.viz_dir = os.path.join(self.base_dir, 'visualizations')
        
        # Создаем базовые директории
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        print(f"Created/checked directories:")
        print(f"- Temporary files: {self.temp_dir}")
        print(f"- Visualizations: {self.viz_dir}")

    def read_and_normalize_audio(self, audio_path):
        """Чтение и нормализация аудио файла"""
        print(f"\nReading audio file: {audio_path}")
        sr, audio = wavfile.read(audio_path)
        
        # Нормализация аудио
        audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0
            
        print(f"Audio loaded: {len(audio)} samples, {sr}Hz, duration: {len(audio)/sr:.2f}s")
        return audio, sr

    def analyze_emotion(self, audio_path, chunk_duration=3.0):
        """
        Анализ эмоций в аудио файле
        :param audio_path: путь к аудио файлу
        :param chunk_duration: длительность чанка в секундах
        """
        try:
            # Получаем ID сессии из пути к аудио
            session_id = os.path.basename(os.path.dirname(audio_path))
            session_viz_dir = os.path.join(self.viz_dir, session_id)
            chunks_viz_dir = os.path.join(session_viz_dir, 'chunks')
            
            # Создаем директории для визуализаций текущей сессии
            os.makedirs(chunks_viz_dir, exist_ok=True)
            print(f"Created visualization directory for session: {session_viz_dir}")
            
            # Загружаем аудио
            audio, sr = self.read_and_normalize_audio(audio_path)
            duration = len(audio) / sr
            
            # Разбиваем на чанки
            chunk_samples = int(chunk_duration * sr)
            chunks = [audio[i:i + chunk_samples] for i in range(0, len(audio), chunk_samples)]
            
            print(f"\nProcessing {len(chunks)} chunks of {chunk_duration}s each")
            
            emotions_timeline = []
            all_emotions = {emotion: 0.0 for emotion in self.emotions}
            chunk_count = 0
            
            # Прогресс бар для обработки чанков
            for i, chunk in enumerate(tqdm(chunks, desc="Analyzing chunks")):
                if len(chunk) < sr: # Пропускаем слишком короткие чанки
                    continue
                
                # Путь для временного файла
                chunk_path = os.path.join(self.temp_dir, f'chunk_{i}.wav')
                viz_path = os.path.join(chunks_viz_dir, f'chunk_{i}_viz.png')
                
                try:
                    # Сохраняем чанк
                    wavfile.write(chunk_path, sr, chunk)
                    
                    # Анализируем эмоции
                    result = self.emotion_classifier(chunk_path)
                    
                    # Преобразуем результаты
                    chunk_emotions = {emotion: 0.0 for emotion in self.emotions}
                    for pred in result:
                        label = pred['label'].lower()
                        if label in self.emotions:
                            chunk_emotions[label] = pred['score'] * 100
                    
                    # Создаем визуализацию чанка
                    self.visualizer.visualize_chunk(chunk, sr, i, chunk_emotions, viz_path)
                    
                    # Добавляем в timeline
                    timestamp = i * chunk_duration
                    emotions_timeline.append({
                        'timestamp': timestamp,
                        'emotions': chunk_emotions
                    })
                    
                    # Обновляем общую статистику
                    for emotion in self.emotions:
                        all_emotions[emotion] += chunk_emotions[emotion]
                    chunk_count += 1
                    
                finally:
                    # Удаляем временный аудио файл
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
            
            # Вычисляем средние значения
            if chunk_count > 0:
                for emotion in all_emotions:
                    all_emotions[emotion] /= chunk_count
            
            dominant_emotion = max(all_emotions.items(), key=lambda x: x[1])[0]
            
            results = {
                'timeline': emotions_timeline,
                'average': all_emotions,
                'dominant_emotion': dominant_emotion,
                'duration': duration,
                'chunks_processed': chunk_count
            }
            
            # Генерируем итоговую визуализацию
            summary_viz_path = os.path.join(session_viz_dir, 'summary.png')
            generate_summary_visualization(audio_path, results, summary_viz_path)
            
            print(f"\nVisualizations saved in: {session_viz_dir}")
            print(f"- Chunks visualizations: {chunks_viz_dir}")
            print(f"- Summary visualization: {summary_viz_path}")
            
            return results
            
        except Exception as e:
            print(f"Error analyzing speech emotion: {e}")
            return None
    
    def __del__(self):
        """Очистка временных файлов"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"\nCleaned up temporary directory: {self.temp_dir}")

def format_emotion_dict(emotions_dict, dominant_emotion=None):
    """Форматированный вывод эмоций"""
    formatted_emotions = [
        f"'{emotion}': {value:.2f}%{' # Dominant' if emotion == dominant_emotion else ''}"
        for emotion, value in emotions_dict.items()
    ]
    return "{\n    " + ",\n    ".join(formatted_emotions) + "\n}"

def main():
    # Пример использования с разными настройками визуализации
    vis_config = {
        'show_waveform': True,
        'show_spectrogram': True,
        'show_emotions': True
    }
    
    analyzer = SpeechEmotionAnalyzer(visualization_config=vis_config)
    
    base_dir = "data/recordings"
    sessions = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))], reverse=True)
    
    if sessions:
        latest_session = sessions[0]
        audio_path = os.path.join(base_dir, latest_session, "audio.wav")
        
        if os.path.exists(audio_path):
            print(f"Analyzing audio from session: {latest_session}")
            results = analyzer.analyze_emotion(audio_path)
            
            if results:
                print("\nSpeech Emotion Analysis Results:")
                print(f"Total duration: {results['duration']:.2f}s")
                print(f"Chunks processed: {results['chunks_processed']}")
                
                print("\nAverage emotions:")
                print(format_emotion_dict(results['average'], results['dominant_emotion']))
                
                print("\nFirst 3 entries in timeline:")
                for entry in results['timeline'][:3]:
                    print(f"\nTime: {entry['timestamp']:.2f}s")
                    print(format_emotion_dict(entry['emotions']))
            else:
                print("Analysis failed")
        else:
            print(f"No audio file found in latest session: {audio_path}")
    else:
        print("No recording sessions found")

if __name__ == "__main__":
    main()