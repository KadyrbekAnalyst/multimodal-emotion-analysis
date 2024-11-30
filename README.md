# Multimodal Emotion Analysis

Проект для анализа эмоций человека с использованием нескольких модальностей: видео (выражение лица), аудио (речь) и текст (расшифровка речи).

## Структура проекта

```
multimodal-emotion-analysis/
├── data/
│   ├── recordings/          # Директория с записями сессий
│   │   └── SESSION_ID/      # Уникальная папка для каждой сессии
│   │       ├── video.mp4    # Видеозапись
│   │       └── audio.wav    # Аудиозапись
│   ├── temp/               # Временные файлы (автоматически очищается)
│   └── visualizations/     # Визуализации анализа
│       └── SESSION_ID/
│           ├── chunks/     # Визуализации по чанкам
│           └── summary.png # Итоговая визуализация
├── src/
│   ├── camera/
│   │   ├── __init__.py
│   │   └── camera_recorder.py    # Модуль записи видео и аудио
│   ├── facial_recognition/
│   │   ├── __init__.py
│   │   └── facial_emotion_detector.py    # Модуль анализа эмоций по видео
│   └── speech_recognition/
│       ├── __init__.py
│       ├── speech_emotion.py     # Модуль анализа эмоций из речи
│       └── visualizer.py         # Модуль визуализации аудио и эмоций
└── requirements.txt
```

## Основные модули

### 1. Camera Recorder (`src/camera/camera_recorder.py`)
- Захват видео с веб-камеры и запись аудио с микрофона
- Детекция лица в реальном времени
- Сохранение записей в формате MP4 (видео) и WAV (аудио)
- Создание уникальной сессии для каждой записи
- **Основные классы:**
  - `CameraRecorder`: управление записью
  - `AudioRecorder`: запись аудио в отдельном потоке

### 2. Facial Emotion Detector (`src/facial_recognition/facial_emotion_detector.py`)
- Анализ эмоций по видеозаписи
- Покадровая обработка видео
- Использует DeepFace для распознавания эмоций
- Вычисление средних значений эмоций
- **Основные функции:**
  - Определение 7 базовых эмоций: angry, disgust, fear, happy, sad, surprise, neutral
  - Временная шкала изменения эмоций
  - Определение доминирующей эмоции

### 3. Speech Emotion Analyzer (`src/speech_recognition/speech_emotion.py`)
- Анализ эмоциональной окраски речи
- Разбиение аудио на чанки для анализа
- Визуализация аудио-характеристик и эмоций
- **Компоненты:**
  - Анализатор эмоций из речи
  - Визуализатор (`visualizer.py`)
  - Генерация графиков и спектрограмм
- **Визуализации включают:**
  - Форму волны
  - Спектрограмму
  - Распределение эмоций
  - Итоговую сводку по всему аудио

## Установка и зависимости

```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt

# Дополнительно для Windows
# Установка ffmpeg с https://www.gyan.dev/ffmpeg/builds/
# Добавить путь к ffmpeg в переменные среды PATH
```

## Использование

### Запись видео и аудио:
```python
from src.camera.camera_recorder import CameraRecorder

recorder = CameraRecorder()
result = recorder.record_video()  # Запись 15-секундного видео
```

### Анализ эмоций из видео:
```python
from src.facial_recognition.facial_emotion_detector import VideoEmotionAnalyzer

analyzer = VideoEmotionAnalyzer()
results = analyzer.analyze_video("path/to/video.mp4")
```

### Анализ эмоций из речи:
```python
from src.speech_recognition.speech_emotion import SpeechEmotionAnalyzer

analyzer = SpeechEmotionAnalyzer(visualization_config={
    'show_waveform': True,
    'show_spectrogram': True,
    'show_emotions': True
})
results = analyzer.analyze_emotion("path/to/audio.wav")
```

## Примечания
- Все модули создают логи своей работы
- Визуализации сохраняются в соответствующих папках сессий
- Временные файлы автоматически очищаются
- Для работы необходима веб-камера и микрофон