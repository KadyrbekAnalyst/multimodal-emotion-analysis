from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import os
import base64
import logging
from datetime import datetime
from ngrok import set_auth_token, connect
import sys
import subprocess
from pydub import AudioSegment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import EmotionAnalysisSystem

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Отключаем предупреждения TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Загружаем каскад Хаара для детекции лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Инициализируем систему анализа
analysis_system = EmotionAnalysisSystem()

# Создаем необходимые директории
os.makedirs('app/static/temp', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

def convert_webm_to_wav(webm_path):
    wav_path = webm_path.replace('.webm', '.wav')
    try:
        # Используем ffmpeg для конвертации
        subprocess.run(['ffmpeg', '-i', webm_path, wav_path], check=True)
        return wav_path
    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        return None

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file'}), 400
            
        video_file = request.files['video']
        if not video_file:
            return jsonify({'error': 'Empty video file'}), 400
            
        # Сохраняем файл временно
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = os.path.join('app', 'static', 'temp', f'video_{session_id}.webm')
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        video_file.save(temp_path)
        logger.info(f"Video saved to {temp_path}")

        # Конвертируем видео в WAV для анализа аудио
        wav_path = convert_webm_to_wav(temp_path)
        if wav_path is None:
            return jsonify({'error': 'Failed to convert audio'}), 500

        # Запускаем анализ
        results = analysis_system.analyze_session({
            'video_path': temp_path,
            'audio_path': wav_path,
            'session_id': session_id
        })

        if results is None:
            return jsonify({'error': 'Analysis failed'}), 500

        # Формируем URL для визуализации
        visualization_url = f'/static/temp/visualizations/visualization_video_{session_id}.webm.png'
        
        return jsonify({
            'status': 'success',
            'results': results,
            'visualization_url': visualization_url
        })
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/check_face', methods=['POST'])
def check_face():
    try:
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'error': 'No image data'}), 400
            
        # Конвертируем base64 в изображение
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Ищем лица
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        return jsonify({
            'faces_found': len(faces) > 0,
            'face_count': len(faces)
        })
        
    except Exception as e:
        logger.error(f"Error checking face: {e}")
        return jsonify({'error': str(e)}), 500

def cleanup_temp_files():
    """Очистка старых временных файлов"""
    temp_dir = 'app/static/temp'
    if os.path.exists(temp_dir):
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path):
                        if (datetime.now() - datetime.fromtimestamp(os.path.getctime(file_path))).total_seconds() > 3600:
                            os.unlink(file_path)
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")

def setup_ngrok():
    """Настройка ngrok"""
    try:
        auth_token = "2qMDpFmlpVUaiccMGxv4YggGYZH_5b9ZW2i2obxFy8bELJU2U"  # Замените на ваш токен
        set_auth_token(auth_token)
        public_url = connect(5000)
        logger.info(f"Public URL: {public_url}")
        return public_url
    except Exception as e:
        logger.error(f"Ngrok setup failed: {e}")
        return None

if __name__ == '__main__':
    try:
        cleanup_temp_files()
        public_url = setup_ngrok()
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        cleanup_temp_files()