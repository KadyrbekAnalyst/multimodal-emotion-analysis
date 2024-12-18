from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import io
import base64
from datetime import datetime
import logging
import sys
import os
from main import EmotionAnalysisSystem

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Инициализация системы анализа
analysis_system = EmotionAnalysisSystem()

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.recording = False
        self.frames = []  # Храним кадры в памяти
        self.audio_frames = []  # Храним аудио в памяти
        
        # Настройки камеры
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        success, frame = self.video.read()
        if success:
            if self.recording:
                # Сохраняем оригинальный кадр для анализа
                self.frames.append(frame.copy())
            
            # Оптимизируем для стриминга
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            ret, jpeg = cv2.imencode('.jpg', frame, encode_param)
            return jpeg.tobytes()
        return None

    def create_video_buffer(self):
        """Создает видео буфер из сохраненных кадров"""
        if not self.frames:
            return None
            
        # Создаем буфер в памяти
        buffer = io.BytesIO()
        
        # Получаем размеры из первого кадра
        height, width = self.frames[0].shape[:2]
        
        # Создаем видео writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(buffer, fourcc, 30.0, (width, height))
        
        # Записываем кадры
        for frame in self.frames:
            out.write(frame)
            
        out.release()
        buffer.seek(0)
        return buffer

camera = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = camera.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                       
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording')
def start_recording():
    try:
        camera.recording = True
        camera.frames = []  # Очищаем предыдущие кадры
        camera.audio_frames = []  # Очищаем предыдущие аудио данные
        return jsonify({"status": "started"})
    except Exception as e:
        logger.error(f"Error starting recording: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/stop_recording')
def stop_recording():
    try:
        camera.recording = False
        
        if not camera.frames:
            return jsonify({"status": "error", "message": "No frames recorded"})
            
        # Создаем буферы видео и аудио
        video_buffer = camera.create_video_buffer()
        
        if not video_buffer:
            return jsonify({"status": "error", "message": "Failed to create video buffer"})
        
        # Анализируем данные
        results = analysis_system.analyze_session({
            'video_data': video_buffer,
            'audio_data': camera.audio_frames,
        })
        
        if not results:
            return jsonify({"status": "error", "message": "Analysis failed"})
            
        # Получаем графики в формате base64
        visualizations = results.get('visualizations', {})
        
        # Очищаем буферы
        camera.frames = []
        camera.audio_frames = []
        
        return jsonify({
            "status": "success",
            "results": {
                "emotions": results.get('fusion_results', {}),
                "visualizations": visualizations
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing recording: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/analyze')
def analyze():
    """Отдельный endpoint для анализа"""
    try:
        latest_results = analysis_system.get_latest_results()
        if latest_results:
            return jsonify({"status": "success", "results": latest_results})
        return jsonify({"status": "error", "message": "No results available"})
    except Exception as e:
        logger.error(f"Error retrieving results: {e}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)