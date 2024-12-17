from flask import Flask, render_template, request, jsonify
import sys
import os

# Добавляем путь к родительской директории
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import EmotionAnalysisSystem

app = Flask(__name__)
analysis_system = EmotionAnalysisSystem()

UPLOAD_FOLDER = 'data/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    
    video_file = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, 'session_video.webm')
    video_file.save(video_path)

    # Анализ видео и аудио
    session_data = {'video_path': video_path, 'audio_path': video_path}
    analysis_results = analysis_system.analyze_session(session_data)

    if analysis_results:
        return jsonify(analysis_results['fusion_results'])
    return jsonify({'error': 'Analysis failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
