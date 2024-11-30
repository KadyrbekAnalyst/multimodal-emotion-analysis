import cv2
import os
import pyaudio
import wave
import threading
import numpy as np
from datetime import datetime
from deepface import DeepFace
import warnings
import shutil
import time

class AudioRecorder:
    def __init__(self, filename, duration):
        self.filename = filename
        self.duration = duration
        self.frames = []
        self.is_recording = False
        
        # Аудио параметры
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
    
    def start_recording(self):
        self.is_recording = True
        self.frames = []
        
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    frames_per_buffer=self.chunk)
        
        print("Starting audio recording...")
        
        try:
            # Увеличим длительность записи на 1 секунду для синхронизации
            total_chunks = int(self.rate / self.chunk * (self.duration + 1))
            for i in range(total_chunks):
                if not self.is_recording:
                    break
                data = stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
            
            print("Audio recording completed")
        
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Сохраняем аудио только если есть записанные фреймы
            if self.frames:
                wf = wave.open(self.filename, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.frames))
                wf.close()
    
    def stop_recording(self):
        self.is_recording = False

class CameraRecorder:
    def __init__(self, record_duration=15, base_dir='data/recordings', detector='opencv'):
        self.record_duration = record_duration
        self.base_dir = base_dir
        self.detector = detector
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, self.timestamp)
        # Создаем только базовую директорию
        os.makedirs(base_dir, exist_ok=True)

    def detect_face(self, frame):
        """
        Detect face and return coordinates
        Returns: (face_detected, face_coords)
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                faces = DeepFace.extract_faces(
                    frame, 
                    detector_backend=self.detector,
                    enforce_detection=True,
                    align=False
                )
                
                if len(faces) > 0:
                    face = faces[0]
                    facial_area = face['facial_area']
                    x = facial_area['x']
                    y = facial_area['y']
                    w = facial_area['w']
                    h = facial_area['h']
                    return True, [x, y, w, h]
                
                return False, None
                
        except Exception as e:
            return False, None
        
    def get_file_paths(self):
        """Генерация путей для файлов"""
        # Создаем директорию сессии только когда она действительно нужна
        os.makedirs(self.session_dir, exist_ok=True)
        video_path = os.path.join(self.session_dir, 'video.mp4')
        audio_path = os.path.join(self.session_dir, 'audio.wav')
        return video_path, audio_path
    def cleanup_session(self):
        """Удаление директории сессии при ошибке"""
        try:
            if os.path.exists(self.session_dir):
                # Даем системе время на освобождение файлов
                time.sleep(1)
                shutil.rmtree(self.session_dir)
                print(f"Cleaned up session directory: {self.session_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up directory: {e}")
    
    def record_video(self):
        """Запись видео и аудио"""
        video_path, audio_path = self.get_file_paths()
        cap = None
        out = None
        
        try:
            # Инициализация видео записи
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Could not open camera")
            
            # Параметры видео
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 30
            
            # Настройка видео записи
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            # Инициализация аудио записи
            audio_recorder = AudioRecorder(audio_path, self.record_duration)
            audio_thread = threading.Thread(target=audio_recorder.start_recording)
            audio_thread.start()
            
            face_detected = False
            frame_count = 0
            max_frames = self.record_duration * fps
            
            print(f"\nStarting recording session: {self.timestamp}")
            print("Please show your face to the camera...")
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Failed to capture frame")
                
                face_found, face_coords = self.detect_face(frame)
                
                if face_found and face_coords:
                    face_detected = True
                    x, y, w, h = face_coords
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Face Detected", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No Face Detected", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                current_time = frame_count / fps
                cv2.putText(frame, f"Time: {current_time:.1f} sec", 
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow('Emotion Recording', frame)
                out.write(frame)
                
                frame_count += 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        except Exception as e:
            print(f"Error during recording: {e}")
            self.cleanup_session()
            return None
            
        finally:
            # Останавливаем запись аудио
            if 'audio_recorder' in locals():
                audio_recorder.stop_recording()
                audio_thread.join()
            
            # Освобождаем ресурсы
            if out is not None:
                out.release()
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
        
        if not face_detected:
            print("Error: No face detected during recording")
            self.cleanup_session()
            return None
        
        print(f"\nRecording completed successfully!")
        print(f"Session directory: {self.session_dir}")
        print(f"Video saved as: video.mp4")
        print(f"Audio saved as: audio.wav")
        
        return {
            'session_id': self.timestamp,
            'session_dir': self.session_dir,
            'video_path': video_path,
            'audio_path': audio_path
        }

def main():
    # Создаем базовую директорию, если её нет
    base_dir = 'data/recordings'
    os.makedirs(base_dir, exist_ok=True)
    
    recorder = CameraRecorder()
    result = recorder.record_video()
    
    if result:
        print("\nRecording Summary:")
        print(f"Session ID: {result['session_id']}")
        print(f"Files location: {result['session_dir']}")
    else:
        print("Recording failed")

if __name__ == "__main__":
    main()