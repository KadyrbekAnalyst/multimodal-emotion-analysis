from src.camera.camera_recorder import CameraRecorder

def main():
    recorder = CameraRecorder()
    video_path = recorder.record_video()
    
    # Далее работа с полученным видео