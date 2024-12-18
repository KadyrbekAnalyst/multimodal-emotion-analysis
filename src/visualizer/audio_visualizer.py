import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

class AudioVisualizer:
    def __init__(self):
        plt.style.use('dark_background')
    
    def create_visualization(self, audio_data, sr, emotions, save_path=None):
        """
        Создает единую визуализацию для аудио файла
        :param audio_data: аудио данные
        :param sr: частота дискретизации
        :param emotions: словарь с эмоциями
        :param save_path: путь для сохранения
        """
        plt.figure(figsize=(15, 10))
        
        # Спектрограмма
        plt.subplot(2, 1, 1)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Audio Spectrogram')
        
        # Распределение эмоций
        plt.subplot(2, 1, 2)
        emotions_list = sorted(emotions.items())
        labels, values = zip(*emotions_list)
        bars = plt.bar(labels, values)
        plt.title('Emotions Distribution from Speech')
        plt.xticks(rotation=45)
        plt.ylabel('Confidence (%)')
        
        # Добавляем значения над столбцами
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()