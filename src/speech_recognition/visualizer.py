import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

class AudioVisualizer:
    def __init__(self, show_waveform=True, show_spectrogram=True, show_emotions=True):
        """
        :param show_waveform: Показывать ли форму волны
        :param show_spectrogram: Показывать ли спектрограмму
        :param show_emotions: Показывать ли распределение эмоций
        """
        self.show_waveform = show_waveform
        self.show_spectrogram = show_spectrogram
        self.show_emotions = show_emotions
        
    def visualize_chunk(self, chunk, sr, chunk_id, emotions=None, save_path=None):
        """
        Визуализация аудио чанка
        :param chunk: аудио данные
        :param sr: частота дискретизации
        :param chunk_id: идентификатор чанка
        :param emotions: словарь с эмоциями и их вероятностями
        :param save_path: путь для сохранения визуализации
        """
        # Определяем количество подграфиков
        n_plots = sum([self.show_waveform, self.show_spectrogram, self.show_emotions])
        if n_plots == 0:
            return None
            
        current_plot = 1
        plt.figure(figsize=(15, 5*n_plots))
        
        # Визуализация формы волны
        if self.show_waveform:
            plt.subplot(n_plots, 1, current_plot)
            time = np.arange(len(chunk)) / sr
            plt.plot(time, chunk)
            plt.title(f'Waveform - Chunk {chunk_id}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            current_plot += 1
        
        # Визуализация спектрограммы
        if self.show_spectrogram:
            plt.subplot(n_plots, 1, current_plot)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(chunk)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram - Chunk {chunk_id}')
            current_plot += 1
        
        # Визуализация эмоций
        if self.show_emotions and emotions:
            plt.subplot(n_plots, 1, current_plot)
            emotions_list = sorted(emotions.items())
            labels, values = zip(*emotions_list)
            plt.bar(labels, values)
            plt.title('Emotions Distribution')
            plt.xticks(rotation=45)
            plt.ylabel('Confidence (%)')
        
        plt.tight_layout()
        
        # Сохраняем или показываем
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def generate_summary_visualization(audio_path, results, save_path=None):
    """
    Генерация итоговой визуализации для всего аудио файла
    :param audio_path: путь к аудио файлу
    :param results: результаты анализа эмоций
    :param save_path: путь для сохранения визуализации
    """
    y, sr = librosa.load(audio_path)
    
    plt.figure(figsize=(15, 12))
    
    # Спектрограмма всего аудио
    plt.subplot(3, 1, 1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Full Audio Spectrogram')
    
    # График изменения эмоций во времени
    plt.subplot(3, 1, 2)
    for emotion in results['timeline'][0]['emotions'].keys():
        times = [entry['timestamp'] for entry in results['timeline']]
        values = [entry['emotions'][emotion] for entry in results['timeline']]
        plt.plot(times, values, label=emotion, alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Confidence (%)')
    plt.title('Emotions Over Time')
    plt.legend()
    
    # Средние эмоции
    plt.subplot(3, 1, 3)
    emotions = results['average']
    plt.bar(emotions.keys(), emotions.values())
    plt.title('Average Emotions')
    plt.xticks(rotation=45)
    plt.ylabel('Confidence (%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()