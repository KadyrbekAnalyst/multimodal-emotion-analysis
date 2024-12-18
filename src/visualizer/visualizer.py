import matplotlib.pyplot as plt
import numpy as np
import os

class EmotionVisualizer:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        plt.style.use('dark_background')

    def create_visualization(self, video_emotions, speech_emotions, text_results, fusion_results, save_path):
        """
        Создает визуализацию результатов анализа эмоций
        """
        plt.figure(figsize=(15, 10))

        # График распределения эмоций для каждой модальности
        plt.subplot(2, 1, 1)
        x = np.arange(len(self.emotions))
        width = 0.25

        # Получаем значения для каждой модальности
        video_values = [video_emotions['average'].get(e, 0) for e in self.emotions]
        speech_values = [speech_emotions['average'].get(e, 0) for e in self.emotions]
        text_values = [text_results['emotions'].get(e, 0) for e in self.emotions]

        # Создаем столбцы для каждой модальности
        plt.bar(x - width, video_values, width, label='Video', color='#FF9999')
        plt.bar(x, speech_values, width, label='Audio', color='#99FF99')
        plt.bar(x + width, text_values, width, label='Text', color='#9999FF')

        plt.xlabel('Emotions')
        plt.ylabel('Confidence (%)')
        plt.title('Emotion Distribution by Modality')
        plt.xticks(x, self.emotions, rotation=45)
        plt.legend()

        # График итоговых результатов
        plt.subplot(2, 1, 2)
        
        # Проверяем формат fusion_results и извлекаем значения
        if isinstance(fusion_results, dict):
            if 'emotions' in fusion_results:
                fusion_values = [fusion_results['emotions'].get(e, 0) for e in self.emotions]
                dominant_emotion = fusion_results.get('dominant_emotion', 'unknown')
            else:
                fusion_values = [fusion_results.get(e, 0) for e in self.emotions]
                # Находим доминирующую эмоцию безопасным способом
                dominant_emotion = max(fusion_results.items(), key=lambda x: float(x[1]) if isinstance(x[1], (int, float, str)) else 0)[0]
        else:
            # Если fusion_results не словарь, используем безопасные значения
            fusion_values = [0] * len(self.emotions)
            dominant_emotion = 'unknown'

        bars = plt.bar(self.emotions, fusion_values, color='#FFB266')
        
        # Выделяем доминирующую эмоцию если она есть в списке эмоций
        if dominant_emotion in self.emotions:
            dominant_idx = self.emotions.index(dominant_emotion)
            bars[dominant_idx].set_color('#FF3333')

        plt.title('Fused Emotions Analysis\nDominant: ' + dominant_emotion.capitalize())
        plt.xlabel('Emotions')
        plt.ylabel('Confidence (%)')
        plt.xticks(rotation=45)

        # Добавляем значения над столбцами
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')

        plt.tight_layout()

        # Создаем директорию если её нет
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Сохраняем визуализацию
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()