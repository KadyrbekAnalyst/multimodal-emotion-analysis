import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os
from datetime import datetime

class EmotionFusion:
    def __init__(self, weights: Dict[str, float] = None):
        """
        :param weights: Словарь с весами для каждой модальности
            {
                'video': float,  # вес для видео анализа
                'audio': float,  # вес для аудио анализа
                'text': float    # вес для текстового анализа
            }
        """
        # Веса по умолчанию (равные)
        self.weights = weights or {
            'video': 0.33,
            'audio': 0.33,
            'text': 0.34
        }
        
        # Проверяем, что сумма весов равна 1
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:  # Допускаем небольшую погрешность
            # Нормализуем веса
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        # Список поддерживаемых эмоций
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def fuse_emotions(self, 
                     video_emotions: Dict[str, float],
                     audio_emotions: Dict[str, float],
                     text_emotions: Dict[str, float]) -> Dict:
        """
        Объединение эмоций из разных модальностей
        """
        try:
            # Проверяем наличие всех необходимых эмоций
            all_emotions = {emotion: 0.0 for emotion in self.emotions}
            
            # Объединяем эмоции с учетом весов
            if video_emotions:
                for emotion in all_emotions:
                    all_emotions[emotion] += video_emotions.get(emotion, 0.0) * self.weights['video']
            
            if audio_emotions:
                for emotion in all_emotions:
                    all_emotions[emotion] += audio_emotions.get(emotion, 0.0) * self.weights['audio']
            
            if text_emotions:
                for emotion in all_emotions:
                    all_emotions[emotion] += text_emotions.get(emotion, 0.0) * self.weights['text']
            
            # Определяем доминирующую эмоцию
            dominant_emotion = max(all_emotions.items(), key=lambda x: x[1])[0]
            
            # Формируем результат
            result = {
                'fused_emotions': all_emotions,
                'dominant_emotion': dominant_emotion,
                'confidence_scores': {
                    'video': self._calculate_confidence(video_emotions),
                    'audio': self._calculate_confidence(audio_emotions),
                    'text': self._calculate_confidence(text_emotions)
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error in emotion fusion: {e}")
            return None

    def _calculate_confidence(self, emotions: Dict[str, float]) -> float:
        """
        Расчет уверенности предсказания для одной модальности
        """
        if not emotions:
            return 0.0
            
        # Находим максимальное значение эмоции
        max_emotion = max(emotions.values())
        # Считаем энтропию распределения
        total = sum(emotions.values())
        if total == 0:
            return 0.0
            
        probabilities = [v/total for v in emotions.values()]
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)
        max_entropy = -np.log2(1/len(emotions))  # максимальная возможная энтропия
        
        # Нормализованная уверенность (0-100%)
        confidence = (1 - entropy/max_entropy) * 100
        return confidence

    def visualize_results(self, results: Dict, save_path: str = None) -> None:
        """
        Визуализация результатов анализа
        """
        if not results:
            return
            
        plt.figure(figsize=(15, 10))
        
        # График объединенных эмоций
        plt.subplot(2, 1, 1)
        emotions = results['fused_emotions']
        plt.bar(emotions.keys(), emotions.values())
        plt.title('Fused Emotions Analysis')
        plt.ylabel('Confidence (%)')
        plt.xticks(rotation=45)
        
        # Добавляем метку для доминирующей эмоции
        dominant_idx = list(emotions.keys()).index(results['dominant_emotion'])
        plt.text(dominant_idx, emotions[results['dominant_emotion']], 
                'Dominant', ha='center', va='bottom')
        
        # График уверенности по модальностям
        plt.subplot(2, 1, 2)
        confidence = results['confidence_scores']
        plt.bar(confidence.keys(), confidence.values())
        plt.title('Confidence Scores by Modality')
        plt.ylabel('Confidence (%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def format_results(results: Dict) -> str:
    """
    Форматирование результатов для вывода
    """
    if not results:
        return "Fusion failed"
        
    output = [
        "Fused Emotions Analysis:",
        "{\n    " + ",\n    ".join([
            f"'{emotion}': {value:.2f}%{' # Dominant' if emotion == results['dominant_emotion'] else ''}"
            for emotion, value in results['fused_emotions'].items()
        ]) + "\n}",
        "\nConfidence Scores:",
        "{\n    " + ",\n    ".join([
            f"'{modality}': {score:.2f}%"
            for modality, score in results['confidence_scores'].items()
        ]) + "\n}"
    ]
    
    return "\n".join(output)

def main():
    # Пример использования
    fusion = EmotionFusion(weights={
        'video': 0.4,
        'audio': 0.3,
        'text': 0.3
    })
    
    # Пример данных
    video_emotions = {
        'angry': 10.0,
        'disgust': 5.0,
        'fear': 15.0,
        'happy': 40.0,
        'sad': 10.0,
        'surprise': 10.0,
        'neutral': 10.0
    }
    
    audio_emotions = {
        'angry': 5.0,
        'disgust': 5.0,
        'fear': 10.0,
        'happy': 50.0,
        'sad': 10.0,
        'surprise': 10.0,
        'neutral': 10.0
    }
    
    text_emotions = {
        'angry': 5.0,
        'disgust': 5.0,
        'fear': 10.0,
        'happy': 45.0,
        'sad': 15.0,
        'surprise': 10.0,
        'neutral': 10.0
    }
    
    results = fusion.fuse_emotions(video_emotions, audio_emotions, text_emotions)
    
    print("\nFusion Results:")
    print(format_results(results))
    
    # Сохраняем визуализацию
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    visualization_path = os.path.join("data", "visualizations", f"fusion_{timestamp}.png")
    os.makedirs(os.path.dirname(visualization_path), exist_ok=True)
    
    fusion.visualize_results(results, visualization_path)
    print(f"\nVisualization saved to: {visualization_path}")

if __name__ == "__main__":
    main()