import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class EmotionFusion:
    def __init__(self, weights: Dict[str, float] = None):
        """
        Инициализация с настройкой весов для каждой модальности
        :param weights: {'video': float, 'audio': float, 'text': float}
        """
        try:
            # Веса по умолчанию
            self.weights = weights or {
                'video': 0.7,  # Больший вес видео, так как лица наиболее информативны
                'audio': 0.2,
                'text': 0.1
            }
            
            # Проверка и нормализация весов
            total_weight = sum(self.weights.values())
            if abs(total_weight - 1.0) > 0.01:
                self.weights = {k: v/total_weight for k, v in self.weights.items()}
            
            self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            plt.style.use('dark_background')  # Используем темную тему для графиков
            
            logger.info("EmotionFusion initialized with weights: %s", self.weights)
            
        except Exception as e:
            logger.error(f"Error initializing EmotionFusion: {e}")
            raise

    def fuse_emotions(self, 
                     video_emotions: Dict[str, float],
                     audio_emotions: Dict[str, float],
                     text_emotions: Dict[str, float]) -> Dict:
        """
        Объединение эмоций из разных модальностей с учетом весов
        """
        try:
            logger.info("Starting emotion fusion")
            
            # Инициализация результирующих эмоций
            fused_emotions = {emotion: 0.0 for emotion in self.emotions}
            
            # Объединяем эмоции с учетом весов
            for emotion in self.emotions:
                if video_emotions:
                    fused_emotions[emotion] += video_emotions.get(emotion, 0.0) * self.weights['video']
                if audio_emotions:
                    fused_emotions[emotion] += audio_emotions.get(emotion, 0.0) * self.weights['audio']
                if text_emotions:
                    fused_emotions[emotion] += text_emotions.get(emotion, 0.0) * self.weights['text']
            
            # Определяем доминирующую эмоцию
            dominant_emotion = max(fused_emotions.items(), key=lambda x: x[1])[0]
            
            # Рассчитываем уверенность для каждой модальности
            confidence_scores = {
                'video': self._calculate_confidence(video_emotions),
                'audio': self._calculate_confidence(audio_emotions),
                'text': self._calculate_confidence(text_emotions)
            }
            
            logger.info(f"Fusion completed. Dominant emotion: {dominant_emotion}")
            
            return {
                'emotions': fused_emotions,
                'dominant_emotion': dominant_emotion,
                'confidence_scores': confidence_scores
            }
            
        except Exception as e:
            logger.error(f"Error during emotion fusion: {e}")
            return None

    def _calculate_confidence(self, emotions: Optional[Dict[str, float]]) -> float:
        """Расчет уверенности для одной модальности"""
        if not emotions:
            return 0.0
            
        try:
            # Находим максимальное значение эмоции
            max_emotion = max(emotions.values())
            # Нормализуем значения
            total = sum(emotions.values())
            
            if total == 0:
                return 0.0
                
            # Рассчитываем уверенность на основе распределения вероятностей
            probabilities = [v/total for v in emotions.values()]
            entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)
            max_entropy = -np.log2(1/len(emotions))
            
            # Преобразуем в проценты
            confidence = (1 - entropy/max_entropy) * 100
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0

    def create_visualization(self, results: Dict, save_path: str) -> None:
        """Создание итоговой визуализации"""
        try:
            if not results:
                logger.error("No results to visualize")
                return
            
            plt.figure(figsize=(15, 10))
            
            # График уверенности по модальностям
            plt.subplot(2, 1, 1)
            confidence_scores = results['confidence_scores']
            bars = plt.bar(confidence_scores.keys(), confidence_scores.values())
            plt.title('Confidence by Modality')
            plt.ylabel('Confidence Score (%)')
            
            # Добавляем значения над столбцами
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            # График итоговых эмоций
            plt.subplot(2, 1, 2)
            emotions = results['emotions']
            bars = plt.bar(emotions.keys(), emotions.values())
            plt.title(f'Fused Emotions (Dominant: {results["dominant_emotion"].capitalize()})')
            plt.xticks(rotation=45)
            plt.ylabel('Confidence (%)')
            
            # Выделяем доминирующую эмоцию
            dominant_idx = list(emotions.keys()).index(results['dominant_emotion'])
            bars[dominant_idx].set_color('#FF3333')
            
            # Добавляем значения над столбцами
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Создаем директорию если нужно
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Сохраняем визуализацию
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            logger.info(f"Visualization saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")

def test_fusion():
    """Тестирование системы объединения эмоций"""
    try:
        fusion = EmotionFusion()
        
        # Тестовые данные
        test_emotions = {
            'video': {'angry': 10, 'happy': 60, 'neutral': 30},
            'audio': {'angry': 20, 'happy': 50, 'neutral': 30},
            'text': {'angry': 15, 'happy': 55, 'neutral': 30}
        }
        
        results = fusion.fuse_emotions(
            test_emotions['video'],
            test_emotions['audio'],
            test_emotions['text']
        )
        
        if results:
            logger.info("Test Results:")
            logger.info(f"Dominant emotion: {results['dominant_emotion']}")
            logger.info(f"Confidence scores: {results['confidence_scores']}")
            
            # Создаем тестовую визуализацию
            save_path = os.path.join('app', 'static', 'temp', 'test_fusion.png')
            fusion.create_visualization(results, save_path)
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_fusion()