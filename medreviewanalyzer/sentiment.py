"""Модуль для анализа общей тональности отзыва."""

from __future__ import annotations

import logging
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, BertTokenizerFast

logger = logging.getLogger("medreviewanalyzer.sentiment")


class SentimentAnalyzer:
    """Класс-оболочка для модели анализа тональности.

    Использует HuggingFace Transformers. Модель загружается в конструкторе.
    """

    def __init__(self, model_name: str = "blanchefort/rubert-base-cased-sentiment"):
        """
        Args:
            model_name: имя модели в HF hub или путь к локальной папке.
        """
        logger.info("Загружаю модель тональности: %s", model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, return_dict=True
        )
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> str:
        """Предсказание тональности для одного текста.

        Args:
            text: текст отзыва.

        Returns:
            Одна из строк: 'Neutral', 'Positive', 'Negative'.
        """
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        label = int(torch.argmax(probs, dim=1).item())

        labels_map: Dict[int, str] = {0: "Neutral", 1: "Positive", 2: "Negative"}
        result = labels_map.get(label, "Neutral")
        logger.debug("Sentiment predict: %s -> %s", text[:50], result)
        return result
