"""Модуль для извлечения аспектов (Aspect Term Extraction) через pyabsa."""

from __future__ import annotations

import contextlib
import io
import logging
from typing import Any, Dict, List

from pyabsa import AspectTermExtraction as ATEPC

logger = logging.getLogger("medreviewanalyzer.aspects")


class AspectAnalyzer:
    """Обёртка над pyabsa AspectExtractor.

    Возвращает список словарей: {"aspect": str, "sentiment": str, "confidence": float}
    """

    def __init__(self, model_name: str = "multilingual"):
        """
        Args:
            model_name: имя контрольной точки pyabsa или 'multilingual' и т.п.
        """
        logger.info("Инициализирую AspectExtractor: %s", model_name)
        # Против шумных логов
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            self.extractor = ATEPC.AspectExtractor(model_name)

        self.extractor = ATEPC.AspectExtractor(model_name)

    def analyze(self, text: str) -> List[Dict[str, Any]]:
        """Анализ одного текста на предмет аспектов.

        Args:
            text: строка отзыва.

        Returns:
            Список найденных аспектов или пустой список.
            В случае ошибки возвращается список с одним элементом {"error": ...}.
        """
        try:
            result = self.extractor.predict(text, print_result=False, save_result=False)
            aspects = result.get("aspect", [])
            sentiments = result.get("sentiment", [])
            confidences = result.get("confidence", [])

            if not aspects:
                return []

            output = [
                {"aspect": a, "sentiment": s, "confidence": round(float(c), 3)}
                for a, s, c in zip(aspects, sentiments, confidences)
            ]
            logger.debug("Аспекты: %s", output)
            return output
        except Exception as exc:  # noqa: BLE001 - ловим все ошибки для стабильности
            logger.exception("Ошибка AspectAnalyzer.analyze")
            return [{"error": str(exc)}]
