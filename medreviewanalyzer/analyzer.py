"""Главный API пакета — класс MedicalReviewAnalyzer."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List
import json
from collections import Counter

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .aspects import AspectAnalyzer
from .categorizer import get_category
from .sentiment import SentimentAnalyzer
from .utils.logging_config import configure_logging
from .utils.metrics import compute_metrics

logger = logging.getLogger("medreviewanalyzer")


class MedicalReviewAnalyzer:
    """Основной класс для анализа отзывов: тональность + аспекты + категории.

    Примеры:
        analyzer = MedicalReviewAnalyzer(verbose=False)
        r = analyzer.analyze_review("Врач отличный, но регистратура медлит.")
    """

    def __init__(self, verbose: bool = False, log_file: str | None = None):
        """
        Args:
            verbose: если True — включает подробное логирование.
            log_file: путь к файлу логов (опционально).
        """
        configure_logging(verbose=verbose, log_file=log_file)
        self.logger = logging.getLogger("medreviewanalyzer")
        self.sentiment_model = SentimentAnalyzer()
        self.aspect_model = AspectAnalyzer()
        self.verbose = verbose

        self.logger.info("MedicalReviewAnalyzer инициализирован (verbose=%s)", verbose)

    def analyze_review(self, text: str) -> Dict[str, Any]:
        """Анализ одного отзыва.

        Args:
            text: текст отзыва.

        Returns:
            Словарь:
              {
                "text": original_text,
                "sentiment": "Positive"|"Neutral"|"Negative",
                "aspects": [
                  {"aspect": str, "sentiment": str, "confidence": float, "category": str},
                  ...
                ]
              }
        """
        self.logger.debug("analyze_review: %s", text[:120])
        sentiment = self.sentiment_model.predict(text)
        aspects = self.aspect_model.analyze(text)

        # Добавляем категорию к каждому аспекту (если он корректный)
        enriched: List[Dict[str, Any]] = []
        for a in aspects:
            if isinstance(a, dict) and "aspect" in a:
                a["category"] = get_category(a["aspect"])
            enriched.append(a)

        result = {"text": text, "sentiment": sentiment, "aspects": enriched}
        self.logger.debug("Результат анализа: %s", result)
        return result

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "review_text",
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Анализ DataFrame с отзывами пакетно и с прогресс-баром."""
        df = df.copy()
        texts = df[text_col].tolist()
        sentiments, aspects_all = [], []

        it = tqdm(
            range(0, len(texts), batch_size),
            disable=not show_progress,
            desc="Анализ отзывов",
        )

        for i in it:
            batch = texts[i : i + batch_size]
            # Анализируем тональности батчем
            sents = [self.sentiment_model.predict(t) for t in batch]
            sentiments.extend(sents)

            # Анализируем аспекты
            batch_aspects = []
            for t in batch:
                a = self.aspect_model.analyze(t)
                for x in a:
                    if isinstance(x, dict) and "aspect" in x:
                        x["category"] = get_category(x["aspect"])
                batch_aspects.append(a)
            aspects_all.extend(batch_aspects)

        df["общая_тональность"] = sentiments
        df["аспекты"] = aspects_all

        # Метрики качества
        stats = compute_metrics(aspects_all)
        self.logger.info("Метрики анализа: %s", stats)

        return df

    def save_analysis_results(
        self, df: pd.DataFrame, stats: dict, results_dir: str = "results"
    ) -> None:
        """Сохраняет результаты анализа датафрейма с графиками.

        Args:
            df: проанализированный датафрейм с колонками 'общая_тональность' и 'аспекты'.
            stats: словарь с метриками анализа (compute_metrics).
            results_dir: путь к каталогу для сохранения.
        """
        import os, json
        import matplotlib.pyplot as plt
        import seaborn as sns
        from collections import Counter

        os.makedirs(results_dir, exist_ok=True)

        # Сохраняем датафрейм
        df_to_save = df.copy()
        # Преобразуем аспекты в строки для Excel
        df_to_save["аспекты"] = df_to_save["аспекты"].apply(
            lambda x: (
                json.dumps(x, ensure_ascii=False) if isinstance(x, list) else str(x)
            )
        )
        excel_path = os.path.join(results_dir, "df_analysis.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            df_to_save.to_excel(writer, sheet_name="reviews", index=False)
            pd.DataFrame([stats]).to_excel(writer, sheet_name="metrics", index=False)

        # График распределения тональностей
        plt.figure(figsize=(6, 4))
        sns.countplot(x="общая_тональность", data=df)
        plt.title("Распределение тональностей")
        plt.savefig(os.path.join(results_dir, "sentiment_distribution.png"))
        plt.close()

        # График распределения категорий аспектов
        all_categories = []
        for aspects in df["аспекты"]:
            if isinstance(aspects, str):
                aspects = json.loads(aspects)
            if isinstance(aspects, list):
                all_categories.extend(
                    [a["category"] for a in aspects if "category" in a]
                )

        counter = Counter(all_categories)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(counter.keys()), y=list(counter.values()))
        plt.xticks(rotation=45, ha="right")
        plt.title("Количество аспектов по категориям")
        plt.savefig(os.path.join(results_dir, "category_distribution.png"))
        plt.close()

        self.logger.info("Результаты анализа сохранены в '%s'", results_dir)
