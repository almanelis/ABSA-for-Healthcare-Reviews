"""Вычисление простых метрик анализа аспектов."""

from __future__ import annotations

from typing import List, Dict, Any


def compute_metrics(all_aspects: List[List[Dict[str, Any]]]) -> Dict[str, float]:
    """Подсчитывает базовые статистики по результатам аспектного анализа."""
    total = len(all_aspects)
    no_aspects = sum(1 for x in all_aspects if not x)
    avg_conf = 0.0
    count_conf = 0

    for arr in all_aspects:
        for a in arr:
            if "confidence" in a:
                avg_conf += a["confidence"]
                count_conf += 1

    return {
        "total_reviews": total,
        "reviews_without_aspects": round(no_aspects / total * 100, 2) if total else 0.0,
        "average_confidence": round(avg_conf / count_conf, 3) if count_conf else 0.0,
    }
