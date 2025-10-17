from functools import lru_cache
from statistics import mean
from pathlib import Path

from rapidfuzz import fuzz
import pymorphy2

from .utils.io import load_yaml_config


morph = pymorphy2.MorphAnalyzer()


@lru_cache(maxsize=1)
def load_categories(path: str | Path | None = None) -> dict[str, list[str]]:
    """Загружает категории из YAML (с кешированием)."""
    default_path = Path(__file__).resolve().parent / "config" / "categories.yaml"
    return load_yaml_config(path or default_path)


_threshold_history: list[int] = []  # для автоподбора порога


def auto_threshold(default: int = 75) -> int:
    """Эвристически подбирает порог на основе истории."""
    if not _threshold_history:
        return default
    avg = int(mean(_threshold_history))
    return min(max(avg, 60), 90)  # держим в разумных пределах


def _normalize_word(word: str) -> str:
    """
    Приводит слово к нормальной (лемматизированной) форме.
    """
    parsed = morph.parse(word)[0]
    return parsed.normal_form


def get_category(aspect: str, threshold: int | None = None) -> str:
    """Как раньше, но теперь:
    - категории берутся из YAML,
    - порог может адаптироваться."""
    categories = load_categories()
    threshold = threshold or auto_threshold()

    aspect = aspect.strip().lower()
    if not aspect:
        return "другое"

    tokens = [_normalize_word(t) for t in aspect.split()]
    best_cat, best_score = "другое", 0

    for cat, kws in categories.items():
        for kw in kws:
            kw_n = _normalize_word(kw)
            for t in tokens:
                score = fuzz.partial_ratio(t, kw_n)
                if score > best_score:
                    best_cat, best_score = cat, score

    _threshold_history.append(best_score)
    return best_cat if best_score >= threshold else "другое"
