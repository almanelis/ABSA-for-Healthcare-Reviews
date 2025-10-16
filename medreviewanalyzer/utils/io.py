"""Утилиты для загрузки YAML конфигураций."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Загружает YAML файл.

    Args:
        path: путь к YAML.

    Returns:
        dict: структура файла.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
