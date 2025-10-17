"""Конфигурация настройки логов и предупреждений."""

from __future__ import annotations

import logging
import os
import sys
import warnings


def suppress_third_party_logs():
    """Подавляет шумные логи и предупреждения от библиотек."""

    # Поведение токенизаторов и трансформеров
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    # Базовая конфигурация logging
    logging.getLogger().setLevel(logging.WARNING)

    # Конкретные логгеры
    for name in (
        "transformers",
        "pyabsa",
        "torch",
        "spacy",
        "huggingface_hub",
        "urllib3",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)

    # Типичные предупреждения
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def configure_logging(verbose: bool = False, log_file: str | None = None):
    """Настройка логирования для пакета.

    Args:
        verbose: если True — устанавливаем INFO уровень и логируем в stdout.
        log_file: если задан — добавляем FileHandler с соответствующим именем.
    """
    logger = logging.getLogger("medreviewanalyzer")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    # Удаляем старые обработчики (при повторном импорте)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Подавляем сторонние шумы после установки нашей конфигурации
    if not verbose:
        suppress_third_party_logs()
