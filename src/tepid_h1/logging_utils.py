"""Structured logging utilities for the framework."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    json_format: bool = False,
) -> logging.Logger:
    """Configure structured logging for the framework."""
    logger = logging.getLogger("tepid_h1")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = _JsonFormatter() if json_format else _TextFormatter()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        return f"[{timestamp}] [{record.levelname:7s}] {record.getMessage()}"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        return json.dumps(log_entry, ensure_ascii=False)


def log_training_step(
    logger: logging.Logger,
    step: int,
    loss: float,
    gradient_norm: float,
    learning_rate: float,
) -> None:
    """Log a training step with structured data."""
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        "",
        0,
        f"Step {step}: loss={loss:.4f}, grad_norm={gradient_norm:.4f}, lr={learning_rate:.6f}",
        (),
        None,
    )
    record.extra_data = {
        "step": step,
        "loss": loss,
        "gradient_norm": gradient_norm,
        "learning_rate": learning_rate,
    }
    logger.handle(record)
