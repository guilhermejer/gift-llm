import json
import logging
from datetime import datetime
from typing import Any


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


def build_http_log_line(payload: dict[str, Any]) -> str:
    """
    Emula o formato de linha do backend Go:
    2026/06/29 17:20:43 {"level":"info",...}
    """
    now = datetime.now()
    prefix = now.strftime("%Y/%m/%d %H:%M:%S")
    return f"{prefix} {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"


def log_http_event(event: dict[str, Any]) -> None:
    logger = logging.getLogger("http")
    logger.info(build_http_log_line(event))
