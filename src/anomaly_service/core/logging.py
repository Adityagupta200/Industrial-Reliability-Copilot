import sys
from loguru import logger


def setup_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        serialize=True,  # JSON logs
        backtrace=False,
        diagnose=False,
        enqueue=True,
    )
