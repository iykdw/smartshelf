import contextlib
import logging
import re
from typing import Any


class LogRedacter(logging.Filter):
    def __init__(self, patterns):
        super().__init__()
        self.patterns = patterns

    def filter(self, record: logging.LogRecord) -> bool:
        for pattern in self.patterns:
            with contextlib.suppress(TypeError):
                record.msg = re.sub(pattern, "<REDACTED>", record.msg)
        return True


def get_logger(files: list[str], streams: list[Any], level, name):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    for file in files:
        handler = logging.FileHandler(file)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s", datefmt="%Y-%m-%d:%H:%M:%S"
            )
        )
        logger.addHandler(handler)
    for stream in streams:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s", datefmt="%Y-%m-%d:%H:%M:%S"
            )
        )
        logger.addHandler(handler)

    SECRET_REGEXES = [r"(key=[ ]?)([^&]*)"]
    logger.addFilter(LogRedacter(SECRET_REGEXES))

    return logger
