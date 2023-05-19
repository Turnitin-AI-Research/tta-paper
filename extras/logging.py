"""Top level utils file."""
from typing import Union
import os
import logging


def get_logger(logger_name: str, logger_level: Union[int, str, None] = None) -> logging.Logger:
    """Create logger, formatter, and console handler and return logger."""
    logger = logging.Logger(logger_name)  # using getLogger instead results in duplicate logs
    logger_level = os.environ.get('LOGLEVEL', 'WARNING').upper() if logger_level is None else logger_level
    logger.setLevel(logger_level)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logger_level)
        formatter = logging.Formatter('%(levelname)s: %(asctime)s: %(name)s: %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger
