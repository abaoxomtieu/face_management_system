import logging
import os
from datetime import datetime
from pathlib import Path

import pytz


class CoreCFG:
    PROJECT_NAME = "FaceRecognition"
    BOT_NAME = str("FaceRecognition")


def get_date_time():
    return datetime.now(pytz.timezone("Asia/Ho_Chi_Minh"))


DATE_TIME = get_date_time().date()
BASE_DIR = os.path.dirname(Path(__file__).parent.parent)
LOG_DIR = os.path.join(BASE_DIR, "logs")


class CustomFormatter(logging.Formatter):
    green = "\x1b[0;32m"
    grey = "\x1b[38;5;248m"
    yellow = "\x1b[38;5;229m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[38;5;31m"
    white = "\x1b[38;5;255m"
    reset = "\x1b[38;5;15m"

    base_format = f"{grey}%(asctime)s | %(name)s | %(threadName)s | {{level_color}}%(levelname)-8s{grey} | {blue}%(module)s:%(lineno)d{grey} - {white}%(message)s"

    FORMATS = {
        logging.INFO: base_format.format(level_color=green),
        logging.WARNING: base_format.format(level_color=yellow),
        logging.ERROR: base_format.format(level_color=red),
        logging.CRITICAL: base_format.format(level_color=bold_red),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def custom_logger(app_name="APP"):
    logger_r = logging.getLogger(name=app_name)
    # Set the timezone to Ho_Chi_Minh
    tz = pytz.timezone("Asia/Ho_Chi_Minh")

    logging.Formatter.converter = lambda *args: datetime.now(tz).timetuple()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())

    logger_r.setLevel(logging.INFO)
    logger_r.addHandler(ch)

    return logger_r


logger = custom_logger(app_name=CoreCFG.PROJECT_NAME)

