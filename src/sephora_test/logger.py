"""setup logger for sephora_test package."""

from logging import INFO, Formatter, StreamHandler, getLogger
from logging.handlers import RotatingFileHandler
from os import getcwd, makedirs, path

ROOT_DIR_LOGS = path.join(getcwd(), "logs")

if not path.exists(ROOT_DIR_LOGS):
    try:
        makedirs(ROOT_DIR_LOGS)
    except OSError as e:
        print(f"Error creating logs directory {ROOT_DIR_LOGS}: {e}")


def setup_logger(name: str, log_path: str = ROOT_DIR_LOGS, level: int = INFO):
    """
    setup logger for sephora_test package.
    args:
        name (str): name of the logger.
        log_path (str): path to the log file.
        level (int): logging level.
    returns:
        logger (logging.Logger): configured logger instance.

    """
    if name.startswith("sephora_test"):
        name = name.replace("sephora_test", "")
    if len(path.split(name)) > 1:
        name = path.split(name)[-1]  # keep the file part of the __file__

    logger = getLogger(name)

    if not logger.handlers:
        formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        try:
            makedirs(log_path, exist_ok=True)
            log_file = path.join(log_path, f"{name}.log")

            handler = RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        except Exception as e:
            print(f"Error setting up file handler for logger {name}: {e}")
        stream_handler = StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    logger.propagate = False
    logger.setLevel(level)
    return logger
