import logging
import os


def setup_logger(
    name="pipeline", log_file="pipeline.log", level=logging.DEBUG, include_location=True
):
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, log_file)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    if include_location:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s",
            datefmt="%m-%d-%Y %I:%M%p",
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%m-%d-%Y %I:%M%p"
        )

    class FlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    file_handler = FlushingFileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # Log initialization only for the first call
    if not hasattr(setup_logger, "initialized"):
        logger.debug(f"Logger initialized, writing to: {os.path.abspath(log_path)}")
        setup_logger.initialized = True

    return logger
