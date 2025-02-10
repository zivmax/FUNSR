import logging
import torch.distributed as dist

logger_initialized = {}


def setup_logger(name="outs", log_file=None, log_level=logging.INFO):
    """Set up a logger with stream and file handlers."""
    if name in logger_initialized:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Stream handler
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler (optional)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    logger_initialized[name] = True
    return logger


def get_logger(name="outs"):
    """Retrieve an existing logger."""
    if name in logger_initialized:
        return logging.getLogger(name)
    else:
        return setup_logger(name=name)


def print_log(message, logger=None, level=logging.INFO):
    """Print a message to the console and/or log file."""
    if logger is None:
        print(message)
    else:
        logger.log(level, message)
