import logging

# Create a logger instance
logger = logging.getLogger("flowstab")

def setup_logger(log_level=logging.INFO):
    """
    Set up the logger for the package.

    Args:
        log_level (int): The logging level (e.g., logging.DEBUG, logging.INFO).
    """
    logger.setLevel(log_level)

    # Create console handler and set level
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger if it doesn't have handlers
    if not logger.hasHandlers():
        logger.addHandler(ch)

def get_logger():
    """Return the logger instance."""
    return logger
