import os
import logging

# Create a logger instance
logger = logging.getLogger("flowstab")

class CustomPathnameFilter(logging.Filter):
    def filter(self, record):
        # Get the full pathname
        full_path = record.pathname
        
        # Split the path into parts
        path_parts = full_path.split(os.sep)
        
        # Limit to the last 2 parts
        if len(path_parts) > 2:
            record.pathname = os.sep.join(path_parts[-2:])
        return True

def setup_logger(log_level=logging.INFO):
    """
    Set up the logger for the package.

    Args:
        log_level (int): The logging level (e.g., logging.DEBUG, logging.INFO).
    """
    logger.setLevel(log_level)

    logger.addFilter(CustomPathnameFilter())

    # Create console handler and set level
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - PID:%(process)d - %(message)s')
    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger if it doesn't have handlers
    if not logger.hasHandlers():
        logger.addHandler(ch)

def get_logger():
    """Return the logger instance."""
    return logger
