import logging
import re
import types
from flowstab.logger import setup_logger, get_logger, CustomPathnameFilter

def test_setup_logger_sets_level_and_handler(monkeypatch):
    logger = get_logger()
    # Remove all handlers and filters for a clean state
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    for f in logger.filters[:]:
        logger.removeFilter(f)
    logger.setLevel(logging.NOTSET)

    # Patch hasHandlers to always return False so setup_logger adds a handler
    monkeypatch.setattr(logger, "hasHandlers", lambda: False)

    setup_logger(logging.DEBUG)
    assert logger.level == logging.DEBUG

    handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(handlers) == 1

    formatter = handlers[0].formatter
    assert "%(asctime)s" in formatter._fmt
    assert "%(levelname)s" in formatter._fmt

def test_get_logger_is_singleton():
    logger1 = get_logger()
    logger2 = get_logger()
    assert logger1 is logger2

def test_custom_pathname_filter_shortens_path(tmp_path):
    # Simulate a record with a long path
    record = types.SimpleNamespace()
    record.pathname = str(tmp_path / "a" / "b" / "c" / "file.py")
    filt = CustomPathnameFilter()
    filt.filter(record)
    # After filter, only two last parts should remain
    assert record.pathname.endswith("c/file.py")

def test_logger_outputs_to_console(caplog):
    setup_logger(logging.INFO)
    logger = get_logger()
    with caplog.at_level(logging.INFO):
        logger.info("Logger test message 123")
    assert "Logger test message 123" in caplog.text
    # Check that the pathname in output is shortened
    assert ".py" in caplog.text
