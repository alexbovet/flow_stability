"""
flow stability

Copyright (C) 2021 Alexandre Bovet <alexandre.bovet@maths.ox.ac.uk>

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation; either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.


"""
import logging

from .flow_stability import FlowStability
from .logger import setup_logger, get_logger

# Default log level
setup_logger()  # Set up the logger with the default level

def set_log_level(level):
    """
    Set the logging level for the package.

    Parameters
    ----------
    level : str
        The logging level as a string (e.g., 'DEBUG', 'INFO').
    """
    level_dict = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    
    if level in level_dict:
        logger = get_logger()
        logger.setLevel(level_dict[level])
        for handler in logger.handlers:
            handler.setLevel(level_dict[level])
    else:
        raise ValueError(f"Invalid log level: {level}. Choose from {list(level_dict.keys())}.")
