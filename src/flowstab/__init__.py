"""
Flow Stability: Temporal Network Flow-Based Clustering and Analysis
==================================================================

This package provides tools for stability analysis and clustering of temporal networks
using flow-based methods. The core component is the :class:`FlowStability` class,
which implements a reproducible, state-tracked analytical pipeline for temporal (contact sequence)
data. The package also includes logging utilities to facilitate debugging and reproducible research.

Main Features
-------------
- Load and process temporal networks (contact sequences) from files or arrays.
- Compute Laplacian and inter-transition matrices for random walks on temporal networks.
- Extract flow-based clusterings using methods such as the Louvain algorithm.
- Track analysis progress and enforce correct computational order through a robust state machine.
- Configurable logging for both development and production.

Logging
-------
The package sets up a default logger on import. You can adjust the logging level:

>>> import flowstab
>>> flowstab.set_log_level("DEBUG")

Usage Example
-------------
A typical workflow for analyzing a temporal network and extracting clusters:

>>> from flowstab import FlowStability
>>> fs = FlowStability()
>>> fs.set_temporal_network(filename="my_contacts.csv")
>>> fs.set_time_scale(10)
>>> fs.compute_laplacian_matrices()
>>> fs.compute_inter_transition_matrices()
>>> fs.set_flow_clustering()
>>> fs.find_louvain_clustering()
>>> print(fs.flow_clustering_forward)

This workflow ensures that all computational steps are performed in the required order;
the state machine will warn or prevent you from skipping prerequisites.

Module Contents
---------------
- FlowStability
    Main class for performing flow stability analysis and clustering.
- set_log_level
    Function to set the global logging level for the package.

Author
------
Alexandre Bovet <alexandre.bovet@maths.ox.ac.uk>


Contributors
............

- Jonas I. Liechti <j-i-l@t4d.ch>

License
-------
GNU Lesser General Public License v3 or later (LGPLv3+).

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
