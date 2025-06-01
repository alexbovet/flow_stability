"""
Flow Stability Analysis
======================

This sub-module provides tools to perform flow stability analysis on temporal 
networks using contact sequences. It enables loading temporal network data, 
computing Laplacian and inter-transition matrices, and extracting flow-based 
clusterings. It leverages a state-tracking system to ensure that computations 
are performed in the correct order and that all required data is available 
before each analysis step.

Key Features
------------
- State-tracked analysis pipeline for reproducibility and robustness.
- Support for loading and representing temporal (contact sequence) networks.
- Computation of Laplacian and inter-transition matrices for random walks.
- Flow integral clustering and community detection (e.g., Louvain clustering).
- Flexible time scale and direction support for forward/backward dynamics.

Examples
--------
A typical usage workflow:

>>> from flowstab.flow_stability import FlowStability
>>> fs = FlowStability()
>>> fs.set_temporal_network(events_table="my_contacts.csv")
>>> fs.set_time_scale(10)
>>> fs.compute_laplacian_matrices()
>>> fs.compute_inter_transition_matrices()
>>> fs.time_direction = 0
>>> fs.set_flow_clustering()
>>> fs.find_louvain_clustering()
>>> print(fs.flow_clustering_forward)

Classes
-------
FlowStability
    Main class for performing flow stability analysis on a temporal network.

States
    Enum of analysis states, used for internal state tracking.

ProcessException
    Exception raised for errors in the flow stability process.
"""
from __future__ import annotations
from typing import Any, Iterator


import numpy as np

from .logger import get_logger
from .helpers import include_doc_from, inverted_iterator
from .state_tracking import StateMeta, OrderedEnum
from .temporal_network import (
    ContTempNetwork,
)
from .network_clustering import (
    FlowIntegralClustering,
)

# get the logger
logger = get_logger()


class ProcessException(Exception):
    pass

class States(OrderedEnum):
    """Defines the stages of a flow stability analysis"""
    INITIAL = 0
    """Initiated an flow stability analysis with no, or incomplete data"""
    TEMP_NW = 1
    """Ready to calculate the Laplacian matrices"""
    LAPLAC = 2
    """Ready to calculate the inter transition matrices"""
    INTER_T = 3
    """Ready to compute the flow integral clustering"""
    CLUSTERING = 4
    """Ready to compute the Louvain clusters."""
    FINAL = 5
    """Computed all that there is."""

register = StateMeta.register  # make register method available as decorator

class FlowStability(metaclass=StateMeta, states=States):
    """
    Conducts flow stability analysis using a temporal network.

    This class loads a temporal network (contact sequence) and
    provides methods for analyzing the stability of flows via
    Laplacian and inter-transition matrices, and clustering.

    Raises
    ------
    ValueError
        If inputs are not of valid types or parameters are inconsistent.
    TypeError
        If provided types are not as expected.
    ProcessException
        For errors during the processing steps.


    """
    def __init__(self, *,
                 temporal_network: ContTempNetwork|None=None,
                 time_scale: int|float|None=None,
                 t_start: int|float|None=None,
                 t_stop: int|float|None=None,
                 time_direction: int|None=None,
                 **kwargs: Any):
        """
        Initialize a flow stability analysis instance.

        Parameters
        ----------
        temporal_network : ContTempNetwork or None, optional
            The temporal network data to analyze.
        time_scale : int, float, or None, optional
            Characteristic time scale for the random walk's transition rate.
        t_start : int, float, or None, optional
            Start time for the analysis.
        t_stop : int, float, or None, optional
            End time for the analysis.
        time_direction : int or None, optional
            Direction of time for the analysis (-1, 0, or 1).
        **kwargs : dict
            Additional keyword arguments for initializing the temporal network.
        """
        self.temporal_network = temporal_network
        self.t_start = t_start
        self.t_stop = t_stop
        self.time_scale = time_scale
        self.time_direction = time_direction
        self._flow_clustering_forward = {}
        self._flow_clustering_backward = {}

    @include_doc_from(ContTempNetwork)
    @property
    def temporal_network(self):
        """
        The temporal network used as basis for the analysis.

        Returns
        -------
        ContTempNetwork or None
            The temporal network currently set for the analysis.
        """
        return self._temporal_network

    @register(next_state=States.TEMP_NW, ignore_none=True)
    @temporal_network.setter
    def temporal_network(self, temporal_network:ContTempNetwork|None):
        """
        Set the temporal network for analysis.

        Parameters
        ----------
        temporal_network : ContTempNetwork or None
            The temporal network data to use. If not a ContTempNetwork instance,
            the property is set to None and a warning is issued.

        .. note::
            You might also use `set_temporal_network` to load temporal network
            data.

        """
        if isinstance(temporal_network, ContTempNetwork):
            self._temporal_network = temporal_network
        else:
            logger.warning(
                f"Object of type {type(temporal_network)} cannot be "
                "used as temporal network. Setting `temporal_network` "
                "attribute to `None` and resetting the state of the analysis."
            )
            self._temporal_network = None
        return None

    @register(minimal_state=States.INITIAL, next_state=States.TEMP_NW)
    @include_doc_from(ContTempNetwork)
    def set_temporal_network(self, **kwargs):
        """
        Set the temporal network for the flow stability analysis.

        Parameters
        ----------
        **kwargs : dict
            Arguments to initialize a ContTempNetwork instance. If no arguments
            are provided, the temporal network is set to None.

        Returns
        -------
        self : FlowStability
            The instance itself.
        """
        if not kwargs:
            self.temporal_network = None
        else:
            # set self.temporal_network
            self.temporal_network = ContTempNetwork(**kwargs)
        return self

    @property
    def time_scale(self):
        """
        Time scales used for random walk transition rates.

        Returns
        -------
        iterator
            Iterator over the time scales. Each value determines the rate of the
            random walk's transitions.

        .. note::
            Single values are alos returned as an iterator.

        """
        return iter(self._time_scale)

    @register(next_state=States.LAPLAC)
    @time_scale.setter
    def time_scale(self, time_scale:None|Iterator|int|float):
        """
        Set the time scale(s) for the random walk's transition rate.

        .. note::
            You might also use `set_time_scale` to directly create a range of
            time scales.

        Parameters
        ----------
        time_scale : None, int, float, or iterator of float
            If None, a default value is used.
            If an int or float, a single time scale is set.
            If an iterator, it must yield float or int values.

        Raises
        ------
        TypeError
            If the input is not None, int, float, or an iterator of numbers.
        """
        if time_scale is None:
            self._time_scale = [None, ]
        elif isinstance(time_scale, (int, float)):
            self._time_scale = [time_scale]
        elif isinstance(time_scale, Iterator):
            self._time_scale = list(time_scale)
        else:
            raise TypeError(f"Invalid type '{type(time_scale)}'.")

    @include_doc_from(np.logspace)
    def set_time_scale(self, value:int|float|None=None, **kwargs):
        """
        Set the time scale(s) for the analysis.

        Parameters
        ----------
        value : int, float, or None, optional
            Characteristic random walk inter-event time. If None and `kwargs` is
            empty, the median inter-event time will be used.
        **kwargs : dict
            Arguments passed to `numpy.logspace` to generate multiple time
            scales.

        Returns
        -------
        None
        """
        if value is not None:
            self.time_scale = value
        elif kwargs:
            self.time_scale = np.logspace(**kwargs)
        else:
            # TODO: Use the median of the inter event times
            self.time_scale = None
        return None

    @property
    def lamda(self,):
        """
        Iterator of lambda values, the inverse of the time scales.

        .. warning::
            This attribute is deprecated and will be removed in future versions.

            Please use `time_scale` which is the inverse of `lambda`, instead.

        Returns
        -------
        iterator of float or None
            Each value is the inverse of a time scale, or None if time scale is
            None.
        """
        return inverted_iterator(self.time_scale)

    @property
    def t_start(self):
        """
        Start time for Laplacian matrix calculation.

        The laplacian matrices will be calculated form the first event time
        before or equal to this time-point.

        Returns
        -------
        float or None
            Start time for Laplacian matrices; events before this time are
            ignored.
        """
        return self._t_start

    @register(next_state=States.TEMP_NW, ignore_none=False)
    @t_start.setter
    def t_start(self, value:int|float|None):
        """
        Set the starting time for the temporal network data.

        .. note::
            Whenever setting this value, the Laplaican matrices will be
            calculated anew.

        Parameters
        ----------
        value : int, float, or None
            The Laplacian matrices will be computed from this time onward.
        """
        self._t_start = value

    @property
    def t_stop(self):
        """
        Stop time for Laplacian matrix calculation.

        The laplacian matrices will be calculated up to the first event time
        after or equal to this timepoint.

        Returns
        -------
        float or None
            End time for Laplacian matrices; events after this time are ignored.
        """
        return self._t_stop

    @register(next_state=States.TEMP_NW, ignore_none=False)
    @t_stop.setter
    def t_stop(self, value: int | float | None):
        """
        Set the stop time for the temporal network data.

        .. note::
            Whenever setting this value, the Laplaican matrices will be
            calculated anew.

        Parameters
        ----------
        value : int, float, or None
            The Laplacian matrices will be computed up to this time.
        """
        self._t_stop = value

    @property
    def time_direction(self):
        """
        Get the time direction for the analysis.

        Returns
        -------
        int or None
            Time direction: -1 (backward), 0 (both), or 1 (forward).
        """
        return self._time_direction

    @register(next_state=States.INTER_T)
    @time_direction.setter
    def time_direction(self, value: int | None):
        """
        Set the time direction for the analysis.

        Parameters
        ----------
        value : int or None
            Can be -1 (backward), 0 (both directions), or 1 (forward). None
            defaults to 0.

        Raises
        ------
        AssertionError
            If value is not None, -1, 0, or 1.
        """
        if value is None:
            value = 0
        else:
            assert value in [-1, 0, 1]
        self._time_direction = value

    @register(minimal_state=States.TEMP_NW, next_state=States.LAPLAC)
    def compute_laplacian_matrices(self, **kwargs):
        """
        Compute Laplacian matrices for the current temporal network.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments passed to ContTempNetwork's
            `compute_laplacian_matrices` method.

        Returns
        -------
        self : FlowStability
            The instance itself.
        """
        kwargs.update(dict(
            t_start=self.t_start,
            t_stop=self.t_stop
        ))
        self._temporal_network.compute_laplacian_matrices(**kwargs)
        return self

    @register(minimal_state=States.LAPLAC, next_state=States.INTER_T)
    def compute_inter_transition_matrices(self, linear_approx=False, **kwargs):
        """
        Compute inter-transition matrices for the temporal network.

        Parameters
        ----------
        linear_approx : bool, optional
            If True, use a linear approximation for the computation.
        **kwargs : dict
            Additional arguments passed to the computation methods.

        Returns
        -------
        self : FlowStability
            The instance itself.
        """
        if linear_approx:
            to_compute = self._temporal_network.compute_lin_inter_transition_matrices
        else:
            to_compute = self._temporal_network.compute_inter_transition_matrices

        _time_scale = None
        if 'time_scale' in kwargs:
            _time_scale = kwargs['time_scale']
        elif 'lamda' in kwargs and kwargs['lamda'] is not None:
            _time_scale = 1 / kwargs['lamba']
        if _time_scale is not None:
            self.time_scale = _time_scale
        for _ts in self.time_scale:
            logger.info(
                f"Computing inter T matrices for time_scale={_ts}."
            )
            if _ts is None:
                _lambda = None
            else:
                _lambda = 1 / _ts
            kwargs.update(dict(
                lamda=_lambda
            ))
            to_compute(**kwargs)
            logger.info("-> done.")
        return self

    @property
    def flow_clustering_forward(self):
        """
        Get the dictionary of forward-time flow integral clustering results.

        Returns
        -------
        dict
            Maps time_scale values to FlowIntegralClustering objects.
        """
        return self._flow_clustering_forward

    @register(next_state=States.CLUSTERING)
    @flow_clustering_forward.setter
    def flow_clustering_forward(
        self,
        flow_clustering: tuple[int | float, FlowIntegralClustering | None]
    ):
        """
        Set a FlowIntegralClustering result for the forward direction.

        Parameters
        ----------
        flow_clustering : tuple
            Tuple of (time_scale, FlowIntegralClustering instance or None).
            Only FlowIntegralClustering instances with reversed_time == False are
            accepted; otherwise, None is set and a warning is issued.
        """
        _time_scale, _flow_clustering = flow_clustering
        if isinstance(_flow_clustering, FlowIntegralClustering):
            assert not _flow_clustering.reversed_time
            self._flow_clustering_forward[_time_scale] = _flow_clustering
        else:
            logger.warning(
                f"Object of type {type(flow_clustering)} cannot be "
                "used for attribute `flow_clustering`. "
                "`temporal_network` attribute is set to `None`."
            )
            self._flow_clustering_forward[_time_scale] = None

    @property
    def flow_clustering_backward(self):
        """
        Get the dictionary of backward-time flow integral clustering results.

        Returns
        -------
        dict
            Maps time_scale values to FlowIntegralClustering objects.
        """
        return self._flow_clustering_backward

    @register(next_state=States.CLUSTERING)
    @flow_clustering_backward.setter
    def flow_clustering_backward(
        self,
        flow_clustering: tuple[int | float, FlowIntegralClustering | None]
    ):
        """
        Set a FlowIntegralClustering result for the backward direction.

        Parameters
        ----------
        flow_clustering : tuple
            Tuple of (time_scale, FlowIntegralClustering instance or None).
            Only FlowIntegralClustering instances with reversed_time == True are
            accepted; otherwise, None is set and a warning is issued.
        """
        _time_scale, _flow_clustering = flow_clustering
        if isinstance(_flow_clustering, FlowIntegralClustering):
            assert _flow_clustering.reversed_time
            self._flow_clustering_backward[_time_scale] = _flow_clustering
        else:
            logger.warning(
                f"Object of type {type(flow_clustering)} cannot be "
                "used for attribute `flow_clustering`. "
                "`temporal_network` attribute is set to `None`."
            )
            self._flow_clustering_backward[_time_scale] = None

    @register(minimal_state=States.INTER_T, next_state=States.CLUSTERING)
    def set_flow_clustering(self, **kwargs):
        """
        Perform flow integral clustering analysis.

        Parameters
        ----------
        **kwargs : dict
            Arguments passed to FlowIntegralClustering.

        Returns
        -------
        self : FlowStability
            The instance itself.
        """
        for _ts in self.time_scale:
            logger.info(
                f"Creating a flow integral clustering for time_scale={_ts}."
            )
            if _ts is None:
                _lambda = None
            else:
                _lambda = 1 / _ts

            kwargs.update(dict(
                T_inter_list=[T.toarray()
                              for T in self.temporal_network.inter_T[_lambda]],
                time_list=self.temporal_network.times,
            ))
            if self.time_direction <= 0:
                logger.info(
                    f"\t- Creating the time backward clustering."
                )
                try:
                    kwargs.update(dict(reverse_time=True))
                    self.flow_clustering_backward = (
                        _ts,
                        FlowIntegralClustering(**kwargs)
                    )
                except ValueError as e:
                    logger.warning(
                        f"Failed to initiate the FlowIntegralClustering: {e}"
                    )
            if self.time_direction >= 0:
                logger.info(
                    f"\t- Creating the time forward clustering."
                )
                try:
                    kwargs.update(dict(reverse_time=False))
                    self.flow_clustering_forward = (
                        _ts,
                        FlowIntegralClustering(**kwargs)
                    )
                except ValueError as e:
                    logger.warning(
                        f"Failed to initiate the FlowIntegralClustering: {e}"
                    )
            logger.info("-> done.")
        return self

    @register(minimal_state=States.CLUSTERING, next_state=States.FINAL)
    def find_louvain_clustering(self, **kwargs):
        """
        Find Louvain clusters for the flow integral clustering.

        Parameters
        ----------
        **kwargs : dict
            Arguments passed to FlowIntegralClustering's
            `find_louvain_clustering` method.

        Returns
        -------
        self : FlowStability
            The instance itself.
        """
        for _ts in self.time_scale:
            logger.info(
                f"Creating a flow integral clustering for time_scale={_ts}."
            )
            if _ts is None:
                _lambda = None
            else:
                _lambda = 1 / _ts
            if self.time_direction <= 0:
                logger.info(
                    f"\tBackwards in time."
                )
                n_loops = self.flow_clustering_backward[
                    _ts].find_louvain_clustering(**kwargs)
            if self.time_direction >= 0:
                logger.info(
                    f"\tForwards in time."
                )
                n_loops = self.flow_clustering_forward[
                    _ts].find_louvain_clustering(**kwargs)
            logger.info("-> done.")
        return self
