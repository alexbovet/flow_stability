"""
# flow stability
#
# Copyright (C) 2021 Alexandre Bovet <alexandre.bovet@maths.ox.ac.uk>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


"""
from __future__ import annotations
from typing import Any, Iterator
from functools import total_ordering
from enum import Enum


import numpy as np

from .logger import get_logger
from .helpers import include_doc_from, inverted_iterator
from ._state_tracking import _StateMeta
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

@total_ordering
class States(Enum):
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

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return f"{self.name} ({self.value})"

register = _StateMeta.register  # make register method available as decorator

class FlowStability(metaclass=_StateMeta, states=States):
    """
    Conducts flow stability analysis using a contact sequence.

    This class loads a contact sequence into a Pandas DataFrame and 
    provides methods for analyzing the stability of the flow.

    Attributes
    ----------
    contact_sequence : pd.DataFrame
        The loaded contact sequence DataFrame.

    Raises
    ------
    ValueError
        If the input is neither a DataFrame nor a valid CSV file path.
    """
    def __init__(self, *,
                 temporal_network: ContTempNetwork|None=None,
                 time_scale: int|float|None=None,
                 t_start: int|float|None=None,
                 t_stop: int|float|None=None,
                 time_direction: int|None=None,
                 **kwargs: Any):
        """
        Initiate a flow stability analysis.


        Parameters
        ----------
        temporal_network: Union[pd.DataFrame, str, Path]
            A Pandas DataFrame containing the contact sequence, a string 
            representing the path to a CSV file, or a Path object pointing 
            to the CSV file.
        time_scale:
            ...
        t_start:
            ...
        t_stop:
            ...
        time_direction : int|None
            ...
        **kwargs : Any
            Optional keyword arguments to pass to `pandas.read_csv()` when 
            loading from a file.
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
        """
        return self._temporal_network

    @register(next_state=States.TEMP_NW, ignore_none=True)
    @temporal_network.setter
    def temporal_network(self, temporal_network:ContTempNetwork|None):
        """Set the temporal network data

        .. note::
            You might also use `set_temporal_network` to load temporal network
            data.

        Parameters
        ----------
        temporal_network : ContTempNetwork
            Temporal network data.
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
        """Setting the temporal network for the flow stability analysis.
        """
        if not kwargs:
            self.temporal_network = None
        else:
            # set self.temporal_network
            self.temporal_network = ContTempNetwork(**kwargs)
        return self

    @property
    def time_scale(self):
        """Inter event time scale of the random walk.
        """
        return iter(self._time_scale)

    @register(next_state=States.LAPLAC)
    @time_scale.setter
    def time_scale(self, time_scale:None|Iterator|int|float):
        """Set the time scale determining the random walks transition rate.

        """
        if time_scale is None:
            self._time_scale = [None, ]
        elif isinstance(time_scale, (int, float)):
            self._time_scale = [time_scale]
        elif isinstance(time_scale, Iterator):
            self._time_scale = list(time_scale)
        else:
            raise TypeError(f"Invalid type '{type(time_scale)}'.")

    @include_doc_from(np.linspace)
    def set_time_scale(self, value:int|float|None=None, **kwargs):
        """Set the time scale determining the random walks transition rate.

        Parameter
        ---------
        value:
            Characteristic random walk inter event time. If set to `None` the
            median inter event time will be used.
        """
        if value is not None:
            self.time_scale = value
        elif kwargs:
            self.time_scale = np.linspace(**kwargs)
        else:
            # TODO: Use the median of the inter event times
            self.time_scale = None
        return None
    @property
    def lamda(self,):
        return inverted_iterator(self.time_scale)

    @property
    def t_start(self):
        """Start time to calculate the Lapalcian matrices from.

        The laplacian matrices will be calculated form the first event time
        before or equal to this time-point.
        """
        return self._t_start

    @register(next_state=States.TEMP_NW, ignore_none=False)
    @t_start.setter
    def t_start(self, value:int|float|None):
        """Set the starting time for the temporal network data.

        The laplacian matrices will be computed from this time onward.

        .. note::
            When setting this value, the Laplaican matrices will be calculated
            anew.
        """
        self._t_start = value

    @property
    def t_stop(self):
        """Stop time until when the Laplacian matrices will be calculated.

        The laplacian matrices will be calculated up to the first event time
        after or equal to this timepoint.
        """
        return self._t_stop

    @register(next_state=States.TEMP_NW, ignore_none=False)
    @t_stop.setter
    def t_stop(self, value:int|float|None):
        """Set the stop time for the temporal network data to include.

        .. note::
            When setting this value, the Laplaican matrices will be calculated
            anew.
        """
        self._t_stop = value


    @property
    def time_direction(self):
        """
        """
        return self._time_direction

    @register(next_state=States.CLUSTERING, ignore_none=False)
    @time_direction.setter
    def time_direction(self, value:int|None):
        """Set the stop time for the temporal network data to include.

        .. note::
            When setting this value, the Laplaican matrices will be calculated
            anew.
        """
        if value is None:
            value = 0
        else:
            assert value in [-1, 0, 1]
        self._time_direction = value

    @register(minimal_state=States.TEMP_NW, next_state=States.LAPLAC)
    @include_doc_from(ContTempNetwork.compute_laplacian_matrices)
    def compute_laplacian_matrices(self, **kwargs):
        """
        """
        kwargs.update(dict(
            t_start = self.t_start,
            t_stop = self.t_stop
        ))
        self._temporal_network.compute_laplacian_matrices(**kwargs)
        return self

    @register(minimal_state=States.LAPLAC, next_state=States.INTER_T)
    @include_doc_from(ContTempNetwork.compute_inter_transition_matrices)
    # @ok_state
    def compute_inter_transition_matrices(self, linear_approx=False, **kwargs):
        """
        """
        if linear_approx:
            to_compute = self._temporal_network.compute_lin_inter_transition_matrices
        else:
            to_compute = self._temporal_network.compute_inter_transition_matrices
        # handle the time_scale parameter explicitely
        _time_scale = None
        if 'time_scale' in kwargs:
            _time_scale = kwargs['time_scale']
        # ###
        # TODO: deprecate with lamda
        elif 'lamda' in kwargs and kwargs['lamda'] is not None:
            _time_scale = 1 / kwargs['lamba']
        if _time_scale is not None:
            self.time_scale = _time_scale
        # ###
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

    @include_doc_from(FlowIntegralClustering)
    @property
    def flow_clustering_forward(self):
        """
        """
        return self._flow_clustering_forward

    @register(next_state=States.CLUSTERING)
    @flow_clustering_forward.setter
    def flow_clustering_forward(
        self,
        flow_clustering:tuple[int|float, FlowIntegralClustering|None]):
        """Set the flow integral clustering object

        .. note::
            You migth also use `set_flow_clustering` to create a instance
            directly.
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

    @include_doc_from(FlowIntegralClustering)
    @property
    def flow_clustering_backward(self):
        """
        """
        return self._flow_clustering_backward

    @register(next_state=States.CLUSTERING)
    @flow_clustering_backward.setter
    def flow_clustering_backward(
        self,
        flow_clustering:tuple[int|float,FlowIntegralClustering|None]):
        """Set the flow integral clustering object

        .. note::
            You migth also use `set_flow_clustering` to create a instance
            directly.
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
    @include_doc_from(FlowIntegralClustering)
    def set_flow_clustering(self, **kwargs):
        """
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
                # run backward
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
                # run forward
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
    @include_doc_from(FlowIntegralClustering.find_louvain_clustering)
    def find_louvain_clustering(self, **kwargs):
        """
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
                # run backward
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

    def run(self, restart:bool=False):
        """Perform a flow stability analysis.

        Parameters
        ----------
        direction:
           Set the temporal direction to perform the flow analysis for.
           Options are:

           - 1: Forward in time
           - -1: Backwards in time
           - 0: Copute both directions
        restart:
           Indicate if the computation should be restarted from the very
           beginning or to only re-run the parts that need to be computed again.

           .. note::
               With `restart=False`:

               - If you call `run` multiple times, only the first call will
                 perform the computations.
               - If you adapt a parameter in-between two `run` calls, then only
                 the parts of the analysis that are affected by this parameter
                 will be re-run agian.
                 Example: If you change the `time_scale` parameter, then the
                 computation of the analysis will be re-run form the computation
                 of the inter transition matrices onwards, but the laplacian
                 matrices will not be re-run, as they do not depend on the
                 `time_scale` parameter.
                 it 
        """
        pass
