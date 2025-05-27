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

    def run(self, restart: bool = False):
        """
        Perform the complete flow stability analysis pipeline.

        Parameters
        ----------
        restart : bool, optional
            If True, restart the computation from the beginning. If False,
            only recompute parts affected by parameter changes.

        Returns
        -------
        None
        """
        pass

Let me know if you want docstrings for other classes or want any further detail or formatting!

Code
Issues 16
Pull requests 3
Actions
Projects 1
Wiki
Security

    Insights

wip on FlowStability instance #59
Draft
j-i-l wants to merge 34 commits into main from 53-unified-interfacing-through-flowstability-class
+4,202 −2,750
Conversation 3
Commits 34
Checks 1
Files changed 21
Draft
wip on FlowStability instance
#59
File filter
Refresh
2 / 21 files viewed
2 changes: 2 additions & 0 deletions 2
.gitignore
Viewed
8 changes: 4 additions & 4 deletions 8
README.md
Viewed
Original file line number 	Original file line 	Diff line number 	Diff line change
@@ -59,13 +59,13 @@ this library with `apt-get install libmkl-rt`.


The main classes are:
The main classes are:
- `ContTempNetwork` in the module `temporal_network` which is used to store and save temporal networks and to compute inter-event transition matrices.
- `ContTempNetwork` in the module `temporal_network` which is used to store and save temporal networks and to compute inter-event transition matrices.
- `FlowIntegralClustering` in the module `flow_stability` which is used to computed the flow stability (integral of covariance) and to find the best forward and backward partition using the Louvain algorithm.
- `FlowIntegralClustering` in the sub-module `network_clustering` which is used to computed the flow stability (integral of covariance) and to find the best forward and backward partition using the Louvain algorithm.


Additional interesting classes and functions are:
Additional interesting classes and functions are:
- `Clustering` and `SparseClustering` in `flow_stability` can be used to directly cluster covariances or integrals of covariances.
- `Clustering` and `SparseClustering` in the `network_clustering` sub-module can be used to directly cluster covariances or integrals of covariances.
- `static_clustering` in `flow_stability` is an helper function to cluster static networks using Markov Stability.
- `static_clustering` in `flow_stability` is an helper function to cluster static networks using Markov Stability.
- `run_multi_louvain` in `flow_stability` helper function to run the Louvain multiple times on the same covariance in order to check the robustness of the partition.
- `run_multi_louvain` in `network_clustering` helper function to run the Louvain multiple times on the same covariance in order to check the robustness of the partition.
- `avg_norm_var_information` in `flow_stability` computes the average Normalized Variation of Information of list of cluster lists obtained with `run_multi_louvain`.
- `avg_norm_var_information` in `network_clustering` computes the average Normalized Variation of Information of list of cluster lists obtained with `run_multi_louvain`.
- `compute_parallel_clustering` in `parallel_clustering`, same than `run_multi_louvain` but in parallel.
- `compute_parallel_clustering` in `parallel_clustering`, same than `run_multi_louvain` but in parallel.
- the `parallel_expm` module contains functions to compute the matrix exponential of very large matrices using different strategies.
- the `parallel_expm` module contains functions to compute the matrix exponential of very large matrices using different strategies.


9 changes: 9 additions & 0 deletions 9
docs/conf.py
Viewed
Original file line number 	Original file line 	Diff line number 	Diff line change
@@ -43,4 +43,13 @@
autoapi_file_patterns = ['*.py', ]
autoapi_file_patterns = ['*.py', ]
autoapi_member_order = "groupwise"
autoapi_member_order = "groupwise"
autoapi_python_class_content = "both"
autoapi_python_class_content = "both"
autoapi_options = [
    'members',
    'undoc-members',
    # 'private-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
    'imported-members',
]
# autoapi_ignore = ["*conf.py", "*setup.py" , "*_cython*.pyx", ]
# autoapi_ignore = ["*conf.py", "*setup.py" , "*_cython*.pyx", ]
40 changes: 33 additions & 7 deletions 40
examples/asymmetric_example.ipynb
Viewed
Original file line number 	Original file line 	Diff line number 	Diff line change
@@ -38,7 +38,7 @@
    "\n",
    "\n",
    "from flowstab.synth_temp_network import Individual, SynthTempNetwork\n",
    "from flowstab.synth_temp_network import Individual, SynthTempNetwork\n",
    "from flowstab.temporal_network import ContTempNetwork\n",
    "from flowstab.temporal_network import ContTempNetwork\n",
    "from flowstab.flow_stability import FlowIntegralClustering"
    "from flowstab.network_clustering import FlowIntegralClustering"
   ]
   ]
  },
  },
  {
  {
@@ -199,7 +199,12 @@
  {
  {
   "cell_type": "code",
   "cell_type": "code",
   "execution_count": 6,
   "execution_count": 6,
   "metadata": {},
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    }
   },
   "outputs": [
   "outputs": [
    {
    {
     "data": {
     "data": {
@@ -353,7 +358,12 @@
  {
  {
   "cell_type": "code",
   "cell_type": "code",
   "execution_count": 7,
   "execution_count": 7,
   "metadata": {},
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    }
   },
   "outputs": [
   "outputs": [
    {
    {
     "data": {
     "data": {
@@ -383,7 +393,13 @@
  {
  {
   "cell_type": "code",
   "cell_type": "code",
   "execution_count": 8,
   "execution_count": 8,
   "metadata": {},
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "scrolled": true
   },
   "outputs": [
   "outputs": [
    {
    {
     "name": "stdout",
     "name": "stdout",
@@ -497,7 +513,12 @@
  {
  {
   "cell_type": "code",
   "cell_type": "code",
   "execution_count": 9,
   "execution_count": 9,
   "metadata": {},
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    }
   },
   "outputs": [
   "outputs": [
    {
    {
     "data": {
     "data": {
@@ -2547,7 +2568,12 @@
  {
  {
   "cell_type": "code",
   "cell_type": "code",
   "execution_count": 12,
   "execution_count": 12,
   "metadata": {},
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    }
   },
   "outputs": [
   "outputs": [
    {
    {
     "name": "stdout",
     "name": "stdout",
@@ -3269,7 +3295,7 @@
   "name": "python",
   "name": "python",
   "nbconvert_exporter": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "pygments_lexer": "ipython3",
   "version": "3.11.8"
   "version": "3.12.8"
  }
  }
 },
 },
 "nbformat": 4,
 "nbformat": 4,
54 changes: 54 additions & 0 deletions 54
examples/basics/minimal_example.py
Viewed
Original file line number 	Original file line 	Diff line number 	Diff line change
@@ -0,0 +1,54 @@
import pandas as pd

from flowstab import FlowStability, set_log_level

set_log_level("INFO")

duration = 600

# URL of the CSV file
url = 'https://zenodo.org/record/4725155/files/mice_contact_sequence.csv.gz'

# Load the CSV file into a DataFrame
mice_contact_df = pd.read_csv(url, compression='gzip')
# limit to the first 60min:
mice_contact_df = mice_contact_df[mice_contact_df['ending_times'] < duration]
# Temporal fix (see #60): reset index
unique_nodes = pd.unique(mice_contact_df[['source_nodes',
                                          'target_nodes']].values.ravel('K'))
node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
mice_contact_df['source_nodes'] = mice_contact_df['source_nodes'].map(node_mapping)
mice_contact_df['target_nodes'] = mice_contact_df['target_nodes'].map(node_mapping)

# initiate the analysis
fs_mice = FlowStability(
    t_start=None,
    t_stop=None)
print(f"{fs_mice.state.current=}")
print(f"{fs_mice.state.next=}")
# use only events within 10min hour
fs_mice.set_temporal_network(
    events_table=mice_contact_df[mice_contact_df['ending_times'] < duration]
)
print(f"{fs_mice.state.current=}")
print(f"{fs_mice.state.next=}")
fs_mice.compute_laplacian_matrices()
print(f"{fs_mice.state.current=}")
print(f"{fs_mice.state.next=}")

# Set the time scale
fs_mice.time_scale = 1
print(f"{fs_mice.state.current=}")
print(f"{fs_mice.state.next=}")

fs_mice.compute_inter_transition_matrices()
print(f"{fs_mice.state.current=}")
print(f"{fs_mice.state.next=}")

fs_mice.set_flow_clustering()
print(f"{fs_mice.state.current=}")
print(f"{fs_mice.state.next=}")

fs_mice.find_louvain_clustering()
print(f"{fs_mice.state.current=}")
print(f"{fs_mice.state.next=}")
51 changes: 40 additions & 11 deletions 51
examples/wild_mice_network.ipynb
Viewed
Original file line number 	Original file line 	Diff line number 	Diff line change
@@ -76,7 +76,7 @@
  {
  {
   "cell_type": "code",
   "cell_type": "code",
   "execution_count": 2,
   "execution_count": 2,
  "id": "28201252",
   "id": "28201252",
   "metadata": {},
   "metadata": {},
   "outputs": [
   "outputs": [
    {
    {
@@ -876,7 +876,13 @@
   "cell_type": "code",
   "cell_type": "code",
   "execution_count": 20,
   "execution_count": 20,
   "id": "9ffe8115",
   "id": "9ffe8115",
   "metadata": {},
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "scrolled": true
   },
   "outputs": [
   "outputs": [
    {
    {
     "data": {
     "data": {
@@ -1500,7 +1506,9 @@
   "cell_type": "code",
   "cell_type": "code",
   "execution_count": 29,
   "execution_count": 29,
   "id": "1f2a7563",
   "id": "1f2a7563",
   "metadata": {},
   "metadata": {
    "scrolled": true
   },
   "outputs": [
   "outputs": [
    {
    {
     "name": "stdout",
     "name": "stdout",
@@ -1812,7 +1820,13 @@
   "cell_type": "code",
   "cell_type": "code",
   "execution_count": 36,
   "execution_count": 36,
   "id": "c0e87970",
   "id": "c0e87970",
   "metadata": {},
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "scrolled": true
   },
   "outputs": [
   "outputs": [
    {
    {
     "data": {
     "data": {
@@ -2248,12 +2262,16 @@
  {
  {
   "cell_type": "markdown",
   "cell_type": "markdown",
   "id": "3beb6055",
   "id": "3beb6055",
   "metadata": {},
   "metadata": {
    "jupyter": {
     "source_hidden": true
    }
   },
   "source": [
   "source": [
    "We see that most mice are in singelton communities, but some form small communities.\n",
    "We see that most mice are in singelton communities, but some form small communities.\n",
    "The backward partition can then be computed by repeating these steps and passing `reverse_time=True` to `FlowIntegralClustering`.\n",
    "The backward partition can then be computed by repeating these steps and passing `reverse_time=True` to `FlowIntegralClustering`.\n",
    "\n",
    "\n",
    "One should then run the clustering many times at different resolution scales. The normalized variation of information can then be computed at each scale using the function `avg_norm_var_information` from `flow_stability`. The function `run_multi_louvain` from `flow_stability` helps you run multiple (serial) repetitions of the clustering.\n",
    "One should then run the clustering many times at different resolution scales. The normalized variation of information can then be computed at each scale using the function `avg_norm_var_information` from `network_clustering`. The function `run_multi_louvain` from the `network_clustering` sub-module helps you run multiple (serial) repetitions of the clustering.\n",
    "\n",
    "\n",
    "There are also helper functions to run parallel repetitions of the clustering and compute the average normalized variation of information in the module `parallel_clustering`. They need to be initialized with a `clustering`, for example `forw_flow.clustering[0]`."
    "There are also helper functions to run parallel repetitions of the clustering and compute the average normalized variation of information in the module `parallel_clustering`. They need to be initialized with a `clustering`, for example `forw_flow.clustering[0]`."
   ]
   ]
@@ -2273,6 +2291,10 @@
   "execution_count": 37,
   "execution_count": 37,
   "id": "4a02b970",
   "id": "4a02b970",
   "metadata": {
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "scrolled": true
    "scrolled": true
   },
   },
   "outputs": [
   "outputs": [
@@ -2383,7 +2405,12 @@
   "cell_type": "code",
   "cell_type": "code",
   "execution_count": 38,
   "execution_count": 38,
   "id": "0bfcf589",
   "id": "0bfcf589",
   "metadata": {},
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    }
   },
   "outputs": [
   "outputs": [
    {
    {
     "name": "stdout",
     "name": "stdout",
@@ -2588,7 +2615,9 @@
   "cell_type": "code",
   "cell_type": "code",
   "execution_count": 41,
   "execution_count": 41,
   "id": "9dda0646",
   "id": "9dda0646",
   "metadata": {},
   "metadata": {
    "scrolled": true
   },
   "outputs": [
   "outputs": [
    {
    {
     "data": {
     "data": {
@@ -4100,7 +4129,7 @@
    }
    }
   ],
   ],
   "source": [
   "source": [
    "from flowstab.flow_stability import Partition, sort_clusters\n",
    "from flowstab.network_clustering import Partition, sort_clusters\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "# mapping between integration time grid and temporal network time (in seconds)\n",
    "# mapping between integration time grid and temporal network time (in seconds)\n",
@@ -4424,7 +4453,7 @@
    }
    }
   ],
   ],
   "source": [
   "source": [
    "from flowstab.flow_stability import norm_mutual_information\n",
    "from flowstab.network_clustering import norm_mutual_information\n",
    "partition_forw_paper_1s = paper_results[1]['partitions_forward_per_week'][0]\n",
    "partition_forw_paper_1s = paper_results[1]['partitions_forward_per_week'][0]\n",
    "partition_back_paper_1s = paper_results[1]['partitions_backward_per_week'][0]\n",
    "partition_back_paper_1s = paper_results[1]['partitions_backward_per_week'][0]\n",
    "\n",
    "\n",
@@ -4669,7 +4698,7 @@
   "name": "python",
   "name": "python",
   "nbconvert_exporter": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "pygments_lexer": "ipython3",
   "version": "3.11.8"
   "version": "3.12.8"
  }
  }
 },
 },
 "nbformat": 4,
 "nbformat": 4,
52 changes: 52 additions & 0 deletions 52
src/flowstab/__init__.py
Changed since last view
Viewed
Original file line number 	Original file line 	Diff line number 	Diff line change
@@ -0,0 +1,52 @@
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
234 changes: 234 additions & 0 deletions 234
src/flowstab/_state_tracking.py
Viewed
Original file line number 	Original file line 	Diff line number 	Diff line change
@@ -0,0 +1,234 @@
from __future__ import annotations
import textwrap
import warnings

from copy import copy
from types import SimpleNamespace
from enum import Enum
from functools import wraps


class State:
    def __init__(self,
                 states:Enum,
                 properties_required:dict|None=None,
                 properties_set:dict|None=None,
                 methods_required:dict|None=None,
                 **kwargs
                 ):
        self._states = states
        self.current = min(self._states, key=lambda state: state.value)
        self.properties_required = copy(properties_required)
        self.methods_required = copy(methods_required)
        self.properties_set = copy(properties_set)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def missing(self):
        _current_state = self.current
        _required_props = self.properties_required.get(
            _current_state, []
        )
        _missing_props = [
            prop for prop in _required_props
            if not self.properties_set[prop]
        ]
        return _missing_props

    @property
    def next(self):
        _to_run = self._next_method.get(self.current)
        _to_set = self.missing
        return _to_set, _to_run


class _StateMeta(type):
    def __new__(cls, name, bases, namespace, states:Enum):
        cls.init_state(namespace=namespace, states=states)

        cls_instance = super().__new__(cls, name, bases, namespace)

        cls_instance.attach_state(namespace=namespace)

        return cls_instance

    @classmethod
    def init_state(cls, namespace, states:Enum):
        state = SimpleNamespace()
        # attach the states
        state.states = states
        _minimal_state_map = {}
        _state_map = {}
        # What method can be run next
        _next_method_map = {}
        # What method needed to be run to get to this state
        _method_required_state = {}
        # What property needs to be set to move from this state
        _property_next_state = {}
        _info = {}
        _howto = {}

        for key, value in namespace.items():
            if callable(value) and hasattr(value, 'next_state'):
                _info[key] = textwrap.dedent(value.__doc__ or "").rstrip()

                _howto[key] = textwrap.dedent(
                        value.__doc__ or "").rstrip()
                _minimal_state_map[key] = value.minimal_state
                _state_map[key] = value.next_state
                _next_method_map[key] = value.minimal_state
                _method_required_state[key] = value.next_state

            elif isinstance(value, property) and hasattr(value.fset, 'next_state'):
                _info[key] = textwrap.dedent(value.__doc__ or "").rstrip()
                _howto[key] = textwrap.dedent(value.fset.__doc__ or "").rstrip()

                _state_map[key] = value.fset.next_state
                _minimal_state_map[key] = getattr(value.fset,
                                                  'minimal_state', None)
                _property_next_state[key] = value.fset.next_state

        _all_states = set(
            _minimal_state_map.values()).union(set(
                _state_map.values()))
        state._all_states = sorted(
            [_state for _state in _all_states if _state is not None]
        )
        _method_required = {s: [] for s in state._all_states if s is not None}
        _next_method = {s: [] for s in state._all_states if s is not None}
        _property_required = {s: [] for s in state._all_states if s is not None}

        for _current_state in state._all_states:
            _included_states = []
            for _before_state in state._all_states:
                _included_states.append(_before_state)
                if _before_state == _current_state:
                    break
            for key, _state in _method_required_state.items():
                if _state is not None and _state in _included_states:
                    _method_required[_current_state].append(key)
            for key, _state in _next_method_map.items():
                if _state is not None and _state in _included_states:
                    _next_method[_current_state] = key
            for key, _state in _property_next_state.items():
                if _state is not None and _state in _included_states:
                    _property_required[_current_state].append(key)

        state.required = _minimal_state_map
        state._next_method = _next_method
        # state.next = _state_map
        # state._property_next_step = _property_next_step
        state.methods_required = _method_required
        state.properties_required = _property_required
        state.info = _info
        state.howto = _howto

        namespace['_state'] = state
        return None

    def attach_state(self, namespace):
        """
        This makes sure to attach the `state` property to the actual class
        """
        # Get the user-defined __init__ method if it exists
        user_init = namespace.get('__init__')
        _state = namespace.pop('_state')

        _property_is_set = {}
        for key, value in namespace.items():
            if isinstance(value, property) and hasattr(value.fset, 'next_state'):
                _property_is_set[key] = False

        # Define a custom __init__ method to initialize instance attributes
        def __init__(self, *args, **kwargs):
            # initiate the state
            self._state = State(**vars(_state))
            # initiate all properties as unset
            self._state.properties_set = copy(_property_is_set)
            # attach the missing method
            if user_init:  # If a user-defined __init__ exists, call it
                user_init(self, *args, **kwargs)

        # Set the custom __init__ method to the class
        self.__init__ = __init__

        # Define the state property
        def state_property(self):
            return self._state

        # def set_state_property(self, value):
        #     self._state.state = value

        # Add the property to the class
        self.state = property(state_property, None)

    @staticmethod
    def register(*, next_state: Enum, minimal_state: Enum|None = None,
                 ignore_none:bool=True):
        def decorator(func):
            if isinstance(func, property):
                # If it's a property, decorate the setter
                if func.fset:
                    func_name = func.fset.__name__
                    original_setter = func.fset

                    @wraps(original_setter)
                    def wrapped_setter(self, value):
                        # Run pre-run checks (if needed)
                        original_setter(self, value)
                        if value is None and ignore_none:
                            self._state.properties_set[func_name] = False
                        else:
                            self._state.properties_set[func_name] = True
                        # set the state back if it was more adavnced
                        self.state.current = min(self.state.current, next_state)
                        # Run post-run checks (if needed)

                    wrapped_setter.minimal_state = minimal_state
                    wrapped_setter.next_state = next_state

                    # Return the property with the decorated setter
                    return property(func.fget, wrapped_setter, func.fdel, func.__doc__)

            func_name = func.__name__
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                _current_state = self.state.current
                # TODO: we might set some properties from kwargs and thus update
                #       the current state.
                # check if the process is in the minimal state
                if _current_state < minimal_state:
                    warnings.warn(
                        f"'{func_name}' can only be run in state "
                        f"'{minimal_state}'. The current state is "
                        f"'{_current_state}'."
                    )
                    _return = None
                else:
                    # check if the required properties are set
                    _required_props = self._state.properties_required.get(
                        minimal_state, []
                    )
                    _missing_props = [
                        prop for prop in _required_props
                        if not self._state.properties_set[prop]
                    ]
                    if _missing_props:
                        _mp_list = '- '+'\n- '.join(_missing_props)
                        warnings.warn(
                            "You need to set the following attributes before you "
                            f"can run '{func_name}':\n{_mp_list}"
                        )
                        _return = None
                    else:
                        _return = func(self, *args, **kwargs)
                        # set the state to after the method call
                        self.state.current = next_state
                return _return

            wrapper.minimal_state = minimal_state
            wrapper.next_state = next_state
            return wrapper

        return decorator
2,816 changes: 417 additions & 2,399 deletions 2,816
src/flowstab/flow_stability.py
Viewed

Large diffs are not rendered by default.
18 changes: 18 additions & 0 deletions 18
src/flowstab/helpers.py
Viewed
Original file line number 	Original file line 	Diff line number 	Diff line change
@@ -0,0 +1,18 @@
"""Unsorted collection of simple helper functions
"""

def include_doc_from(cls):
    """Decorator to include the docstring from a specified class."""
    def decorator(func):
        func.__doc__ += "\n" + (cls.__doc__ or "")
        return func
    return decorator

def inverted_iterator(iterator):
    """Create an inverted iterator
    """
    for val in iterator:
        if val is None:
            yield val
        else:
            yield 1. / val
46 changes: 46 additions & 0 deletions 46
src/flowstab/logger.py
Viewed
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
        Compute Laplacian matrices for the temporal network.

        Parameters
        ----------
        **kwargs : dict
            Arguments passed to the underlying ContTempNetwork method.

        Returns
        -------
        self : FlowStability
            The instance itself.
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
        Compute inter-transition matrices for the temporal network.

        Parameters
        ----------
        linear_approx : bool, optional
            If True, use a linear approximation for the computation.
        **kwargs : dict
            Arguments passed to the underlying ContTempNetwork method.

        Returns
        -------
        self : FlowStability
            The instance itself.
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
