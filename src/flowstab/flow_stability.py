"""#
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
from typing import Any

import warnings
import textwrap
import functools

from pathlib import Path

import pandas as pd

from .logger import get_logger

from .temporal_network import (
    ContTempNetwork,
)
from .network_clustering import (
    FlowIntegralClustering,
)

# get the logger
logger = get_logger()


def include_doc_from(cls):
    """Decorator to include the docstring from a specified class."""
    def decorator(func):
        func.__doc__ += "\n" + (cls.__doc__ or "")
        return func
    return decorator

class ProcessException(Exception):
    pass

def ok_state(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Get the name of the method being called
        method_name = method.__name__
        required_state = self._run_at.get(method_name)
        if self._state < required_state:
            raise ProcessException(
                f"`{method_name}` cannot be run yet. Run the "
                "`next_step` method to see what steps should be perforemd next"
            )
        
        # Call the original method
        _return = method(self, *args, **kwargs)

        self._state = self._next_step[method_name]
        return _return 
    return wrapper

class FlowStability:
    """
    Conducts flow stability analysis using a contact sequence.

    This class loads a contact sequence into a Pandas DataFrame and 
    provides methods for analyzing the stability of the flow.

    Parameters
    ----------
    temporal_network: Union[pd.DataFrame, str, Path]
        A Pandas DataFrame containing the contact sequence, a string 
        representing the path to a CSV file, or a Path object pointing 
        to the CSV file.
    **kwargs : Any
        Optional keyword arguments to pass to `pandas.read_csv()` when 
        loading from a file.

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
                 temporal_network: pd.DataFrame|str|Path|None=None,
                 time_scale: int|float|None=None,
                 t_start: int|float|None=None,
                 t_stop: int|float|None=None,
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
        """
        # this variable track the progress of the analysis - enum would be good
        self._state = 0
        self._temporal_network = None
        self._time_scale = None
        # TODO: deprecate with lamda parameter
        self._lamda = None
        # Prepare all property relate information
        self._init_info()
        # Set what attributes are required when
        self._init_required_next()
        if isinstance(temporal_network, pd.DataFrame):
            # Create a ContTempNetwork instance directly from the DataFrame
            self.set_temporal_network(events_table=temporal_network)
            logger.debug("Pandas dataframe provided and used as events table.")
        elif isinstance(temporal_network, (str, Path)):
            try:
                # Convert Path to string if it's a Path object
                events_table = pd.read_csv(str(temporal_network), **kwargs)
                # Create a ContTempNetwork instance using the loaded DataFrame
                self.set_temporal_network(events_table=events_table)
                logger.debug("Loading events from csv file.")
            except FileNotFoundError:
                raise ValueError(
                    f"The file at {temporal_network} was not found."
                )
            except pd.errors.EmptyDataError:
                raise ValueError(
                    f"The file at {temporal_network} is empty."
                )
            except pd.errors.ParserError:
                raise ValueError(
                    f"The file at {temporal_network} could not be parsed."
                )
        else:
            self.set_temporal_network()

    def _init_info(self):
        self._properties = []
        self._info= {}
        self._howto = {}
        for attr in dir(self):
            prop = getattr(self.__class__, attr, None)
            if isinstance(prop, property):
                self._properties.append(attr)
                self._info[attr] = textwrap.dedent(prop.__doc__ or "").rstrip()
                self._howto[attr] = textwrap.dedent(
                        prop.fset.__doc__ or "").rstrip() if prop.fset else None

    def _init_required_next(self, ):
        self._required_next = {
            0: [
                'temporal_network'
                ],
            1: [
                'time_scale'
            ],
        }
        self._required_at = {val: step
                             for step, values in self._required_next.items()
                             for val in values}
        self._run_next = {
            1: 'compute_laplacian_matrices',
            1.1: 'compute_inter_transition_matrices'
        }
        self._run_at = {fct: step for step, fct in self._run_next.items()}
        # define the next state for any method
        self._next_step = {
            'compute_laplacian_matrices': 1.1,
            'compute_inter_transition_matrices': 1.2,
        }

    def _values(self, ):
        _values = {}
        for attr in dir(self):
            # Check if the attribute is a property
            prop = getattr(self.__class__, attr, None)
            if isinstance(prop, property):
                _values[attr] = getattr(self, attr)
        return _values

    def info(self, attribute:str|None=None):
        """Provide information on the attribute in question.

        If not attribute is provided, then informaiton on all relevant
        attributes is returned.
        """
        if attribute is None:
            return self._info
        else:
            return self._info.get(attribute)
    
    def howto(self, attribute:str|None=None):
        """Provide details on how to set the attribute.
        """
        if attribute is None:
            return self._howto
        else:
            return self._howto.get(attribute)
            
    def print_status(self):
        # Get all attributes of the class
        for attr in self._properties:
            value = getattr(self, attr)
            docstring = self._info[attr]
            setter_docstring = self._howto[attr]
            print(f"Property: {attr}, Value: {value}, Docstring: {docstring}")
            if setter_docstring:
                print(f"  Setter Docstring: {setter_docstring}")

    def next_step(self, ):
        _next_steps = {}
        # first determine what parameter should next be set
        _values = self._values()
        _next_to_set = []
        min_step = 10  # should be the end 
        for attr, step in self._required_at.items():
            print(attr, step, min_step)
            if _values.get(attr) is None:
                if step == min_step:
                    _next_to_set.append(attr)
                elif step < min_step:
                    _next_to_set = [attr]
                    min_step = step
        _next_steps['parameter'] = tuple(_next_to_set)
        # determine what steps must be run next
        # TODO
        return _next_steps

    def _ready_for(self, step):
        """Internal method to make sure the step in the analysis is doable
        """
        # TODO: This should be a method rather than a step
        required_for = self._required_attributes(next_step=step)
        required_values = {attr: getattr(self, attr) for attr in required_for}
        for attr, value in required_values.items():
            if value is None:
                logger.info(f"Missing mandatory attribute: '{attr}':"
                            f"\n{self.howto(attr)}")
                return False
        return True
        # if self._state >= step:
        #     return True
        # else:
        #     logger.warning(
        #         "The flow stability analysis is not ready for this step."
        #     )
        #     return False
        
    def _required_attributes(self, next_step):
        required_attrs = [attr
                          for step, attrs in self._required_next.items()
                          for attr in attrs if step < next_step]
        return required_attrs

    @include_doc_from(ContTempNetwork)
    def set_temporal_network(self, *args, **kwargs):
        """
        Setting the temporal network for the flos stability analysis
        """
        if not args and not kwargs:
            self._temporal_network = None
            self._state = 0
            logger.info(
                "Setting the temporal network with no data -> "
                "RESETTING ANALYSIS.")
        else:
            # set self.temporal_network
            self.temporal_network = ContTempNetwork(*args,
                                                    **kwargs)
            # reset the progress to first stage
            self._state = 1
        return self

    @include_doc_from(ContTempNetwork.compute_laplacian_matrices)
    @ok_state
    def compute_laplacian_matrices(self, *args, **kwargs):
        """
        """
        self._temporal_network.compute_laplacian_matrices(*args,
                                                            **kwargs)
        return self

    @include_doc_from(ContTempNetwork.compute_inter_transition_matrices)
    @ok_state
    def compute_inter_transition_matrices(self, **kwargs):
        """
        """
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
        self._temporal_network.compute_inter_transition_matrices(**kwargs)
        return self

    # NOTE: we need to properly handle the _state when setting properties
    #       a _state might require multiple properties so it cannot be set
    #       to completed if a single property is set. Ideally we perform a
    #       check if all required properties are set for a given stage
    # NOTE: if a property from a previous stage was changed then the _state
    #       should be set back. Otherwise the property might get ignored and
    #       we end up in an inconsistent state.

    @property
    def temporal_network(self):
        """
        """
        return self._temporal_network

    @temporal_network.setter
    @include_doc_from(ContTempNetwork)
    def temporal_network(self, temporal_network:ContTempNetwork):
        """Set the temporal network data.
        """
        # TODO: do so sanity checks
        self._temporal_network = temporal_network
        return None

    @property
    def lamda(self):
        """
        """
        return self._time_scale

    @lamda.setter
    def lamda(self, value):
        """Setting the random walk rate.

        This method will be removed in favor of `time_scale`.
        """
        warnings.warn(
            "The parameter `lamda` is deprecated and will removed in "
            "future releases!\nTo set the random walk rate use the "
            "`time_scale` variable instead, which is the inverse of `lamda`."
        )
        if value is not None:
            self._time_scale = 1 / value

    @property
    def time_scale(self):
        """Inter event time scale of the random walk.
        """
        return self._time_scale

    @time_scale.setter
    def time_scale(self, value:int|float|None):
        """Set the time scale determining the random walks transition rate.

        Parameter
        ---------
        value:
          Characteristic random walk inter event time. If set to `None` the
          median inter event time will be used.
        """
        if self._time_scale != value:
            self._time_scale = value
            # ###
            # TODO: This should be removed
            if self._time_scale is not None:
                self._lamda = 1 / self._time_scale
            else:
                self._lamda = None
            # ###

    # Facade for FlowIntegralClustering
    @ok_state
    @include_doc_from(FlowIntegralClustering)
    def set_flow_clustering(self, integral_time_grid, *args, **kwargs):
        """
        """
        # NOTE: we could set this up so that one could directly provide
        #       arguments for the integral clustering without providing
        #       a temporal network
        if self._ready_for(step=1.2):
            kwargs.update(dict(
                T_inter_list=[T.toarray()
                              for T in self.temporal_network.inter_T[self._lamda]],
                time_list=self.temporal_network.times,
            ))
            
        try:
            self.flow_clustering = FlowIntegralClustering(
                    *args,
                    integral_time_grid=integral_time_grid,
                    **kwargs)
        except ValueError as e:
            logger.warning(
                f"Falied to initiate the FlowIntegralClustering: {e}"
            )
        return self

    def run(self, direction:int=1, restart:bool=False):
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

