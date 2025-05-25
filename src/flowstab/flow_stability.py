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

import warnings
import textwrap
import inspect

from functools import wraps, total_ordering
from pathlib import Path
from enum import Enum

import numpy as np
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

def inverted_iterator(iterator):
    """Create an inverted iterator
    """
    for val in iterator:
        yield 1. / val

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
    """Ready to ..."""

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)

def state(next_state:Enum, state:Enum|None=None):
    def decorator(method):
        if isinstance(method, property):
            method.fset.state = state
            method.fset.next_state = next_state
        else:
            method.state = state
            method.next_state = next_state
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if state is not None and self._state < state:
                # we're not ready to run this step
                print(
                    f"Analysis is in state '{self._state}' but a method of state '{state}' is called"
                )
            else:
                _return = method(self, *args, **kwargs)
                self._state = next_state
        return wrapper
    return decorator

def ok_state(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # Get the name of the method being called
        method_name = method.__name__
        print(f"{method_name=}")
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

class AutoTrackMeta(type):
    def __new__(cls, name, bases, dct):
        cls._method_stage = {}
        cls._method_next_stage = {}
        def prop_change(func, prop_name):
            def inner(self, value):
                func(self, value)
                self._property_update(property_name=prop_name)
            return inner
        def method_wrapper(func, method_name):
            def inner(self, *args, **kwargs):
                # check if the analysis is ready
                missing_params, steps_to_run = self._missing(
                    method_name=method_name
                )
                if not missing_params and not steps_to_run:
                    _return= func(self, *args, **kwargs)
                else:
                    print(f"{missing_params=}")
                    print(f"{steps_to_run=}")
                    # provide info about what it missing
                    _return = None
                return _return
            return inner
        for key, value in dct.items():
            print(f"{key=} - {value=}")
            if isinstance(value, property):
                setter = value.fset
                if setter:
                    cls._method_stage[setter] = getattr(setter, 'state', None)
                    cls._method_next_stage[setter] = getattr(setter, 'next_state', None)
                    value = property(value.fget, prop_change(setter, key))
                dct[key] = value
            elif callable(value):
                process_methods = getattr(cls, 'process_methods', [])
                if key in process_methods:
                    cls._method_stage[value] = getattr(value, 'state', None)
                    cls._method_next_stage[value] = getattr(value, 'next_state', None)
                    dct[key] = method_wrapper(value, False, is_special=True)
        # TODO: YOU ARE HERE! THIS SHOULD NOT RETURN ONE FOR t_start!
        print(cls._method_stage)
        print(cls._method_next_stage)
        return super().__new__(cls, name, bases, dct)

class FlowStability(metaclass=AutoTrackMeta):
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
    # This holds a list of all methods relevant to the flow stability analysis
    process_methods = [
        'compute_laplacian_matrices',
        'compute_inter_transition_matrices',
    ]
    state_parameters = {
        States.INITIAL: [],
        States.TEMP_NW: [],
    }
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
        **kwargs : Any
            Optional keyword arguments to pass to `pandas.read_csv()` when 
            loading from a file.
        """
        # this variable track the progress of the analysis - enum would be good
        self._init_state_map()
        # Prepare all property relate information
        self._init_info()
        # Set what attributes are required when
        self._init_required_next()
        # ###
        # Init internal objects
        # - temporal_network data
        self.temporal_network = temporal_network
        self.t_start = t_start
        self.t_stop = t_stop
        self.time_scale = time_scale

    @include_doc_from(ContTempNetwork)
    @property
    def temporal_network(self):
        """
        """
        return self._temporal_network

    @temporal_network.setter
    def temporal_network(self, temporal_network:ContTempNetwork|None):
        """Set the temporal network data
        """
        _unset = 0
        if isinstance(temporal_network, ContTempNetwork):
            self._temporal_network = temporal_network
        else:
            logger.warning(
                f"Object of type {type(temporal_network)} cannot be "
                "used as temporal network. Setting `temporal_network` "
                "attribute to `None` and resetting the state of the analysis."
            )
            self._temporal_network = None
            _unset = 1
        return _unset

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
        return self._time_scale

    @time_scale.setter
    def time_scale(self, time_scale:None|Iterator|int|float):
        """Set the time scale determining the random walks transition rate.

        """
        if time_scale is None:
            self._time_scale = iter([])
        elif isinstance(time_scale, (int, float)):
            self._time_scale = iter([time_scale])
        elif isinstance(time_scale, Iterator):
            self._time_scale = iter(time_scale)
        else:
            raise TypeError(f"Invalid type '{type(time_scale)}'.")
        # INFO: lamda is going to be removed in future releases
        self._set_lamda()
        # TODO: set state
        print(inspect.currentframe().f_code.co_name)

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
            self.time_scale = np.nditer(np.linspace(**kwargs))
        else:
            # TODO: Use the median of the inter event times
            self.time_scale = None
        return None

    def _set_lamda(self):
        """
        """
        self._lamda = inverted_iterator(iterator=self.time_scale)


    @property
    def lamda(self):
        """
        """
        return self._lamda

    @lamda.setter
    def lamda(self, value:int|float|Iterator|None=None):
        """Setting the random walk rate.

        This method will be removed in favor of `time_scale`.
        """
        warnings.warn(
            "The parameter `lamda` is deprecated and will removed in "
            "future releases!\nTo set the random walk rate use the "
            "`time_scale` variable instead, which is the inverse of `lamda`."
        )
        if value is None:
            _lamda_iter = iter([])
        elif isinstance(value, (int, float)):
            _lamda_iter = iter([value,])
        elif isinstance(value, Iterator):
            _lamda_iter = iter(value)
        else:
            raise TypeError(f"Invalid type '{type(value)}'")

        self._time_scale = inverted_iterator(iterator=_lamda_iter)

    @property
    def t_start(self):
        """Start time to calculate the Lapalcian matrices from.

        The laplacian matrices will be calculated form the first event time
        before or equal to this timepoint.
        """
        return self.t_start

    @state(next_state=States.TEMP_NW)
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
        return self.t_start

    @t_stop.setter
    def t_stop(self, value:int|float|None):
        """Set the stop time for the temporal network data to include.

        .. note::
            When setting this value, the Laplaican matrices will be calculated
            anew.
        """
        self._t_stop = value

    def _get_needed_params(method_name:str)->list:
        """Provide a list of required parameters for running a method
        """
        compute_laplacian_matrices = []

        

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

    def _missing(self, method_name:str)->tuple[list,list]:
        """Compute a list of parameters and methods to set/run beforehand
        """
        # What parameters are required to run this method
        needed_params = self._get_needed_params(method_name=method_name)
        # Which one of these parameters are not yet set?
        missing_params = self._get_missing_params(params=needed_params)

        # Are we in the state in which this method can be run?
        required_state = self._get_needed_state(method_name=method_name)
        
        return missing_params, steps_to_run

    def _ready_for(self, method_name:str):
        """Check if the analysis is all set for running this method.
        """

        #TODO: switch for using the method name
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

    @include_doc_from(ContTempNetwork.compute_laplacian_matrices)
    @state(state=States.TEMP_NW, next_state=States.LAPLAC)
    def compute_laplacian_matrices(self, *args, **kwargs):
        """
        """
        self._temporal_network.compute_laplacian_matrices(*args,
                                                            **kwargs)
        return self

    @include_doc_from(ContTempNetwork.compute_inter_transition_matrices)
    @state(state=States.LAPLAC, next_state=States.INTER_T)
    # @ok_state
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

    def _init_state_map(self,):
        """Initialize the mapping between properties and states of the analysis
        """
        # Set the analysis to the initial state
        self._state = States.INITIAL
        # Define a mapping from property to the state at which it is relevant
        self._affect_after = dict(
            temporal_network=States.TEMP_NW,
            t_start=States.TEMP_NW,
            t_stop=States.TEMP_NW,
            time_scale=States.LAPLAC,
        )

    def _property_update(self, property_name:str):
        """Synchronising the state of the analysis with the changed parameter

        This method makes sure that any change in a parameter leads the analysis
        to be set back to the earliest state at which the changed parameter is
        of relevance.
        """
        logger.debug(
            f"Setting property `{property_name}`"
        )
        self._state = min(self._state,
                            self._affect_after[property_name])

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

