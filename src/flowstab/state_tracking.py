"""
State tracking utilities for a multi-step analysis

This sub-module implements a state machine system for analysis classes,
enabling fine-grained tracking of computation progress through method and
property transitions. It provides a metaclass (`StateMeta`) and decorator
(`register`) that augment classes with state management, enforcing method
order and prerequisite satisfaction based on user-defined enumerated states.

Usage Example
-------------
To use state tracking in a custom analysis class:

    from enum import Enum
    from flowstab.state_tracking import StateMeta, OrderedEnum

    class MyStates(OrderedEnum):
        INIT = 0
        STEP1 = 1
        STEP2 = 2

    class MyAnalysis(metaclass=StateMeta, states=MyStates):
        @property
        def data(self):
            '''Access the data attribute.'''
            return self._data

        @register(next_state=MyStates.STEP1)
        @data.setter
        def data(self, value):
            '''Set the data attribute.'''
            self._data = value

        @register(minimal_state=MyStates.INIT, next_state=MyStates.STEP1)
        def do_step_1(self):
            '''Perform analysis step requiring data.'''
            pass

        @register(minimal_state=MyStates.STEP1, next_state=MyStates.STEP2)
        def do_step_2(self):
            '''Perform analysis step requiring data.'''
            pass

This setup ensures:

- method `do_step_1` is run first
- whenever `data` is set, `run_step_2` must be re-run
- `run_step_2` can only be run after `run_step_1` was run once and after `data`
  is set.
"""
from __future__ import annotations
import textwrap
import warnings

from functools import total_ordering
from copy import copy
from types import SimpleNamespace
from enum import Enum
from functools import wraps

@total_ordering
class OrderedEnum(Enum):
    """
    Enum with total ordering based on values.

    Allows comparison operators (<, <=, >, >=, ==, !=) between enum members,
    using their assigned values.
    """
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value == other.value
        return NotImplemented

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return f"{self.name} ({self.value})"

class State:
    """
    State machine for managing analysis workflow progress.

    Holds information about the current state, required properties, and
    required methods. Used internally by classes using StateMeta.
    """
    def __init__(self,
                 states:OrderedEnum,
                 properties_required:dict|None=None,
                 properties_set:dict|None=None,
                 methods_required:dict|None=None,
                 **kwargs
                 ):
        """
        Initialize the State object.

        Parameters
        ----------
        states : Enum
            Enumeration of possible states.
        properties_required : dict or None, optional
            Mapping of state to required property names.
        properties_set : dict or None, optional
            Mapping of property names to their set status.
        methods_required : dict or None, optional
            Mapping of state to required method names.
        **kwargs
            Additional attributes to set.
        """
        self._states = states
        self.current = min(self._states, key=lambda state: state.value)
        self.properties_required = copy(properties_required)
        self.methods_required = copy(methods_required)
        self.properties_set = copy(properties_set)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def missing(self):
        """
        List of required properties not yet set for the current state.

        Returns
        -------
        list
            Property names still required for the current state.
        """
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
        """
        Get required properties still missing and the next method for the state.

        Returns
        -------
        tuple
            (list of missing properties, next method name or None)
        """
        _to_run = self._next_method.get(self.current)
        _to_set = self.missing
        return _to_set, _to_run
        

class StateMeta(type):
    """
    Metaclass for state-tracked analysis classes.

    This metaclass augments target classes with state management, attaching
    state tracking attributes and logic based on user-defined states and
    decorators.
    """
    def __new__(cls, name, bases, namespace, states:Enum):
        """
        Create a new class with state tracking.

        Parameters
        ----------
        name : str
            Class name.
        bases : tuple
            Base classes.
        namespace : dict
            Class namespace.
        states : Enum
            Enum defining possible states.
        """
        cls.init_state(namespace=namespace, states=states)

        cls_instance = super().__new__(cls, name, bases, namespace)

        cls_instance.attach_state(namespace=namespace)

        return cls_instance
    
    @classmethod
    def init_state(cls, namespace, states:Enum):
        """
        Populate the given namespace with state tracking metadata.

        Parameters
        ----------
        namespace : dict
            The class namespace to mutate.
        states : Enum
            Enum of possible states.
        """
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
        Attach the state property and custom __init__ to the class.

        Parameters
        ----------
        namespace : dict
            The class namespace to mutate.
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
        """
        Decorator to register a method or property setter for state management.

        Parameters
        ----------
        next_state : Enum
            State the object will be in after successful call.
        minimal_state : Enum or None, optional
            Minimal state required to execute the method/setter.
        ignore_none : bool, optional
            If True, value=None will mark the property unset.

        Returns
        -------
        function
            Decorated method or property setter.
        """
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
