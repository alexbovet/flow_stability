import pytest
from enum import Enum
from src.flowstab.state_tracking import StateMeta
from functools import total_ordering

@total_ordering
class MyStates(Enum):
    INIT = 0
    PROP_SET = 1
    METHOD_RUN = 2

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return f"{self.name} ({self.value})"

register = StateMeta.register  # make register method available as decorator

class MyClass(metaclass=StateMeta, states=MyStates):
    def __init__(self):
        self._value = None

    @property
    def value(self):
        return self._value

    @register(next_state=MyStates.PROP_SET)
    @value.setter
    def value(self, v):
        self._value = v

    @register(next_state=MyStates.METHOD_RUN, minimal_state=MyStates.PROP_SET)
    def run(self):
        """Run some process after value is set"""
        self._ran = True

@pytest.fixture(scope='function')
def obj():
    return MyClass()

def test_initial_state(obj):
    assert obj.state.current == MyStates.INIT
    assert not obj.state.properties_set['value']
    assert 'value' in obj.state.properties_required[MyStates.PROP_SET]

def test_set_property_and_state(obj):
    """Setting a property sets to the next_state if current is more advanced
    """
    obj.value = 10
    assert obj.state.current == MyStates.INIT
    obj.state.current = MyStates.METHOD_RUN
    obj.value = 10
    assert obj.state.current == MyStates.PROP_SET
    assert obj.state.properties_set['value']

def test_missing_property_blocks_method(obj):
    # run() requires value to be set
    with pytest.warns(UserWarning,
                      match="can only be run in state"):
        obj.run()
    # set the state requried for run() but don't set the property
    obj.state.current = MyStates.PROP_SET
    with pytest.warns(UserWarning,
                      match="You need to set the following attributes"):
        obj.run()
    obj.value = 10
    # Should not advance state
    assert obj.state.current != MyStates.METHOD_RUN
    obj.run()
    assert obj.state.current == MyStates.METHOD_RUN

def test_method_advances_state(obj):
    obj.value = 10
    obj.state.current = MyStates.PROP_SET
    obj.run()
    assert obj.state.current == MyStates.METHOD_RUN

def test_missing_properties(obj):
    # At INIT, value is required for PROP_SET
    assert obj.state.missing == []
    obj.state.current = MyStates.PROP_SET
    assert obj.state.missing == ['value']

def test_next_property_and_method(obj):
    assert obj.state.next == ([], None)
    obj.state.current = MyStates.PROP_SET
    assert obj.state.next == (['value',], 'run')
    obj.value = 42
    assert obj.state.next == ([], 'run')
