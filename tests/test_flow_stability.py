import pytest
import logging
import numpy as np
from flowstab import FlowStability, set_log_level

@pytest.fixture
def minimal_temp_network():
    # This fixture should return a ContTempNetwork instance as expected by FlowStability.
    # You may need to adjust this if ContTempNetwork needs specific arguments.
    from flowstab.temporal_network import ContTempNetwork
    import pandas as pd
    df = pd.DataFrame({"source_nodes": [0],
                       "target_nodes": [1],
                       "starting_times": [0],
                       "ending_times": [1]})
    return ContTempNetwork(events_table=df)

def test_empty_init_sets_temporal_network_none():
    fs = FlowStability()
    assert fs.temporal_network is None

def test_init_with_network(minimal_temp_network):
    fs = FlowStability(temporal_network=minimal_temp_network)
    assert fs.temporal_network is minimal_temp_network

def test_set_temporal_network(minimal_temp_network):
    fs = FlowStability()
    fs.temporal_network = minimal_temp_network
    assert fs.temporal_network is minimal_temp_network

def test_set_temporal_network_wrong_type_logs_warning(caplog):
    fs = FlowStability()
    with caplog.at_level("WARNING"):
        fs.temporal_network = 12345  # Should trigger warning and set None
    assert fs.temporal_network is None
    assert "cannot be used as temporal network" in caplog.text

def test_set_time_scale_and_getter():
    fs = FlowStability()
    fs.time_scale = 2.0
    # Should return an iterator with the single value
    vals = list(fs.time_scale)
    assert vals == [2.0]

def test_set_time_scale_with_list():
    fs = FlowStability()
    fs.time_scale = iter([1.0, 2.0])
    vals = list(fs.time_scale)
    assert vals == [1.0, 2.0]

def test_set_time_scale_invalid_type():
    fs = FlowStability()
    with pytest.raises(TypeError):
        fs.time_scale = "invalid"

def test_set_time_scale_with_set_time_scale_method():
    fs = FlowStability()
    fs.set_time_scale(value=4.0)
    vals = list(fs.time_scale)
    assert vals == [4.0]

def test_properties_t_start_t_stop_time_direction():
    fs = FlowStability()
    fs.t_start = 1.5
    assert fs.t_start == 1.5
    fs.t_stop = 5.0
    assert fs.t_stop == 5.0
    fs.time_direction = -1
    assert fs.time_direction == -1
    with pytest.raises(AssertionError):
        fs.time_direction = 2  # invalid, must be -1, 0, or 1

def test_set_temporal_network_method(minimal_temp_network, caplog):

    set_log_level("INFO")
    with caplog.at_level(logging.INFO):
        fs = FlowStability()
        fs.set_temporal_network(events_table=minimal_temp_network.events_table)
        print(caplog.text)
    assert isinstance(fs.temporal_network, type(minimal_temp_network))

# You may add more advanced integration tests if you have the full dependency chain (e.g., test compute_laplacian_matrices)
