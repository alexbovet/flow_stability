import pytest
import os
import pickle
import logging

import numpy as np

from flowstab import FlowStability, set_log_level

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

def test_setting_time_scale():
    # set with simple value
    fs = FlowStability()
    fs.time_scale = 2.0
    # Should return an iterator with the single value
    vals = list(fs.time_scale)
    assert vals == [2.0]
    # check the `set_time_scale` method with simple values
    fs = FlowStability()
    fs.set_time_scale(value=4.0)
    vals = list(fs.time_scale)
    assert vals == [4.0]
    # check the `set_time_scale` method whith `np.logspace'
    fs = FlowStability()
    start, stop, num = 1, 100, 20
    fs.set_time_scale(start=start, stop=stop, num=num)
    ref_vals = np.logspace(start=start, stop=stop, num=num)
    vals = np.array(list(fs.time_scale))
    np.testing.assert_allclose(vals, ref_vals)
    fs = FlowStability()
    fs.time_scale = [1.0, 2.0]
    vals = list(fs.time_scale)
    assert vals == [1.0, 2.0]
    # make sure invalid types are detected
    fs = FlowStability()
    with pytest.raises(TypeError):
        fs.time_scale = "invalid"
    # make sure inpossible values are rejected
    fs = FlowStability()
    with pytest.raises(ValueError) as exc_info:
        fs.time_scale = [-1111, 0, 1]
    assert '-1111' in str(exc_info.value)
    assert '0' in str(exc_info.value)

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
    set_log_level("INFO")
    with caplog.at_level(logging.INFO):
        fs = FlowStability()
        fs.set_temporal_network(events_table=minimal_temp_network.events_table)
        print(caplog.text)
    assert isinstance(fs.temporal_network, type(minimal_temp_network))

def test_flow_stability_import_export(minimal_temp_network, caplog, tmp_path):
        fs = FlowStability()
        fs.set_temporal_network(events_table=minimal_temp_network.events_table)
        # hash the object
        assert hash(fs)
        # Export the fs to a pickle
        with open(os.path.join(tmp_path, 'FS_minimal.p'), 'wb') as fobj:
            pickle.dump(fs, fobj)
        # Load it
        with open(os.path.join(tmp_path, 'FS_minimal.p'), 'rb') as fobj:
            fs1 = pickle.load(fobj)
        assert hash(fs1)

def test_fs_help(capsys, minimal_temp_network):
    fs = FlowStability(temporal_network=minimal_temp_network)
    fs.help()
    captured = capsys.readouterr()
    assert 'next steps' in captured.out.lower()
    fs.help('set_temporal_network')
    captured = capsys.readouterr()
    assert 'some details about' in captured.out.lower()
