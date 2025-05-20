import pytest
import numpy as np

from flowstab import FlowStability, set_log_level

from .conftest import get_temporal_network_df_minimal

def test_FlowStability_initiation(caplog, get_temporal_network_df_minimal):
    """
    """
    set_log_level("DEBUG")
    # print info for empty network
    with caplog.at_level('INFO'):
        fs_empty = FlowStability()
    assert 'no data' in  caplog.text.lower()
    assert fs_empty.progress == 0
    caplog.clear()
    minimal_events_table = get_temporal_network_df_minimal
    with caplog.at_level('DEBUG'):
        fs_minimal = FlowStability(temporal_network=minimal_events_table)
    assert 'data frame' in  caplog.text.lower()
    assert fs_minimal.progress == 1

def test_FlowStability_buildup(caplog, get_temporal_network_df_minimal):
    """
    """
    lamda = 0.1
    set_log_level("DEBUG")
    # print info for empty network
    with caplog.at_level('INFO'):
        fs_empty = FlowStability(temporal_network=1)
    assert 'no data' in  caplog.text.lower()
    caplog.clear()
    assert fs_empty.progress == 0
    # adding network
    fs_empty.set_temporal_network(events_table=get_temporal_network_df_minimal)
    assert fs_empty.progress == 1
    fs_empty.compute_laplacian_matrices()
    assert fs_empty.progress == 1.1
    fs_empty.compute_inter_transition_matrices(lamda=lamda)
    assert fs_empty.progress == 1.2
    integral_time_grid = np.linspace(0, 7, 8, endpoint=True)
    fs_empty.set_flow_clustering(integral_time_grid=integral_time_grid)
    # assert fs_empty.
