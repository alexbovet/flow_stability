import pytest

import os
import pickle
import tempfile

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix


from flowstab.temporal_network import ContTempNetwork

class TestContTempNetwork:
    def setup_method(self):
        self.source_nodes = [0, 1]
        self.target_nodes = [1, 2]
        self.starting_times = [0.5, 1.0]
        self.ending_times = [1.0, 1.5]

        self.extra_attrs = {"attr1": [True, False]}
        self.events_table = pd.DataFrame({
            "source_nodes": self.source_nodes,
            "target_nodes": self.target_nodes,
            "starting_times": self.starting_times,
            "ending_times": self.ending_times
        })
        self.temp_dir = tempfile.gettempdir()
        self.tmp_pkl = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        self.tmp_json = tempfile.NamedTemporaryFile(suffix='.json',
                                                    delete=False)

    def teardown_method(self):
        temp_dir = tempfile.gettempdir()
        for file in os.listdir(temp_dir):
            if 'temp.pkl' in file or 'temp.json' in file:
                os.remove(os.path.join(temp_dir, file))

    def test_init_with_events_table(self):
        network = ContTempNetwork(events_table=self.events_table)
        assert network.events_table.equals(self.events_table)

    def test_init_with_source_and_target_nodes(self):
        network = ContTempNetwork(source_nodes=self.source_nodes,
                                  target_nodes=self.target_nodes,
                                  starting_times=self.starting_times,
                                  ending_times=self.ending_times)
        assert isinstance(network, ContTempNetwork)

    def test_init_without_source_nodes(self):
        with pytest.raises(AssertionError):
            ContTempNetwork(target_nodes=[1, 2])

    def test_init_without_target_nodes(self):
        with pytest.raises(AssertionError):
            ContTempNetwork(source_nodes=[0, 1])

    def test_init_with_invalid_source_node_type(self):
        with pytest.raises(AssertionError):
            ContTempNetwork(source_nodes=['a', 'b'], target_nodes=[1, 2])

    def test_init_with_invalid_target_node_type(self):
        with pytest.raises(AssertionError):
            ContTempNetwork(source_nodes=[0, 1], target_nodes=['a', 'b'])

    def test_relabel_nodes(self):
        # create a network without relabeling nodes
        network = ContTempNetwork(source_nodes=[1, 2], target_nodes=[2, 3],
                                  starting_times=self.starting_times,
                                  ending_times=self.ending_times)
        original_labels = network.node_to_label_dict

        # check that the node labels have been correctly converted to integers
        new_network = ContTempNetwork(events_table=network.events_table,
                                      relabel_nodes=False)
        assert original_labels == {l: n
                                   for l, n in enumerate(sorted([1, 2, 3]))}
        # The new network should not have the correct ids
        np.testing.assert_equal(new_network.node_array,np.array([0,1,2]))

    @pytest.fixture
    def saved_network(self):
        network = ContTempNetwork(source_nodes=self.source_nodes,
                                  target_nodes=self.target_nodes,
                                  starting_times=self.starting_times,
                                  ending_times=self.ending_times)
        with open(self.tmp_pkl.name, 'wb') as f:
            pickle.dump(network, f)
            return network

    @pytest.fixture
    def get_loaded_network(self):
        def loaded_network():
            with open(self.tmp_pkl.name, 'rb') as f:
                return pickle.load(f)
        return loaded_network

    def test_save_and_load_pickle(self, saved_network, get_loaded_network):
        assert isinstance(saved_network, ContTempNetwork)
        loaded_network = get_loaded_network()
        assert isinstance(loaded_network, ContTempNetwork)
        sn_et = saved_network.events_table
        ln_et = loaded_network.events_table
        pd.testing.assert_series_equal(sn_et.source_nodes, ln_et.source_nodes)
        pd.testing.assert_series_equal(sn_et.target_nodes, ln_et.target_nodes)
        pd.testing.assert_series_equal(sn_et.starting_times, ln_et.starting_times)
        pd.testing.assert_series_equal(sn_et.ending_times, ln_et.ending_times)

    def test_relabel_nodes_with_custom_labels(self):
        custom_labels = {"node0": 3, "node1": 4}
        network = ContTempNetwork(source_nodes=["node0", "node1"],
                                  target_nodes=["node1", "node2"],
                                  starting_times=self.starting_times,
                                  ending_times=self.ending_times,
                                  node_to_label_dict=custom_labels)
        new_network = ContTempNetwork(events_table=network.events_table)
        nw_node_labels = set(new_network.node_to_label_dict.items())
        custom_labels = set(new_network.node_to_label_dict.items())
        assert all(nl in nw_node_labels for nl in custom_labels)

    def test_merge_overlapping_events(self):
        # create a network with overlapping events
        source_nodes = [0, 0]
        target_nodes = [1, 2]
        starting_times = [0.5, 1.0]
        ending_times = [1.0, 1.5]
        extra_attrs = {"attr1": [True, False]}
        events_table = pd.DataFrame({
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
            "starting_times": starting_times,
            "ending_times": ending_times
        })
        network = ContTempNetwork(events_table=events_table,
                                  merge_overlapping_events=True)
        assert network._overlapping_events_merged

    def test_compute_times(self):
        # create a network and check that the compute times dictionary is empty
        network = ContTempNetwork()
        assert not network._compute_times

def test_ContTempNetworkErrors():
    with pytest.raises(AssertionError):
        ContTempNetwork(source_nodes=[0, 1], target_nodes=[1])

    with pytest.raises(Exception):
        ContTempNetwork(events_table=pd.DataFrame({"source_nodes": [0, 1]}))

def test_ContTempInstNetwork():
    """
    """
    from flowstab.temporal_network import ContTempInstNetwork
    pass

def test_lin_approx_trans_matrix():
    """
    """
    from flowstab.temporal_network import lin_approx_trans_matrix
    pass

def test_compute_stationary_transition():
    """
    """
    from flowstab.temporal_network import compute_stationary_transition
    pass

def test_compute_subspace_expm():
    """
    """
    from flowstab.temporal_network import compute_subspace_expm
    pass

def test_csc_row_normalize():
    """
    """
    from flowstab.temporal_network import csc_row_normalize
    pass

def test_find_spectral_gap():
    """
    """
    from flowstab.temporal_network import find_spectral_gap
    pass

def test_remove_nnz_rowcol():
    """
    """
    from flowstab.temporal_network import remove_nnz_rowcol
    pass

def test_rebuild_nnz_rowcol():
    """
    """
    from flowstab.temporal_network import numpy_rebuild_nnz_rowcol
    pass

def test_sparse_lapl_expm():
    """
    """
    from flowstab.temporal_network import sparse_lapl_expm
    pass

def test_sparse_lin_approx():
    """
    """
    from flowstab.temporal_network import sparse_lin_approx
    pass

def test_sparse_stationary_trans():
    """
    """
    from flowstab.temporal_network import sparse_stationary_trans
    pass

def test_set_to_ones():
    """
    """
    from flowstab.temporal_network import set_to_ones
    pass

def test_set_to_zeroes():
    """
    """
    from flowstab.temporal_network import set_to_zeroes
    pass
