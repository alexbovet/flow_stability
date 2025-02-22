import pytest

import os
import pickle
import tempfile

from types import SimpleNamespace
from copy import copy

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix


from flowstab.temporal_network import ContTempNetwork

class TestTempNetwork:
    def setup_method(self):
        # ###
        # folder setup
        self.temp_dir = tempfile.gettempdir()
        self.tmp_json = tempfile.NamedTemporaryFile(suffix='.json',
                                                    delete=False)
        # ###
        # Test data
        # create a minimmal network
        self.minimal = SimpleNamespace()
        self.minimal.source_nodes = [1, 2]
        self.minimal.target_nodes = [2, 3]
        self.minimal.starting_times = [0.5, 1.0]
        self.minimal.ending_times = [1.0, 1.5]
        self.minimal.extra_attrs = {"attr1": [True, False]}
        self.minimal.events_table = self._to_df(self.minimal)
        self.minimal.nodes = self._get_nodes(self.minimal)
        self.minimal.node_label_id_map = self._get_label_id_map(self.minimal)
        self.minimal.tmp_pkl = tempfile.NamedTemporaryFile(suffix='.pkl',
                                                           delete=False)
        self.minimal.tmp_json = tempfile.NamedTemporaryFile(suffix='.json',
                                                    delete=False)
        self.minimal_instant = copy(self.minimal)
        del(self.minimal_instant.ending_times)
        self.minimal_instant.events_table.drop(ContTempNetwork._STOPS, axis=1)

        # create a simple network
        self.simple = SimpleNamespace()
        # we assume 10 nodes, and each starting a connection in order
        self.simple.source_nodes = list(range(1, 11))
        # target nodes are also in order
        self.simple.target_nodes = list(range(2,11)) + [1]
        self.simple.starting_times = [0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
        self.simple.ending_times =   [3, 1, 2, 7, 4, 5, 6, 6, 6, 7]
        self.simple.events_table = self._to_df(self.simple)
        self.simple.nodes = self._get_nodes(self.simple)
        self.simple.node_label_id_map = self._get_label_id_map(self.simple)
        self.simple.tmp_pkl = tempfile.NamedTemporaryFile(suffix='.pkl',
                                                          delete=False)
        self.simple.tmp_json = tempfile.NamedTemporaryFile(suffix='.json',
                                                    delete=False)
        self.simple_instant = copy(self.simple)
        del(self.simple.ending_times)
        self.simple.events_table.drop(ContTempNetwork._STOPS, axis=1)
        # ###
        # gather all networks
        self.networks = [self.minimal, self.simple]

    @staticmethod
    def _to_df(network: SimpleNamespace):
        """Convert a network from a namespace ot a data frame
        """
        as_df = pd.DataFrame({
            "source_nodes": network.source_nodes,
            "target_nodes": network.target_nodes,
            "starting_times": network.starting_times,
        })
        ending_times = getattr(network, ContTempNetwork._STOPS, None)
        if ending_times is not None:
            as_df[ContTempNetwork._STOPS] = ending_times
        return as_df

    @staticmethod
    def _get_instance(network: SimpleNamespace, **params):
        return ContTempNetwork(
            source_nodes=network.source_nodes,
            target_nodes=network.target_nodes,
            starting_times=network.starting_times,
            ending_times=getattr(network, ContTempNetwork._STOPS, None),
            **params
        )

    @staticmethod
    def _get_nodes(network: SimpleNamespace):
        """Get a sorted list of nodes
        """
        nodes = set()
        nodes.update(network.source_nodes)
        nodes.update(network.target_nodes)
        return sorted(nodes)

    @staticmethod
    def _get_label_id_map(network: SimpleNamespace):
        """Get the mapping from node labels to internal ID
        """
        return {node: _id for _id, node in enumerate(network.nodes)}

    def teardown_method(self):
        temp_dir = tempfile.gettempdir()
        for file in os.listdir(temp_dir):
            if 'temp.pkl' in file or 'temp.json' in file:
                os.remove(os.path.join(temp_dir, file))

    def test_init_with_events_table(self):
        for network in self.networks:
            temp_network = ContTempNetwork(events_table=network.events_table)
            assert temp_network.events_table.equals(network.events_table)

    def test_init_with_source_and_target_nodes(self):
        for network in self.networks:
            temp_network = ContTempNetwork(
                source_nodes=network.source_nodes,
                target_nodes=network.target_nodes,
                starting_times=network.starting_times,
                ending_times=getattr(network, ContTempNetwork._STOPS, None)
            )
            assert isinstance(temp_network, ContTempNetwork)

    def test_init_without_source_nodes(self):
        for network in self.networks:
            with pytest.raises(AssertionError):
                ContTempNetwork(target_nodes=network.target_nodes)

    def test_init_without_target_nodes(self):
        for network in self.networks:
            with pytest.raises(AssertionError):
                ContTempNetwork(source_nodes=network.source_nodes)

    def test_init_inconsistent_event_numbers_type(self):
        """Make sure we detect variable numbers of events
        """
        with pytest.raises(AssertionError):
            # error in source and target nodes
            ContTempNetwork(source_nodes=[1, 2, 3], target_nodes=[1, 2],
                            starting_times = [0, 0], ending_times = [1, 1])
            # not enough ending times
            ContTempNetwork(source_nodes=[1, 2], target_nodes=[1, 2],
                            starting_times = [0, 0], ending_times = [1])

    def test_init_missing_params(self):
        """Make sure we detect variable numbers of events
        """
        with pytest.raises(AssertionError):
            # missing starting times
            ContTempNetwork(source_nodes=[1, 2], target_nodes=[1, 2],)

    def test_init_with_inconsistent_node_type(self):
        with pytest.raises(TypeError):
            # int and str cannot be compared > type error
            ContTempNetwork(source_nodes=[0, 1], target_nodes=['a', 'b'],
                            starting_times = [0, 0], ending_times = [1, 1])

    def test_relabel_nodes(self):
        """
        """
        # create a network without relabeling nodes
        for network in self.networks:
            temp_network = self._get_instance(network, relabel_nodes=True)
            map_to_original_labels = temp_network.node_to_label_dict
            map_labels_to_ids = {
                _id: label for label, _id in map_to_original_labels.items()
            }
            # check that the node labels have been correctly converted to int
            new_temp_network = ContTempNetwork(
                events_table=temp_network.events_table,
                relabel_nodes=False
            )
            assert map_labels_to_ids == network.node_label_id_map
            assert map_to_original_labels == {
                _id: label for _id, label in enumerate(network.nodes)
            }
            # The new network should not have the correct ids (since we did not
            # number them consistently when initiating (minimal and sample)
            with pytest.raises(AssertionError):
                np.testing.assert_equal(new_temp_network.node_array,
                                        np.array(network.nodes))

    @pytest.fixture
    def saved_network(self):
        def _get_network(network: SimpleNamespace):
            temp_network = self._get_instance(network)
            with open(network.tmp_pkl.name, 'wb') as f:
                pickle.dump(temp_network, f)
                return temp_network
        return _get_network

    @pytest.fixture
    def get_loaded_network(self):
        def loaded_network(network):
            with open(network.tmp_pkl.name, 'rb') as f:
                return pickle.load(f)
        return loaded_network

    def test_save_and_load_pickle(self, saved_network, get_loaded_network):
        for network in self.networks:
            temp_network = saved_network(network=network)
            assert isinstance(temp_network, ContTempNetwork)
            loaded_network = get_loaded_network(network=network)
            assert isinstance(loaded_network, ContTempNetwork)
            sn_et = temp_network.events_table
            ln_et = loaded_network.events_table
            pd.testing.assert_series_equal(sn_et.source_nodes,
                                           ln_et.source_nodes)
            pd.testing.assert_series_equal(sn_et.target_nodes,
                                           ln_et.target_nodes)
            pd.testing.assert_series_equal(sn_et.starting_times,
                                           ln_et.starting_times)
            pd.testing.assert_series_equal(sn_et.ending_times,
                                           ln_et.ending_times)

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

    def test_laplacian_computation(self):
        for network in self.networks:
            temp_network = ContTempNetwork(
                events_table=network.events_table,
            )
            temp_network.compute_laplacian_matrices()


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
