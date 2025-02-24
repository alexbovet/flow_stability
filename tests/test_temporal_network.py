import pytest

import os
import pickle
import tempfile

from types import SimpleNamespace
from copy import copy

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix


from flowstab.temporal_network import ContTempNetwork, ContTempInstNetwork

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
        self.minimal.ending_times = [1.5, 2.0]
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
        self.minimal_instant.events_table = self.minimal.events_table.drop(
            ContTempNetwork._ENDINGS, axis=1
        )

        # create a simple network
        self.simple = SimpleNamespace()
        # we assume 10 nodes, and each starting a connection in order
        self.simple.source_nodes = list(range(1, 11))
        # target nodes are also in order
        self.simple.target_nodes = list(range(2,11)) + [1]
        self.simple.starting_times = [0, 0.5, 1, 2, 3, 4, 4, 5, 5, 5]
        self.simple.ending_times =   [3, 1, 2, 7, 4, 5, 6, 6, 6, 7]
        self.simple.events_table = self._to_df(self.simple)
        self.simple.nodes = self._get_nodes(self.simple)
        self.simple.node_label_id_map = self._get_label_id_map(self.simple)
        self.simple.tmp_pkl = tempfile.NamedTemporaryFile(suffix='.pkl',
                                                          delete=False)
        self.simple.tmp_json = tempfile.NamedTemporaryFile(suffix='.json',
                                                    delete=False)
        self.simple_instant = copy(self.simple)
        del(self.simple_instant.ending_times)
        self.simple_instant.events_table = self.simple.events_table.drop(
            ContTempNetwork._ENDINGS, axis=1
        )
        # ###
        # gather all networks
        self.networks = [self.minimal, self.minimal_instant,
                         self.simple, self.simple_instant]
        self.minimals = [self.minimal, self.minimal_instant]

    @staticmethod
    def _to_df(network: SimpleNamespace):
        """Convert a network from a namespace ot a data frame
        """
        as_df = pd.DataFrame({
            "source_nodes": network.source_nodes,
            "target_nodes": network.target_nodes,
            "starting_times": network.starting_times,
        })
        ending_times = getattr(network, ContTempNetwork._ENDINGS, None)
        if ending_times is not None:
            as_df[ContTempNetwork._ENDINGS] = ending_times
        return as_df

    @staticmethod
    def _get_instance(network: SimpleNamespace, use_df=False, **params):
        if use_df:
            return ContTempNetwork(events_table=network.events_table, **params)
        else:
            return ContTempNetwork(
                source_nodes=network.source_nodes,
                target_nodes=network.target_nodes,
                starting_times=network.starting_times,
                ending_times=getattr(network, ContTempNetwork._ENDINGS, None),
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
            # loading the data frame as is should not change it
            temp_network = ContTempNetwork(
                events_table=network.events_table,
                use_as_is=True)
            assert temp_network._events_table.equals(network.events_table)
            # loading a data frame will overwrite the IDs
            with pytest.raises(AssertionError):
                temp_network = ContTempNetwork(
                    events_table=network.events_table
                )
                assert temp_network._events_table.equals(network.events_table)

    def test_init_with_source_and_target_nodes(self):
        for network in self.networks:
            temp_network = ContTempNetwork(
                source_nodes=network.source_nodes,
                target_nodes=network.target_nodes,
                starting_times=network.starting_times,
                ending_times=getattr(network, ContTempNetwork._ENDINGS, None)
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

    @pytest.fixture
    def saved_network(self):
        def _get_network(network: SimpleNamespace, use_df=False, **params):
            temp_network = self._get_instance(network, use_df=use_df, **params)
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
            # save it wikhout 'use_as_is'
            temp_network = saved_network(network=network, use_df=True)
            # now with used as is
            temp_network_ua = saved_network(network=network, use_df=True,
                                            use_as_is=True)
            assert isinstance(temp_network, ContTempNetwork)
            assert isinstance(temp_network_ua, ContTempNetwork)
            # check if the use as is property is carreid along
            assert not temp_network._use_as_is
            assert temp_network_ua._use_as_is
            # for the converted network, we expect the _events_table to differ
            # essential columns of the internal df
            _mandatory_cols = temp_network._events_table[
                [col for col in temp_network._MANDATORY]
            ]
            _mandatory_cols_ua = temp_network_ua._events_table[
                [col for col in temp_network_ua._MANDATORY]
            ]
            # essential columns of the property df
            mandatory_cols = temp_network.events_table[
                [col for col in temp_network._MANDATORY]
            ]
            mandatory_cols_ua = temp_network_ua.events_table[
                [col for col in temp_network._MANDATORY]
            ]

            nw_et = network.events_table
            # check if the nodes were relabeled as expected
            for colname in [temp_network._SOURCES, temp_network._TARGETS]:
                # the property should match in both cases (use as is and not)
                assert temp_network.events_table[colname].equals(
                    nw_et[colname]
                )
                assert temp_network_ua.events_table[colname].equals(
                    nw_et[colname]
                )
                # for the use as is, also the internal should match
                assert temp_network_ua._events_table[colname].equals(
                    nw_et[colname]
                )
                with pytest.raises(AssertionError):
                    # the internal df should not match if not ua
                    assert temp_network._events_table[colname].equals(
                        nw_et[colname]
                    )

            # first save is again
            temp_network = saved_network(network=network, use_df=False)
            # load the temp network
            loaded_network = get_loaded_network(network=network)
            assert isinstance(loaded_network, ContTempNetwork)
            sn_et = temp_network._events_table
            ln_et = loaded_network._events_table
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

    def test_time_grid(self):
        for network in self.networks:
            temp_network = ContTempNetwork(
                events_table=network.events_table,
            )
            temp_network._compute_time_grid()

    def test_laplacian_computation(self):
        """Check if the instant version is equivalent to the non-instant
        version.
        """

        for network in self.minimals:
            temp_network = ContTempNetwork(
                events_table=network.events_table,
            )
            if not hasattr(network, temp_network._ENDINGS):
                # if we are dealing with an instantaneous network
                print('\n\nis instant!\n\n')
                inst_temp_network = ContTempInstNetwork(
                    events_table=network.events_table,
                )
                # use the method form the child class
                inst_temp_network.compute_laplacian_matrices()
                print(f"{list(map(lambda x: x.toarray(), inst_temp_network.laplacians))=}")
            else:
                # if not instantaneous, simply compute the laplacian
                temp_network.compute_laplacian_matrices()
                print(f"{list(map(lambda x: x.toarray(), temp_network.laplacians))=}")


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
