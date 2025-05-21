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

import gzip
import os
import pickle
import time
from functools import partial

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import (
    coo_matrix,
    csc_matrix,
    csr_matrix,
    diags,
    dok_matrix,
    eye,
    isspmatrix,
    isspmatrix_csr,
    lil_matrix,
)
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh, expm

from .parallel_expm import compute_subspace_expm_parallel
from .sparse_stoch_mat import inplace_csr_row_normalize, SparseStochMat

from .logger import get_logger

# get the logger
logger = get_logger()

class ContTempNetwork:
    """Continuous time temporal network
    
    Parameters
    ----------
    source_nodes: Python list
        List of source nodes of each event, ordered according to `starting_times`.
    
    target_nodes: Python list
        List of target nodes of each event.
        
    starting_times: Python list
        List of starting times of each event.
        
    ending_times: Python list
        List of ending times of each event.
        
    extra_attrs: Dict
        Extra event attributes. Must be given in a dict with {attr_name: list_of_values},
        where list_of_values has the same order and length as `source_nodes`.
        
    relabel_nodes: boolean
        Relabel nodes from 0 to num_nodes and save original labels in 
        self.node_to_label_dict. Default is `True`
        
    reset_event_table_index: boolean
        Reset the index of the `events_table` DataFrame. Default is `True`.
        
    node_to_label_dict: Python dict
        If `relabel_nodes` is `False, this can be used to save the original labels
        of the nodes.
        
    merge_overlapping_events: boolean
        Check for overlapping events (between the same pair of nodes) 
        and merges them. Default is `False`.
        
    events_table: Pandas Dataframe
        Dataframe with columns 'source_nodes', 'target_nodes', 'starting_times'
        and 'ending_times' and index corresponding to event index. Used for
        instantiating a new ConTempNetwork from the event_table of an other one.
    
    
    """
    # parametrize the column names > single place to change them:
    _SOURCES = "source_nodes"
    _TARGETS = "target_nodes"
    _STARTS = "starting_times"
    _ENDINGS = "ending_times"
    _MANDATORY = [_SOURCES, _TARGETS, _STARTS]
    _ESSENTIAL = [_SOURCES, _TARGETS, _STARTS, _ENDINGS]
    # to hold endings - starts
    _DURATIONS = "durations"
    # for instantaneous event this is the duration to use
    _DEFAULT_DURATION = 1
    def __init__(self, *,
                 source_nodes=[],
                 target_nodes=[],
                 starting_times=[],
                 ending_times=[],
                 extra_attrs=None,
                 relabel_nodes=True,
                 reset_event_table_index=True,
                 node_to_label_dict=None,
                 merge_overlapping_events=False,
                 events_table=None,
                 **kwargs):

        if events_table is None:
            assert len(source_nodes) == len(target_nodes) == \
                   len(starting_times) == len(ending_times)

            if relabel_nodes:
               # relabel nodes from 0 to num_nodes and save
                # original labels in self.node_to_label_dict
                all_nodes = set()
                all_nodes.update(source_nodes)
                all_nodes.update(target_nodes)
                self.label_to_node_dict = {l : n for n, l in enumerate(sorted(all_nodes))}
                self.node_to_label_dict = {n : l for l, n in self.label_to_node_dict.items()}

                source_nodes = [self.label_to_node_dict[n] for n in source_nodes]
                target_nodes = [self.label_to_node_dict[n] for n in target_nodes]
            else:
                self.node_to_label_dict=node_to_label_dict

            data={"source_nodes" : source_nodes,
                                                  "target_nodes" : target_nodes,
                                                  "starting_times" : starting_times,
                                                  "ending_times" : ending_times}
            columns=["source_nodes","target_nodes",
                                             "starting_times","ending_times"]

            if extra_attrs is not None:
                assert isinstance(extra_attrs, dict)

                for attr_name, val_list in extra_attrs.items():
                    assert len(val_list) == len(source_nodes)
                    data[attr_name] = val_list
                    columns.append(attr_name)

            self.events_table = pd.DataFrame(data=data,
                                             columns=columns)

            self.events_table.sort_values(by=["starting_times", "ending_times"],
                                          inplace=True)

        else:
            if isinstance(events_table, (str, Path)):
                try:
                    # Convert Path to string if it's a Path object
                    self.events_table = pd.read_csv(str(events_table), **kwargs)
                    logger.debug("Loading events from csv file.")
                except FileNotFoundError:
                    raise ValueError(
                        f"The file at {events_table} was not found."
                    )
                except pd.errors.EmptyDataError:
                    raise ValueError(
                        f"The file at {events_table} is empty."
                    )
                except pd.errors.ParserError:
                    raise ValueError(
                        f"The file at {events_table} could not be parsed."
                    )
            elif isinstance(events_table, pd.DataFrame):
                self.events_table = events_table
            else:
                raise ValueError(
                    "`events_table` must be a pandas DataFrame or the"
                    "path to a CSV file. "
                    f"'{type(events_table)} is not acceptable."
                )                               
            reset_event_table_index = False     
            if relabel_nodes:                   
                all_nodes = set()
                all_nodes.update(events_table.source_nodes.tolist())
                all_nodes.update(events_table.target_nodes.tolist())
                self.label_to_node_dict = {l : n for n, l in enumerate(sorted(all_nodes))}
                self.node_to_label_dict = {n : l for l, n in self.label_to_node_dict.items()}
            else:
                self.node_to_label_dict=node_to_label_dict


        if reset_event_table_index:
            self.events_table.reset_index(inplace=True, drop=True)

        self.node_array = np.sort(pd.unique(self.events_table[["source_nodes",
                                    "target_nodes"]].values.ravel("K")))

        self.num_nodes = self.node_array.shape[0]

        self.num_events = self.events_table.shape[0]

        self.start_time = self.events_table.starting_times.min()

        self.end_time = self.events_table.ending_times.max()

        self.events_table["durations"] = self.events_table.ending_times - \
                            self.events_table.starting_times

        # to record compute times
        self._compute_times = {}

        self._overlapping_events_merged = False
        if merge_overlapping_events:
            num_merged = 1
            while num_merged != 0:
                num_merged = self._merge_overlapping_events()
            self._overlapping_events_merged = True


        self.is_directed = False
        self.instantaneous_events = False


    def __repr__(self):
        return str(self.__class__) + \
              f" with {self.num_nodes} nodes and {self.num_events} events"

    def save(self, filename,
             matrices_list=None,
             attributes_list=None):
        """Save graph event_table and matrices as pickle file
        
        Parameters
        ----------
        filename: str
            Filename where to save. The extension is automatically added.

        matrices_list: list of strings
            List of names of matrices to save.
            The default list is:
                `matrices_list = ['laplacians',
                                    'adjacencies',
                                    'inter_T',
                                    'inter_T_lin',
                                    'T',
                                    'T_lin',
                                    '_stationary_trans',
                                    'delta_inter_T',
                                    'delta_inter_T_lin',]`
        attributes_list: list of strings
            List of attribute names to save.
            The default list is:
                `attributes_list = ['node_to_label_dict',
                                  'events_table',
                                  'times',
                                  'time_grid',
                                  'num_nodes',
                                  '_compute_times',
                                  '_t_start_laplacians',
                                  '_k_start_laplacians',
                                  '_t_stop_laplacians',
                                  '_k_stop_laplacians',
                                  '_overlapping_events_merged',]`
                                    
            
        """
        save_dict = dict()


        matrices = ["laplacians",
                      "adjacencies",
                      "inter_T",
                      "inter_T_lin",
                      "T",
                      "T_lin",
                      "_stationary_trans",
                      "delta_inter_T",
                      "delta_inter_T_lin"]

        if matrices_list is None:
            matrices_list = matrices

        attributes = ["node_to_label_dict",
                      "events_table",
                      "times",
                      "time_grid",
                      "num_nodes",
                      "_compute_times",
                      "_t_start_laplacians",
                      "_k_start_laplacians",
                      "_t_stop_laplacians",
                      "_k_stop_laplacians",
                      "_overlapping_events_merged",
                      "is_directed"]

        if attributes_list is None:
            attributes_list = attributes

        for attr in attributes_list:
            if hasattr(self, attr):
                save_dict[attr] = getattr(self,attr)


        for mat in matrices_list:
            if hasattr(self, mat):
                save_dict[mat] = getattr(self,mat)


        with open(os.path.splitext(filename)[0] + ".pickle", "wb") as fopen:
            pickle.dump(save_dict, fopen)


    @classmethod
    def load(cls, filename,
             matrices_list=None,
             attributes_list=None):
        """Load network from file and returns a ContTempNetwork instance.
        
        Parameters
        ----------
        filename: str
            Filename where the network is saved save. The extension is automatically added.

        matrices_list: list of strings
            List of names of matrices to load.
            The default list is:
                `matrices_list = ['laplacians',
                                    'adjacencies',
                                    'inter_T',
                                    'inter_T_lin',
                                    'T',
                                    'T_lin',
                                    '_stationary_trans',
                                    'delta_inter_T',
                                    'delta_inter_T_lin',]`
        attributes_list: list of strings
            List of attribute names to load.
            The default list is:
                `attributes_list = ['node_to_label_dict',
                                  'events_table',
                                  'times',
                                  'time_grid',
                                  'num_nodes',
                                  '_compute_times',
                                  '_t_start_laplacians',
                                  '_k_start_laplacians',
                                  '_t_stop_laplacians',
                                  '_k_stop_laplacians',
                                  '_overlapping_events_merged',]`
                                    
            
            
        """
        matrices = ["laplacians",
                      "adjacencies",
                      "inter_T",
                      "inter_T_lin",
                      "T",
                      "T_lin",
                      "_stationary_trans",
                      "delta_inter_T",
                      "delta_inter_T_lin"]

        if matrices_list is None:
            matrices_list = matrices

        attributes = ["node_to_label_dict",
                      "events_table",
                      "times",
                      "time_grid",
                      "num_nodes",
                      "_compute_times",
                      "_t_start_laplacians",
                      "_k_start_laplacians",
                      "_t_stop_laplacians",
                      "_k_stop_laplacians",
                      "_overlapping_events_merged",
                      "is_directed"]

        if attributes_list is None:
            attributes_list = attributes


        # all in a pickle file

        graph_dict = pd.read_pickle(os.path.splitext(filename)[0] +".pickle")

        events_table = graph_dict.pop("events_table")

        net = cls(events_table=events_table,
                           relabel_nodes=False,
                           node_to_label_dict=graph_dict.pop("node_to_label_dict"))

        for k,val in graph_dict.items():
            if k in matrices_list:
                setattr(net, k, val)
            if k in attributes_list:
                setattr(net, k, val)

        return net


    def save_inter_T(self, filename, lamda=None, round_zeros=True, tol=1e-8,
                                   compressed=False,
                                   save_delta=False,
                                   replace_existing=False):
        """ 
        
            
        Saves all the inter transition matrices in `self.inter_T[lamda]` 
        in a pickle file togheter  with a dictionary including parameters: 
        `_k_start_laplacians`, `_k_stop_laplacians`, `_t_start_laplacians`,
        `_t_stop_laplacians`, `_t_stop_laplacians`, `times_k_start_to_k_stop+1` 
        (= self.times.values[self._k_start_laplacians:self._k_stop_laplacians+1])
        `num_nodes` and `_compute_times`.
            
        if `save_delta` is True, creates delta_inter_T if it is 
        not already present and saves it together with 
        inter_T[lamda][0] in a pickle file.
        otherwise, saves inter_T[lamda] directly (good if used with
        SparseStochMat instances).
            
        Parameters
        ----------
            filename: str
            Filename where the data is saved (.pickle or .gz).
        lamda: float or int, optional.
            Use to save to results for a specific lamba. If None, the results
            for all the computed lambdas will be saved. Default is None.
        round_zeros: bool.
            If True, values smaller than tol*max(abs(inter_T_k)) will be set to zero
            to preserve sparsity. Default is True.
        tol: float
            See round_zeros. Default is 1e-8.
        compressed: bool
            Used to compress the file with gzip. Default is False.
        save_delta: bool
            If True, creates delta_inter_T if it is 
            not already present and saves it together with 
            inter_T[lamda][0]. Only the differences between two consecutives
            inter_Ts are saved in order to minimize file size. 
            Must not be used if `use_sparse_stoch` was used in `compute_inter_transition_matrices`.
        replace_existing: bool
            If True, erase and replace files if they already exists. 
            Default is False.
            
        Returns
        -------
            None
            
                        
        """
        assert hasattr(self, "inter_T"), f"PID {os.getpid()} : nothing saved, compute inter_T first."


        ext = os.path.splitext(filename)[-1]

        file = filename

        if compressed:
            if ext != ".gz":
                file += ".gz"
        elif ext != ".pickle":
            file += ".pickle"

        skipping = False
        if os.path.exists(file):
            if replace_existing:
                print("PID ", os.getpid(), f" : , file {file} already exists, replacing it.")
            else:
                print("PID ", os.getpid(), f" : , file {file} already exists, skipping.")
                skipping = True

        if not skipping:


            save_dict = {}
            save_dict["_k_start_laplacians"] = self._k_start_laplacians
            save_dict["_k_stop_laplacians"] = self._k_stop_laplacians
            save_dict["_t_start_laplacians"] = self._t_start_laplacians
            save_dict["_t_stop_laplacians"] = self._t_stop_laplacians
            save_dict["times_k_start_to_k_stop+1"] = self.times.values[\
                                self._k_start_laplacians:self._k_stop_laplacians+1]
            save_dict["num_nodes"] = self.num_nodes
            save_dict["_compute_times"] = self._compute_times

            if save_delta:
                assert not isinstance(self.inter_T[lamda][0],
                                  SparseStochMat),\
                                "inter_T must not be SparseStochMat"

                if not hasattr(self, "delta_inter_T") and \
                                    hasattr(self, "inter_T"):
                    if lamda is not None:
                        self._compute_delta_trans_mat(lamda, round_zeros=round_zeros,
                                                      tol=tol)
                    else:
                        #computes it for all lamda
                        for lamda in self.inter_T.keys():
                            self._compute_delta_trans_mat(lamda, round_zeros=round_zeros,
                                                      tol=tol)

                if (lamda is not None) and (lamda not in self.delta_inter_T.keys()):
                    self._compute_delta_trans_mat(lamda, round_zeros=round_zeros,
                                                      tol=tol)

                if hasattr(self, "delta_inter_T"):
                    save_dict["inter_T"] = dict()
                    save_dict["is_delta_trans"] = True

                    if lamda is None:
                        lamdas = self.delta_inter_T.keys()
                    else:
                        lamdas = [lamda]

                    for lamda in lamdas:
                        save_dict["inter_T"][lamda] = dict()
                        save_dict["inter_T"][lamda]["delta_inter_T"] = self.delta_inter_T[lamda]
                        save_dict["inter_T"][lamda]["trans_mat0"] = self.inter_T[lamda][0].copy()
                        if round_zeros:
                            set_to_zeroes(save_dict["inter_T"][lamda]["trans_mat0"],
                                          tol=tol)

                text = "delta trans mats"

            else:

                save_dict["inter_T"] = dict()
                save_dict["is_sparse_stoch"] = True

                if lamda is None:
                    lamdas = self.inter_T.keys()
                else:
                    lamdas = [lamda]

                for lamda in lamdas:
                    assert isinstance(self.inter_T[lamda][0],
                                      SparseStochMat),\
                                    "inter_T needs to be SparseStochMat"

                    save_dict["inter_T"][lamda] = []
                    for interT in self.inter_T[lamda]:
                        if round_zeros:
                            interT.set_to_zeroes(tol)
                        save_dict["inter_T"][lamda].append(interT.to_dict())

                text = "sparse stoch trans mats"


            if compressed:

                print("PID ", os.getpid(), " : "," saving " + text + " to " + file)

                with gzip.open(file,
                               "wb", compresslevel=2) as fopen:
                    pickle.dump(save_dict, fopen)
            else:

                print("PID ", os.getpid(), " : "," saving " + text + " to " + file)

                with open(file, "wb") as fopen:
                    pickle.dump(save_dict, fopen)




    def save_inter_T_lin(self, filename, lamda=None, round_zeros=True, tol=1e-8,
                                   compressed=False,
                                   replace_existing=True,
                                   save_delta=False):
        """Creates delta_inter_T_lin if it is not already present and
        saves it together with inter_T_lin[lamda][t_s][0] in a pickle file.
                        
        """
        assert hasattr(self, "inter_T_lin"), f"PID {os.getpid()} : nothing saved, compute inter_T_lin first."


        ext = os.path.splitext(filename)[-1]

        file = filename

        if compressed:
            if ext != ".gz":
                file += ".gz"
        elif ext != ".pickle":
            file += ".pickle"

        skipping = False
        if os.path.exists(file):
            if replace_existing:
                print("PID ", os.getpid(), f" : , file {file} already exists, replacing it.")
            else:
                print("PID ", os.getpid(), f" : , file {file} already exists, skipping.")
                skipping = True

        if not skipping:

            save_dict = {}
            save_dict["_k_start_laplacians"] = self._k_start_laplacians
            save_dict["_k_stop_laplacians"] = self._k_stop_laplacians
            save_dict["_t_start_laplacians"] = self._t_start_laplacians
            save_dict["_t_stop_laplacians"] = self._t_stop_laplacians
            save_dict["times_k_start_to_k_stop+1"] = self.times.values[\
                                self._k_start_laplacians:self._k_stop_laplacians+1]

            save_dict["num_nodes"] = self.num_nodes
            save_dict["_compute_times"] = self._compute_times

            if save_delta:
                if not hasattr(self, "delta_inter_T_lin") and \
                                    hasattr(self, "inter_T_lin"):
                    if lamda is not None:
                        self._compute_delta_trans_mat(lamda, round_zeros=round_zeros,
                                                      tol=tol)
                    else:
                        #computes it for all lamda
                        for lamda in self.inter_T_lin.keys():
                            self._compute_delta_trans_mat(lamda, round_zeros=round_zeros,
                                                      tol=tol)

                if (lamda is not None) and (lamda not in self.delta_inter_T_lin.keys()):
                    self._compute_delta_trans_mat(lamda, round_zeros=round_zeros,
                                                      tol=tol)


                if hasattr(self, "delta_inter_T_lin"):


                    save_dict["inter_T_lin"] = dict()
                    save_dict["is_delta_trans"] = True

                    if lamda is None:
                        lamdas = self.delta_inter_T_lin.keys()
                    else:
                        lamdas = [lamda]

                    for lamda in lamdas:
                        save_dict["inter_T_lin"][lamda] = dict()
                        for t_s in self.inter_T_lin[lamda].keys():
                            save_dict["inter_T_lin"][lamda][t_s] = dict()
                            save_dict["inter_T_lin"][lamda][t_s]["delta_inter_T_lin"] = self.delta_inter_T_lin[lamda][t_s]
                            save_dict["inter_T_lin"][lamda][t_s]["trans_mat_lin0"] = self.inter_T_lin[lamda][t_s][0].copy()
                            if round_zeros:
                                set_to_zeroes(save_dict["inter_T_lin"][lamda][t_s]["trans_mat_lin0"],
                                          tol=tol)

                text = "delta trans mats"

            else:

                save_dict["inter_T_lin"] = dict()
                save_dict["is_sparse_stoch"] = True

                if lamda is None:
                    lamdas = self.delta_inter_T_lin.keys()
                else:
                    lamdas = [lamda]

                for lamda in lamdas:
                    save_dict["inter_T_lin"][lamda] = dict()
                    for t_s in self.inter_T_lin[lamda].keys():
                        assert isinstance(self.inter_T_lin[lamda][t_s][0],
                                          SparseStochMat),\
                                        "inter_T needs to be SparseStochMat"

                        save_dict["inter_T_lin"][lamda][t_s] = []
                        for interT in self.inter_T_lin[lamda][t_s]:
                            if round_zeros:
                                interT.set_to_zeroes(tol=tol)
                            save_dict["inter_T_lin"][lamda][t_s].append(interT.to_dict())

                text = "sparse stoch trans mats"



            if compressed:

                print("PID ", os.getpid(), " : "," saving " + text + " to " + file)

                with gzip.open(file,
                               "wb", compresslevel=2) as fopen:
                    pickle.dump(save_dict, fopen)
            else:

                print("PID ", os.getpid(), " : "," saving " + text + " to " + file)

                with open(file, "wb") as fopen:
                    pickle.dump(save_dict, fopen)


    @staticmethod
    def load_inter_T(filename):
        """Loads inter_T and inter_T_lin from 'filename' that was saved with
        save_inter_T.
            
        returns a dictionary with the inter_T restored.
            
        """
        ext = os.path.splitext(filename)[-1]


        if ext not in [".pickle",".gz"]:
            # detect extension
            if os.path.exists(filename + ".pickle"):
                ext = ".pickle"
                filename += ".pickle"
            elif os.path.exists(filename + ".gz"):
                ext = ".gz"
                filename += ".gz"
            elif os.path.exists(filename + ".pickle.gz"):
                ext = ".pickle.gz"
                filename += ".pickle.gz"
            else:
                raise FileNotFoundError(filename)

        if ext == ".gz" or ext == ".pickle.gz":
            with gzip.open(filename,
                           "rb") as fopen:
                load_dict = pickle.load(fopen)

        else:
            with open(filename, "rb") as fopen:
                load_dict = pickle.load(fopen)

        return_dict = {"_k_start_laplacians" : load_dict["_k_start_laplacians"],
                       "_k_stop_laplacians" : load_dict["_k_stop_laplacians"],
                       "_t_start_laplacians" : load_dict["_t_start_laplacians"],
                       "_t_stop_laplacians" : load_dict["_t_stop_laplacians"],
                       "num_nodes" : load_dict["num_nodes"],
                       "times_k_start_to_k_stop+1" : load_dict["times_k_start_to_k_stop+1"]}

        # rebuild inter_T from delta_inter_T
        if "inter_T" in load_dict.keys():
            return_dict["inter_T"] = dict()

            if load_dict.get("is_sparse_stoch", False):

                for lamda in load_dict["inter_T"].keys():
                    return_dict["inter_T"][lamda] = \
                        [SparseStochMat(**mat_dict) for mat_dict in \
                                          load_dict["inter_T"][lamda]]

            else:
                for lamda in load_dict["inter_T"].keys():
                    return_dict["inter_T"][lamda] = \
                        [load_dict["inter_T"][lamda]["trans_mat0"]]

                    for dT in load_dict["inter_T"][lamda]["delta_inter_T"]:
                        return_dict["inter_T"][lamda].append(\
                                 return_dict["inter_T"][lamda][-1] + dT)


        if "inter_T_lin" in load_dict.keys():
            return_dict["inter_T_lin"] = dict()

            if load_dict.get("is_sparse_stoch", False):

                for lamda in load_dict["inter_T_lin"].keys():
                    return_dict["inter_T_lin"][lamda] = dict()

                    for t_s in load_dict["inter_T_lin"][lamda].keys():

                        return_dict["inter_T_lin"][lamda][t_s] = \
                            [SparseStochMat(**mat_dict) for mat_dict in \
                                              load_dict["inter_T_lin"][lamda][t_s]]

            else:
                for lamda in load_dict["inter_T_lin"].keys():
                    return_dict["inter_T_lin"][lamda] = dict()

                    for t_s in load_dict["inter_T_lin"][lamda].keys():

                        return_dict["inter_T_lin"][lamda][t_s] = \
                            [load_dict["inter_T_lin"][lamda][t_s]["trans_mat_lin0"]]

                        for dT in load_dict["inter_T_lin"][lamda][t_s]["delta_inter_T_lin"]:
                            return_dict["inter_T_lin"][lamda][t_s].append(\
                                     return_dict["inter_T_lin"][lamda][t_s][-1]+dT)



        del load_dict

        return return_dict

    def save_T(self, filename, lamda=None, round_zeros=True, tol=1e-8,
                                   compressed=False):
        """ 
        saves a dict with 'T' as key and net.T as item with other 
        useful attributes.
            
        also works with SparseStochMat.
            
        only works if net.T[lamda] is a matrix and not a list of matrices,
        i.e. if compute_transition_matrices was ran without save_intermediate.
                        
        """
        assert hasattr(self, "T"), f"PID {os.getpid()} : nothing saved, compute inter_T first."

        save_dict = {}
        save_dict["_k_start_laplacians"] = self._k_start_laplacians
        save_dict["_k_stop_laplacians"] = self._k_stop_laplacians
        save_dict["_t_start_laplacians"] = self._t_start_laplacians
        save_dict["_t_stop_laplacians"] = self._t_stop_laplacians
        save_dict["times_k_start_to_k_stop+1"] = self.times.values[\
                            self._k_start_laplacians:self._k_stop_laplacians+1]
        save_dict["num_nodes"] = self.num_nodes
        save_dict["_compute_times"] = self._compute_times



        save_dict["T"] = dict()

        if lamda is None:
            lamdas = self.T.keys()
        else:
            lamdas = [lamda]

        for lamda in lamdas:
            if isinstance(self.T[lamda], SparseStochMat):
                save_dict["is_sparse_stoch"] = True
                if round_zeros:
                    T = self.T[lamda].copy()
                    T.set_to_zeroes(tol)
                else:
                    T = self.T[lamda]
                save_dict["T"][lamda] = T.to_dict()

                text = "SparseStochMat T"

            elif isspmatrix_csr(self.T[lamda]):

                if round_zeros:
                    T = self.T[lamda].copy()
                    set_to_zeroes(T, tol)
                else:
                    T = self.T[lamda]
                save_dict["T"][lamda] = T

                text = "csr T"

            else:
                raise TypeError("T must be csr or SparseStochMat.")


        ext = os.path.splitext(filename)[-1]

        file = filename

        if compressed:

            if ext != ".gz":
                file += ".gz"

            print("PID ", os.getpid(), " : "," saving " + text + " to " + file)

            with gzip.open(file,
                           "wb", compresslevel=2) as fopen:
                pickle.dump(save_dict, fopen)
        else:
            ext = os.path.splitext(filename)[-1]
            if ext != ".pickle":
                file += ".pickle"

            print("PID ", os.getpid(), " : "," saving " + text + " to " + file)

            with open(file, "wb") as fopen:
                pickle.dump(save_dict, fopen)

    def save_T_lin(self, filename, lamda=None, round_zeros=True, tol=1e-8,
                                   compressed=False):
        """ 
        saves a dict with 'T_lin' as key and net.T_lin as item with other 
        useful attributes.
            
        also works with SparseStochMat instance.
            
        only works if net.T_lin[lamda][t_s] is a matrix and not a list of matrices,
        i.e. if compute_transition_matrices was ran without save_intermediate.
                        
        """
        assert hasattr(self, "T_lin"), f"PID {os.getpid()} : nothing saved, compute inter_T_lin first."

        save_dict = {}
        save_dict["_k_start_laplacians"] = self._k_start_laplacians
        save_dict["_k_stop_laplacians"] = self._k_stop_laplacians
        save_dict["_t_start_laplacians"] = self._t_start_laplacians
        save_dict["_t_stop_laplacians"] = self._t_stop_laplacians
        save_dict["times_k_start_to_k_stop+1"] = self.times.values[\
                            self._k_start_laplacians:self._k_stop_laplacians+1]
        save_dict["num_nodes"] = self.num_nodes
        save_dict["_compute_times"] = self._compute_times


        save_dict["T_lin"] = dict()

        if lamda is None:
            lamdas = self.T_lin.keys()
        else:
            lamdas = [lamda]

        for lamda in lamdas:
            save_dict["T_lin"][lamda] = dict()
            for t_s in self.T_lin[lamda].keys():

                if isinstance(self.T_lin[lamda][t_s], SparseStochMat):
                    save_dict["is_sparse_stoch"] = True
                    if round_zeros:
                        T = self.T_lin[lamda][t_s].copy()
                        T.set_to_zeroes(tol)
                    else:
                        T = self.T_lin[lamda][t_s]

                    save_dict["T_lin"][lamda][t_s] = T.to_dict()

                    text = "SparseStochMat T_lin"

                elif isspmatrix_csr(self.T_lin[lamda][t_s]):

                    if round_zeros:
                        T = self.T_lin[lamda][t_s].copy()
                        set_to_zeroes(T, tol)
                    else:
                        T = self.T_lin[lamda][t_s]
                    save_dict["T_lin"][lamda][t_s] = T

                    text = "csr T_lin"

                else:
                    raise TypeError("T must be csr or SparseStochMat.")


        ext = os.path.splitext(filename)[-1]

        file = filename

        if compressed:

            if ext != ".gz":
                file += ".gz"

            print("PID ", os.getpid(), " : "," saving " + text + " to " + file)

            with gzip.open(file,
                           "wb", compresslevel=2) as fopen:
                pickle.dump(save_dict, fopen)
        else:
            ext = os.path.splitext(filename)[-1]
            if ext != ".pickle":
                file += ".pickle"

            print("PID ", os.getpid(), " : "," saving " + text + " to " + file)

            with open(file, "wb") as fopen:
                pickle.dump(save_dict, fopen)

    @staticmethod
    def load_T(filename):
        """Loads T and T_lin from 'filename' that was saved with
        save_T.
            
        returns a dictionary with the T restored.
            
        """
        ext = os.path.splitext(filename)[-1]


        if ext not in [".pickle",".gz"]:
            # detect extension
            if os.path.exists(filename + ".pickle"):
                ext = ".pickle"
                filename += ".pickle"
            elif os.path.exists(filename + ".gz"):
                ext = ".gz"
                filename += ".gz"
            elif os.path.exists(filename + ".pickle.gz"):
                ext = ".pickle.gz"
                filename += ".pickle.gz"
            else:
                raise FileNotFoundError(filename)

        if ext == ".gz" or ext == ".pickle.gz":
            with gzip.open(filename,
                           "rb") as fopen:
                load_dict = pickle.load(fopen)

        else:
            with open(filename, "rb") as fopen:
                load_dict = pickle.load(fopen)

        return_dict = {"_k_start_laplacians" : load_dict["_k_start_laplacians"],
                       "_k_stop_laplacians" : load_dict["_k_stop_laplacians"],
                       "_t_start_laplacians" : load_dict["_t_start_laplacians"],
                       "_t_stop_laplacians" : load_dict["_t_stop_laplacians"],
                       "num_nodes" : load_dict["num_nodes"],
                       "times_k_start_to_k_stop+1" : load_dict["times_k_start_to_k_stop+1"]}


        if "T" in load_dict.keys():
            return_dict["T"] = dict()

            if load_dict.get("is_sparse_stoch", False):

                for lamda in load_dict["T"].keys():
                    return_dict["T"][lamda] = \
                        SparseStochMat(**load_dict["T"][lamda])

            else:
                for lamda in load_dict["T"].keys():
                    return_dict["T"][lamda] = load_dict["T"][lamda]


        if "T_lin" in load_dict.keys():
            return_dict["T_lin"] = dict()

            if load_dict.get("is_sparse_stoch", False):

                for lamda in load_dict["T_lin"].keys():
                    return_dict["T_lin"][lamda] = dict()

                    for t_s in load_dict["T_lin"][lamda].keys():

                        return_dict["T_lin"][lamda][t_s] = \
                            SparseStochMat(**load_dict["T_lin"][lamda][t_s])

            else:
                for lamda in load_dict["T_lin"].keys():
                    return_dict["T_lin"][lamda] = dict()

                    for t_s in load_dict["T_lin"][lamda].keys():

                        return_dict["T_lin"][lamda][t_s] = \
                                            load_dict["T_lin"][lamda][t_s]



        del load_dict

        return return_dict


    def compute_static_adjacency_matrix(self,
                               start_time=None,
                               end_time=None):
        """Returns the adjacency matrix of the static network built from the
        aggregagted edge activity between `start_time` and `end_time`.

        Parameters
        ----------
        start_time : float or int, optional
            Starting time for the aggregation. The default is None, i.e. the 
            start time of the entire temporal network.
        end_time : float or int, optional
            Ending time for the aggregation. The default is None, i.e. the 
            end time of the entire temporal network.

        Returns
        -------
        CSR sparse matrix
            Symmetric adjacency matrix, where element ij is equal to the 
            aggregated time during which egde ij was active after `start_time` 
            and before `end_time`.

        """
        if start_time is None:
            start_time = self.start_time

        if end_time is None:
            end_time = self.end_time

        mask = np.logical_and(self.events_table.starting_times < end_time,
                              self.events_table.ending_times > start_time)

        # loop on events
        data = []
        cols = []
        rows = []
        for ev in self.events_table.loc[mask].itertuples():
            data.append(min(ev.ending_times, end_time) - max(ev.starting_times, start_time))
            rows.append(ev.source_nodes)
            cols.append(ev.target_nodes)

        A = coo_matrix((data, (rows,cols)),
                       shape=(self.num_nodes, self.num_nodes))

        return A + A.T


    def _compute_time_grid(self):
        """Create `self.time_grid`, a dataframe with ('times', 'id') as index,
        wre `id` is the index of the corresponding event in `events_table`,
        and column 'is_start' which is True is the ('times', 'id') 
        corresponds to a starting event.
        Also creates `self.times`, an array with all the times values.
            
        """
        self.time_grid = pd.DataFrame(
            columns=["times", "id", "is_start"],
            index=range(self.events_table.shape[0]*2)
        )
        self.time_grid.iloc[:self.events_table.shape[0],[0,1]] = \
                     self.events_table.reset_index()[["starting_times","index"]]
        self.time_grid["is_start"] = False
        self.time_grid.loc[0:self.events_table.shape[0]-1,"is_start"] = True

        self.time_grid.iloc[self.events_table.shape[0]:,[0,1]] = \
                      self.events_table.reset_index()[["ending_times","index"]]

        self.time_grid.times = pd.to_numeric(self.time_grid.times)

        self.time_grid.sort_values("times", inplace=True)

        # group events with same times
        self.time_grid.set_index(["times", "id"], inplace=True)

        self.time_grid.sort_index(inplace=True)

        self.times = self.time_grid.index.levels[0]


    def _get_closest_time(self, t):
        """Return closest smaller or equal time in `times` and its index"""
        if not hasattr(self, "times"):
            self._compute_time_grid()

        if t not in self.times:
        # take the largest smaller time
            if t <= self.times[0]:
                t = self.times[0]
            else:
                t = self.times[self.times <= t].max()

        k = int(np.where(self.times == t)[0])

        return t, k


    def compute_laplacian_matrices(self, t_start=None, t_stop=None,
                                   save_adjacencies=False):
        """Computes the laplacian matrices and saves them in `self.laplacians`
            
            Computes from the first event time (in `self.times`) before or
            equal to `t_start` until the event time index before `t_stop`.
            
            Laplacians are computed from `self.times[self._k_start_laplacians]`
            until `self.times[self._k_stop_laplacians-1]`.
            
            The laplacian at step k, is the random walk laplacian
            between `times[k]`. and `times[k+1]`.
            
        Parameters
        ----------
        t_start : float or int, optional
            The default is None, i.e. starts at the beginning of times.
            The computation starts from the first time index before or equal to t_start.
            The corresponding starting time index is saved in `self._k_start_laplacians`
            and the real starting time is `self.times[self._k_start_laplacians]` which 
            is saved in `self._t_start_laplacians`.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations. Default is end of times.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Similarily to `t_start`, the corresponding times are saved in 
            `self._k_stop_laplacians` and `self._t_stop_laplacians`.
        save_adjacencies : bool, optional
            Default is False. Use to save adjacency matrices in `self.adjacencies`.

        Returns
        -------
        None.            

        """
        logger.info(f"PID {os.getpid()}: Computing Laplacians")

        if not hasattr(self, "time_grid"):
            self._compute_time_grid()

        # instantaneous adjacency matrix
        A = lil_matrix((self.num_nodes, self.num_nodes),
                       dtype=np.float64)

        #identity
        I = eye(self.num_nodes,
                dtype=np.float64).tocsc()

        #degree array
        degrees = np.zeros(self.num_nodes, dtype=np.float64)
        # inverse degrees diagonal matrix
        Dm1 = I.copy()
        # self loop matrix
        S = I.copy()


        self.laplacians = []
        if save_adjacencies:
            self.adjacencies = []

        # set boundary conditions : L_k is the laplacian during t_k and t_k+1
        if t_start is None:
            self._t_start_laplacians = self.times[0]
            self._k_start_laplacians = 0
        else:
            t, k = self._get_closest_time(t_start)
            self._t_start_laplacians = t
            self._k_start_laplacians = k
        if t_stop is None:
            self._t_stop_laplacians = self.times[-1]
            self._k_stop_laplacians = len(self.times)-1
        else:
            t, k = self._get_closest_time(t_stop)
            self._t_stop_laplacians = t
            self._k_stop_laplacians = k

        t0 = time.time()

        if self._k_start_laplacians > 0:
            # initial conditions, we have to find the adjacency mat just before
            # _k_start_laplacians.

            t_km1 = self.times[self._k_start_laplacians-1]

            # find events that have started before or at t_k-1
            # and were still occuring at t_k-1
            mask_ini = (self.events_table.starting_times <= \
                                                t_km1) & \
                        (self.events_table.ending_times > \
                                                t_km1)

            for event in self.events_table.loc[mask_ini][["source_nodes",
                                          "target_nodes"]].itertuples():

                if A[event.source_nodes, event.target_nodes] != 1:
                        A[event.source_nodes, event.target_nodes] = 1
                        A[event.target_nodes, event.source_nodes] = 1

                        degrees[event.source_nodes] += 1
                        degrees[event.target_nodes] += 1

                if degrees[event.source_nodes] == 0:
                        S.data[event.source_nodes] = 1
                        Dm1.data[event.source_nodes] = 1
                else:
                    S.data[event.source_nodes] = 0
                    Dm1.data[event.source_nodes] = 1/degrees[event.source_nodes]

                if degrees[event.target_nodes] == 0:
                    S.data[event.target_nodes] = 1
                    Dm1.data[event.target_nodes] = 1
                else:
                    S.data[event.target_nodes] = 0
                    Dm1.data[event.target_nodes] = 1/degrees[event.target_nodes]


        # time grid for this time range
        time_grid_range = self.time_grid.loc[\
                  (self.time_grid.index.get_level_values("times") >= \
                       self._t_start_laplacians) & \
                  (self.time_grid.index.get_level_values("times") < \
                       self._t_stop_laplacians)]

        for k, (tk, time_ev) in enumerate(time_grid_range.groupby(level="times")):
            if not k%1000:
                logger.info(
                    f"{k} over "
                    f"{self._k_stop_laplacians - self._k_start_laplacians}"
                )
                logger.info( f"{time.time()-t0:.2f}s")

            meet_id = time_ev.index.get_level_values("id")
            # starting or ending events
            is_starts = time_ev.is_start.values


            events_k = [self.events_table.loc[mid,["source_nodes","target_nodes"]].astype(np.int64) \
                                        for mid in meet_id.values]

            #update instantaneous matrices
            for event, is_start in zip(events_k, is_starts):
                # unweighted, undirected
                if is_start:
                    # if they are not already connected (can happen if the
                    # opposite event overlap)
                    if A[event.source_nodes, event.target_nodes] != 1:
                        A[event.source_nodes, event.target_nodes] = 1
                        A[event.target_nodes, event.source_nodes] = 1

                        degrees[event.source_nodes] += 1
                        degrees[event.target_nodes] += 1

                elif A[event.source_nodes, event.target_nodes] > 0:
                    A[event.source_nodes, event.target_nodes] = 0
                    A[event.target_nodes, event.source_nodes] = 0

                    degrees[event.source_nodes] -= 1
                    degrees[event.target_nodes] -= 1

                # update self loops
                if degrees[event.source_nodes] == 0:
                    S.data[event.source_nodes] = 1
                    Dm1.data[event.source_nodes] = 1
                else:
                    S.data[event.source_nodes] = 0
                    Dm1.data[event.source_nodes] = 1/degrees[event.source_nodes]

                if degrees[event.target_nodes] == 0:
                    S.data[event.target_nodes] = 1
                    Dm1.data[event.target_nodes] = 1
                else:
                    S.data[event.target_nodes] = 0
                    Dm1.data[event.target_nodes] = 1/degrees[event.target_nodes]

            # Laplacian L(tk)
            Acsc = A.tocsc()
            # T_D = Dm1 @ (Acsc + S)
            # L = I - T_D

            self.laplacians.append(I - Dm1 @ (Acsc + S))
            if save_adjacencies:
                self.adjacencies.append(A.copy())

        t_end = time.time()-t0
        self._compute_times["laplacians"] = t_end
        logger.info(f"Finished in {t_end}")

    def compute_inter_transition_matrices(self, *, lamda=None, t_start=None,
                                          t_stop=None, fix_tau_k=False,
                                          use_sparse_stoch=False,
                                          dense_expm=True):
        """Computes interevent transition matrices as T_k(lamda) = expm(-tau_k*lamda*L_k).
        
        The transition matrix T_k is saved in `self.inter_T[lamda][k]`, where 
        self.inter_T is a dictionary with lamda as keys and lists of transition
        matrices as values.
        
        will compute from self.times[self._k_start_laplacians] until 
        self.times[self._k_stop_laplacians-1]
        
        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]
        
        Parameters
        ----------
        lamda : float, optional
            Random walk rate, dynamical resolution parameter. The default (None)
            is 1 over the median inter event time.
        t_start : float or int, optional
            Starting time, passed to `compute_laplacian_matrices` if the 
            Laplacians have not yet been computed.
            Otherwise is not used.
            The computation starts at self.times[self._k_start_laplacians].
            The default is None, i.e. starts at the beginning of times.
        t_stop : float or int, optional
            Same than `t_start` but for the ending time of computations.
            Computations stop at self.times[self._k_stop_laplacians-1].
            Default is end of times.
        fix_tau_k : bool, optional
            If true, all interevent times (tau_k) in the formula above are set to 1. 
            This decouples the dynamic scale from the length of event which
            is useful for temporal networks with instantaneous events.
            The default is False.
        use_sparse_stoch : bool, optional
            Whether to use custom sparse stochastic matrix format to save the
            inter transition matrices. Especially useful for large networks as 
            the matrix exponential is then computed on each connected component 
            separately (more memory efficient). The default is False.
        dense_expm : bool, optional
            Whether to use the dense version of the matrix exponential algorithm
            at each time steps. Recommended for not too large networks. 
            The inter trans. matrices are still saved as sparse scipy matrices
            as they usually have many zero values. The default is True. Has no
            effect is use_sparse_stoch is True.

        Returns
        -------
        None.

        """
        # NOTE: We might drop such tests if we check for process consistency
        #       on a higher leverl (i.e. FlowStability)
        if not hasattr(self, "laplacians"):
            self.compute_laplacian_matrices(t_start=t_start, t_stop=t_stop)
        if not hasattr(self, "inter_T"):
            self.inter_T = dict()


        if lamda is None:
            logger.info("Taking lamda as 1/tau_w with tau_w = median "
                        "interevent time")
            lamda = 1/np.median(np.diff(self.times))

        # new value of lamda, we need to recompute
        if lamda not in self.inter_T.keys():

            logger.info(
                f"Computing interevent transition matrices for lamda={lamda}"
            )
            self.inter_T[lamda] = []

            t0 = time.time()
            for k, tk in enumerate(self.times[self._k_start_laplacians:
                                              self._k_stop_laplacians]):
                if not k%1000:
                    logger.info(
                        f"{k} over "
                        f"{self._k_stop_laplacians-1-self._k_start_laplacians}"
                    )
                    logger.info(f"{time.time()-t0:.2f}s")

                if fix_tau_k:
                    tau_k = 1.0
                else:
                    tau_k = self.times[self._k_start_laplacians+k+1] - tk

                if use_sparse_stoch:
                    self.inter_T[lamda].append(sparse_lapl_expm(self.laplacians[k],
                                                                       tau_k*lamda,
                                                                       dense_expm=dense_expm))
                elif self.laplacians[k].getnnz() == 0:
                    # expm of zero = identity
                    self.inter_T[lamda].append(eye(self.num_nodes,format="csr"))
                elif dense_expm:
                    self.inter_T[lamda].append(csr_matrix(expm(-tau_k*lamda*self.laplacians[k].toarray())))
                else:
                    self.inter_T[lamda].append(expm(-tau_k*lamda*self.laplacians[k]).tocsr())

            if len(self.inter_T[lamda]) == 0:
                logger.info("no events, trans. matrix = identity")
                # is there was no event, the transition is identity
                if use_sparse_stoch:
                    self.inter_T[lamda].append(SparseStochMat.create_diag(size=self.num_nodes))
                else:
                    self.inter_T[lamda].append(eye(self.num_nodes,
                                                      dtype=np.float64,
                                                      format="csr"))

            t_end = time.time()-t0

            self._compute_times["inter_T_" + str(lamda)]  = t_end

            logger.info(
                f"Finished computing interevent transition matrices in {t_end}"
            )
        logger.info(
            f"Interevent transition matrices already computed for lamda={lamda}"
        )

    def compute_lin_inter_transition_matrices(self, lamda=None, t_start=None,
                                              t_stop=None,
                                              t_s=10, fix_tau_k=False,
                                              use_sparse_stoch=False):
        """Compute interevent transition matrices as a linear approximation of
        expm(-tau_k*lamda*L_k) based on the discrete time transition matrix.
            
        `t_s` is the time value for which the linear approximation reaches the
        stationary transition matrix (default is `t_s=10`).
        
        The transition matrix T_k_lin is saved in 
        `self.inter_T_lin[lamda][t_s][k]`,
        where `self.inter_T_lin` is a dictionary with lamda as keys and
        lists of transition matrices as values.
            
        will compute from self.times[self._k_start_laplacians]
        until self.times[self._k_stop_laplacians-1]
            
        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]
            
        """
        I = eye(self.num_nodes,
                dtype=np.float64, format="csr")

        if not hasattr(self, "_stationary_trans"):
            self._compute_stationary_transition(
                t_start=t_start, t_stop=t_stop,
                use_sparse_stoch=use_sparse_stoch
            )

        if not hasattr(self, "inter_T_lin"):
            self.inter_T_lin = dict()

        if lamda is None:
            logger.info(
                "taking lamda as 1/tau_w with tau_w = median interevent time"
            )
            lamda = 1/np.median(np.diff(self.times))

        compute = True

        if lamda in self.inter_T_lin.keys():
            if t_s in self.inter_T_lin[lamda].keys():
                compute = False
                logger.info(
                    "Interevent transition matrices already computed "
                    f"for lamda={lamda}, t_s={t_s}"
                )

        if compute:
            if lamda not in self.inter_T_lin.keys():
                self.inter_T_lin[lamda] = dict()
                self.inter_T_lin[lamda][t_s] = []
            else:
                self.inter_T_lin[lamda][t_s] = []

            logger.info(
                "Computing interevent linear transition matrices for "
                f"lamda={lamda}, t_s={t_s}"
            )

            t0 = time.time()
            for k, tk in enumerate(self.times[self._k_start_laplacians:
                                              self._k_stop_laplacians]):
                if not k%1000:
                    logger.info(
                        f"{k} over "
                        f"{self._k_stop_laplacians-1-self._k_start_laplacians}"
                    )
                    logger.info(f"{time.time()-t0:.2f}s")

                if fix_tau_k:
                    tau_k = 1.0
                else:
                    tau_k = self.times[self._k_start_laplacians+k+1] - tk

                Lcsr = self.laplacians[k].tocsr()

                if use_sparse_stoch:

                    #get non zero indices
                    nonzerosum_rowcols = ~np.logical_and(Lcsr.getnnz(1)==0,
                                                         Lcsr.getnnz(0)==0)

                    nz_rowcols, = (nonzerosum_rowcols).nonzero()

                    self.inter_T_lin[lamda][t_s].append(\
                           sparse_lin_approx(I - Lcsr,
                                             tau_k*lamda,
                                             Pi=self._stationary_trans[k],
                                             t_s=t_s,
                                             nz_rowcols=nz_rowcols))
                else:

                    self.inter_T_lin[lamda][t_s].append(\
                           lin_approx_trans_matrix(I - Lcsr,
                                                   tau_k*lamda,
                                                   Pi=self._stationary_trans[k],
                                                   t_s=t_s))

            if len(self.inter_T_lin[lamda][t_s]) == 0:
                logger.info("no events, lin. trans. matrix = identity")
                # is there was no event, the transition is identity
                if use_sparse_stoch:
                    self.inter_T_lin[lamda].append(SparseStochMat.create_diag(size=self.num_nodes))
                else:
                    self.inter_T_lin[lamda][t_s].append(eye(self.num_nodes,
                                                      dtype=np.float64,
                                                      format="csr"))

            t_end = time.time()-t0
            self._compute_times[f"tran_matrices_lin_{lamda}_{t_s}"] = t_end

            logger.info("Finished computing linear interevent transition "
                        f"matrices in {t_end}")



    def compute_transition_matrices(self, lamda=None, t_start=None, t_stop=None,
                                    save_intermediate=True,
                                    reverse_time=False,
                                    force_csr=False,
                                    tol=None):
        """Compute transition matrices and saves them in a dict of lists `self.T[lamda]`
        where `self.T[lamda][k]` is the product of all interevent transition 
        matrices from t_0 to t_k computed with lamda.
        """
        if not hasattr(self, "inter_T") or \
                lamda not in self.inter_T.keys():
            raise Exception("Compute inter_T first.")


        if not hasattr(self, "T"):
            self.T = dict()

        if reverse_time:
            k_init = len(self.inter_T[lamda])-1
            k_range = reversed(range(k_init))
            logger.info("Reversed time computation.")
        else:
            k_init = 0
            k_range = range(1,len(self.inter_T[lamda]))


        if lamda not in self.T.keys():
            if save_intermediate:
                if force_csr:
                    # forcing the first matrix to csr, will ensure that
                    # all products are done in csr format,
                    # since CSR @ SparseStochMat t is not implemented
                    self.T[lamda] = [self.inter_T[lamda][k_init].tocsr()]
                else:
                    self.T[lamda] = [self.inter_T[lamda][k_init]]

                if tol is not None:
                    set_to_zeroes(self.T[lamda][0],tol)
                    inplace_csr_row_normalize(self.T[lamda][0])
            else:
                if force_csr:
                    self.T[lamda] = self.inter_T[lamda][k_init].tocsr()
                else:
                    self.T[lamda] = self.inter_T[lamda][k_init]

                if tol is not None:
                    set_to_zeroes(self.T[lamda],tol)
                    inplace_csr_row_normalize(self.T[lamda])




            logger.info("Computing transition matrix")

            t0 = time.time()

            for k in k_range:
                if not k%1000:
                    logger.info(f"{k} over {len(self.inter_T[lamda])}")
                    logger.info(f"{time.time()-t0:.2f}s")

                Tk = self.inter_T[lamda][k]
                if tol is not None:
                    set_to_zeroes(Tk,tol)
                    inplace_csr_row_normalize(Tk)

                if save_intermediate:

                    self.T[lamda].append(self.T[lamda][-1] @ Tk)

                    #normalize T to correct precision errors
                    if tol is not None:
                        set_to_zeroes(self.T[lamda][-1],tol)

                    inplace_csr_row_normalize(self.T[lamda][-1])
                else:
                    self.T[lamda] = self.T[lamda] @ Tk
                    if tol is not None:
                        set_to_zeroes(self.T[lamda], tol)

                    #normalize T to correct precision errors
                    inplace_csr_row_normalize(self.T[lamda])

            t_end = time.time()-t0

            self._compute_times[f"trans_matrix_{lamda}_rev{reverse_time}"] = t_end

            logger.info(f"Finished in {t_end:.2f}s")
        logger.info(
            f"Transition matrices already computed for lamda={lamda}"
        )

    def compute_lin_transition_matrices(self, lamda=None,
                                        t_start=None, t_stop=None,
                                        t_s=10, save_intermediate=True,
                                        reverse_time=False):
        """Compute transition matrices and saves them in a dict of lists `self.T_lin[lamda][t_s]`
        where `self.T_lin[lamda][t_s][k]` is the product of all interevent transition 
        matrices from t_0 to t_k computed with lamda and t_s.
        """
        if not hasattr(self, "inter_T_lin") \
                or lamda not in self.inter_T_lin.keys() \
                or t_s not in self.inter_T_lin[lamda].keys():
            raise Exception("Compute inter_T_lin first.")

        if not hasattr(self, "T_lin"):
            self.T_lin = dict()

        compute = True

        if lamda in self.T_lin.keys():
            if t_s in self.T_lin[lamda].keys():
                compute = False
                logger.info("Transition matrices already computed "
                            f"for lamda={lamda}, t_s={t_s}")

        if compute:

            if reverse_time:
                k_init = len(self.inter_T_lin[lamda][t_s])-1
                k_range = reversed(range(k_init))
                logger.info("Reversed time computation.")
            else:
                k_init = 0
                k_range = range(1,len(self.inter_T_lin[lamda][t_s]))


            # initial conditions
            if lamda not in self.T_lin.keys():
                self.T_lin[lamda] = dict()
                if save_intermediate:
                    self.T_lin[lamda][t_s] = [self.inter_T_lin[lamda][t_s][k_init]]
                else:
                    self.T_lin[lamda][t_s] = self.inter_T_lin[lamda][t_s][k_init]

            if t_s not in self.T_lin[lamda].keys():
                if save_intermediate:
                    self.T_lin[lamda][t_s] = [self.inter_T_lin[lamda][t_s][k_init]]
                else:
                    self.T_lin[lamda][t_s] = self.inter_T_lin[lamda][t_s][k_init]

            logger.info(
                f"Computing transition matrix for lamda={lamda}, t_s={t_s}"
            )

            t0 = time.time()

            for k in k_range:
                if not k%1000:
                    logger.info(f"{k} over {len(self.inter_T_lin[lamda][t_s])}")
                    logger.info(f"{time.time()-t0:.2f}s")
                if save_intermediate:
                    self.T_lin[lamda][t_s].append(self.T_lin[lamda][t_s][-1] @ \
                                      self.inter_T_lin[lamda][t_s][k])

                    #normalize T to correct precision errors
                    inplace_csr_row_normalize(self.T_lin[lamda][t_s][-1])
                else:
                    self.T_lin[lamda][t_s] = self.T_lin[lamda][t_s] @ \
                                      self.inter_T_lin[lamda][t_s][k]

                    #normalize T to correct precision errors
                    inplace_csr_row_normalize(self.T_lin[lamda][t_s])

            t_end = time.time()-t0

            self._compute_times[f"trans_matrix_lin_{lamda}_{t_s}_rev{reverse_time}"] = t_end

            logger.info(f"Finished in {t_end}s")
        logger.info(
            f"Transition matrices already computed for lamda={lamda}"
        )

    def _compute_stationary_transition(self, t_start=None,
                                               t_stop=None,
                                               use_sparse_stoch=True):

        if not hasattr(self, "laplacians"):
            self.compute_laplacian_matrices(t_start=t_start,
                                            t_stop=t_stop)
        logger.info("Computing stationary transition matrices")

        self._stationary_trans = []

        I = eye(self.num_nodes,
                dtype=np.float64, format="csc")

#        I = np.eye(self.num_nodes, dtype=np.float64)

        t0 = time.time()
        for k in range(len(self.laplacians)):
            if not k%1000:
                logger.info(f"{k} over {len(self.laplacians)}")
                logger.info(f"{time.time()-t0:.2f}s")

            if use_sparse_stoch:
                self._stationary_trans.append(\
                  sparse_stationary_trans(I - self.laplacians[k]))

            else:
                self._stationary_trans.append(\
                      compute_stationary_transition(I - self.laplacians[k]))


        t_end = time.time()-t0

        self._compute_times["_stationary_trans"] = t_end

        logger.info(f"Stationary transition matrices computation took {t_end}s")




    def _merge_overlapping_events(self):
        """Merge temporally overlapping undirected event between each pair of nodes.
            
        """
        events_to_keep = np.ones(self.events_table.shape[0],dtype=bool)

        A = self.compute_static_adjacency_matrix()

        # loop over nodes
        for i, n1 in enumerate(self.node_array):

            for n2 in (A[n1,:] > 0).nonzero()[1]:


                mask_12 = np.logical_and(self.events_table.source_nodes.values == n1,
                                         self.events_table.target_nodes.values == n2)

                mask_21 = np.logical_and(self.events_table.source_nodes.values == n2,
                                         self.events_table.target_nodes.values == n1)

                #sort by starting times
                evs = self.events_table.loc[np.logical_or(mask_12, mask_21)].\
                                sort_values(by=["starting_times", "ending_times"])

                evs_list = list(evs.itertuples())

                # event to compare
                ev1 = evs_list[0]
                merged = 0
                for k in range(1,len(evs_list)):
                    ev2 = evs_list[k]
                    # if ev2 overlaps with ev1, merge them, otherwise ev2 becomes
                    # ev1
                    if ev2.starting_times < ev1.ending_times:
                        #merge
                        events_to_keep[ev2.Index] = False
                        self.events_table.loc[ev1.Index,"ending_times"] = ev2.ending_times
                        ev1._replace(ending_times=ev2.ending_times)
                        merged += 1
                    else:
                        ev1 = ev2
                logger.info(f"n1,n2 ({n1},{n2}): {merged} merged")

        num_merged = (events_to_keep == False).sum()
        logger.info(f"Merged {num_merged} events.")

        self.events_table = self.events_table.loc[events_to_keep]

        self.events_table.reset_index(inplace=True, drop=True)

        self.num_nodes = self.node_array.shape[0]

        self.num_events = self.events_table.shape[0]

        self.start_time = self.events_table.starting_times.min()

        self.end_time = self.events_table.ending_times.max()

        self._compute_time_grid()

        return num_merged

    def _compute_delta_trans_mat(self, lamda, round_zeros=True, tol=1e-8):
        """Comptes and put in a attribute `delta_inter_T` the matrix differences
        between each consecutive inter_T.
            
        self.delta_inter_T[lamda][k] = self.inter_T[lamda][k+1] - self.inter_T[lamda][k]
            
        The length of self.delta_inter_T[lamda] is len(self.inter_T[lamda]) - 1
            
        """
        if hasattr(self, "inter_T") and lamda in self.inter_T.keys():


            if not hasattr(self, "delta_inter_T"):
                self.delta_inter_T = dict()

            if lamda not in self.delta_inter_T.keys():

                self.delta_inter_T[lamda] = [self.inter_T[lamda][k+1] - \
                                                  self.inter_T[lamda][k] for k in \
                                                  range(len(self.inter_T[lamda])-1)]

                if round_zeros:
                    for M in self.delta_inter_T[lamda]:
                        set_to_zeroes(M, tol=tol)

            else:
                print("PID ", os.getpid(), " : ",
                      f"delta_inter_T has already been computed with lamda={lamda}")

        else:
            print("PID ", os.getpid(), " : ", "delta_inter_T has not been computed")


        if hasattr(self, "inter_T_lin") and lamda in self.inter_T_lin.keys():


            if not hasattr(self, "delta_inter_T_lin"):
                self.delta_inter_T_lin = dict()

            if lamda not in self.delta_inter_T_lin.keys():

                self.delta_inter_T_lin[lamda] = dict()

                for t_s in self.inter_T_lin[lamda].keys():

                    self.delta_inter_T_lin[lamda][t_s] = [self.inter_T_lin[lamda][t_s][k+1] - \
                                                      self.inter_T_lin[lamda][t_s][k] for k in \
                                                      range(len(self.inter_T_lin[lamda][t_s])-1)]

                    if round_zeros:
                        for M in self.delta_inter_T_lin[lamda][t_s]:
                            set_to_zeroes(M, tol=tol)

            else:
                print("PID ", os.getpid(), " : ",
                      f"delta_inter_T_lin has already been computed with lamda={lamda}")

        else:
            print("PID ", os.getpid(), " : ", "delta_inter_T_lin has not been computed")


class ContTempInstNetwork(ContTempNetwork):
    """Continuous time temporal network with instantaneous events
    
        This is a subclass of ContTempNetwork for continuous time temporal
        networks where events do not have a duration.
        
        In practice, it is implemented as a ContTempNetwork where 
        ending_times_k = starting_times_k+1 and where durations (tau_k) = 1  
        for all events for the computation of the transition matrices.
        
    Parameters
    ----------
    source_nodes: Python list
        List of source nodes of each event, ordered according to `starting_times`
    
    target_nodes: Python list
        List of target nodes of each event
        
    starting_times: Python list
        List of starting times of each event
                
    relabel_nodes: boolean
        Relabel nodes from 0 to num_nodes and save original labels in 
        self.node_to_label_dict. Default is `True`
        
    reset_event_table_index: boolean
        Reset the index of the `events_table` DataFrame. Default is `True`.
        
    node_to_label_dict: Python dict
        If `relabel_nodes` is `False, this can be used to save the original labels
        of the nodes.
                
    events_table: Pandas Dataframe
        Dataframe with columns 'source_nodes', 'target_nodes', 'starting_times'
        and 'ending_times' and index corresponding to event index. Used for
        instantiating a new ConTempNetwork from the event_table of an other one.
    
    
    """

    def __init__(self, source_nodes=[],
                        target_nodes=[],
                        starting_times=[],
                        relabel_nodes=True,
                        reset_event_table_index=True,
                        node_to_label_dict=None,
                        events_table=None):

        if events_table is None:


            utimes = np.unique(starting_times)
            end_times_map = {utimes[k] : utimes[k+1] for k in range(utimes.size-1)}
            end_times_map[utimes[-1]] = utimes[-1]+1

            ending_times = [end_times_map[t] for t in starting_times]

            super().__init__(source_nodes=source_nodes,
                             target_nodes=target_nodes,
                             starting_times=starting_times,
                             ending_times=ending_times,
                             relabel_nodes=relabel_nodes,
                             reset_event_table_index=reset_event_table_index,
                             node_to_label_dict=node_to_label_dict,
                             merge_overlapping_events=False,
                             events_table=events_table)


        self.events_table["durations"] = [1.0]*self.events_table.shape[0]
        self.instantaneous_events = True




    def compute_laplacian_matrices(self, t_start=None,
                                   t_stop=None, save_adjacencies=False):
        """Compute all laplacian matrices and saves them in self.laplacians
        
        Computes from the first time index before or equal to t_start until
        the time index before t_stop.
            
        laplacians are computed from self.times[self._k_start_laplacians]
        until self.times[self._k_stop_laplacians-1]
            
        The laplacian at step k, is the random walk laplacian
        between times[k] and times[k+1]
        """
        logger.info("Computing Laplacians")

        if not hasattr(self, "time_grid"):
            self._compute_time_grid()

        # instantaneous adjacency matrix
        A = dok_matrix((self.num_nodes, self.num_nodes),
                       dtype=np.float64)

        #identity
        I = eye(self.num_nodes,
                dtype=np.float64).tocsc()

        #degree array
        degrees = np.zeros(self.num_nodes, dtype=np.float64)
        # inverse degrees diagonal matrix
        Dm1 = I.copy()
        # self loop matrix
        S = I.copy()


        self.laplacians = []
        if save_adjacencies:
            self.adjacencies = []

        # set boundary conditions : L_k is the laplacian during t_k and t_k+1
        if t_start is None:
            self._t_start_laplacians = self.times[0]
            self._k_start_laplacians = 0
        else:
            t, k = self._get_closest_time(t_start)
            self._t_start_laplacians = t
            self._k_start_laplacians = k
        if t_stop is None:
            self._t_stop_laplacians = self.times[-1]
            self._k_stop_laplacians = len(self.times)-1
        else:
            t, k = self._get_closest_time(t_stop)
            self._t_stop_laplacians = t
            self._k_stop_laplacians = k

        t0 = time.time()


        # time grid for this time range
        time_grid_range = self.time_grid.loc[\
                  (self.time_grid.index.get_level_values("times") >= \
                       self._t_start_laplacians) & \
                  (self.time_grid.index.get_level_values("times") < \
                       self._t_stop_laplacians)]

        for k, (tk, time_ev) in enumerate(time_grid_range.groupby(level="times")):
            if not k%1000:
                logger.info(
                    f"{k} over "
                    f"{self._k_stop_laplacians - self._k_start_laplacians}"
                )
                print(f"PID {os.getpid()} : {time.time()-t0:.2f}s")

            meet_id = time_ev.index.get_level_values("id")
            # starting or ending events
            is_starts = time_ev.is_start.values

            events_k = self.events_table.loc[meet_id,
                              ["source_nodes","target_nodes"]].astype(np.int64)

            #update instantaneous matrices
            for event, is_start in zip(events_k.itertuples(), is_starts):
                # unweighted, undirected
                if is_start:
                    # if they are not already connected (can happen if the
                    # opposite event overlap)
                    if A[event.source_nodes, event.target_nodes] != 1:
                        A[event.source_nodes, event.target_nodes] = 1
                        A[event.target_nodes, event.source_nodes] = 1

                        degrees[event.source_nodes] += 1
                        degrees[event.target_nodes] += 1

                        S.data[event.source_nodes] = 0
                        Dm1.data[event.source_nodes] = 1/degrees[event.source_nodes]

                        S.data[event.target_nodes] = 0
                        Dm1.data[event.target_nodes] = 1/degrees[event.target_nodes]

                else:
                    #end of meeting
                    # no need for instantaneous events
                    pass


            # Laplacian L(tk)
            Acsc = A.tocsc()
            # T_D = Dm1 @ (Acsc + S)
            # L = I - T_D

            self.laplacians.append(I - Dm1 @ (Acsc + S))
            if save_adjacencies:
                self.adjacencies.append(A.copy())

            # reset matrices
            A.clear()
            S.data.fill(1.0)
            Dm1.data.fill(1.0)
            degrees.fill(0.0)

        t_end = time.time()-t0
        self._compute_times["laplacians"] = t_end
        logger.info(f"Finished in {t_end}")




    def compute_inter_transition_matrices(self, lamda=None, t_start=None, t_stop=None,
                                    use_sparse_stoch=False,
                                    dense_expm=True):
        """Compute interevent transition matrices as T_k(lamda) = expm(-lamda*L_k).
        
        Here, for instantaneous events, all events are assumed to have the 
        same duration of unit time (i.e. tau_k =1 for all k).
        
        The transition matrix T_k is saved in `self.inter_T[lamda][k]`,
        where self.inter_T is a dictionary with lamda as keys and
        lists of transition matrices as values.
            
        will compute from self.times[self._k_start_laplacians]
        until self.times[self._k_stop_laplacians-1]
            
        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]   
        """
        super().compute_inter_transition_matrices(lamda=lamda,
                                                  t_start=t_start,
                                                  t_stop=t_stop,
                                                  fix_tau_k=True,
                                                  use_sparse_stoch=use_sparse_stoch,
                                                  dense_expm=dense_expm)


    def compute_lin_inter_transition_matrices(self, lamda=None, t_start=None, t_stop=None,
                                    t_s=10,
                                    use_sparse_stoch=False):
        """Compute interevent transition matrices as a linear approximation of
        expm(-lamda*L_k) based on the discrete time transition matrix.
            
        Here, for instantaneous events, all events are assumed to have the 
        same duration of unit time (i.e. tau_k =1 for all k).
            
        `t_s` is the time value for which the linear approximation reaches the
        stationary transition matrix (default is `t_s=10`).
        
        The transition matrix T_k_lin is saved in 
        `self.inter_T_lin[lamda][t_s][k]`,
        where `self.inter_T_lin` is a dictionary with lamda as keys and
        lists of transition matrices as values.
            
        will compute from self.times[self._k_start_laplacians]
        until self.times[self._k_stop_laplacians-1]
            
        the transition matrix at step k, is the probability transition matrix
        between times[k] and times[k+1]
            
        """
        super().compute_lin_inter_transition_matrices(\
                                                lamda=lamda,
                                                t_start=t_start,
                                                t_stop=t_stop,
                                                t_s=t_s,
                                                fix_tau_k=True,
                                                use_sparse_stoch=use_sparse_stoch)


def lin_approx_trans_matrix(T, t, Pi=None,t_s=10):
    r"""Linear approximation of a continuous time transition matrix :math:`T(t) = e^{-tL}` based on
    an interpolation between :math:`I` and :math:`T` and 
    a second interpolation between :math:`T` and :math:`\Pi`, the transition 
    matrix at stationarity.
        
    For each connected component of the graph, :math:`T(t)` is approximated as
        
    .. math::
        \tilde{T}(t) & = & (1-t) I + tT \text{ for } 0\leq t \leq 1 \\
            \tilde{T}(t) & = & \frac{1}{1-t_s}[T(t-t_s) + \Pi(1-t)] \text{ for } 1 < t \leq t_s \\
            \tilde{T}(t) & = & \Pi \text{ for } t > t_s
            
    where  :math:`t_s` is the mixing time of the random walk (default is `t_s=10`).
            
            
    Parameters
    ----------
        T : scipy.sparse csr matrix
        Transition matrix of the discrete time random walk
    t : float
        Time, greater or equal to zero.
    Pi : scipy.sparse matrix
        Transition matrix of the discrete time random walk at stationarity.
        If None, it will be computed from T.
    t_s : float
        Stationarity time at which the interpolation reaches Pi.
            
    Returns
    -------
        Tapprox : scipy.sparse.csr matrix
        Linear approximation of expmL at time t.
            
    """
    assert isspmatrix_csr(T)

    num_nodes = T.shape[0]
    I = eye(num_nodes,
                dtype=np.float64, format="csr")

    if t < 0:
        raise ValueError("t must be >= 0")

    elif t <= 1:
        return (1-t)*I + t*T.tocsr()
    else:

        if Pi is None:
            Pi = compute_stationary_transition(T)

        if t < t_s:

            return (1/(1-t_s))*(T*(t-t_s)+Pi*(1-t))
        else:
            return Pi


def compute_stationary_transition(T):
    """Compute the transition matrix at stationarity for matrix `T`

    Parameters
    ----------
        T : scipy.sparse matrix or numpy.ndarray
        Transition matrix of the discrete time random walk.
        Can be a CSC or CSR matrix.

    Returns
    -------
        Pi : scipy.sparse.csr matrix
        Stationary transition matrix.
            
    """
    num_nodes = T.shape[0]

    # otherwise 0 values may count as an edge
    T.eliminate_zeros()
    T.sort_indices()

    n_comp, comp_labels = connected_components(T,directed=False)
    comp_sizes = np.bincount(comp_labels)
    cmp_to_indices = {cmp : (comp_labels == cmp).nonzero()[0] for \
                          cmp in range(n_comp)}

    # constructors for sparse array
    data = np.zeros((comp_sizes**2).sum(), dtype=np.float64)
    indices = np.zeros((comp_sizes**2).sum(), dtype=np.int32)
    indptr = np.zeros(num_nodes+1, dtype=np.int32)

    # vector of degrees (number of nonzero elements of each row)
    if isspmatrix(T):
        degs = np.diff(T.indptr)
    else:
        degs = (T > 0).sum(1)

    data_ind = 0
    for row in range(num_nodes):
        cmp = comp_labels[row]
        cmp_degs = degs[cmp_to_indices[cmp]]
        data[data_ind:data_ind+comp_sizes[cmp]] = \
                                                cmp_degs/cmp_degs.sum()
        indices[data_ind:data_ind+comp_sizes[cmp]] = \
                        cmp_to_indices[cmp]
        indptr[row] = data_ind
        data_ind += comp_sizes[cmp]
    indptr[num_nodes] = data_ind

    # Stationary transition matrix

    return csr_matrix((data, indices, indptr), shape=(num_nodes,num_nodes),
                    dtype=np.float64)

def compute_subspace_expm(A, n_comp=None, comp_labels=None,
                           thresh_ratio=None,
                           normalize_rows=True):
    """Compute the exponential matrix of `A` by applying expm on each connected
    subgraphs defined by A and recomposing it to return expm(A).

    Parameters
    ----------
        A : scipy.sparse.csc_matrix
        
    thresh_ratio: float, optional.
        Threshold ratio used to trim negligible values in the resulting matrix.
        Values smaller than `max(expm(A))/thresh_ratio` are set to 
        zero. Default is None.
    normalize_rows: bool, optional.
        Whether rows of the resulting matrix are normalized to sum to 1.
        

    Returns
    -------
        expm(A) : scipy.sparse.csr_matrix
        matrix exponential of A
            
    """
    num_nodes = A.shape[0]

    # otherwise 0 values may count as an edge
    A.eliminate_zeros()
    A.sort_indices()

    if (n_comp is None) or (comp_labels is None):
        n_comp, comp_labels = connected_components(A,directed=False)
    comp_sizes = np.bincount(comp_labels)
    cmp_indices = [(comp_labels == cmp).nonzero()[0] for \
                          cmp in range(n_comp)]

    logger.info(f"subspace_expm with {n_comp} components")

    # constructors for sparse array
    data = np.zeros((comp_sizes**2).sum(), dtype=np.float64)
    indices = np.zeros((comp_sizes**2).sum(), dtype=np.int32)
    indptr = np.zeros(num_nodes+1, dtype=np.int32)

    # if nproc == 1:
    #     expm_func = lambda M: expm(M)
    # else:
    #     expm_func = lambda M: compute_parallel_expm(M, nproc=nproc,
    #                                                 thresh_ratio=None,
    #                                                 normalize_rows=False)
    subnets_expms = []
    for i, cmp_ind in enumerate(cmp_indices):
        logger.info(
            f"Computing component {i} over {n_comp}, with size {cmp_ind.size}"
        )

        subnets_expms.append(expm(A[cmp_ind,:][:,cmp_ind]).toarray())

    # reconstruct csr sparse matrix
    logger.info("Reconstructing expm mat")
    data_ind = 0
    for row in range(num_nodes):
        cmp = comp_labels[row]
        cmp_expm = subnets_expms[cmp]
        sub_expm_row, = np.where(cmp_indices[cmp] == row)

        data[data_ind:data_ind+comp_sizes[cmp]] = cmp_expm[sub_expm_row,:]

        indices[data_ind:data_ind+comp_sizes[cmp]] = cmp_indices[cmp]

        indptr[row] = data_ind

        data_ind += comp_sizes[cmp]

    indptr[num_nodes] = data_ind

    expmA = csr_matrix((data, indices, indptr), shape=(num_nodes,num_nodes),
                    dtype=np.float64)

    if thresh_ratio is not None:
        expmA.data[expmA.data<expmA.data.max()/thresh_ratio] = 0.0
        expmA.eliminate_zeros()
    if normalize_rows:
        inplace_csr_row_normalize(expmA)

    return expmA




def csc_row_normalize(X):
    """Row normalize scipy sparse csc matrices.
    returns a copy of X row-normalized and in CSC format.
    """
    X = X.tocsr()

    for i in range(X.shape[0]):
        row_sum = X.data[X.indptr[i]:X.indptr[i+1]].sum()
        if row_sum != 0:
            X.data[X.indptr[i]:X.indptr[i+1]] /= row_sum

    return X.tocsc()

def find_spectral_gap(L):
    """L is assummed to be connected"""
    Lcsr = L.tocsr()

    I = eye(L.shape[0],
                dtype=np.float64,
                format="csr")

    degs = np.diff((I-Lcsr).indptr)

    D12 = diags(np.sqrt(degs),
                format="csr")
    Dm12 = diags(1/np.sqrt(degs),
                format="csr")

    Lsym = D12 @ Lcsr @ Dm12

    # stationary solution
    Pi = np.vstack([degs/degs.sum()]*L.shape[0])


    gap = eigsh(Lsym.toarray()-Pi,1,sigma=0,
                return_eigenvectors=False)

    return gap


def remove_nnz_rowcol(L):
    """Returns a CSC or CSR matrix where the indices with zero row and column
    cols have been removed, also return an array of the indices of rows/columns
    with non-zero values and the (linear) size of L.
        
        
    Returns
    -------
    L_small, nonzero_indices, size
        
    """
    # indicies with zero sum row AND col
    nonzerosum_rowcols = ~np.logical_and(L.getnnz(1)==0,
                                        L.getnnz(0)==0)

    nonzero_indices, = (nonzerosum_rowcols).nonzero()

    return L[nonzerosum_rowcols][:,nonzerosum_rowcols], nonzero_indices, L.shape[0]





def numpy_rebuild_nnz_rowcol(T_data,
                              T_indices,
                              T_indptr,
                              zero_indices):
    """Returns a CSR matrix (data,indices,rownnz, shape) built from the CSR
    matrix T_small but with
    added row-colums at zero_indicies (with 1 on the diagonal)
        
     
    """
    n_rows = T_indptr.size-1 + zero_indices.size

    data = np.zeros(T_data.size+zero_indices.size,
                                   dtype=np.float64)
    indices = np.zeros(T_data.size+zero_indices.size,
                                   dtype=np.int32)
    indptr = np.zeros(n_rows+1,
                                   dtype=np.int32)
    new_col_inds = np.zeros(T_indptr.size-1,
                                   dtype=np.int32)
    Ts_indices = np.zeros(T_indices.size,
                                   dtype=np.int32)
    zero_set = set(zero_indices)


    # map col indices to new positions
    k = 0
    for i in range(n_rows):
        if i not in zero_set:
            new_col_inds[k] = i
            k +=1



    for k,i in enumerate(T_indices):
        Ts_indices[k] = new_col_inds[i]

    row_id_small_t = -1
    data_ind = 0
    for row_id in range(n_rows):
        row_id_small_t +=1
        if row_id in zero_set:
            # add a row with just 1 on the diagonal
            data[data_ind] = 1.0
            indices[data_ind] = row_id
            indptr[row_id+1] = indptr[row_id]+1

            row_id_small_t -= 1
            data_ind += 1

        else:
            row_start = T_indptr[row_id_small_t]
            row_end = T_indptr[row_id_small_t+1]

            num_data_row = row_end - row_start

            data[data_ind:data_ind+num_data_row] = T_data[row_start:row_end]
            indices[data_ind:data_ind+num_data_row] = Ts_indices[row_start:row_end]
            indptr[row_id+1] = indptr[row_id]+num_data_row

            data_ind += num_data_row


    return (data, indices, indptr, n_rows)




def sparse_lapl_expm(L, fact, dense_expm=True, nproc=1,
                     thresh_ratio=None,
                     normalize_rows=True):
    """ 
    computes the matrix exponential of a laplacian L, expm(-fact*L), 
    considering only the non-zeros rows/cols of L


    Parameters
    ----------
    L : scipy sparse csc matrix
        Laplacian matrix with large proportion of zero rows/cols.
    fact : float
        factor in front of the laplacian
    dense_expm : boolean
        Whether to compute the expm on the small Laplacian as a dense
        or sparse array. Default is True.
    nproc : int, optional
        number of parallel processes for dense_expm=False. The default is 1.
    thresh_ratio: float, optional.
        Threshold ratio used to trim negligible values in the resulting matrix.
        Values smaller than `max(expm(A))/thresh_ratio` are set to 
        zero. For dense_expm=False. Default is None.
    normalize_rows: bool, optional.
        Whether rows of the resulting matrix are normalized to sum to 1.
        For dense_expm=False
        
    Returns
    -------
    expm(-fact*L) : `SparseStochMat` object
        Transition matrix 

    """
    if L.getnnz() == 0: #zero matrix
        # return identity
        return SparseStochMat.create_diag(L.shape[0])

    L_small, nz_inds, size = remove_nnz_rowcol(L)


    if nproc == 1:
        expm_func = partial(compute_subspace_expm,
                            A=-fact*L_small,
                            thresh_ratio=thresh_ratio,
                            normalize_rows=normalize_rows)

    else:
        expm_func = partial(compute_subspace_expm_parallel,
                            A=-fact*L_small,
                            nproc=nproc,
                            thresh_ratio=thresh_ratio,
                            normalize_rows=normalize_rows)


    if dense_expm:
        T_small = csr_matrix(expm(-fact*L_small.toarray()))
    else:

        # for large networks, try subspace expm
        L_small.eliminate_zeros()
        if L_small.shape[0] >= 1000:
            n_comp, comp_labels = connected_components(L_small,directed=False)
            if n_comp > 1 :
                T_small = expm_func(n_comp=n_comp,
                                    comp_labels=comp_labels)
            else:
                T_small = expm(-fact*L_small).tocsr()
        else:
            T_small = expm(-fact*L_small).tocsr()

    return SparseStochMat(size, T_small.data, T_small.indices,
                            T_small.indptr, nz_inds)


def sparse_lin_approx(T, t, Pi=None, t_s=10, nz_rowcols=None):
    """Linear approximation of a continuous time transition matrix
        for sparse transition matrices.
        
        Performs computation using `lin_approx_trans_matrix` 
        on a smallest L matrices with no zeros 
        row/cols and returns a SparseStochMat 
    

    Parameters
    ----------
    T : scipy sparse csr matrix
        Original full size laplacian matrix.
    t : float
        Interpolation time.
    Pi : scipcy sparse csr matrix, optional
        Transition matrix at stationarity. Same shape that T. 
        The default is None, i.e. computed from T.
    t_s : float, optional
        Stationarity time at which the interpolation reaches Pi. The default is 10.
    nz_rowcols : ndarray of int32
        indices of T of nonzero offdiagonal rows/cols to build a SparseStochMat 

    Returns
    -------
    Tapprox : SparseStochMat object
            Linear approximation at time t.

    """
    T_ss = SparseStochMat.from_full_csr_matrix(T, nz_rowcols=nz_rowcols)

    if Pi is None:
        Pi_small = compute_stationary_transition(T_ss.T_small)
    elif isinstance(Pi, SparseStochMat):
        Pi_small = Pi.T_small
    elif isinstance(Pi, csr_matrix):
        Pi_small = SparseStochMat.from_full_csr_matrix(Pi, nz_rowcols=nz_rowcols).T_small
    else:
        raise TypeError("Pi must be a csr or SparseStochMat.")

    Tapprox_small = lin_approx_trans_matrix(T_ss.T_small,
                                        t=t,
                                        Pi=Pi_small,
                                        t_s=t_s)

    return SparseStochMat(T_ss.size, Tapprox_small.data, Tapprox_small.indices,
                            Tapprox_small.indptr, T_ss.nz_rowcols)



def sparse_stationary_trans(T):
    """Parameters
    ----------
    T : scipy sparse csr matrix
        Discrete time transition matrix.

    Returns
    -------
    Pi : scipy sparse csr matrix
        Transition matrix at stationarity

    """
    T_ss = SparseStochMat.from_full_csr_matrix(T.tocsr())

    Pi_small = compute_stationary_transition(T_ss.T_small)

    return SparseStochMat(T_ss.size, Pi_small.data, Pi_small.indices,
                            Pi_small.indptr, T_ss.nz_rowcols)


def set_to_ones(Tcsr, tol=1e-8):
    """In place replaces ones in sparse matrix that are, within the tolerence, close
    to ones with actual ones
    """
    Tcsr.data[np.abs(Tcsr.data - 1) <= tol] = 1


def set_to_zeroes(Tcsr, tol=1e-8, relative=True, use_absolute_value=False):
    """In place replaces zeroes in sparse matrix that are, within the tolerence, close
    to zero with actual zeroes.
    If tol is None, does nothing
    """
    if tol is not None:
        if isinstance(Tcsr, SparseStochMat):
            Tcsr.set_to_zeroes(tol, relative=relative)
        elif isinstance(Tcsr, (csr_matrix,csc_matrix)):
            if Tcsr.data.size > 0:
                if relative:
                    # tol = tol*np.abs(Tcsr.data).max()
                    # finding the max of the absolute value without making a copy of the whole array
                    tol = tol*np.abs([Tcsr.data.min(),Tcsr.data.max()]).max()

                if use_absolute_value:
                    Tcsr.data[np.abs(Tcsr.data) <= tol] = 0
                else:
                    Tcsr.data[Tcsr.data <= tol] = 0

                Tcsr.eliminate_zeros()
        else:
            raise TypeError("Tcsr must be csc,csr or SparseStochMat")
