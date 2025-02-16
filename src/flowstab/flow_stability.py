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
from __future__ import annotations
from typing import Collection

import importlib.util
import os
import time
from array import array
from copy import deepcopy
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csc_matrix, csr_matrix, diags, eye, isspmatrix_csr
from scipy.sparse.linalg import eigs

from .sparse_stoch_mat import (
    USE_SPARSE_DOT_MKL,
    inplace_csr_matmul_diag,
    inplace_diag_matmul_csr,
    sparse_autocov_mat,
    sparse_gram_matrix,
    sparse_matmul,
    SparseStochMat,
)
from .temporal_network import (
    inplace_csr_row_normalize,
    set_to_zeroes,
    sparse_lapl_expm
)

USE_CYTHON = True
if importlib.util.find_spec("cython") is not None:
    from _cython_fast_funcs import (
        compute_S,
        cython_nmi,
        cython_nvi,
        sum_Sout,
        sum_Sto
    )
else:
    print("Could not load cython functions")
    USE_CYTHON = False



class Partition:
    """A node partition that can be described as a list of node sets
    and a node to cluster dict.
    """

    def __init__(self,
                 num_nodes:int,
                 cluster_list:Collection|None=None,
                 node_to_cluster_dict:dict|None=None,
                 check_integrity:bool=False):
        """
        Parameters
        ----------
        num_nodes:
          The number of nodes in the partition
        cluster_list:
          A list of clusters with each cluster being a set of nodes
        node_to_custer_dict:
          A mapping that maps each node to the index of the corresponding
          cluster in `cluster_list`
        """

        self.num_nodes = num_nodes

        if cluster_list is None and node_to_cluster_dict is None:
            # default clustering is one node per cluster
            self.cluster_list = [set([n]) for n in range(self.num_nodes)]
            self.node_to_cluster_dict = {n : n for n in range(self.num_nodes)}

        elif cluster_list is not None and node_to_cluster_dict is None:
            self.cluster_list = cluster_list
            self.node_to_cluster_dict = {}
            for i, clust in enumerate(self.cluster_list):
                for node in clust:
                    self.node_to_cluster_dict[node] = i

        elif cluster_list is None and node_to_cluster_dict is not None:
            self.node_to_cluster_dict = node_to_cluster_dict
            self.cluster_list = [set() for _ in \
                                 range(max(node_to_cluster_dict.values()) + 1)]
            for node, clust in node_to_cluster_dict.items():
                self.cluster_list[clust].add(node)

        elif cluster_list is not None and node_to_cluster_dict is not None:
            raise ValueError("cluster_list and node_to_cluster_dict " +\
                             "cannot be provided together")

        self.remove_empty_clusters()
        if check_integrity:
            self.check_integrity()

    def __repr__(self):

        return f"Partition with {self.num_nodes} nodes and {self.get_num_clusters()} clusters."


    def move_node(self,node, c_f):
        """Moves a node to cluster `c_f`"""
        # initial cluster
        c_i = self.node_to_cluster_dict[node]

        if c_i != c_f:
            self.node_to_cluster_dict[node] = c_f
            self.cluster_list[c_i].remove(node)
            self.cluster_list[c_f].add(node)
        else:
            print(f"Warning, node is already in {c_f}")


    def remove_empty_clusters(self):
        """Removes empty clusters from `cluster_list` and  `node_to_cluster_dict`
        and reindexes clusters.
        """
        self.cluster_list = [c for c in self.cluster_list if len(c) > 0 ]
        self.node_to_cluster_dict = {}
        for i, clust in enumerate(self.cluster_list):
            for node in clust:
                self.node_to_cluster_dict[node] = i

    def get_num_clusters(self, non_empty=False):

        if non_empty:
            return len([c for c in self.cluster_list if len(c)>0])
        else:
            return len(self.cluster_list)

    def get_indicator(self, sparse=True):
        """Returns an `N x c` indicator matrix `H` such that each row of `H`
        correspond to one node and is all zeros except for a one 
        indicating the cluster to which it belongs:
                
            `h_ik = 1` iff node i is in cluster k, and zero otherwise.
                
        if sparse=True, returns a csc sparse matrix
            
        """
        if sparse:
            # each column correspond to a cluster
            Hdata = np.ones(self.num_nodes, dtype=np.int32)
            Hindices = np.zeros(self.num_nodes, dtype=np.int32)
            Hindptr = np.zeros(self.get_num_clusters()+1, dtype=np.int32)

            H = csc_matrix((self.num_nodes,self.get_num_clusters()),
                     dtype=np.int32)


            for i,c in enumerate(self.cluster_list):
                nc = len(c)
                Hindptr[i+1] = Hindptr[i] + nc
                Hindices[Hindptr[i]:Hindptr[i+1]] = list(c)

            return csc_matrix((Hdata, Hindices, Hindptr), shape=(self.num_nodes,self.get_num_clusters()),
                     dtype=np.int32)
        else:
            H = np.zeros((self.num_nodes,self.get_num_clusters()),
                         dtype=int)

            for i,c in enumerate(self.cluster_list):
                H[list(c),i] = 1

            return H

    def iter_cluster_node_index(self):
        """Returns an iterator giving a list of node indices for each cluster"""
        for c in self.cluster_list:
            yield list(c)

    def __str__(self):

        return str(self.cluster_list)

    def check_integrity(self):
        """Check that all nodes are in only one cluster,
        i.e. non-overlapping clusters whose union is the full set of nodes.
        """
        inter = set()
        total_len = 0
        num_clusters = self.get_num_clusters()
        for i in range(num_clusters):
            total_len += len(self.cluster_list[i])
            for j in range(num_clusters):
                if i != j:
                    inter.update(self.cluster_list[i].intersection(self.cluster_list[j]))

        if len(inter) > 0:
            raise ValueError("Overlapping clusters.")

        if total_len != self.num_nodes:
            raise ValueError("Some nodes have no clusters.")



class BaseClustering:
    """BaseClustering.
        
    Base Class for autocovariance matrix clustering with the Louvain algorithm.
        
    At least `T` or `S` must be given to initialize the clustering.
    
    Clusters can either be initilized with a cluster_list or a node_to_cluster_dict.


    Parameters
    ---------- 
    T: numpy.ndarrays
        NxN transition matrix, T[i,j] is the probability of going from node i to
        node j between t1 and t2.
        
    p1: numpy.ndarrays
        Nx1 probability density at t1. Default is the uniform probability.
        
    p2: numpy.ndarrays
        Nx1 probability density at t2. Default is p1 @ T.
        
    S: numpy.ndarrays
        NxN covariance matrix. Default is diag(p1) @ T - outer(p1,p2).
        
    source_cluster_list: list
        list of set of nodes describing the source partition. Default is singleton
        clusters.
        
    source_node_to_cluster_dict: dict
        dictionary with mapping between nodes and source cluster number. Default is singleton
        clusters.
        
    target_cluster_list: list
        list of set of nodes describing the target partition. Default is singleton
        clusters.
        
    target_node_to_cluster_dict: dict
        dictionary with mapping between nodes and target cluster number. Default is singleton
        clusters.
            
    rnd_state: np.random.RandomState
        Random state object. Default creates a new one.
        
    rnd_seed: int
        Seed for the random object. Default is a random seed.
        
    S_threshold: float
        Smallest values of S. Used to trim insignificantly small values.
    
    """

    def __init__(self, T=None, p1=None, p2=None, S=None,
                       source_cluster_list=None,
                       source_node_to_cluster_dict=None,
                       target_cluster_list=None,
                       target_node_to_cluster_dict=None,
                       rnd_state=None, rnd_seed=None,
                       S_threshold=None):

        if T is None and S is None:
            raise ValueError("At least T or S must be provided")

        if T is None:
            #only is S provided, T will only be used to look for neighours.
            # consider a fully connected network.
            self.num_nodes = S.shape[0]
            self.T = np.ones_like(S)
            self.T /= self.T.sum(1)[:,np.newaxis]

        else:
            self.num_nodes = T.shape[0]

            if not isinstance(T,np.ndarray):
                raise TypeError("T must be a numpy array.")

            if isinstance(T,np.matrix):
                raise TypeError("T must be a numpy array, not a numpy matrix.")

            assert np.all(np.logical_or(np.isclose(T.sum(1),np.ones(T.shape[1])),
                                        np.isclose(T.sum(1),np.zeros(T.shape[1])))),\
                                        "Transition matrix must be stochastic with possible zero rows"

            self.T = T.copy()

        if p1 is None:
            # uniform distribution
            p1 = np.ones(self.num_nodes)/self.num_nodes

        if p2 is None:
            p2 = p1 @ self.T

        if not (isinstance(p1, np.ndarray) and \
                isinstance(p2, np.ndarray)):
            raise TypeError("p1 and p2 must be numpy arrays.")

        if isinstance(p1, np.matrix) or \
                isinstance(p2, np.matrix):
            raise TypeError("p1 and p2 must be numpy arrays, not numpy matrices.")


        self.p1 = p1

        self.p2 = p2

        self.S_threshold = S_threshold

        if S is None:
            # compute stability matrix
            self._S = self._compute_S(S_threshold=S_threshold)
        else:
            if not isinstance(S, np.ndarray):
                raise TypeError("S must be numpy arrays.")
            if isinstance(S,np.matrix):
                raise TypeError("S must be numpy arrays, not numpy matrices.")

            assert S.shape == self.T.shape, "T and S must have the same shape."

            self._S = S.copy()

            if S_threshold is not None:
                self._S[np.where(np.abs(S)<S_threshold)] = 0

        # initialize clusters
        self.source_part = Partition(self.num_nodes,
                                     cluster_list=source_cluster_list,
                                     node_to_cluster_dict=source_node_to_cluster_dict)

        self.target_part = Partition(self.num_nodes,
                                     cluster_list=target_cluster_list,
                                     node_to_cluster_dict=target_node_to_cluster_dict)


        # random number generator
        if rnd_state is not None:
            self._rnd_state = rnd_state
        else:
            self._rnd_state = np.random.RandomState(rnd_seed)


        # list of out and in neighbors arrays, include potential self loops
        self._out_neighs = []
        self._in_neighs = []
        self._neighs = []
        for node in range(self.num_nodes):
            self._out_neighs.append(np.nonzero(self.T[node,:] > 0)[0].tolist())
            self._in_neighs.append(np.nonzero(self.T[:,node] > 0)[0].tolist())
            self._neighs.append(list(set(self._out_neighs[node] + self._in_neighs[node])))





    def _compute_S(self, S_threshold=None):
        """Computes the internal matrix comparing probabilities for each
        node
                S[i,j] = p1[i]*T[i,j] - p1[i]*p2[j]
                
        Saves the matrix in `self._S`.
        """
        if USE_CYTHON:
            S = compute_S(self.p1,self.p2,self.T)
        else:
            S = np.diag(self.p1) @ self.T - np.outer(self.p1,self.p2)

        if S_threshold is not None:
            S[np.where(np.abs(S)<S_threshold)] = 0

        return S



    def _compute_clustered_autocov(self, partition=None):
        """Compute the clustered autocovariance matrix based on `source_part`
        and `target_part`.
            
        `partition` is a tuple with `(source_part, target_part)`.
            
        Default partitions are `self.source_part` and `self.target_part`.
        """
        if partition is None:
            source_part = self.source_part
            target_part = self.target_part
        else:
            source_part, target_part = partition

        num_s_clusters = source_part.get_num_clusters()
        num_t_clusters = target_part.get_num_clusters()

        R = np.zeros((num_s_clusters,num_t_clusters))

        # get indices for correct broadcasting
        t_cluster_to_node_list = {ic : np.array(cl)[np.newaxis,:] for ic,cl in \
                                enumerate(target_part.iter_cluster_node_index())}

        s_cluster_to_node_list = {ic : np.array(cl)[:,np.newaxis] for ic,cl in \
                            enumerate(source_part.iter_cluster_node_index())}

        for s in range(num_s_clusters):
            if len(source_part.cluster_list[s]) > 0:
                for t in range(num_t_clusters):
                    if len(target_part.cluster_list[t]) > 0 :
                        # idx = np.ix_(cluster_to_node_list[s], cluster_to_node_list[t])
                        R[s,t] = self._S[s_cluster_to_node_list[s],
                                         t_cluster_to_node_list[t]].sum()

        return R

    @staticmethod
    def _find_optimal_flow(R):
        """For a given clustered autocovariance matrix `R`, returns the optimal
        flow, i.e. the perfect matching between clusters that give the maximum
        stability
            
        Returns
        -------
            flow_stab, flow_map, flow_map_inv
            
        """
        row_ind, col_ind = linear_sum_assignment(1-R)

        return (R[row_ind, col_ind].sum(),
                {s:t for s,t in zip(row_ind,col_ind)},
                {t:s for s,t in zip(row_ind,col_ind)})


    def compute_stability(self, R=None):
        """Returns the stability
            
        """
        raise NotImplementedError


    def _compute_new_R_moveto(self, k, c_i,
                                       c_f,
                                       Rold,
                                       partition=None):
        """Return the new clustered autocov matrix obtained from `Rold`
        by moving node `k` from `c_i = (c_i_s,c_i_t)` into bi-cluster 
        `c_f = (c_f_s,c_f_t)`.
            
        If given, the list of original clusters is given by 
        `partition = (source_part, target_part)`.
        Otherwise is taken from `self.source_part` and `self.target_part`.
            
        `Rold` should be the output of `_compute_delta_R_moveout`.
            
        `c_f` may be an empty cluster
        """
        c_i_s, c_i_t = c_i
        c_f_s, c_f_t = c_f

        if partition is None:
            source_part = self.source_part
            target_part = self.target_part
        else:
            source_part, target_part = partition

        Rnew = Rold.copy()

        num_s_clusters = source_part.get_num_clusters()
        num_t_clusters = target_part.get_num_clusters()

        t_cluster_to_node_list = {ic : np.array(cl, dtype=int)[np.newaxis,:] for ic,cl in \
                                enumerate(target_part.iter_cluster_node_index())}

        s_cluster_to_node_list = {ic : np.array(cl, dtype=int)[:,np.newaxis] for ic,cl in \
                            enumerate(source_part.iter_cluster_node_index())}

        for s in range(num_s_clusters):
            Rnew[s,c_f_t] +=  self._S[s_cluster_to_node_list[s],k].sum()

        for t in range(num_t_clusters):
            Rnew[c_f_s,t] +=  self._S[k,t_cluster_to_node_list[t]].sum()

        # we remove Skk from the c_i because it's not there anymore
        Rnew[c_i_s,c_f_t] -= self._S[k,k]
        Rnew[c_f_s,c_i_t] -= self._S[k,k]

        # we add the diagonal because it was not in the sums
        Rnew[c_f_s,c_f_t] += self._S[k,k]

        return Rnew


    def _compute_new_R_moveout(self, k, c_i,
                                 partition=None,
                                 Rold=None):
        """Return the  new clustered autocov matrix obtained from `Rold`
        by moving node k out of the bi-cluster `c_i = (c_i_s, c_i_t)`.
            
        If given, the list of original clusters is given by 
        `partition = (source_part, target_part)`.
        Otherwise is taken from `self.source_part` and `self.target_part`.
            
        If `Rold` is not given, it will be recomputed.
            
        c_i is assumed to be non-empty!
            
        """
        c_i_s, c_i_t = c_i

        if partition is None:
            source_part = self.source_part
            target_part = self.target_part
        else:
            source_part, target_part = partition

        if Rold is None:
            Rold = self._compute_clustered_autocov(partition=(source_part,target_part))

        Rnew = Rold.copy()

        if k not in source_part.cluster_list[c_i_s] or \
           k not in target_part.cluster_list[c_i_t]:
            raise ValueError("node k must be in bicluster (c_i_s, c_i_t)")

        num_s_clusters = source_part.get_num_clusters()
        num_t_clusters = target_part.get_num_clusters()


        t_cluster_to_node_list = {ic : np.array(cl, dtype=int)[np.newaxis,:] for ic,cl in \
                            enumerate(target_part.iter_cluster_node_index())}

        s_cluster_to_node_list = {ic : np.array(cl, dtype=int)[:,np.newaxis] for ic,cl in \
                            enumerate(source_part.iter_cluster_node_index())}


        for s in range(num_s_clusters):
            Rnew[s,c_i_t] -=  self._S[s_cluster_to_node_list[s],k].sum()

        for t in range(num_t_clusters):
            Rnew[c_i_s,t] -=  self._S[k,t_cluster_to_node_list[t]].sum()

        # we add S[k,k] because it was counted twice in the sums
        Rnew[c_i_s,c_i_t] += self._S[k,k]


        return Rnew

    def _potential_new_clusters(self, node):
        """Returns a set of potential source and target clusters where to move
        `node`.
        """
        raise NotImplementedError



    def _compute_subsets_connectedness(self, s1, s2):
        r"""`s1` and `s2` must be two lists of nodes.
        
        Returns
        -------
            
        .. math::
            \sum_{i\in s_1}\sum_{j\in s_2} S_{i,j}
                
        The fraction of walkers going from s1 to s2 minus the expected
        value of the same quantity.                
            
        """
        #reshape for correct indexing
        s1 = [[i] for i in s1]

        return self._S[s1,s2].sum()


    def _louvain_move_nodes(self,
                           delta_r_threshold=np.finfo(float).eps,
                           n_sub_iter_max=1000,
                           verbose=False,
                           print_num_loops=False):
        """Return delta_r_tot, n_loop
        
        """
        delta_r_tot = 0
        delta_r_loop = 1
        n_loop = 1


        R = self._compute_clustered_autocov()

        stability = self.compute_stability(R)

        while (delta_r_loop > delta_r_threshold) and (n_loop < n_sub_iter_max):

            delta_r_loop = 0

            if verbose:
                print("\n-------------")
                print("Louvain sub loop number " + str(n_loop))

            if print_num_loops:
                if not n_loop%100:
                    print("       PID ", os.getpid(),
                          f" starting sub loop: {n_loop}")

            # shuffle order to process the nodes
            node_ids = np.arange(self.num_nodes)
            self._rnd_state.shuffle(node_ids)

            for node in node_ids:
                # test gain of stability if we move node to neighbours communities

                # initial cluster of node
                c_i = self._get_node_cluster(node)

                if verbose >1:
                    print(f"++ treating node {node} from cluster {c_i}")

                # new R if we move node out of (c_i_s,c_i_t)
                R_out = self._compute_new_R_moveout(node, c_i,
                                                    Rold=R)

                # find potential communities where to move node
                comms = self._potential_new_clusters(node)


                delta_r_best = 0
                c_f_best = c_i
                for c_f in comms:
                    if c_f != c_i:
                        # new R if we move node there
                        Rnew = self._compute_new_R_moveto(node, c_i,
                                                              c_f,
                                                              R_out)
                        # total gain of moving node
                        delta_r = self.compute_stability(Rnew) - stability

                        if verbose >= 10:
                            print(" -- checking cluster: ", c_f)
                            print(" -- delta_r: ", delta_r)
                        # we use `>=` to allow for more mixing (can be useful)
                        if delta_r >= delta_r_best:
                            delta_r_best = delta_r
                            c_f_best = c_f
                            Rnew_best = Rnew

                if c_f_best != c_i:
                    #move node to best_source cluster
                    self._move_node_to_cluster(node,c_f_best)


                    delta_r_loop += delta_r_best
                    stability += delta_r_best
                    R = Rnew_best

                    if verbose > 1:
                        print(f"moved node {node} from cluster {c_i} to cluster {c_f_best}")

                # else do nothing
                elif verbose > 1:
                        print(f"node {node} in clusters ({c_i}) has not moved")




            if verbose:
                print("\ndelta r loop : " + str(delta_r_loop))
                print("delta r total : " + str(delta_r_tot))
                print("number of clusters : " + \
                      str(self._get_num_clusters()))
                if verbose>1:
                    print("** clusters : ")
                    for cl in self._get_cluster_list().items():
                        print("** ", cl)

                if delta_r_loop == 0:
                    print("No changes, exiting.")

            delta_r_tot += delta_r_loop

            n_loop += 1

        # remove empty clusters
        self._remove_empty_clusters()

        return delta_r_tot, n_loop


    def _aggregate_clusters(self, partition=None):
        """For each of the c_s source clusters given by `source_part`
        and c_t clusters given by `target_part`, aggregates 
        the corresponding nodes in a new meta node.
            
        `partition=(source_part, target_part)`.
            
        Returns `T`, `p1` and `p2` the corresponding c_sxc_t transition matrix
        a 1xc_s and a 1xc_t probability vector.
         
        """
        if partition is None:
            source_part = self.source_part
            target_part = self.target_part
        else:
            source_part, target_part = partition

        num_s_clusters = source_part.get_num_clusters()
        num_t_clusters = target_part.get_num_clusters()

        p1 = np.zeros(num_s_clusters)
        p2 = np.zeros(num_t_clusters)
        T = np.zeros((num_s_clusters,num_t_clusters))
        S = np.zeros((num_s_clusters,num_t_clusters))

        # get indices for correct broadcasting
        t_cluster_to_node_list = {ic : np.array(cl)[np.newaxis,:] for ic,cl in \
                                enumerate(target_part.iter_cluster_node_index())}

        s_cluster_to_node_list = {ic : np.array(cl)[:,np.newaxis] for ic,cl in \
                            enumerate(source_part.iter_cluster_node_index())}

        for s in range(num_s_clusters):
            p1[s] = self.p1[s_cluster_to_node_list[s]].sum()
            for t in range(num_t_clusters):
                p2[t] = self.p2[t_cluster_to_node_list[t]].sum()

                # idx = np.ix_(cluster_to_node_list[s], cluster_to_node_list[t])
                T[s,t] = self.T[s_cluster_to_node_list[s],
                               t_cluster_to_node_list[t]].sum()

                S[s,t] = self._S[s_cluster_to_node_list[s],
                               t_cluster_to_node_list[t]].sum()

        # renormalize T
        T = T/T.sum(axis=1)[:, np.newaxis]
        #also normalize p1 and p2 for rounding errors
        p1 = p1/p1.sum()
        p2 = p2/p2.sum()

        return T, p1, p2, S

    def _get_updated_partition(self, og_old_part, meta_new_part):
        """Returns an updated version of original og_old_part corresponding
        to the meta_new_part
        """
        og_old_part.remove_empty_clusters()
        meta_new_part.remove_empty_clusters()

        new_node_to_cluster_dict = dict()
        # this is the state before moving nodes
        for meta_node, original_nodes in enumerate(og_old_part.cluster_list):

            for node in original_nodes:

                # this is the state after moving nodes
                new_node_to_cluster_dict[node] = \
                    meta_new_part.node_to_cluster_dict[meta_node]


        return Partition(og_old_part.num_nodes,
                         node_to_cluster_dict=new_node_to_cluster_dict)

    def find_louvain_clustering(self, delta_r_threshold=np.finfo(float).eps,
                                n_meta_iter_max=1000,
                                n_sub_iter_max=1000,
                                verbose=False,
                                rnd_seed=None,
                                save_progress=False,
                                print_num_loops=False):
        """Returns n_meta_loop
        
        `rnd_seed` is used to set to state of the random number generator.
        Default is to keep the current state
           
        """
        # implement loop control based on delta r difference

        # random numbers from seed rnd_seed
        if rnd_seed is not None:
            self._rnd_state.seed(rnd_seed)

        if save_progress:
            self._save_progress()

        # if the autocovariance is all zeros, return the full partiton
        if self._check_if_S_is_zero():
            self._update_to_fullpart()
            if verbose:
                print("S is all zeros")

            return 0

        # initial clustering
        cluster_dict = deepcopy(self._get_cluster_list())

        meta_clustering = self.__class__(T=self.T, p1=self.p1, p2=self.p2,
                                         S=self._S,
                                         rnd_state=self._rnd_state,
                                         **cluster_dict)

        delta_r_meta_loop = 1
        n_meta_loop = 0
        stop = False
        while not ((delta_r_meta_loop <= delta_r_threshold) or stop or \
                                                   (n_meta_loop > n_meta_iter_max)):

            n_meta_loop += 1

            if verbose:
                print("\n**************")
                print("* Louvain meta loop number " + str(n_meta_loop))


            if print_num_loops:
                print("       PID ", os.getpid(),
                      f" starting meta loop: {n_meta_loop}")


            # sub loop
            delta_r_meta_loop, n_sub_loops = meta_clustering._louvain_move_nodes(delta_r_threshold,
                                                n_sub_iter_max,
                                                verbose,
                                                print_num_loops)

            if print_num_loops:
                print("       PID ", os.getpid(),
                      f" finished meta loop: {n_meta_loop}, number of total sub loops: {n_sub_loops}")

            #update original node partition
            self._update_original_partition(meta_clustering)


            if (all([len(c) == 1 for clust_list in self._get_cluster_list().values() \
                     for c in clust_list])) or \
                   n_meta_loop > n_meta_iter_max or \
                   delta_r_meta_loop <= delta_r_threshold:
                stop = True # we've reached the best partition
                if verbose:
                    print("Reached best partition.")
            else:
                # cluster aggregation

                if verbose:
                    print("Aggregating clusters.")
                T, p1, p2, S = self._aggregate_clusters()
                if verbose:
                    print("Finished aggregating clusters.")

                meta_clustering = self.__class__(T=T, p1=p1, p2=p2, S=S,
                                             rnd_state=self._rnd_state)

            if save_progress:
                self._save_progress()


            if verbose:
                print("\n*  delta r meta loop : " + str(delta_r_meta_loop))
                print("*  number of clusters : " + \
                      str(self._get_num_clusters()))

                if verbose>1:
                    print("** clusters : ")
                    for cl in self._get_cluster_list().items():
                        print("** ", cl)
                print("*  end of meta loop num : ", n_meta_loop)


        return n_meta_loop


    def _save_progress(self):
        if not hasattr(self,"source_part_progress"):
            self.source_part_progress = []
        if not hasattr(self,"target_part_progress"):
            self.target_part_progress = []

        if not hasattr(self, "stability_progress"):
            self.stability_progress = []

        self.source_part_progress.append(deepcopy(self.source_part))
        self.target_part_progress.append(deepcopy(self.target_part))

        self.stability_progress.append(self.compute_stability())

    def _get_node_cluster(self, node):
        """Returns the bi-cluster `(c_s, c_t)` in which `node` is."""
        raise NotImplementedError


    def _move_node_to_cluster(self, node, c):
        """Move `node` to the bi-cluster `c`."""
        raise NotImplementedError

    def _get_cluster_list(self):
        """Returns a dictionary of source and target clusters lists."""
        raise NotImplementedError

    def _get_num_clusters(self):
        """Returns a tuple with the number of source and target clusters."""
        raise NotImplementedError

    def _remove_empty_clusters(self):
        """Remove empty source and target clusters."""
        self.source_part.remove_empty_clusters()
        self.target_part.remove_empty_clusters()

    def _update_original_partition(self, meta_clustering):
        """Update the partion of `self` based on `meta_clustering`"""
        self.target_part = self._get_updated_partition(self.target_part,
                                                     meta_clustering.target_part)

        self.source_part = self._get_updated_partition(self.source_part,
                                                     meta_clustering.source_part)

    def _update_to_fullpart(self):

        self.target_part = Partition(num_nodes=self.num_nodes,
                                     cluster_list=[set(range(self.num_nodes))])
        self.source_part = Partition(num_nodes=self.num_nodes,
                                     cluster_list=[set(range(self.num_nodes))])

    def _check_if_S_is_zero(self, thresh=1e-6):

        return np.abs(self._S).max()< thresh/(self.num_nodes**2)


class Clustering(BaseClustering):
    """Symmetric Clustering.
    
    Finds the best partition that optimizes the stability between two times 
    defined as the trace of the clustered autocovariance matrix.
        
    
    At least `T` must be given to initialize the clustering.
    
    Clusters can either be initilized with a cluster_list or a node_to_cluster_dict.


    Parameters
    ---------- 
    T: numpy.ndarrays
        NxN transition matrix, T[i,j] is the probability of going from node i to
        node j between t1 and t2.
        
    p1: numpy.ndarrays
        Nx1 probability density at t1. Default is the uniform probability.
        
    p2: numpy.ndarrays
        Nx1 probability density at t2. Default is p1 @ T.
        
    S: numpy.ndarrays
        NxN covariance matrix. Default is diag(p1) @ T - outer(p1,p2).
        
    cluster_list: list
        list of set of nodes describing the partition. Default is singleton
        clusters.
        
    node_to_cluster_dict: dict
        dictionary with mapping between nodes and cluster number. Default is singleton
        clusters.
        
    rnd_state: np.random.RandomState
        Random state object. Default creates a new one.
        
    rnd_seed: int
        Seed for the random object. Default is a random seed.
    
    S_threshold: float
        Smallest values of S. Used to trim insignificantly small values.
        
    """

    def __init__(self, p1=None, p2=None,T=None, S=None,
                       cluster_list=None,
                       node_to_cluster_dict=None,
                       rnd_state=None, rnd_seed=None,
                       S_threshold=None):

        super().__init__(p1=p1, p2=p2,T=T, S=S,
                       source_cluster_list=cluster_list,
                       source_node_to_cluster_dict=node_to_cluster_dict,
                       target_cluster_list=None,
                       target_node_to_cluster_dict=None,
                       rnd_state=rnd_state,
                       rnd_seed=rnd_seed,
                       S_threshold=S_threshold)

        # create an alias for source partition since we only need one partition
        self.partition = self.source_part
        self.target_part = None


    def compute_stability(self, R=None):
        """Returns the stability of the clusters given in `cluster_list`
        computed between times `t1` and `t2`
            
        Here, for symmetric clustering, we only care about the diagonal
        of R.
            
        """
        if R is None:
            R = self._compute_clustered_autocov()

        return R.sum()


    def _compute_clustered_autocov(self, partition=None):
        """Compute the clustered autocovariance matrix based on `partition`.
            
        Default partition is `self.source_part`.
            
        Here, for symmetric clustering, we only care about the diagonal
        of R.
        """
        if partition is None:
            partition = self.source_part

        num_clusters = partition.get_num_clusters()

        # diagonal of R
        R = np.zeros(num_clusters)

        # get indices for correct broadcasting
        cluster_to_node_list = {ic : np.array(cl) for ic,cl in \
                            enumerate(partition.iter_cluster_node_index())}

        for s in range(num_clusters):
            if len(partition.cluster_list[s]) > 0:
                idx = np.ix_(cluster_to_node_list[s], cluster_to_node_list[s])
                R[s] = self._S[idx].sum()

        return R

    def _compute_delta_stab_moveto(self, k, c_f,
                                 partition=None):
        """Return the gain in stability obtained by moving node
        k into community c_f.
            
        If given, the list of original clusters is given by `partition` and
        otherwise is taken from `self.source_part`.
            
        c_f may be an empty cluster
        """
        if partition is None:
            partition = self.partition

        if k in partition.cluster_list[c_f]:
            raise ValueError("node k must not be in cluster c_f")

        # indexes of nodes in c_f
        ix_cf = list(partition.cluster_list[c_f])

        # gain in stability from moving node k to community c_f
        if USE_CYTHON:
            delta_r1 = sum_Sto(self._S, k, ix_cf)
        else:
            delta_r1 = self._S[k,ix_cf].sum() \
                        + self._S[ix_cf,k].sum() \
                        + self._S[k,k]

        return delta_r1


    def _compute_delta_stab_moveout(self, k, c_i,
                                 partition=None):
        """Return the gain in stability obtained by moving node
        k out of community c_i.
            
        If given, the list of clusters is given by `partition` and
        otherwise is taken from `self.partition`.
            
        c_i is assumed to be non-empty!
            
        """
        if partition is None:
            partition = self.partition

        if k not in partition.cluster_list[c_i]:
            raise ValueError("node k must be in cluster c_i")


        # indexes of nodes in c_i
        ix_ci = list(partition.cluster_list[c_i])

        # gain in stability from moving node k out of community c_i
        if USE_CYTHON:
            delta_r2 = sum_Sout(self._S, k, ix_ci)
        else:
            delta_r2 = - self._S[k,ix_ci].sum() \
                       - self._S[ix_ci,k].sum() \
                       + self._S[k,k]
                   # we add S[k,k] because it was counted twice in the sums

        return delta_r2


    def _louvain_move_nodes(self,
                           delta_r_threshold=np.finfo(float).eps,
                           n_sub_iter_max=1000,
                           verbose=False,
                           print_num_loops=False):
        """Return delta_r_tot, n_loop
        
        """
        delta_r_tot = 0
        delta_r_loop = 1
        n_loop = 1


        while (delta_r_loop > delta_r_threshold) and (n_loop < n_sub_iter_max):

            delta_r_loop = 0

            moved_nodes = set()

            if verbose:
                print("\n-------------")
                print("Louvain sub loop number " + str(n_loop))

            if print_num_loops:
                if not n_loop%100:
                    print("       PID ", os.getpid(),
                          f" starting sub loop: {n_loop}")

            # shuffle order to process the nodes
            node_ids = np.arange(self.num_nodes)
            self._rnd_state.shuffle(node_ids)

            for node in node_ids:
                # test gain of stability if we move node to neighbours communities

                # initial cluster of node
                c_i = self._get_node_cluster(node)

                if verbose >1:
                    print(f"++ treating node {node} from cluster {c_i}")

                # delta stab if we move node out of (c_i_s,c_i_t)
                r_out = self._compute_delta_stab_moveout(node, c_i)

                if verbose > 10:
                    print("+++ delta_r_out: ", r_out)

                # find potential communities where to move node
                comms = self._potential_new_clusters(node)


                delta_r_best = 0
                c_f_best = c_i
                for c_f in comms:
                    if c_f != c_i:
                        # new delta r if we move node there
                        r_in = self._compute_delta_stab_moveto(node, c_f)
                        if verbose > 10:
                            print(" --- testing cluster: ",c_f)
                            print(" --- delta_r_in: ", r_in)
                        # total gain of moving node
                        delta_r = r_out + r_in

                        if verbose >= 10:

                            print(" --- delta_r: ", delta_r)
                        # we use `>=` to allow for more mixing (can be useful)
                        if delta_r >= delta_r_best:
                            delta_r_best = delta_r
                            c_f_best = c_f

                if c_f_best != c_i:
                    #move node to best_source cluster
                    self._move_node_to_cluster(node,c_f_best)

                    moved_nodes.add(node)

                    delta_r_loop += delta_r_best

                    if verbose > 1:
                        print(f"moved node {node} from cluster {c_i} to cluster {c_f_best}")

                # else do nothing
                elif verbose > 1:
                        print(f"node {node} in clusters ({c_i}) has not moved")




            if verbose:
                print("\ndelta r loop : " + str(delta_r_loop))
                print("delta r total : " + str(delta_r_tot))
                print("number of clusters : " + \
                      str(self._get_num_clusters()))
                print(f"moved nodes: {moved_nodes}")
                if verbose>1:
                    print("** clusters : ")
                    for cl in self._get_cluster_list().items():
                        print("** ", cl)

                if delta_r_loop == 0:
                    print("No changes, exiting.")

            delta_r_tot += delta_r_loop

            n_loop += 1

        # remove empty clusters
        self._remove_empty_clusters()

        return delta_r_tot, n_loop

    def _potential_new_clusters(self, node):
        """Returns a set of potential source and target clusters where to move
        `node`.
            
        The current cluster of node is not included.
        """
        comms = {self.partition.node_to_cluster_dict[neigh] for \
                                                neigh in self._neighs[node]}

        # remove initial community if it is there
        comms.discard(self.partition.node_to_cluster_dict[node])

        return comms

    def _compute_new_R_moveto(self, k, c_i,
                                       c_f,
                                       Rold,
                                       partition=None):


       raise Exception("Not used in Clustering")

    def _compute_new_R_moveout(self, k, c_i,
                                 partition=None,
                                 Rold=None):

       raise Exception("Not used in Clustering")


    def _aggregate_clusters(self, partition=None):
        """For each of the c clusters given by `partition`, aggregates
        the corresponding nodes in a new meta node.
            
        Default `partition` is `self.source_part`.
            
        Returns `T`, `p1`, `p2` and `S` the corresponding cxc transition matrix
        a 1xc, a 1xc probability vector and the cxc covariance matrix.
         
        """
        if partition is None:
            partition = self.source_part

        num_clusters = partition.get_num_clusters()

        p1 = np.zeros(num_clusters)
        p2 = np.zeros(num_clusters)
        T = np.zeros((num_clusters,num_clusters))
        S = np.zeros((num_clusters,num_clusters))

        # get indices for correct broadcasting
        t_cluster_to_node_list = {ic : np.array(cl)[np.newaxis,:] for ic,cl in \
                                enumerate(partition.iter_cluster_node_index())}

        s_cluster_to_node_list = {ic : np.array(cl)[:,np.newaxis] for ic,cl in \
                            enumerate(partition.iter_cluster_node_index())}

        for s in range(num_clusters):
            p1[s] = self.p1[s_cluster_to_node_list[s]].sum()
            for t in range(num_clusters):
                p2[t] = self.p2[t_cluster_to_node_list[t]].sum()

                # idx = np.ix_(cluster_to_node_list[s], cluster_to_node_list[t])
                T[s,t] = self.T[s_cluster_to_node_list[s],
                               t_cluster_to_node_list[t]].sum()
                S[s,t] = self._S[s_cluster_to_node_list[s],
                               t_cluster_to_node_list[t]].sum()
        # renormalize T
        T = T/T.sum(axis=1)[:, np.newaxis]
        #also normalize p1 and p2 for rounding errors
        p1 = p1/p1.sum()
        p2 = p2/p2.sum()

        return T, p1, p2, S

    def _save_progress(self):
        if not hasattr(self,"partition_progress"):
            self.partition_progress = []

        if not hasattr(self, "stability_progress"):
            self.stability_progress = []

        self.partition_progress.append(deepcopy(self.source_part))

        self.stability_progress.append(self.compute_stability())

    def _get_node_cluster(self, node):
        """Returns the bi-cluster `(c_s, c_t)` in which `node` is."""
        return self.source_part.node_to_cluster_dict[node]


    def _move_node_to_cluster(self, node, c):
        """Move `node` to the bi-cluster `c`."""
        self.source_part.move_node(node, c)

    def _get_cluster_list(self):
        """Returns a dictionary of source and target clusters lists."""
        return {"cluster_list" : self.source_part.cluster_list}

    def _get_num_clusters(self):
        """Returns a tuple with the number of source and target clusters."""
        return self.source_part.get_num_clusters(non_empty=True)

    def _remove_empty_clusters(self):
        """Remove empty source and target clusters."""
        self.source_part.remove_empty_clusters()

    def _update_original_partition(self, meta_clustering):
        """Update the partion of `self` based on `meta_clustering`"""
        self.source_part = self._get_updated_partition(self.source_part,
                                                     meta_clustering.source_part)

    def _update_to_fullpart(self):

        self.source_part = Partition(num_nodes=self.num_nodes,
                                     cluster_list=[set(range(self.num_nodes))])


    def find_louvain_clustering(self, delta_r_threshold=np.finfo(float).eps,
                                n_meta_iter_max=1000,
                                n_sub_iter_max=1000,
                                verbose=False,
                                rnd_seed=None,
                                save_progress=False,
                                print_num_loops=False):
        """Returns n_meta_loop
        
        `rnd_seed` is used to set to state of the random number generator.
        Default is to keep the current state
           
        """
        n_loop = super().find_louvain_clustering(delta_r_threshold=delta_r_threshold,
                                n_meta_iter_max=n_meta_iter_max,
                                n_sub_iter_max=n_sub_iter_max,
                                verbose=verbose,
                                rnd_seed=rnd_seed,
                                save_progress=save_progress,
                                print_num_loops=print_num_loops)

        self.partition = self.source_part
        self.target_part = None

        if print_num_loops:
            print("       PID ", os.getpid(), f" total num. of meta loops: {n_loop}")

        return n_loop




class SparseClustering(Clustering):
    """Symmetric Clustering using sparse matrices.
    
    Finds the best partition that optimizes the stability between two times 
    defined as the trace of the clustered autocovariance matrix.
        
    
    At least `T` must be given to initialize the clustering.
    
    Clusters can either be initilized with a cluster_list or a node_to_cluster_dict.


    Parameters
    ---------- 
    T: scipy csr_sparse matrix
        NxN transition matrix, T[i,j] is the probability of going from node i to
        node j between t1 and t2.
        
    p1: numpy.ndarrays
        Nx1 probability density at t1. Default is the uniform probability.
        
    p2: numpy.ndarrays
        Nx1 probability density at t2. Default is p1 @ T.
        
    S: sparse_autocov_mat
        NxN covariance matrix. Default is diag(p1) @ T - outer(p1,p2).
        
    cluster_list: list
        list of set of nodes describing the partition. Default is singleton
        clusters.
        
    node_to_cluster_dict: dict
        dictionary with mapping between nodes and cluster number. Default is singleton
        clusters.
        
    rnd_state: np.random.RandomState
        Random state object. Default creates a new one.
        
    rnd_seed: int
        Seed for the random object. Default is a random seed.
    
    S_threshold: float
        Smallest values of S. Used to trim insignificantly small values.
        
    """

    def __init__(self, p1=None, p2=None,T=None, S=None,
                       cluster_list=None,
                       node_to_cluster_dict=None,
                       rnd_state=None, rnd_seed=None):

        if T is None and S is None:
            raise ValueError("At least T or S must be provided")

        if T is None:
            assert isinstance(S, sparse_autocov_mat), "S must be a sparse_autocov_mat."

            # only if S provided, T will only be used to look for neighours.
            # so set T to S.PT.
            self.num_nodes = S.shape[0]
            self.T = S.PT

        else:

            self.num_nodes = T.shape[0]

            if not (isinstance(T, SparseStochMat) or isspmatrix_csr(T)):
                raise TypeError("T must be a csr or SparseStochMat.")

            # assert np.allclose(T.sum(1),np.ones(T.shape[1])),\
            #                     "Transition matrix must be stochastic"

            self.T = T.copy()

        if p1 is None:
            # uniform distribution
            p1 = np.ones(self.num_nodes)/self.num_nodes

        if p2 is None:
            p2 = p1 @ self.T

        if not (isinstance(p1, np.ndarray) and \
                isinstance(p2, np.ndarray)):
            raise TypeError("p1 and p2 must be numpy arrays.")

        if isinstance(p1, np.matrix) or \
                isinstance(p2, np.matrix):
            raise TypeError("p1 and p2 must be numpy arrays, not numpy matrices.")


        self.p1 = p1

        self.p2 = p2


        if S is None:
            # compute stability matrix
            self._S = self._compute_S()

        else:
            if not isinstance(S, sparse_autocov_mat):
                raise TypeError("S must be a sparse_autocov_mat.")
            assert S.shape == self.T.shape, "T and S must have the same shape."

            self._S = S.copy()


        # initialize clusters
        self.source_part = Partition(self.num_nodes,
                                     cluster_list=cluster_list,
                                     node_to_cluster_dict=node_to_cluster_dict)

        # create an alias for source partition since we only need one partition
        self.partition = self.source_part
        self.target_part = None


        # random number generator
        if rnd_state is not None:
            self._rnd_state = rnd_state
        else:
            self._rnd_state = np.random.RandomState(rnd_seed)


        # list of out and in neighbors arrays, include potential self loops
        self._out_neighs = []
        self._in_neighs = []
        self._neighs = []
        for node in range(self.num_nodes):
            self._out_neighs.append([self._S.PT.indices[i] for i in \
                                     range(self._S.PT.indptr[node],self._S.PT.indptr[node+1])])

            self._in_neighs.append([self._S.PTcsc.indices[i] for i in \
                                     range(self._S.PTcsc.indptr[node],self._S.PTcsc.indptr[node+1])])

            self._neighs.append(list(set(self._out_neighs[node] + self._in_neighs[node])))


    def _compute_S(self):
        """Computes the internal matrix comparing probabilities for each
        node as a sparse_autocov_mat
            
                S[i,j] = p1[i]*T[i,j] - p1[i]*p2[j]
                
        Saves the matrix in `self._S`.
        """
        return sparse_autocov_mat.from_T(self.T,
                                         self.p1,
                                         self.p2)


    def compute_stability(self, R=None):
        """Returns the stability of the clusters given in `cluster_list`
        computed between times `t1` and `t2`
            
        Here, for symmetric clustering, we only care about the diagonal
        of R.
            
        """
        if R is None:
            R = self._compute_clustered_autocov()

        return R.sum()


    def _compute_clustered_autocov(self, partition=None):
        """Compute the clustered autocovariance matrix based on `partition`.
            
        Default partition is `self.source_part`.
            
        Here, for symmetric clustering, we only care about the diagonal
        of R.
        """
        if partition is None:
            partition = self.source_part

        num_clusters = partition.get_num_clusters()

        # diagonal of R
        R = np.zeros(num_clusters)

        # get indices for correct broadcasting
        cluster_to_node_list = {ic : array("i",cl) for ic,cl in \
                            enumerate(partition.iter_cluster_node_index())}

        for s in range(num_clusters):
            if len(partition.cluster_list[s]) > 0:
                # idx = np.ix_(cluster_to_node_list[s], cluster_to_node_list[t])
                R[s] = self._S.get_submat_sum(cluster_to_node_list[s],
                                              cluster_to_node_list[s])
        return R

    def _compute_delta_stab_moveto(self, k, c_f,
                                 partition=None):
        """Return the gain in stability obtained by moving node
        k into community c_f.
            
        If given, the list of original clusters is given by `partition` and
        otherwise is taken from `self.source_part`.
            
        c_f may be an empty cluster
        """
        if partition is None:
            partition = self.partition

        if k in partition.cluster_list[c_f]:
            raise ValueError("node k must not be in cluster c_f")

        # indexes of nodes in c_f
        ix_cf = array("i",partition.cluster_list[c_f]) # use typed array for maximum speed

        # gain in stability from moving node k to community c_f

        return self._S._compute_delta_S_moveto(k, ix_cf)


    def _compute_delta_stab_moveout(self, k, c_i,
                                 partition=None):
        """Return the gain in stability obtained by moving node
        k out of community c_i.
            
        If given, the list of clusters is given by `partition` and
        otherwise is taken from `self.partition`.
            
        c_i is assumed to be non-empty!
            
        """
        if partition is None:
            partition = self.partition

        if k not in partition.cluster_list[c_i]:
            raise ValueError("node k must be in cluster c_i")


        # indexes of nodes in c_i
        ix_ci = array("i",partition.cluster_list[c_i])

        # gain in stability from moving node k out of community c_i
        return self._S._compute_delta_S_moveout(k, ix_ci)





    def _aggregate_clusters(self, partition=None, verbose=False):
        """For each of the c clusters given by `partition`, aggregates
        the corresponding nodes in a new meta node.
            
        Default `partition` is `self.source_part`.
            
        Returns `T`, `p1`, `p2` and `S` the corresponding cxc transition matrix
        a 1xc, a 1xc probability vector and the cxc covariance matrix.
         
        """
        if partition is None:
            partition = self.source_part

        if partition.get_num_clusters() == self.num_nodes:
            # nothing to aggregate

            return self.T, self.p1, self.p2, self._S

        else:
            S = self._S.aggregate(list(partition.iter_cluster_node_index()))

            # pass p1 and p2 as None since they could be floats
            return S.PT, None, None, S


    def _check_if_S_is_zero(self):

        return self._S.is_all_zeros()




class FlowIntegralClustering:
    """FlowIntegralClustering.
        
    Class to finds the best partition that optimizes the integral of the 
    autocovariance between two times computed as 
    
    int_t1^t2 [diag(p1) @ T(t1,t) @ np.diag(1/pt) @ T(t1,t).T @ np.diag(p1) - outer(p1,p1)] dt
    
    `T_list` or `T_inter_list` must be given to initialize the clustering.
    
    Clusters can either be initilized with a cluster_list or a node_to_cluster_dict.


    Parameters
    ---------- 
    T_list: list of scipy.sparse.csr matrices or numpy.ndarray
        list of K succesive NxN transition matrices, Tk[i,j] is the probability of
        going from node i to node j between `t1` and `tk+1`.
        
    T_inter_list: list of scipy.sparse.csr matrices or numpy.ndarray
        list of K succesive NxN inter event transition matrices, 
        T_inter_k[i,j] is the probability of
        going from node i to node j between `tk` and `tk+1`.
        
    p1: numpy.ndarrays
        Nx1 probability density at t1. Default is the uniform probability.
        
    time_list: list or numpy.array
        list of K+1 time instants corresponding to the `tk` of the transition
        matrices. Default is unit times.
        
    integral_time_grid: list
        list of times until which to compute the integral. The final times and
        indices used are stored in _t_integral_grid and _k_integral_grid.
        Default is [time_list[0], time_list[-1]]
        
    reverse_time: bool
        Whether to reverse time when computing T_list and I_list from T_inter_list,
        for backward flow stability. Default is False (forward flow stability).
        If T_list is provided in input, it must have been computed with the 
        corresponding time direction.
        
    rtol : float
        Relative tolerance to set I values to zero. 
        Values smaller than I.max()*rtol are set to zero to 
        keep I sparse.
                    
    """

    def __init__(self, T_list=None, p1=None, time_list=None,
                       T_inter_list=None,
                       integral_time_grid=None,
                       reverse_time=False,
                       rtol=1e-8,
                       verbose=False):

        if T_list is None and T_inter_list is None:
            raise ValueError("T_list or T_inter_list must be provided")


        if T_list is not None:
            is_sparse = False
            is_nparray = False
            is_sparse_stoch = False
            if isspmatrix_csr(T_list[0]):
                is_sparse = True
            elif isinstance(T_list[0],np.ndarray):
                is_nparray = True
                if isinstance(T_list[0],np.matrix):
                    raise TypeError("T_inter_list must contain numpy arrays" + \
                                     " (not numpy matrices)" + \
                                     " or scipy CSR matrices.")

            elif isinstance(T_list[0], SparseStochMat):
                is_sparse_stoch = True

            if not (is_sparse or is_nparray or is_sparse_stoch):
                raise TypeError("T_list must contain numpy arrays" + \
                                     ", scipy CSR matrices or SparseStochMat.")

            self.T_list = T_list
        else:
            is_sparse = False
            is_nparray = False
            is_sparse_stoch = False

            if isspmatrix_csr(T_inter_list[0]):
                is_sparse = True
            elif isinstance(T_inter_list[0],np.ndarray):
                is_nparray = True
                if isinstance(T_inter_list[0],np.matrix):
                    raise TypeError("T_inter_list must contain numpy arrays" + \
                                     " (not numpy matrices)" + \
                                     " or scipy CSR matrices.")

            elif isinstance(T_inter_list[0], SparseStochMat):
                is_sparse_stoch = True

            if not (is_sparse or is_nparray or is_sparse_stoch):
                raise TypeError("T_inter_list must contain numpy arrays" + \
                                     ", scipy CSR matrices or SparseStochMat.")

            if reverse_time:
                T_inter_list = T_inter_list[::-1]

            self.T_inter_list=T_inter_list

            self.T_list = self._compute_T_list(T_inter_list, is_nparray, verbose=verbose)

        self.is_nparray = is_nparray
        self.is_sparse = is_sparse
        self.is_sparse_stoch = is_sparse_stoch

        self.num_nodes = self.T_list[0].shape[0]

        if time_list is None:
            time_list = list(range(len(self.T_list)+1))

        self.time_list=np.array(time_list)

        if reverse_time:
            self.time_list = self.time_list[::-1]

        if integral_time_grid is None:
            self.integral_time_grid = [self.time_list[0], self.time_list[-1]]
            self._k_integral_grid = [0,len(self.time_list)-1]
            self._t_integral_grid = [self.time_list[0], self.time_list[-1]]
        else:
            # check if the ordering of integral_time_grid matched the direction of time
            dire = np.diff(integral_time_grid).mean()
            if reverse_time and dire > 0:
                integral_time_grid = integral_time_grid[::-1]
            if not reverse_time and dire < 0:
                integral_time_grid = integral_time_grid[::-1]

            self.integral_time_grid = integral_time_grid
            # indices and times where to store the integral values
            self._k_integral_grid = []
            self._t_integral_grid = []

            for t in self.integral_time_grid:
                if t not in self.time_list:
                # take the largest smaller time
                    if not reverse_time and t <= self.time_list[0] or reverse_time and t >= self.time_list[0]:
                        t = self.time_list[0]
                    elif not reverse_time:
                        t = self.time_list[self.time_list <= t].max()
                    else:
                        t = self.time_list[self.time_list >= t].min()


                k = int(np.where(self.time_list == t)[0])
                self._k_integral_grid.append(k)
                self._t_integral_grid.append(t)

        if p1 is None:
            #uniform distribution
            p1 = np.ones(self.num_nodes, dtype=np.float64)/self.num_nodes

        self.p1 = p1

        PT_list = self._compute_integral(self.T_list, self.time_list,
                                        rtol=rtol,
                                        verbose=verbose)


        if is_nparray:
            self.I_list = [PT - np.outer(self.p1,self.p1) for PT in PT_list]
        else:
            # if p1 uniform:
            if (self.p1 == self.p1[0]).all():
                pp1 = self.p1[0]
            else:
                pp1 = self.p1

            self.I_list = [sparse_autocov_mat(PT, pp1, pp1, PT_symmetric=True) for PT in PT_list]


        self.clustering = {}

        self.partition = {}

    def _compute_T_list(self, T_inter_list, is_nparray, verbose=False):
        """Computes the list of transition matrices Tk from t0 to tk using
        the interevent transition matrices
        """
        if verbose:
            print("PID ", os.getpid(), " : computing T_list")

        if is_nparray:
            T_list = [T_inter_list[0]]
        else:
            T_list = [T_inter_list[0].tocsr()]

        for k in range(1,len(T_inter_list)):
            if is_nparray:
                T_list.append(T_list[-1] @ T_inter_list[k])
            else:
                T_list.append(sparse_matmul(T_list[-1], T_inter_list[k].tocsr()))

                # to correct precision errors
                inplace_csr_row_normalize(T_list[-1])

        return T_list

    def _compute_integral(self, T_list, time_list, verbose=False,
                          rtol=1e-8):
        """Computes time integral of P(t1) @ Tk @ P(1/pk) @ Tk.T @ P(t1) as
        
        P(t1) @ \frac{1}{T}\\int_0^T T(t) @ P(1/p(t)) @ T(t).T * dt @ P(t1)
        
        rtol : float
            Relative tolerance to set I values to zero. 
            Values smaller than I.max()*rtol are set to zero to 
            keep I sparse.
        """
        dt_list = np.diff(time_list)
        if time_list[-1] < time_list[0]:
            #reverse time
            dt_list *= -1


        total_time = 0

        IPT_list = []

        if self.is_nparray:
            IPT = np.zeros((self.num_nodes,self.num_nodes), dtype=np.float64)
            # diag(p)
            P = lambda d: np.diag(d)
            keep_sparse = False
        elif self.is_sparse or self.is_sparse_stoch:
            IPT = csr_matrix((self.num_nodes,self.num_nodes), dtype=np.float64)
            # diag(p)
            keep_sparse = True


        t0 = time.time()

        if verbose:
            print("PID ", os.getpid(), " : computing integral")

        tkm1 = time.time()
        for k, (Tk, dtk) in enumerate(zip(T_list, dt_list)):
            # Tk is the transition from t0 to tk
            if verbose and not k%1000:
                print("PID ", os.getpid(), " : ",k, " over " ,
                      len(T_list), f" took {time.time()-tkm1:.2f}s")
                tkm1 = time.time()


            if self.is_nparray:
                # dense matrix version
                pk = self.p1 @ Tk
                # in order to avoid nan in PTk due to 0 * np.inf
                pk[np.where(pk == 0)] = 1

                PTk = Tk @ P(1/pk) @ Tk.T * dtk
            else:
                # sparse matrix version
                pk = sparse_matmul(self.p1, Tk.tocsr(),
                                    verbose=verbose>=10,
                                    log_message="pk")

                # in order to avoid nan in PTk due to 0 * np.inf
                pk[np.where(pk == 0)] = 1

                # we compute Tk @ Pk^-1 @ Tk.T as (Tk @ Pk^-1/2) @ (Tk @ Pk^-1/2)^T

                PTk = Tk.copy().tocsr()

                inplace_csr_matmul_diag(PTk,np.sqrt(1/pk))

                PTk = sparse_gram_matrix(PTk, transpose=True,
                                            verbose=verbose>=10,
                                            log_message="ITPTk")

                PTk.data *= dtk # operating on data avoids making a copy here.


            if keep_sparse:
                set_to_zeroes(PTk, rtol, use_absolute_value=True)
                set_to_zeroes(IPT, rtol, use_absolute_value=True)
                PTk.eliminate_zeros()
                IPT.eliminate_zeros()

            IPT = IPT + PTk

            total_time += dtk

            if k+1 in self._k_integral_grid: # this step was the integral from tk to tk_+1
                if self.is_nparray:
                    IPT_list.append(P(self.p1) @ IPT @ P(self.p1) * (1/total_time))
                else:
                    # multiply on left and right by P1
                    IPT_copy = IPT.copy()
                    inplace_diag_matmul_csr(IPT_copy, self.p1)
                    inplace_csr_matmul_diag(IPT_copy, self.p1)
                    IPT_copy.eliminate_zeros()
                    IPT_copy.data *= (1/total_time)
                    if USE_SPARSE_DOT_MKL:
                        # this means that only the upper triangular part of IPT
                        # was computed
                        IPT_copy = IPT_copy + IPT_copy.T - diags(IPT_copy.diagonal())
                    IPT_list.append(IPT_copy)


        t1 = time.time()

        if verbose:
            print(f"integral took {t1-t0:.2f}s")



        return IPT_list


    def find_louvain_clustering(self, k=0,
                                delta_r_threshold=np.finfo(float).eps,
                                n_meta_iter_max=1000,
                                n_sub_iter_max=1000,
                                verbose=False,
                                rnd_seed=None,
                                save_progress=False,
                                cluster_list=None,
                                node_to_cluster_dict=None,
                                rnd_state=None,
                                S_threshold=None):
        """Louvain algorithm to find the best partition.
        
        The best partition is saved in `self.partition[k]
        

        Parameters
        ----------
        k : int
            Index of the covariance integral to cluster. self.I_list[k] will be used.
        delta_r_threshold : float, optional
            Minimal value for an improvement of the quality function. The default is np.finfo(float).eps.
        n_meta_iter_max : int, optional
            Maximum number of meta iterations. The default is 1000.
        n_sub_iter_max : int, optional
            Maximum number of sub iterations. The default is 1000.
        verbose : bool, int, optional
            Degree of verbosity. The default is False.
        rnd_seed : int
            Seed for the random object. Default is a random seed.
        save_progress : bool, optional
            Whether to save the progress in the Clustering.partition_progress 
            and Clustering.stability_progress. The default is False.
        cluster_list : list of sets, optional
            list of set of nodes describing the partition. Default is singleton
            clusters.
        node_to_cluster_dict : dict, optional
            dictionary with mapping between nodes and cluster number. Default is singleton
            clusters.
        rnd_state : np.random.RandomState
            Random state object. Default creates a new one.
        S_threshold : float.
            Smallest values of the covariance. Used to trim insignificantly small values. Default is None (no thresholding)

        Returns
        -------
        n_loop : int
            Number of meta loops of the louvain algorithm.


        """
        if k in self.clustering.keys():
            print(f"index {k} already computed")
        else:

            if self.is_nparray:
                self.clustering[k] = Clustering(p1=self.p1, p2=None,
                                  T=self.T_list[self._k_integral_grid[k+1]-1],
                                  S=self.I_list[k],
                                  cluster_list=cluster_list,
                                  node_to_cluster_dict=node_to_cluster_dict,
                                  rnd_state=rnd_state, rnd_seed=rnd_seed,
                                  S_threshold=S_threshold)
            else:
                self.clustering[k] = SparseClustering(p1=self.p1, p2=None,
                                  S=self.I_list[k],
                                  cluster_list=cluster_list,
                                  node_to_cluster_dict=node_to_cluster_dict,
                                  rnd_state=rnd_state, rnd_seed=rnd_seed)

            n_loop = self.clustering[k].find_louvain_clustering(delta_r_threshold=delta_r_threshold,
                                n_meta_iter_max=n_meta_iter_max,
                                n_sub_iter_max=n_sub_iter_max,
                                verbose=verbose,
                                rnd_seed=rnd_seed,
                                save_progress=save_progress)

            self.partition[k] = self.clustering[k].partition

            return n_loop




def jaccard_distance(clusters1, clusters2):
    """Returns the Jaccard distance between two clustering.
    
    inputs can be node_to_cluster dictionaries or cluster lists of node sets
    """
    # convert to node_to_cluster_dict
    if isinstance(clusters1, list):
        node_to_cluster_dict = {}
        for i, clust in enumerate(clusters1):
            for node in clust:
                node_to_cluster_dict[node] = i
        clusters1 = node_to_cluster_dict

    if isinstance(clusters2, list):
        node_to_cluster_dict = {}
        for i, clust in enumerate(clusters2):
            for node in clust:
                node_to_cluster_dict[node] = i
        clusters2 = node_to_cluster_dict


    from itertools import combinations

    same = 0
    diff = 0
    for n1,n2 in combinations(clusters1.keys(), 2):
        same_in_1 = clusters1[n1] == clusters1[n2]
        same_in_2 = clusters2[n1] == clusters2[n2]

        if same_in_1 and same_in_2:
            same += 1
        if not same_in_1 and same_in_2:
            diff += 1
        if same_in_1 and not same_in_2:
            diff += 1

def norm_mutual_information(clusters1, clusters2):
    """Returns the normalized mutial information between two
    non-overlapping clustering.
        
    The mutual information is normalized by the max of the 
    two individual entropies.
        
    .. math::
        NMI = (H(C1)+H(C2)-H(C1,C2))/max(H(C1),H(C2))
    
    inputs can be node_to_cluster dictionaries, cluster lists of node sets
    or instances of Partition.
    """
    # convert to list of sets
    if isinstance(clusters1, dict):
        cluster_list = [set() for _ in \
                        range(max(clusters1.values()) + 1)]
        for node, clust in clusters1.items():
                cluster_list[clust].add(node)
        clusters1 = cluster_list

    if isinstance(clusters2, dict):
        cluster_list = [set() for _ in \
                        range(max(clusters2.values()) + 1)]
        for node, clust in clusters2.items():
                cluster_list[clust].add(node)
        clusters2 = cluster_list

    if isinstance(clusters1, Partition):
        clusters1 = clusters1.cluster_list

    if isinstance(clusters2, Partition):
        clusters2 = clusters2.cluster_list

    if not isinstance(clusters1[0], set):
        # make sure it is a list of sets
        clusters1 = [set(c) for c in clusters1]

    if not isinstance(clusters2[0], set):
        # make sure it is a list of sets
        clusters2 = [set(c) for c in clusters2]

    # num nodes
    N = sum(len(clust) for clust in clusters1)
    n1 = len(clusters1)
    n2 = len(clusters2)

    if USE_CYTHON:
        return cython_nmi(clusters1, clusters2, N, n1, n2)
    else:
        # loop over pairs of clusters
        p1 = np.zeros(n1) # probs to belong to clust1
        p12 = np.zeros(n1*n2) # probs to belong to clust1 & clust2
        k = 0
        for i,clust1 in enumerate(clusters1):
            p1[i] = len(clust1)/N
            for j, clust2 in enumerate(clusters2):
                p12[k] = len(clust1.intersection(clust2))/N
                k += 1

        p2 = np.array([len(clust2)/N for clust2 in clusters2])

        # Shannon entropies
        H1 = - np.sum(p1[p1 !=0]*np.log2(p1[p1 !=0]))
        H2 = - np.sum(p2[p2 !=0]*np.log2(p2[p2 !=0]))
        H12 = - np.sum(p12[p12 !=0]*np.log2(p12[p12 != 0]))

        # Mutual information
        MI = H1 + H2 - H12

        return MI/max((H1,H2))


def norm_var_information(clusters1, clusters2, N=None, use_clust_list=False):
    r"""Returns the normalized variation of information between two
    non-overlapping clustering.
        
    .. math::

        \hat{V}(C_1,C_2) = ({H(C1|C2)+H(C2|C1)})/{log_2 N}
    
    inputs can be node_to_cluster dictionaries, cluster lists of node sets
    or instances of Partition.
    """
    if not use_clust_list:

        # convert to list of sets
        if isinstance(clusters1, dict):
            cluster_list = [set() for _ in \
                            range(max(clusters1.values()) + 1)]
            for node, clust in clusters1.items():
                    cluster_list[clust].add(node)
            clusters1 = cluster_list

        if isinstance(clusters2, dict):
            cluster_list = [set() for _ in \
                            range(max(clusters2.values()) + 1)]
            for node, clust in clusters2.items():
                    cluster_list[clust].add(node)
            clusters2 = cluster_list

        if isinstance(clusters1, Partition):
            clusters1 = clusters1.cluster_list

        if isinstance(clusters2, Partition):
            clusters2 = clusters2.cluster_list


    if not isinstance(clusters1[0], set):
        # make sure it is a list of sets
        clusters1 = [set(c) for c in clusters1]

    if not isinstance(clusters2[0], set):
        # make sure it is a list of sets
        clusters2 = [set(c) for c in clusters2]

    # num nodes
    if N is None:
        N = sum(len(clust) for clust in clusters1)

    n1 = len(clusters1)
    n2 = len(clusters2)

    # trivial cases
    if n1 == n2 == N:
        return 0.0

    clusters1 = sorted(clusters1,key=lambda c:min(c))
    clusters2 = sorted(clusters2,key=lambda c:min(c))

    # other trivial case
    if (n1 == n2) and (clusters1 == clusters2):
        return 0.0

    if USE_CYTHON:
        return cython_nvi(clusters1, clusters2, N)
    else:
        # loop over pairs of clusters
        VI = 0.0
        for i in range(n1):
            clust1 = clusters1[i]
            ni = len(clust1)
            n_inter = 0
            for j in range(n2):
                clust2 = clusters2[j]
                nij = len(clust1.intersection(clust2))
                n_inter += nij
                if nij > 0:
                    nj = len(clust2)
                    VI -= nij*np.log2((nij**2)/(ni * nj))

                if n_inter >= ni:
                    # we have found all the possible intersections
                    break

        return VI/(N*np.log2(N))

def avg_norm_var_information(clusters_lists_list, num_samples=None):
    """Returns the average normalized variation of information for the
    clusters_list in clusters_lists_list.
        
    By default uses all N*(N-1)/2 combinations possible of cluster_list
    pairs, where N = len(clusters_lists_list)
    or uses num_samples pairs if num_samples < N*(N-1)/2.
        
    """
    num_repeat = len(clusters_lists_list)

    max_num_varinf_samples = int(num_repeat*(num_repeat-1)/2)


    if num_samples is not None:
        num_samples = min(num_samples, max_num_varinf_samples)
    else:
        num_samples = max_num_varinf_samples


    nvarinf_samples_idx = np.random.choice(range(max_num_varinf_samples),
                                                       num_samples,
                                                       replace=False)

    nvarinf_samples = [s for i,s in enumerate(combinations(clusters_lists_list,2)) \
                                               if i in nvarinf_samples_idx]

    N = sum(len(clust) for clust in clusters_lists_list[0])

    return np.mean([norm_var_information(c1,c2, N=N,
                        use_clust_list=True) for c1,c2 in nvarinf_samples])



def static_clustering(A, t=1, rnd_seed=None, discrete_time_rw=False,
                      linearized=False,
                      directed=False):
    """Initializes a clustering instance to optimize the continuous time
    Markov stability from the graph given by the adjacency matrix `A`.

    Parameters
    ----------
    A : scipy sparse csr matrix or numpy ndarray
        Adjacency matrix.
    t : int or float, optional
        Markov time (resolution parameter). The default is 1.
    rnd_seed : int, optional
        The default is None.
    discrete_time_rw : bool, optional
        If true, powers of the transition matrix to the power of `t` are used for varying resolution.
        `t` must be an int. The default is False.
    linearized : bool, optional
        If true, the linearized version of the Markov Stability is used.
        if linearized=false and discrete_time_rw==false, the matrix exponential
        of the random walk Laplacian is used to compute the transition matrix (slow).
        The default is False.
    directed : bool, optional
        If true, the network must be strongly connected. The default is False.

    Raises
    ------
    TypeError
        'A must be numpy array or scipy csr.'.
    ValueError
        'sum of degrees equal 0'.

    Returns
    -------
    Instance of `Clustering' or 'SparseClustering'
        Clustering instance initialized for Markov Stability optimization.

    """
    from scipy.linalg import expm

    if isinstance(A, csr_matrix):
        is_csr = True
        D = lambda d : diags(d, format="csr")
        I = lambda size : eye(size, format="csr")
        power = lambda M, n : M.__pow__(n)
    elif isinstance(A, np.ndarray) and not isinstance(A, np.matrix):
        is_csr = False
        D = lambda d : np.diag(d)
        I = lambda size : np.eye(size)
        power = lambda M, n : np.linalg.matrix_power(M, n)
    else:
        raise TypeError("A must be numpy array or scipy csr.")

    # degrees vector
    degs = np.asarray(A.sum(axis=1)).squeeze()

    degs_m1 = degs.copy()
    degs_m1[degs_m1==0] = 1
    degs_m1 = 1/degs_m1

    # A with selfloops
    A_sl = A.copy()

    A_sl[degs==0,degs==0] = 1
    # DTRW Transition matrix
    Tstat = D(degs_m1) @ A_sl


    # stationary solution
    if degs.sum() == 0:
        pi = degs
        raise ValueError("sum of degrees equal 0.")
    elif directed:

        if is_csr:

            w,v = eigs(Tstat.T, k=1, which="LM")

            assert np.allclose(np.abs(w[0]) ,1)

            pi = v.squeeze().real
            pi = pi/pi.sum()

        else:
            w, v = np.linalg.eig(Tstat.T)

            assert np.allclose(np.abs(w[0]) ,1)

            pi = v[:,0].real
            pi = pi/pi.sum()
    else:
        pi = degs/degs.sum()

    # transition matrix at time t
    if discrete_time_rw and isinstance(t,int):

        T = power(Tstat,t)

    elif linearized:
        T = I(degs.size) - t*(I(degs.size)-Tstat)

    else:
        # Random walk laplacian
        Lrw = I(degs.size) - Tstat

        # should be done
        if is_csr:
            print("Warning: the exponential of a sparse matrix is usually not sparse anymore.")
            T = sparse_lapl_expm(Lrw, t).tocsr()
        else:
            T = expm(-t*Lrw)

    if is_csr:
        return SparseClustering(p1=pi,p2=pi,T=T, rnd_seed=rnd_seed)
    else:
        return Clustering(p1=pi,p2=pi,T=T, rnd_seed=rnd_seed)


# TODO: move to helper scripts
def n_random_seeds(n):
    # generate n random seeds

    return [int.from_bytes(os.urandom(4), byteorder="big") for \
                                      _ in range(n)]


def run_multi_louvain(clustering, num_repeat, **kwargs):
    """Helper function to run multiple (serial) instances of the Louvain algorithm for the
    clustering given in `clutering`.

    Parameters
    ----------
    clustering : Clustering or SparseClustering instance
        
    num_repeat : int
        Number of repetition of the algorithm.
    **kwargs : key word args
        Arguments passed when initializing the clusterings.

    Returns
    -------
    n_loops : list
        list with the number of louvain loop for each repetition.
    cluster_lists : list
        list with the cluster lists of each repetition..
    stabilities : list
        list with the value of the stability for each repetition.
    seeds : list
        list with the random seeds used in each repetition.

    """
    cluster_lists = []
    stabilities = []
    seeds = n_random_seeds(num_repeat)
    n_loops = []

    for seed in seeds:

        # create a copy of clustering with a different seed
        c = clustering.__class__(p1=clustering.p1,
                                 p2=clustering.p2,
                                 S=clustering._S,
                                 T=clustering.T,
                                 rnd_seed=seed)

        n_loop = c.find_louvain_clustering(**kwargs)

        n_loops.append(n_loop)
        stabilities.append(c.compute_stability())
        cluster_lists.append(c.partition.cluster_list)


    return n_loops, cluster_lists, stabilities, seeds



def sort_clusters(cluster_list_to_sort, cluster_list_model, thresh_ratio=0.3):
    """Quick heuristic to sort a list of clusters in order to closely match another list
    """
    clust_similarity_lists = []
    for clust in cluster_list_to_sort:
        jaccs = []
        for class_clust in cluster_list_model:
            jaccs.append(len(clust.intersection(class_clust))/len(clust.union(class_clust)))
        clust_similarity_lists.append(jaccs)

    #now sort
    clust_similarity_matrix = np.array(clust_similarity_lists)
    new_clust_order = []
    all_clusts = list(range(clust_similarity_matrix.shape[0]))

    zero_clusts = (clust_similarity_matrix.sum(1) == 0).nonzero()[0].tolist()
    for z in zero_clusts:
        all_clusts.remove(z)

    while len(new_clust_order) < len(cluster_list_to_sort) - len(zero_clusts):
        for cla in range(clust_similarity_matrix.shape[1]):
            # loop on classes and sort according to most similar

            sorted_comms = clust_similarity_matrix[all_clusts,cla].argsort()[::-1]
            scores = clust_similarity_matrix[all_clusts,cla][sorted_comms]
            if scores.max() > 0:
                scores /= scores.max()
                for c, s in zip(sorted_comms,scores):
                    if s >= thresh_ratio and all_clusts[c] not in new_clust_order:
                            new_clust_order.append(all_clusts[c])

        # update all_clusts
        for n in new_clust_order:
            if n in all_clusts:
                all_clusts.remove(n)

    return [cluster_list_to_sort[i] for i in new_clust_order + zero_clusts]

