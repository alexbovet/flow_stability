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

import os
import time
from itertools import combinations
from multiprocessing import Pool, RawArray

import numpy as np
from scipy.sparse import csr_matrix

from .flow_stability import Clustering, SparseClustering, norm_var_information
from .sparse_stoch_mat import SparseAutocovMat

# A global dictionary storing the variables passed from the initializer.
var_dict = {}


def compute_parallel_clustering(clustering, num_repeat=50, nproc=4,
                                verbose=True, n_meta_iter_max=1000,
                                n_sub_iter_max=1000,
                                clust_verbose=False, print_num_loops=False):

    global var_dict
    # create arrays to share between processes

    p1raw = RawArray("d", clustering.p1)
    p2raw = RawArray("d", clustering.p2)
    N = clustering.num_nodes

    if type(clustering) is Clustering:
        Sraw = RawArray("d", clustering._S.flat)
        Traw = RawArray("d", clustering.T.flat)

        params_list = [(seed, verbose, n_meta_iter_max, n_sub_iter_max,
                        clust_verbose, print_num_loops) for \
                        seed in n_random_seeds(num_repeat)]

        if verbose:
            print("**** PID ", os.getpid(), f"starting pool of {nproc} processes for the clustering loop.")

        with Pool(nproc,
                  initializer=_init_sub_worker,
                  initargs=(Sraw, Traw, N, p1raw, p2raw),
                    ) as pool:
            clusters, stabilites, seeds = \
                zip(*pool.map(_compute_sym_clustering_fct, params_list,
                              chunksize=1))


        var_dict = {} # deleting shared arrays

        return clusters, stabilites, seeds



    elif type(clustering) is SparseClustering:
        PTindices = RawArray("l", clustering._S.PT.indices)
        PTindptr = RawArray("l", clustering._S.PT.indptr)
        PTdata = RawArray("d", clustering._S.PT.data)



        params_list = [(seed, verbose, n_meta_iter_max, n_sub_iter_max,
                        clust_verbose, print_num_loops) for \
                        seed in n_random_seeds(num_repeat)]

        if verbose:
            print("**** PID ", os.getpid(), f"starting pool of {nproc} processes for the clustering loop.")

        with Pool(nproc,
                  initializer=_init_sub_worker_sparse,
                  initargs=(PTdata, PTindices,
                            PTindptr, N,
                            p1raw, p2raw),
                    ) as pool:
            clusters, stabilites, seeds = \
                zip(*pool.map(_compute_sym_sparse_clustering_fct, params_list,
                              chunksize=1))

        var_dict = {} # deleting shared arrays

        return clusters, stabilites, seeds


def n_random_seeds(n):

    return [int.from_bytes(os.urandom(4), byteorder="big") for \
                                      _ in range(n)]

def _init_sub_worker(Sraw, Traw, N, p1raw, p2raw):
    # reconstruct A from shared arrays

    global var_dict

    var_dict["N"] = N
    var_dict["S"] = np.frombuffer(Sraw, dtype=np.float64).reshape((N,N))
    var_dict["p1"] = np.frombuffer(p1raw, dtype=np.float64)
    var_dict["p2"] = np.frombuffer(p2raw, dtype=np.float64)
    var_dict["T"] = np.frombuffer(Traw, dtype=np.float64).reshape((N,N))


def _init_sub_worker_sparse(PTdata, PTindices, PTindptr, N, p1raw, p2raw, PT_symmetric=True):
    # reconstruct A from shared arrays

    PT = csr_matrix((np.frombuffer(PTdata, dtype=np.float64),
                                  np.frombuffer(PTindices, dtype=np.int64),
                                  np.frombuffer(PTindptr, dtype=np.int64)),
                                 shape=(N,N))

    S = SparseAutocovMat(PT,
                           np.frombuffer(p1raw, dtype=np.float64),
                           np.frombuffer(p2raw, dtype=np.float64),
                           PT_symmetric=PT_symmetric)

    global var_dict

    var_dict["N"] = N
    var_dict["S"] = S
    var_dict["p1"] = S.p1
    var_dict["p2"] = S.p2
    var_dict["T"] = S.PT



def _compute_sym_clustering_fct(params):

    seed, verbose, n_meta_iter_max, n_sub_iter_max, clust_verbose, print_num_loops  = params

    t0 = time.time()
    if verbose:
        print("**** PID ", os.getpid(), "starting clustering at",
              time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),
              f" with seed {seed}")


    clustering = Clustering(p1=var_dict["p1"],
                                  p2=var_dict["p2"],
                                  T=var_dict["T"],
                                  S=var_dict["S"],
                                  rnd_seed=seed)

    n_loops = clustering.find_louvain_clustering(n_meta_iter_max=n_meta_iter_max,
                                                 n_sub_iter_max=n_sub_iter_max,
                                                verbose=clust_verbose,
                                                print_num_loops=print_num_loops)

    if verbose:
        print("**** PID ", os.getpid(), f"sym clust took {time.time()-t0:.4f}s, finished at",
              time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()), f", in {n_loops} loops.")

    return (clustering.partition.cluster_list, clustering.compute_stability(), seed)

def _compute_sym_sparse_clustering_fct(params):

    seed, verbose, n_meta_iter_max, n_sub_iter_max, clust_verbose, print_num_loops  = params

    t0 = time.time()
    if verbose:
        print("**** PID ", os.getpid(), "starting clustering at",
              time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),
              f" with seed {seed}")


    clustering = SparseClustering(p1=var_dict["p1"],
                                  p2=var_dict["p2"],
                                  T=var_dict["T"],
                                  S=var_dict["S"],
                                  rnd_seed=seed)

    n_loops = clustering.find_louvain_clustering(n_meta_iter_max=n_meta_iter_max,
                                                 n_sub_iter_max=n_sub_iter_max,
                                                verbose=clust_verbose,
                                                print_num_loops=print_num_loops)

    if verbose:
        print("**** PID ", os.getpid(), f"sym clust took {time.time()-t0:.4f}s, finished at",
              time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()), f", in {n_loops} loops.")

    return (clustering.partition.cluster_list, clustering.compute_stability(), seed)


def compute_parallel_nvi(list_of_cluster_lists, N, nproc=4, verbose=True,
                         num_varinf_samples=None):

    num_repeat = len(list_of_cluster_lists)

    max_num_varinf_samples = int(num_repeat*(num_repeat-1)/2)
    if num_varinf_samples is None or num_varinf_samples > max_num_varinf_samples:
        num_varinf_samples = max_num_varinf_samples

    nvarinf_samples_idx = np.random.choice(range(max_num_varinf_samples),
                                                       num_varinf_samples,
                                                       replace=False)

    nvarinf_samples = [s for i,s in enumerate(combinations(list_of_cluster_lists,2))\
                                               if i in nvarinf_samples_idx]


    params = [(c1,c2, N, verbose) for c1, c2 in nvarinf_samples]

    if nproc == 1:
        nvarinf_sym_vals = list(map(_compute_nvi_sample, params))

    else:
        print("**** PID ", os.getpid(), f"starting pool of {nproc} processes for the NVI loop.")
        with Pool(nproc) as pool:
            nvarinf_sym_vals = list(pool.map(_compute_nvi_sample, params,
                              chunksize=1))

    return nvarinf_sym_vals

def _compute_nvi_sample(params):

    c1, c2, N, verbose = params

    tnvi = time.time()
    if verbose:
        print("**** PID ", os.getpid(), "starting NVI at",
              time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))

    nvi = norm_var_information(c1, c2, N,
                               use_clust_list=True)
    if verbose:
        print("PID ", os.getpid(), f" --- nvi sample took {time.time()-tnvi:.4f}s, finished at",
              time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))

    return nvi
