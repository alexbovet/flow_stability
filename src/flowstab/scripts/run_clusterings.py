# Flow stability for dynamic community detection.
# Bovet, A., Delvenne, J.-C. & Lambiotte, R.  Sci. Adv. 8, eabj3063 (2022).
# https://www.science.org/doi/10.1126/sciadv.abj3063
#
# Copyright (C) 2021 Alexandre Bovet <alexandre.bovet@math.uzh.ch>
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

"""Clusters the covariance integrals computed with `run_cov_integrals.py` (step 2).
The clusterings are computed with the Louvain algorithm for each file containing the result of step 2 in {savedir}.
The Louvain algorithm is repeated `num_repeat` times and the best partition (maximizing the flow stability) is selected.
The ensemble of partitions is used to compute the variation of information.
The computations for the different files is spread of `nproc_files` in parallel. It is also possible to parallelize the 
repeated clusterings using `nproc_clustering`. This requires nproc_files*nproc_clustering cores.

A number of metadata (e.g. all the parameters used to run the code) is saved together with the results in a dictionary.

The results of the clustering are saved with the following keys:
- 'clust_counter' : All the different partitions found at each run of the Louvain algorithm togheter with the number of times they appeared.
- 'stabilites' : All the values of the flow stability for each run.
- 'seeds' : Random seeds for each run.
- 'nvarinf' : Average normalized variation of information computed over the ensemble of partitions.
- 'avg_stab': Average flow stability computed over the ensemble of partitions.
- 'avg_nclust' : Average number of clusters computed over the ensemble of partitions.
- 'best_cluster' : Best partition (the one with the max flow stability from the ensemble).
- 'best_stab' : Flow stability value of the best partition.

The results are saved in files named f'{savedir}/clusters_{netname}_tau_w{tau_w:.3e}_PT_{int_start:06d}_to_{int_stop:06d}.pickle'

"""
import os
import pickle
import sys
import time
import traceback
from argparse import SUPPRESS, ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor as futurePool
from itertools import combinations
from multiprocessing import Pool, RawArray

import hickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components

from ..FlowStability import SparseClustering, norm_var_information, static_clustering
from ..SparseStochMat import (
    inplace_csr_matmul_diag,
    inplace_csr_row_normalize,
    inplace_diag_matmul_csr,
    sparse_autocov_mat,
)
from ..temporal_network import set_to_zeroes

# raise Exception


#%%

ap = ArgumentParser(prog="run_clusterings",
                    description=__doc__,
                    add_help=False,
                    formatter_class=ArgumentDefaultsHelpFormatter)

required = ap.add_argument_group("required arguments")
optional = ap.add_argument_group("optional arguments")

# Add back help
optional.add_argument(
    "-h",
    "--help",
    action="help",
    default=SUPPRESS,
    help="show this help message and exit",
)


required.add_argument("--datadir", default="", type=str,
                help="Location of the integrals of covariance matrices.")

required.add_argument("--savedir", default="", type=str,
                      help="Where the results will be saved.")

optional.add_argument("--net_name", default="synthtemp_heira_big", type=str)

optional.add_argument("--nproc_files", default=4, type=int,
                help="Number of processes over which to split files to work on.")

optional.add_argument("--nproc_clustering", default=1, type=int,
                help="Number of processes over which to split clustering iterations.")

optional.add_argument("--num_repeat", default=50, type=int,
                      help="Number of repetitions of the Louvain method. This is used to compute the variation of information at each scale. The partition maximizing the flow stability is saved.  ")

optional.add_argument("--n_meta_iter_max", default=100, type=int,
                      help="Maximum number of meta iterations of the Louvain method.")

optional.add_argument("--n_sub_iter_max", default=100, type=int,
                      help="Maximum number of sub iterations of the Louvain method.")

optional.add_argument("--verbose", default=0, type=int)

optional.add_argument("--clust_verbose", default=0, type=int,
                      help="Verbosity of the Louvain method.")

optional.add_argument("--compute_static_clustering", action="store_true",
                      help="Whether to compute the clustering of the static networks. Only possible if static adjacency matrices have been computed in `run_laplacians_transmats`")

optional.add_argument("--save_clustering_before_nvarinf", action="store_true",
                help="Saves clustering results in tempfile before computing NVAR info.")

optional.add_argument("--not_compute_clustering", action="store_false",
                      help="Use to not compute the clusterings.")

optional.add_argument("--num_varinf_samples", default=None, type=int,
                help="Number of samples to use to compute the variation of information. Default is the max value, i.e. 1/2*num_repeat*(num_repeat-1).")

optional.add_argument("--only_largest_comp", action="store_true",
                help="Compute the clustering only on the largest weakly connected component." +\
                    " All the nodes outside the LCC are aggregated in a single meta node "+\
                        "that is in position zero. The indices of the nodes in the LCC are saved in `largest_comp_nodes`.")

optional.add_argument("--separate_comps", action="store_true",
                help="Compute the clustering on each weakly connected component separately." +\
                    " The partitions are then merged in a single partition. Can make clustering faster for large networks. ")

optional.add_argument("--init_p1", action="store_true",
                help="For non-homogeneous initial distribution, must be used with --direction.")

optional.add_argument("--direction", default="forward",
                help="'forward' or 'backward', used with --init_p1.")

optional.add_argument("--print_num_loops",  action="store_true",
                      help="print the number of loops of the Louvain method.")

optional.add_argument("--integral_rtol", default=None, type=float,
                help="Values smaller than max(integral)*rtol are set to zero in the sparse integral.")

optional.add_argument("--not_normalize_cov_mat",  action="store_false",
                help="Do not normalize the covariance matrix before clustering. The normalization corrects for small numerical errors.")

optional.add_argument("--num_norm_iter", default=10, type=int,
                help="Number of iteration of the normalization of the covariance matrix.")


inargs = vars(ap.parse_args())
datadir = inargs["datadir"]
savedir = inargs["savedir"]
nproc_files = inargs["nproc_files"]
nproc_clustering = inargs["nproc_clustering"]

num_repeat = inargs["num_repeat"]
n_meta_iter_max = inargs["n_meta_iter_max"]
n_sub_iter_max = inargs["n_sub_iter_max"]
verbose = inargs["verbose"]
clust_verbose = inargs["clust_verbose"]
net_name = inargs["net_name"]
compute_static_clustering = inargs["compute_static_clustering"]
compute_clustering = inargs["not_compute_clustering"]
compute_nvarinf = compute_clustering

num_varinf_samples = inargs["num_varinf_samples"]
only_largest_comp = inargs["only_largest_comp"]
separate_comps = inargs["separate_comps"]
save_clustering_before_nvarinf = inargs["save_clustering_before_nvarinf"]
init_p1 = inargs["init_p1"]
direction = inargs["direction"]
print_num_loops = inargs["print_num_loops"]

integral_rtol = inargs["integral_rtol"]
normalize_cov_mat = inargs["not_normalize_cov_mat"]
num_norm_iter = inargs["num_norm_iter"]

max_num_varinf_samples = int(num_repeat*(num_repeat-1)/2)
if num_varinf_samples is None or num_varinf_samples > max_num_varinf_samples:
    num_varinf_samples = max_num_varinf_samples
#%%
if datadir == "":
    raise Exception("datadir must be given")

if savedir == "":
    raise Exception("savedir must be given")


print("Arguments:")
for item in inargs.items():
    print(item)


if only_largest_comp and separate_comps:
    raise ValueError("cannot have both only_largest_comp and separate_comps")

#%%

files = [f for f in os.listdir(datadir) if f.startswith(net_name + "_tau_w") and \
         ("temp_integration" not in f) and \
             ("pTpint" not in f)]

if init_p1:
    if direction == "forward":
        files = [f for f in files if "PTforw" in f]
    elif direction == "backward":
        files = [f for f in files if "PTback" in f]
    else:
        raise ValueError("direction must be forward or backward")



#%% load autocov



def load_autocov_int(file, init_p1=False, direction=None):

    largest_comp_nodes = None

    ITPT_sum = None
    max_S_row_diff = None
    ITPT_sum_norm = None
    max_S_row_diff_norm = None
    normalization = None

    if init_p1==False:

    # this is the case with uniform p1

        # S is the autocov integral I
        # we need to recreate
        # I = P1 (1/(t2-t1))int_t1^t2 T Pt T^T dt P1 - p1^T p1

        print("PID ", os.getpid(),
              f", autocov integral with homogeneous p1. loading {file}")

        if os.path.splitext(file)[-1] == ".hickle":
            autocov = hickle.load(os.path.join(datadir,file))
        elif os.path.splitext(file)[-1] == ".pickle":
            with open(os.path.join(datadir,file), "rb") as fopen:
                autocov = pickle.load(fopen)
        else:
            raise ValueError("File extension not recognised.")

        int_time = autocov["integration_time"]
        ITPT = autocov["ITPT"]
        set_to_zeroes(ITPT, integral_rtol)

        ITPT.data *= (1/int_time)

        # only the upper triangular part is saved.
        ITPT = ITPT + ITPT.T - diags(ITPT.diagonal())

        num_nodes = ITPT.shape[0]

        p1 = np.ones(num_nodes)/num_nodes

        inplace_diag_matmul_csr(ITPT, p1)
        inplace_csr_matmul_diag(ITPT, p1)

        # total sum should be 1
        ITPT_sum = ITPT.data.sum()
        print("PID ", os.getpid(),
              f", ITPT sum {ITPT_sum}")

        max_S_row_diff = np.abs(np.array(ITPT.sum(0)).squeeze() - p1).max()
        print("PID ", os.getpid(),
              f", max(row_sum|ITPT - pTp|) {max_S_row_diff}")
        # normalize ITPT such that sum(S) = 0
        if normalize_cov_mat:
            normalization = "sym_approx"

            # now normalize:
            for _ in range(num_norm_iter):
                inplace_csr_row_normalize(ITPT, 1/num_nodes)
                ITPT = (ITPT+ITPT.T)/2

            ITPT_sum_norm = ITPT.data.sum()

            print("PID ", os.getpid(),
                  f", ITPT sum after {normalization} normalization {ITPT_sum_norm}")

            max_S_row_diff_norm = np.abs(np.array(ITPT.sum(0)).squeeze() - p1).max()

            print("PID ", os.getpid(),
                  f", max(row_sum|ITPT - pTp|) after normalization {max_S_row_diff_norm}")

        S = sparse_autocov_mat(PT=ITPT,
                               p1 = 1/num_nodes,
                               p2 = 1/num_nodes,
                               PT_symmetric=True)
        p2 = p1
        T = S.PT


    elif init_p1 == True:
        print("PID ", os.getpid(),
              f", {direction} autocov integral with init p1. loading {file}")

        if os.path.splitext(file)[-1] == ".hickle":
            autocov = hickle.load(os.path.join(datadir,file))
        elif os.path.splitext(file)[-1] == ".pickle":
            with open(os.path.join(datadir,file), "rb") as fopen:
                autocov = pickle.load(fopen)
        else:
            raise ValueError("File extension not recognised.")

        int_time = autocov["integration_time"]

        normalization = None

        if direction == "forward":
            ITPT_key = "ITPT"
            p_key = "p1"
        elif direction == "backward":
            ITPT_key = "ITPTback"
            p_key = "p2"
        else:
            raise ValueError("direction must be forward or backward")

        ITPT = autocov[ITPT_key]
        set_to_zeroes(ITPT, integral_rtol)

        ITPT.data *= (1/int_time)

        # total sum (for upper triangular) should be 1
        ITPT_sum = ITPT.data.sum()*2 - ITPT.diagonal().sum()
        print("PID ", os.getpid(),
              f", ITPT sum {ITPT_sum}")
        p =  autocov[p_key].toarray().squeeze()


        ITPT = ITPT + ITPT.T - diags(ITPT.diagonal())

        max_S_row_diff = np.abs(np.array(ITPT.sum(0)).squeeze() - p).max()
        print("PID ", os.getpid(),
              f", max(row_sum|ITPT - pTp|) {max_S_row_diff}")

        if normalize_cov_mat:
            normalization = "sym_approx"
            # find zero rows where p>0 and add min value to the diagonal:
            zero_rows = np.where(p>0)[0][(ITPT.sum(1)[np.where(p>0)] == 0.0).nonzero()[0]]
            print("PID ", os.getpid(),
                  f", {len(zero_rows)}, rows with p>0 and ITPT = 0")
            for i in zero_rows:
                ITPT[i,i] = integral_rtol*ITPT.data.max()

            # now normalize:
            for _ in range(num_norm_iter):
                inplace_csr_row_normalize(ITPT, p)
                ITPT = (ITPT+ITPT.T)/2

            ITPT_sum_norm = ITPT.data.sum()

            print("PID ", os.getpid(),
                  f", ITPT sum after {normalization} normalization {ITPT_sum_norm}")

            max_S_row_diff_norm = np.abs(np.array(ITPT.sum(0)).squeeze() - p).max()

            print("PID ", os.getpid(),
                  f", max(row_sum|ITPT - pTp|) after normalization {max_S_row_diff_norm}")


        S = sparse_autocov_mat(PT=ITPT,
                           p1 = p,
                           p2 = p,
                           PT_symmetric=True)
        p2 = p
        p1 = p
        T = S.PT


    ncomp = None
    label = None

    S_list = None
    comp_nodes_list = None
    singletons = None

    if only_largest_comp or separate_comps:

        if ITPT.indices.dtype == np.int64:
            if (ITPT.indices.max() < 2**31) and (ITPT.indptr.max() < 2**31):
                ITPT.indices = np.array(ITPT.indices, dtype=np.int32)
                ITPT.indptr = np.array(ITPT.indptr, dtype=np.int32)
            else:
                raise ValueError("ITPT needs int64, but connected_components needs int32. stopping")

        ncomp, label = connected_components(ITPT, directed="False", connection="weak")

        print("PID ", os.getpid(),
              f", ncomp = {ncomp}, top five comp sizes: {np.sort(np.bincount(label))[::-1][:5]}")

        if only_largest_comp:
            mask_lcc = label == np.where(np.bincount(label) == \
                                         np.bincount(label).max())[0][0]

            largest_comp_nodes = mask_lcc.nonzero()[0]

            idx_list = [(~mask_lcc).nonzero()[0]]
            idx_list += [[i] for i in largest_comp_nodes]

            # aggregate all the nodes outside of the LCC in a single meta node

            print("PID ", os.getpid(),
                  "aggregating unconnected components.")

            S = S.aggregate(idx_list)

            p1 = S.p1
            p2 = S.p2
            T = S.PT
        if separate_comps:
            # we do each components separately

            if init_p1:
                # first find all the nodes with p=0 that will always be grouped in the
                # meta node
                pzero_nodes = set((~(S.p1>0)).nonzero()[0])
            else:
                pzero_nodes = set()

            # then loop over all components that are not in the pzero set
            S_list = []
            comp_nodes_list = []
            singletons = []
            for cmp in np.unique(label):
                mask_cmp = label == cmp
                comp_nodes = mask_cmp.nonzero()[0]

                if comp_nodes.size == 1:
                    if comp_nodes[0] in pzero_nodes:
                        # this is a node in pzero
                        pass
                    else:
                        singletons.append(comp_nodes[0])
                else:

                    idx_list = [(~mask_cmp).nonzero()[0]]
                    idx_list += [[i] for i in comp_nodes]

                    print("PID ", os.getpid(),
                          f"aggregating components other than {cmp} of size {comp_nodes.size}.")


                    S_list.append(S.aggregate(idx_list))
                    comp_nodes_list.append(comp_nodes)


            print("PID ", os.getpid(),
                  f"aggregated {len(comp_nodes_list)} components, found {len(singletons)} singletons and {len(pzero_nodes)} nodes with zero prob. density.")


    return S, T, p1, p2, largest_comp_nodes, S_list, comp_nodes_list, singletons, normalization, ncomp, label, max_S_row_diff, ITPT_sum, max_S_row_diff_norm, ITPT_sum_norm

#%% loop functions
# A global dictionary storing the variables passed from the initializer.
var_dict = {}


def _init_sub_worker(file, PTdata, PTindices, PTindptr, N, p1raw, p2raw, PT_symmetric=True):
    # reconstruct A from shared arrays

    PT = csr_matrix((np.frombuffer(PTdata, dtype=np.float64),
                                  np.frombuffer(PTindices, dtype=np.int64),
                                  np.frombuffer(PTindptr, dtype=np.int64)),
                                 shape=(N,N))

    S = sparse_autocov_mat(PT,
                           np.frombuffer(p1raw, dtype=np.float64),
                           np.frombuffer(p2raw, dtype=np.float64),
                           PT_symmetric=PT_symmetric)

    # make sure that shared arrays from different files are not mixed
    var_dict[file] = {}
    var_dict[file]["N"] = N
    var_dict[file]["S"] = S
    var_dict[file]["p1"] = S.p1
    var_dict[file]["p2"] = S.p2
    var_dict[file]["T"] = S.PT


def _init_sub_worker_list(file, PTdata_list, PTindices_list, PTindptr_list,
                          N_list, p1raw_list, p2raw_list,
                          comp_nodes_list, singletons, PT_symmetric=True):
    # reconstruct A from shared arrays for list of components

    # make sure that shared arrays from different files are not mixed
    var_dict[file] = {}

    var_dict[file]["S_list"] = []

    for PTdata, PTindices, PTindptr, p1raw, p2raw, N in \
            zip(PTdata_list, PTindices_list, PTindptr_list,
                p1raw_list, p2raw_list, N_list):


        PT = csr_matrix((np.frombuffer(PTdata, dtype=np.float64),
                                      np.frombuffer(PTindices, dtype=np.int64),
                                      np.frombuffer(PTindptr, dtype=np.int64)),
                                     shape=(N,N))

        var_dict[file]["S_list"].append(sparse_autocov_mat(PT,
                               np.frombuffer(p1raw, dtype=np.float64),
                               np.frombuffer(p2raw, dtype=np.float64),
                               PT_symmetric=PT_symmetric))


    var_dict[file]["comp_nodes_list"] = comp_nodes_list
    var_dict[file]["singletons"] = singletons

def _compute_clustering_fct(params):

    file, seed = params

    t0 = time.time()
    if verbose:
        print("**** PID ", os.getpid(), "starting clustering at",
              time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),
              f" with seed {seed} from file {file}")


    clustering = SparseClustering(p1=var_dict[file]["p1"],
                                  p2=var_dict[file]["p2"],
                                  T=var_dict[file]["T"],
                                  S=var_dict[file]["S"],
                                  rnd_seed=seed)

    n_loops = clustering.find_louvain_clustering(n_meta_iter_max=n_meta_iter_max,
                                                 n_sub_iter_max=n_sub_iter_max,
                                                verbose=clust_verbose,
                                                print_num_loops=print_num_loops)

    if verbose:
        print("**** PID ", os.getpid(), f"sym clust took {time.time()-t0:.4f}s, finished at",
              time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()), f", in {n_loops} loops.")

    return (clustering.partition.cluster_list, clustering.compute_stability(), seed)

def _compute_sym_list_clustering_fct(params):
    # for computing clustering of a list of components

    file, seed = params

    t0 = time.time()
    if verbose:
        print("**** PID ", os.getpid(), "starting list clustering at",
              time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),
              f" with seed {seed} from file {file}")

    cluster_list = []
    stability = 0
    # loop over all components and add their clusters to the golbal cluster_list
    for S, comp_nodes in zip(var_dict[file]["S_list"], var_dict[file]["comp_nodes_list"]) :
        clustering = SparseClustering(p1=S.p1,
                                      p2=S.p2,
                                      T=S.PT,
                                      S=S,
                                      rnd_seed=seed)

        _ = clustering.find_louvain_clustering(n_meta_iter_max=n_meta_iter_max,
                                                     n_sub_iter_max=n_sub_iter_max,
                                                    verbose=clust_verbose,
                                                    print_num_loops=print_num_loops)

        if clustering.partition.get_num_clusters() == 1:
            # S is all zero, only one cluster
            cluster_list.append(set(comp_nodes))
        else:
            # first cluster is the meta nodes with all the other components
            assert len(clustering.partition.cluster_list[0]) == 1 and clustering.partition.node_to_cluster_dict[0] == 0

            # map of node id in the aggregated network to the real network
            node_map = {i+1: node for i, node in enumerate(comp_nodes)}

            for clust in clustering.partition.cluster_list[1:]:
                cluster_list.append({node_map[n] for n in clust})

            stability += clustering.compute_stability()

    # add singletons
    cluster_list += [{n} for n in var_dict[file]["singletons"]]

    if verbose:
        print("**** PID ", os.getpid(), f"list sym clust took {time.time()-t0:.4f}s, finished at",
              time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()), f', for {len(var_dict[file]["comp_nodes_list"])} components.')

    return (cluster_list, stability, seed)

def compute_static_clustering_fct(params):
    A, seed = params

    t0 = time.time()
    if verbose:
        print("**** PID ", os.getpid(), "starting clustering at",
              time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),
              f" with seed {seed}")

    stat_clustering = static_clustering(A, rnd_seed=seed)

    n_loops = stat_clustering.find_louvain_clustering(n_meta_iter_max=n_meta_iter_max,
                                                 n_sub_iter_max=n_sub_iter_max,
                                       verbose=clust_verbose,
                                       print_num_loops=print_num_loops)

    if verbose:
        print("**** PID ", os.getpid(), f"static clust took {time.time()-t0:.4f}s, finished at",
              time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()), f", in {n_loops} loops.")

    return (stat_clustering.partition.cluster_list, stat_clustering.compute_stability(), seed)

# TODO: move to helper scripts
def n_random_seeds(n):

    return [int.from_bytes(os.urandom(4), byteorder="big") for \
                                      _ in range(n)]


def compute_nvi_sample(params):

    c1, c2, N = params

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
#%% clustering

def worker(file_args):

    (file, compute_static_clustering, compute_clustering, compute_nvarinf,
        save_clustering_before_nvarinf) = file_args
    t0 = time.time()

    savefile = os.path.join(savedir,"clusters_" + file)

    largest_comp_nodes = None

    normalization = None

    ncomp = None

    comp_label = None


    if os.path.exists(savefile):
        print("PID ", os.getpid(), "file already exists, skipping", savefile)
    else:
        print("PID ", os.getpid(), " starting file ", file)
        try:



            S, T, p1, p2, largest_comp_nodes, S_list, comp_nodes_list, \
                singletons, normalization, ncomp, label, max_S_row_diff, \
                    ITPT_sum, max_S_row_diff_norm, \
                        ITPT_sum_norm = load_autocov_int(file,
                                                         init_p1,
                                                         direction)




            if compute_static_clustering:
                print("PID ", os.getpid(), " loading adjacency matrix for", file)
                #get int start and stop
                extracts = os.path.splitext(file)[0].split("_")
                # dont care about order for static network
                int_start, int_stop = sorted((int(extracts[-3]),int(extracts[-1])))

                try:
                    A = pd.read_pickle(os.path.join(datadir,net_name + \
                                f"_static_A_{int_start:06d}_to_{int_stop:06d}.pickle"))
                    # symmetrize A
                    A = (A + A.T)/2
                except FileNotFoundError:
                    print("PID ", os.getpid(), "Adjacency file not found, skipping",
                          file=sys.stdout)
                    print("PID ", os.getpid(), "File not found : ", os.path.join(datadir,net_name + \
                            f"_static_A_{int_start:06d}_to_{int_stop:06d}.pickle"),
                            file=sys.stderr)


                    compute_static_clustering = False

            if compute_clustering or compute_static_clustering:
                print("PID ", os.getpid(), "computing symmetric clusters ", ", file ", file)


                t0 = time.time()
                # tclust = time.time()

                # check if temp results have already been computed:
                tmpsavefile = os.path.join(savedir,"tmpclusters_" + file)
                if os.path.exists(tmpsavefile):
                    print("PID ", os.getpid(), " loading temp res from ", tmpsavefile)
                    with open(tmpsavefile, "rb") as fopen:
                        tmpres = pickle.load(fopen)

                    clust_counter_sym =  tmpres["clust_counter_sym"]
                    sym_stabilites = tmpres["sym_stabilites"]
                    avg_stab_sym = tmpres["avg_stab_sym"]
                    avg_nclust_sym = tmpres["avg_nclust_sym"]
                    best_cluster_sym = tmpres["best_cluster_sym"]
                    best_stab_sym = tmpres["best_stab_sym"]
                    largest_comp_nodes = tmpres["largest_comp_nodes"]
                    ncomp = tmpres["ncomp"]
                    comp_label = tmpres["comp_label"]
                    max_S_row_diff = tmpres["max_S_row_diff"]
                    ITPT_sum = tmpres["ITPT_sum"]
                    max_S_row_diff_norm = tmpres["max_S_row_diff_norm"]
                    ITPT_sum_norm = tmpres["ITPT_sum_norm"]

                    save_clustering_before_nvarinf = False
                    compute_clustering = False
                    compute_nvarinf = True





                if compute_clustering:
                    params_list = [(file, seed) for seed in n_random_seeds(num_repeat)]

                    if nproc_clustering == 1:

                        if separate_comps:
                            var_dict[file] = {}
                            var_dict[file]["S_list"] = S_list
                            var_dict[file]["comp_nodes_list"] = comp_nodes_list
                            var_dict[file]["singletons"] = singletons
                            sym_clusters, sym_stabilites, sym_seeds = \
                                zip(*map(_compute_sym_list_clustering_fct, params_list))

                        else:
                            var_dict[file] = {}
                            var_dict[file]["p1"] = p1
                            var_dict[file]["p2"] = p2
                            var_dict[file]["T"] = T
                            var_dict[file]["S"] = S
                            sym_clusters, sym_stabilites, sym_seeds = \
                                zip(*map(_compute_clustering_fct, params_list))

                    else:
                        print("**** PID ", os.getpid(), f"starting pool of {nproc_clustering} processes for the clustering loop.")

                        if separate_comps:
                            PTindices_list = [RawArray("l",S.PT.indices) for S in S_list]
                            PTindptr_list = [RawArray("l",S.PT.indptr) for S in S_list]
                            PTdata_list = [RawArray("d", S.PT.data) for S in S_list]
                            p1raw_list = [RawArray("d", S.p1) for S in S_list]
                            p2raw_list = [RawArray("d", S.p2) for S in S_list]
                            N_list = [S.PT.shape[0] for S in S_list]

                            with Pool(nproc_clustering,
                                      initializer=_init_sub_worker_list,
                                      initargs=(file, PTdata_list,
                                                PTindices_list, PTindptr_list,
                                                N_list, p1raw_list, p2raw_list,
                                                comp_nodes_list, singletons),
                                        ) as pool:
                                sym_clusters, sym_stabilites, sym_seeds = \
                                    zip(*pool.map(_compute_sym_list_clustering_fct, params_list,
                                                  chunksize=1))


                        else:
                            # create arrays to share between processes
                            PTindices = RawArray("l",S.PT.indices)
                            PTindptr = RawArray("l",S.PT.indptr)
                            PTdata = RawArray("d", S.PT.data)
                            p1raw = RawArray("d", p1)
                            p2raw = RawArray("d", p2)
                            N = S.PT.shape[0]

                            with Pool(nproc_clustering,
                                      initializer=_init_sub_worker,
                                      initargs=(file,
                                                PTdata, PTindices,
                                                PTindptr, N,
                                                p1raw, p2raw),
                                        ) as pool:
                                sym_clusters, sym_stabilites, sym_seeds = \
                                    zip(*pool.map(_compute_clustering_fct, params_list,
                                                  chunksize=1))

                    if file in var_dict:
                        # free global memory
                        del var_dict[file]




                if compute_static_clustering:
                    params_list = [(A, seed) for seed in n_random_seeds(num_repeat)]

                    if nproc_clustering == 1:
                        sym_clusters_static, sym_stabilites_static, sym_seeds_static  = \
                            zip(*map(compute_static_clustering_fct, params_list))

                    else:
                        print("**** PID ", os.getpid(), f"starting pool of {nproc_clustering} processes for the clustering loop.")
                        with Pool(nproc_clustering) as pool:
                            sym_clusters_static, sym_stabilites_static, sym_seeds_static  = \
                                zip(*pool.map(compute_static_clustering_fct, params_list,
                                              chunksize=1))



                t1 = time.time()

                print("PID ", os.getpid(), f" symm clust, took {t1-t0:.2f}s, file ", file)
                print("PID ", os.getpid(), " computing sym nvi ", ", file ", file)

                if compute_clustering:


                    clust_counter_sym = Counter([tuple(sorted([tuple(sorted(c)) for c in clust])) \
                                                 for clust in sym_clusters])

                    best_cluster_sym = sym_clusters[np.argmax(sym_stabilites)]

                    best_stab_sym = max(sym_stabilites)

                    avg_stab_sym = np.mean(sym_stabilites)

                    avg_nclust_sym = np.mean([len(c) for c in sym_clusters])

                    if save_clustering_before_nvarinf:
                        tmpres = {}
                        tmpres["clust_counter_sym"] = clust_counter_sym
                        tmpres["sym_stabilites"] = sym_stabilites
                        tmpres["avg_stab_sym"] = avg_stab_sym
                        tmpres["avg_nclust_sym"] = avg_nclust_sym
                        tmpres["best_cluster_sym"] = best_cluster_sym
                        tmpres["best_stab_sym"] = best_stab_sym
                        tmpres["largest_comp_nodes"] = largest_comp_nodes
                        tmpres["ncomp"] = ncomp
                        tmpres["comp_label"] = comp_label
                        tmpres["ITPT_sum"] = ITPT_sum
                        tmpres["max_S_row_diff"] = max_S_row_diff
                        tmpres["ITPT_sum_norm"] = ITPT_sum_norm
                        tmpres["max_S_row_diff_norm"] = max_S_row_diff_norm
                        tmpres["inargs"] = inargs

                        # already saving clustering
                        tmpsavefile = os.path.join(savedir,"tmpclusters_" + file)
                        print("PID ", os.getpid(), " saving temp res to ", tmpsavefile)
                        with open(tmpsavefile, "wb") as fopen:
                            pickle.dump(tmpres, fopen)

                if compute_nvarinf:

                    nvarinf_samples_idx = np.random.choice(range(max_num_varinf_samples),
                                                       num_varinf_samples,
                                                       replace=False)

                    nvarinf_samples = [s for i,s in enumerate(combinations(sym_clusters,2)) if i in nvarinf_samples_idx]


                    if verbose:
                        print("PID ", os.getpid(), " Number of varinf samples ", len(nvarinf_samples))


                    params = [(c1,c2, T.shape[0]) for c1, c2 in nvarinf_samples]

                    if nproc_clustering == 1:
                        nvarinf_sym_vals = list(map(compute_nvi_sample, params))

                    else:
                        print("**** PID ", os.getpid(), f"starting pool of {nproc_clustering} processes for the NVI loop.")
                        with Pool(nproc_clustering) as pool:
                            nvarinf_sym_vals = list(pool.map(compute_nvi_sample, params,
                                              chunksize=1))



                    nvarinf_sym = np.mean(nvarinf_sym_vals)


                if compute_static_clustering:

                    nvarinf_samples_idx = np.random.choice(range(max_num_varinf_samples),
                                                       num_varinf_samples,
                                                       replace=False)

                    nvarinf_samples = [s for i,s in enumerate(combinations(sym_clusters_static,2)) if i in nvarinf_samples_idx]

                    nvarinf_sym_static = np.mean([norm_var_information(c1,c2, N=T.shape[0],
                                                                use_clust_list=True) for c1,c2 in nvarinf_samples])

                    clust_counter_sym_static = Counter([tuple(sorted([tuple(sorted(c)) for c in clust])) \
                                                 for clust in sym_clusters_static])

                    best_cluster_sym_static = sym_clusters_static[np.argmax(sym_stabilites_static)]

                    best_stab_sym_static = max(sym_stabilites_static)

                    avg_stab_sym_static = np.mean(sym_stabilites_static)

                    avg_nclust_sym_static = np.mean([len(c) for c in sym_clusters_static])



                t2 = time.time()
                print("PID ", os.getpid(), f" symm nvi, took {t2-t1:.2f}s, file ", file)

            print("PID ", os.getpid(), " saving to file", savefile )

            res = {"num_repeat" : num_repeat,
                   "compute_static_clustering" : compute_static_clustering,
                   "compute_clustering" : compute_clustering,
                   "largest_comp_nodes" : largest_comp_nodes,
                   "ncomp" : ncomp,
                   "comp_label" : comp_label,
                   "max_S_row_diff" : max_S_row_diff,
                   "ITPT_sum" : ITPT_sum,
                   "max_S_row_diff_norm" : max_S_row_diff_norm,
                   "ITPT_sum_norm" : ITPT_sum_norm,
                   "normalization" : normalization,
                   "init_p1" : init_p1,
                   "direction" : direction,
                   "inargs" : inargs}

            if compute_clustering:

                res["clust_counter"] = clust_counter_sym
                res["stabilites"] = sym_stabilites
                res["seeds"] = sym_seeds
                res["nvarinf"] = nvarinf_sym
                res["avg_stab"] = avg_stab_sym
                res["avg_nclust"] = avg_nclust_sym
                res["best_cluster"] = best_cluster_sym
                res["best_stab"] = best_stab_sym




            if compute_static_clustering:
                res["stabilites_static"] = sym_stabilites_static
                res["clust_counter_static"] = clust_counter_sym_static
                res["seeds_static"] = sym_seeds_static
                res["nvarinf_static"] = nvarinf_sym_static
                res["avg_stab_static"] = avg_stab_sym_static
                res["avg_nclust_static"] = avg_nclust_sym_static
                res["best_cluster_static"] = best_cluster_sym_static
                res["best_stab_static"] = best_stab_sym_static



            with open(savefile, "wb") as fopen:
                pickle.dump(res, fopen)

        except Exception:
            print("PID ", os.getpid(), "-+-+-+ Exception at file:", file,
                  file=sys.stdout)
            print("PID ", os.getpid(), "-+-+-+ Exception at file:", file,
                   file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    print("+++ PID ", os.getpid(), "finished in ", time.time()-t0)

def main():
    t00 = time.time()
    print(f"starting pool of {nproc_files} processes")
    with futurePool(nproc_files) as p:
        for _ in p.map(worker,
                   [(file, compute_static_clustering, compute_clustering,
                     compute_nvarinf, save_clustering_before_nvarinf) for file in files]):
            pass


    print(f"***** Finished! in {time.time()-t00}")
#%% main pool
if __name__ == "__main__":
    main()
