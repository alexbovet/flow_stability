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

"""Computes the integral of the random walk covariance matrices.
This is done on a coarser temporal grid than the grid used for the slices of the 1st step (run_laplacians_transmats.py).
The grid step size is defined by `int_length` in units of slices. By default, the integral of the covariance
will be computed starting and ending from all combinations of the grid steps. Using `--only_from_start_and_finish`
will compute the forward integrals from the first time point until each following grid steps and 
the backward integrals from the last time point until each previous grid steps.
The resolutions (waiting times) are read from the saved inter-event transition matrices.
For efficiency, only the first term (the sparse part) of the autocovariance integral is computed and saved.
The results are saved in files named
{savedir}/{net_name}_tau_w{tau_w}_PT_{initial_grid_point}_to_{final_grid_point}.{ext}
It will use sparse-dot-mkl to do multithreaded sparse matrix products, if it is installed.

The file {initial_grid_point}_to_{final_grid_point} contains the autocovariance integral between
t1 and t2, where t1 is the *start time* of int_start and t2 is also the *start time* 
of int_stop (also valid when int_start > int_stop).

The value saved is the upper triangular part of:

    int_t1^t2 T(t_1,t) P(t)^{-1} T(t_1,t)^T dt 
    
"""
import gc
import os
import re
import sys
import time
import traceback
from argparse import SUPPRESS, ArgumentDefaultsHelpFormatter, ArgumentParser
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
import psutil
from scipy.sparse import csr_matrix, eye, triu

from ..sparse_stoch_mat import (
    inplace_csr_matmul_diag,
    inplace_csr_row_normalize,
    sparse_gram_matrix,
    sparse_matmul,
)
from ..temporal_network import ContTempNetwork, set_to_zeroes

# raise Exception

#%%

ap = ArgumentParser(prog="run_cov_integrals",
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
                help="Location of the inter trans. matrices.")

required.add_argument("--savedir", default="", type=str,
                help="Where the results will be saved.")

optional.add_argument("--ncpu", default=4, type=int,
                      help="Size of the multiprocessing pool.")

optional.add_argument("--net_name", default="tempnet", type=str,
                help="Name that used when saving results,")

optional.add_argument("--num_points", default=50, type=int,
                      help="Number of steps of the grid overwhich the integral results will be saved.")

optional.add_argument("--int_length",default=None, type=int,
                help="Length of a single grid interval. Used to set the number of intervals instead of num_points")

optional.add_argument("--int_list", default=[], type=int, nargs="+",
                help="List of intervals used for the integral. Used instead of num_points or int_length.")

optional.add_argument("--t_s", default=10, type=float,
                      help="Stationarity time parameter for the linear approximation. Should be the same than the one used for the computation of the inter trans. mat.")

optional.add_argument("--tol", default=1e-8, type=float,
                help="Values smaller than max(T)*tol are set to zero in sparse transition matrices.")

optional.add_argument("--integral_rtol", default=None, type=float,
                help="Values smaller than max(integral)*rtol are set to zero in the sparse integral.")

optional.add_argument("--time_direction", default="both", type=str,
                help="can be 'forward','backward' or 'both'. Default is 'both'.")

optional.add_argument("--only_expm_transmats", action="store_true",
                help="uses only trans. mat. computed with the expm.")

optional.add_argument("--only_lin_transmats", action="store_true",
                help="uses only trans. mat. computed with the linear approx.")

optional.add_argument("--only_from_start_and_finish", action="store_true",
                help="instead of computing every combinations of start and finish, will compute every integrals forward from start and backward from finish.")

optional.add_argument("--only_from_start", action="store_true",
                help="instead of computing every combinations of start and finish, will compute every integrals forward from start.")

optional.add_argument("--only_from_finish", action="store_true",
                help="instead of computing every combinations of start and finish, will compute every integrals backward from finish.")

optional.add_argument("--only_one_interval", action="store_true",
                help="instead of computing every combinations of start and finish, will compute from every start but only for one interval.")

optional.add_argument("--verbose", action="store_true")

optional.add_argument("--verbose_sparse_matmul", action="store_true",
                      help="show computation times of sparse matrix multiplications.")

optional.add_argument("--print_mem_usage", action="store_true",
                help="print memory usage.")

optional.add_argument("--print_interval", default=100, type=int,
                help="Controls how often memory usage is printed.")

optional.add_argument("--save_intermediate", action="store_true",
                help="Saves, and loads, intermediate steps in order to be able to restart computation.")


inargs = vars(ap.parse_args())
datadir = inargs["datadir"]
savedir = inargs["savedir"]
ncpu = inargs["ncpu"]
net_name = inargs["net_name"]
num_points = inargs["num_points"]
int_length = inargs["int_length"]
int_list = inargs["int_list"]
t_s = inargs["t_s"]
tol = inargs["tol"]
integral_rtol = inargs["integral_rtol"]

time_direction = inargs["time_direction"]
if time_direction == "both":
    rev_time_list = [False, True]
elif time_direction == "forward":
    rev_time_list = [False]
elif time_direction == "backward":
    rev_time_list = [True]

only_expm_transmats = inargs["only_expm_transmats"]
only_lin_transmats = inargs["only_lin_transmats"]
only_one_interval = inargs["only_one_interval"]

verbose = inargs["verbose"]
verbose_sparse_matmul = inargs["verbose_sparse_matmul"]

save_intermediate = inargs["save_intermediate"]

use_expm_transmats = True
use_lin_transmats = True
if only_expm_transmats:
    use_lin_transmats = False
if only_lin_transmats:
    use_expm_transmats = False

only_from_start_and_finish = inargs["only_from_start_and_finish"]
only_from_start = inargs["only_from_start"]
only_from_finish = inargs["only_from_finish"]


print_mem_usage = inargs["print_mem_usage"]

print_interval = inargs["print_interval"]
#%%

if datadir == "":
    raise Exception("datadir must be given")

if savedir == "":
    raise Exception("savedir must be given")

print("Arguments:")
for item in inargs.items():
    print(item)
#%%
all_files = os.listdir(datadir)

# get num nodes
num_nodes = None
i = 0

while num_nodes is None:
    compression = None
    if all_files[i][-3:] == ".gz":
        compression = "gzip"
    res = pd.read_pickle(os.path.join(datadir, all_files[i]),
                         compression=compression)
    if isinstance(res, dict):
        num_nodes = res.get("num_nodes")
    i += 1


# extract intervals and tau_ws
intervals = set()
tau_ws = set()
for file in all_files:
    for extract in os.path.splitext(file)[0].split("_"):
        if re.search("int[0-9]{6}",extract):
            intervals.add(int(extract[3:]))
        elif extract.startswith("w"):
            tau_ws.add(float(extract[1:]))

intervals = sorted(list(intervals))
tau_ws = sorted(list(tau_ws))

num_small_grid_points = len(intervals)

if len(int_list) == 0:

    if int_length is None:
        int_length = num_small_grid_points//num_points

    # we can go +1 over num_small_grid_points because the last interval is never loaded
    # we compute up to the begining of the last interval.
    all_inds = list(range(0,num_small_grid_points+1,int_length))



else:
    all_inds = int_list

#%%

def load_trans_mat(k_range, tau_w, lin, reverse_time):
    """loads, computes and returns the transition matrix computed from k_range[0]
    to k_range[-1] and the corresponding time duration.
        
    used to quickly start again an integral computation.
    """
    if reverse_time:
        time_direction_str = "reversed_"
    else:
        time_direction_str = ""


    if lin:

        T_file = os.path.join(datadir, net_name + \
                         f"_tau_w{tau_w:.3e}" + "_int{k:06d}__" + \
                         time_direction_str + "lin_trans_mat")


    else:
        T_file = os.path.join(datadir, net_name + \
                         f"_tau_w{tau_w:.3e}" + "_int{k:06d}__" + \
                         time_direction_str + "trans_mat")




    T = eye(num_nodes, format="csr", dtype=np.float64)

    integr_time = 0

    for k in k_range:
        TM = ContTempNetwork.load_T(T_file.format(k=k))

        if lin:
            Tk = TM["T_lin"][1/tau_w][t_s]
        else:
            Tk = TM["T"][1/tau_w]

        set_to_zeroes(Tk, tol=tol)
        inplace_csr_row_normalize(Tk)

        T = T @ Tk
        set_to_zeroes(T, tol=tol)
        inplace_csr_row_normalize(T)

        integr_time += abs(TM["_t_stop_laplacians"]-TM["_t_start_laplacians"])

    return T, integr_time


def worker(ind_start_tau_w):

    ind_start, tau_w, reverse_time = ind_start_tau_w

    # all_inds is a list of interval index that forms the large grid point
    # ind_start is the index in all_inds from where to start

    int_start = all_inds[ind_start]

    print("PID ", os.getpid(), f"interval start {int_start} for tau_w {tau_w}")

    inter_trans_file0 = os.path.join(datadir, net_name + \
                     f"_tau_w{tau_w:.3e}" + "_int{k:06d}_")


    ITPT_file = os.path.join(savedir, net_name + \
                     f"_tau_w{tau_w:.3e}") + \
                          "_PT_{0:06d}_to_{1:06d}.pickle"
    ITPT_lin_file = os.path.join(savedir, net_name + \
                     f"_tau_w{tau_w:.3e}") + \
                          "_PT_lin_{0:06d}_to_{1:06d}.pickle"


    ITPT_temp_file = os.path.join(savedir, net_name + \
                     f"_tau_w{tau_w:.3e}") + \
                          "_temp_integration_{0:06d}_to_{1:06d}.pickle"
    ITPT_lin_temp_file = os.path.join(savedir, net_name + \
                     f"_tau_w{tau_w:.3e}") + \
                          "_temp_integration_lin_{0:06d}_to_{1:06d}.pickle"

    # initialize matrices
    # I is the the sum of two integrals I = (1/T) * P1 (int TPT) P1 - p1p1
    # (we only need the int TPT part) called ITPT
    ITPT = csr_matrix((num_nodes,num_nodes), dtype=np.float64)

    ITPT_lin = csr_matrix((num_nodes,num_nodes), dtype=np.float64)

    Tk = eye(num_nodes, format="csr", dtype=np.float64)
    Tk_lin = eye(num_nodes, format="csr", dtype=np.float64)

    p1 = np.ones(num_nodes, dtype=np.float64)/num_nodes

    _int_start = int_start



    if reverse_time:
        int_stops = all_inds[ind_start-1::-1]
        d = -1
        if only_one_interval:
            int_stops = [all_inds[ind_start-1]]
    else:
        int_stops = all_inds[ind_start+1:]
        if only_one_interval:
            int_stops = [all_inds[ind_start+1]]
        d = 1
    try:

        integration_time = 0
        integration_time_lin = 0

        # check if all files have already been computed if not loads already
        # existing files an update T and integration time accordingly
        all_expm_exists = False
        all_lin_exists = False
        last_expm_exist = -1
        last_lin_exist = -1


        if use_expm_transmats:
            integrals_expm_exists = np.array([os.path.isfile(ITPT_file.format(int_start, int_stop)) or \
                                    os.path.isfile(ITPT_file.format(int_start, int_stop)+".gz") \
                                      for int_stop in int_stops])

            if integrals_expm_exists.cumprod().nonzero()[0].size > 0:
                #find the longest continuous stretch of True
                last_expm_exist = integrals_expm_exists.cumprod().nonzero()[0].max()

                # if its all True (no need to compute anything)
                all_expm_exists = last_expm_exist +1 == integrals_expm_exists.size

                if not all_expm_exists:
                    #load the last integral
                    int_stop = int_stops[last_expm_exist]
                    if os.path.isfile(ITPT_file.format(int_start, int_stop)):
                        ITPT_load = pd.read_pickle(ITPT_file.format(int_start, int_stop))
                        print("PID ", os.getpid(), " loading I ", ITPT_file.format(int_start, int_stop))
                    else:
                        ITPT_load = pd.read_pickle(ITPT_file.format(int_start, int_stop)+".gz")
                        print("PID ", os.getpid(), " loading I ", ITPT_file.format(int_start, int_stop)+".gz")

                    if reverse_time:
                        load_range = range(int_start-1,int_stop-1,d)
                    else:
                        load_range = range(int_start,int_stop,d)

                    Tk, integration_time = load_trans_mat(load_range,
                                           tau_w, lin=False, reverse_time=reverse_time)
                    ITPT = ITPT_load["ITPT"]

                    # update initial condition
                    _int_start = int_stops[last_expm_exist]
                    int_stops = int_stops[last_expm_exist+1:]
                    #/!\ this won't work if expm and lin are done at the same time
                    assert use_lin_transmats != use_expm_transmats

                    del ITPT_load

        if use_lin_transmats:
            integrals_lin_exists = np.array([os.path.isfile(ITPT_lin_file.format(int_start, int_stop)) or \
                                    os.path.isfile(ITPT_lin_file.format(int_start, int_stop)+".gz") \
                                      for int_stop in int_stops])

            if integrals_lin_exists.cumprod().nonzero()[0].size > 0:
                #find the longest continuous stretch of True
                last_lin_exist = integrals_lin_exists.cumprod().nonzero()[0].max()

                # if its all True (no need to compute anything)
                all_lin_exists = last_lin_exist +1 == integrals_lin_exists.size

                if not all_lin_exists:
                    #load the last integral
                    int_stop = int_stops[last_lin_exist]
                    if os.path.isfile(ITPT_lin_file.format(int_start, int_stop)):
                        print("PID ", os.getpid(), " loading I ", ITPT_lin_file.format(int_start, int_stop))
                        ITPT_lin_load = pd.read_pickle(ITPT_lin_file.format(int_start, int_stop))

                    else:
                        print("PID ", os.getpid(), " loading I ", ITPT_lin_file.format(int_start, int_stop)+".gz")
                        ITPT_lin_load = pd.read_pickle(ITPT_lin_file.format(int_start, int_stop)+".gz")

                    if reverse_time:
                        load_range = range(int_start-1,int_stop-1,d)
                    else:
                        load_range = range(int_start,int_stop,d)

                    Tk_lin, integration_time_lin = load_trans_mat(load_range,
                                           tau_w, lin=True, reverse_time=reverse_time)

                    ITPT_lin = ITPT_lin_load["ITPT"]

                    assert use_lin_transmats != use_expm_transmats

                    del ITPT_lin_load

                    # update initial condition
                    _int_start = int_stops[last_lin_exist]
                    int_stops = int_stops[last_lin_exist+1:]



        compute_expm = use_expm_transmats
        compute_lin = use_lin_transmats

        if all_expm_exists:
            # no need to compute expm int
            compute_expm = False
            print("PID ", os.getpid(),
                  f" expm trans integral from int {int_start} for tau_w {tau_w} already computed")
        if all_lin_exists:
            # no need to compute lin int
            compute_lin = False
            print("PID ", os.getpid(),
                  f" lin trans integral from int {int_start} for tau_w {tau_w} already computed")

        if compute_expm and compute_lin and last_expm_exist > -1 and last_lin_exist > 1:
            raise NotImplementedError


        if compute_expm or compute_lin:
            for int_stop in int_stops:

                t0 = time.time()

                print("PID ", os.getpid(),
                      f" computing trans from int {int_start} to int {int_stop} for tau_w {tau_w}")


                if reverse_time:
                    k_range = range(_int_start-1,int_stop-1,d)
                else:
                    k_range = range(_int_start,int_stop,d)
                for k in k_range:

                    if verbose:
                        print("PID ", os.getpid(),
                          f" -- k = {k} over {k_range[-1]}")


                    # load T list
                    if compute_expm:

                        if save_intermediate and os.path.isfile(ITPT_temp_file.format(int_start, k+1)):
                            #load this step that has already been computed

                            print("PID ", os.getpid(), " loading temp step ", ITPT_temp_file.format(int_start, k+1))
                            ITPT_temp_load = pd.read_pickle(ITPT_temp_file.format(int_start, k+1))

                            assert ITPT_temp_load["last_treated_interval"] == k

                            ITPT = ITPT_temp_load["ITPT"]
                            Tk = ITPT_temp_load["Tk"]
                            integration_time = ITPT_temp_load["integration_time"]

                            del ITPT_temp_load

                        else:



                            inter_Ts = ContTempNetwork.load_inter_T(inter_trans_file0.format(k=k) + \
                                       "_inter_trans_mat")

                            tl = time.time()
                            num_l = len(inter_Ts["inter_T"][1/tau_w])
                            # integrate T
                            for l, (inter_Tk, dtk) in enumerate(zip(inter_Ts["inter_T"][1/tau_w][::d],
                                               d * np.diff(inter_Ts["times_k_start_to_k_stop+1"][::d]))):



                                set_to_zeroes(inter_Tk, tol=tol)
                                inplace_csr_row_normalize(inter_Tk)

                                Tk = sparse_matmul(Tk,inter_Tk.tocsr(),
                                                    verbose=verbose_sparse_matmul,
                                                    log_message="Tk")
                                set_to_zeroes(Tk, tol=tol)
                                inplace_csr_row_normalize(Tk)


                                pk = sparse_matmul(p1, Tk.tocsr(),
                                                    verbose=verbose_sparse_matmul,
                                                    log_message="pk")
                                # in order to avoid nan in Ik due to 0 * np.inf
                                pk[np.where(pk == 0)] = 1

                                ITPTk = Tk.copy().tocsr()

                                # we do (Tk @ Pk^-1/2) @ (Tk @ Pk^-1/2)^T for ITPTk
                                inplace_csr_matmul_diag(ITPTk,np.sqrt(1/pk))


                                ITPTk = sparse_gram_matrix(ITPTk, transpose=True,
                                                            verbose=verbose_sparse_matmul,
                                                            log_message="ITPTk")

                                ITPTk.data *=  dtk # operating on data avoids making a copy here.

                                if integral_rtol is not None:
                                    set_to_zeroes(ITPTk, integral_rtol)

                                ITPT = ITPT + ITPTk

                                if integral_rtol is not None:
                                    set_to_zeroes(ITPT, integral_rtol)

                                if verbose:
                                    if not l%print_interval:
                                        print("PID ", os.getpid(), f" -- k = {k} over {k_range[-1]}, integrating {l} over {num_l},",
                                              f"took {time.time()-tl:0.3f},",
                                              f", ITPT nnz = {ITPT.nnz},",
                                              f", ITPT size (GB) = {(ITPT.data.nbytes + ITPT.indptr.nbytes + ITPT.indices.nbytes)/1024**3:0.6f}.")
                                        if print_mem_usage:
                                            minf = psutil.virtual_memory()
                                            print("PID ", os.getpid(), f"Memory info (GB): used {minf.used/1024**3:0.3f} ({minf.percent}%), available {minf.available/1024**3:0.3f}, active {minf.active/1024**3:0.3f}, inactive {minf.inactive/1024**3:0.3f}, buffers {minf.buffers/1024**3:0.3f}" )
                                        tl = time.time()


                            integration_time += abs(inter_Ts["_t_stop_laplacians"]-inter_Ts["_t_start_laplacians"])


                            if save_intermediate:
                                if integration_time > 0:
                                    print("PID ", os.getpid(), " saving intermediate results to ",
                                          ITPT_temp_file.format(int_start,  k+1))

                                    pd.to_pickle({"ITPT" : ITPT,
                                          "interval_start" : int_start,
                                          "last_treated_interval" : k,
                                          "integration_time" : integration_time,
                                          "Tk": Tk},
                                        ITPT_temp_file.format(int_start, k+1))

                    if compute_lin:
                        tk = time.time()
                        inter_Ts_lin = ContTempNetwork.load_inter_T(inter_trans_file0.format(k=k) + \
                                   "_lin_inter_trans_mat")
                        if verbose:
                            print("PID ", os.getpid(),
                              f" -- loading Tk_lin took = {time.time()-tk}")

                        tk = time.time()

                        for inter_Tk_lin, dtk in zip(inter_Ts_lin["inter_T_lin"][1/tau_w][t_s][::d],
                                           d * np.diff(inter_Ts_lin["times_k_start_to_k_stop+1"][::d])):


                            set_to_zeroes(inter_Tk_lin, tol=tol)
                            inplace_csr_row_normalize(inter_Tk_lin)

                            Tk_lin = sparse_matmul(Tk_lin,inter_Tk_lin.tocsr(),
                                                verbose=verbose_sparse_matmul,
                                                log_message="Tk_lin")

                            set_to_zeroes(Tk_lin, tol=tol)
                            inplace_csr_row_normalize(Tk_lin)

                            pk_lin = sparse_matmul(p1, Tk_lin.tocsr(),
                                                verbose=verbose_sparse_matmul,
                                                log_message="pk_lin")

                            # in order to avoid nan in Ik due to 0 * np.inf
                            pk_lin[np.where(pk_lin == 0)] = 1

                            # we do (Tk @ Pk^-1/2) @ (Tk @ Pk^-1/2)^T for ITPTk
                            ITPTk_lin = Tk_lin.copy().tocsr()
                            inplace_csr_matmul_diag(ITPTk_lin,np.sqrt(1/pk_lin))


                            ITPTk_lin = sparse_gram_matrix(ITPTk_lin, transpose=True,
                                                        verbose=verbose_sparse_matmul,
                                                        log_message="ITPTk_lin")

                            ITPTk_lin.data *= dtk

                            if integral_rtol is not None:
                                set_to_zeroes(ITPTk_lin, integral_rtol)

                            ITPT_lin = ITPT_lin + ITPTk_lin

                            if integral_rtol is not None:
                                set_to_zeroes(ITPT_lin, integral_rtol)

                        integration_time_lin += abs(inter_Ts_lin["_t_stop_laplacians"]-inter_Ts_lin["_t_start_laplacians"])

                        if verbose:
                            print("PID ", os.getpid(),
                              f" -- integrating Tk_lin took = {time.time()-tk}")



                # saving results
                if compute_expm:
                    if integration_time > 0:
                        print("PID ", os.getpid(), " saving to ", ITPT_file.format(int_start, int_stop))

                        pd.to_pickle({"ITPT" : triu(ITPT),
                                      "integration_time" : integration_time},
                                     ITPT_file.format(int_start, int_stop))
                    else:
                        print("PID ", os.getpid(), f" integration time is zero, not saving {int_start} to {int_stop}")

                if compute_lin:

                    if integration_time_lin > 0:
                        tk = time.time()

                        print("PID ", os.getpid(), "saving to ", ITPT_lin_file.format(int_start, int_stop))

                        pd.to_pickle({"ITPT" : triu(ITPT_lin),
                                      "integration_time" : integration_time_lin},
                                     ITPT_lin_file.format(int_start, int_stop))

                        if verbose:
                            print("PID ", os.getpid(),
                              f" -- saving ITPT_lin took = {time.time()-tk}")

                    else:
                        print("PID ", os.getpid(), f" lin integration time is zero, not saving {int_start} to {int_stop}")

                t1 = time.time()
                print("PID ", os.getpid(), f"finished in {t1 - t0:.2f}" )

                _int_start = int_stop


        del ITPT
        del ITPT_lin
        del Tk
        del Tk_lin
        gc.collect()

    except Exception:
        print("PID ", os.getpid(), "-+-+-+ Exception at int_start=", int_start,
              " int_stop=", int_stop, " tau_w=", tau_w, "ind_start=", ind_start,
              file=sys.stdout)
        print("PID ", os.getpid(), "-+-+-+ Exception at int_start=", int_start,
              " int_stop=", int_stop, " tau_w=", tau_w, "ind_start=", ind_start,
               file=sys.stderr)

        traceback.print_exc(file=sys.stderr)




#%%
def main():
    t00 = time.time()

    ind_starts_tau_ws = []


    if only_from_start:
        ind_starts_tau_ws.extend([(0, tau_w, False) for tau_w in tau_ws])
    elif only_from_finish:
        ind_starts_tau_ws.extend([(len(all_inds)-1, tau_w, True) for tau_w in tau_ws])
    else:
        for reverse_time in rev_time_list:
            if reverse_time:
                if only_from_start_and_finish:
                    ind_starts_tau_ws.extend([(len(all_inds)-1, tau_w, reverse_time) for tau_w in tau_ws])
                else:
                    ind_starts_tau_ws.extend([(ind_start, tau_w, reverse_time) for \
                                          ind_start, tau_w in product(range(1,len(all_inds)), tau_ws)])
            elif only_from_start_and_finish:
                ind_starts_tau_ws.extend([(0, tau_w, reverse_time) for tau_w in tau_ws])
            else:
                ind_starts_tau_ws.extend([(ind_start, tau_w, reverse_time) for \
                                      ind_start, tau_w in product(range(len(all_inds)-1), tau_ws)])

    print(ind_starts_tau_ws)

    print(f"starting pool of {ncpu} cpus")
    with Pool(ncpu) as p:
        work = p.map_async(worker, ind_starts_tau_ws)
        data = work.get()


    print(f"***** Finished! in {time.time()-t00:.2f}")

# combination of ind_start and all_inds
if __name__ == "__main__":
    main()
