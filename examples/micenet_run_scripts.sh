#!/bin/bash


#
# Flow stability for dynamic community detection https://arxiv.org/abs/2101.06131v2
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


# This script helps to manage the computation of flow stability dynamic communities
# for large networks by parallelizing and saving intermediary results.
# It consists consecutively calling three python scripts:
#
# 1) run_laplacians_transmats.py: computes the inter event transition matrices. This is done by first splitting
# the network in a given number of slices (controlled by `num_slices` or `slice_length`) and spreading the computation
# for each slice over `ncpu` parallel jobs.
# The results for each slice and each value of the resolution (`tau_w_list`) are saved in separate files 
# named {savedir}/{net_name}_tau_w{tau_w}_int{slice_number}__inter_trans_mat.{ext}.
#
# 2) run_cov_integrals.py: computes the integral of the random walk covariance matrices. 
# This is done on a coarser temporal grid than the grid used for the slices of the 1st step.
# The grid step size is defined by `int_length` in units of slices. By default, the integral of the covariance
# will be computed starting and ending from all combinations of the grid steps. Using `--only_from_start_and_finish`
# will compute the forward integrals from the first time point until each following grid steps and 
# the backward integrals from the last time point until each previous grid steps.
# The resolutions (waiting times) are read from the saved inter-event transition matrices.
# For efficiency, only the first term (the sparse part) of the autocovariance integral is computed and saved.
# The results are saved in files named
# {savedir}/{net_name}_tau_w{tau_w}_PT_{initial_grid_point}_to_{final_grid_point}.{ext}
# It will use sparse-dot-mkl to do multithreaded sparse matrix products, if it is installed.
#
# 3) run_clusterings.py: clusters the covariance integrals computed at step 2. 
# The clusterings are computed with the Louvain algorithm for each file containing the result of step 2 in {savedir}.
# The Louvain algorithm is repeated `num_repeat` times and the best partition (maximizing the flow stability) is selected.
# The ensemble of partitions is used to compute the variation of information.
# The computations for the different files is spread of `nproc_files` in parallel. It is also possible to parallelize the 
# repeated clusterings using `nproc_clustering`. This requires nproc_files*nproc_clustering cores.



# We set the number of threads to the number of threads per core, since we use multiprocessing.
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

# number of cores
NCPU=3 

# jobname (used for log files)
JNAME="micenet"

# name of the temporal network file
NETFILENAME="micenet_2017_02_28_to_2017_05_01.pickle"

# path to the directory with the python scripts
RUNDIR="../parallel_scripts"

DATADIRBASE="mice_data"

mkdir -p $DATADIRBASE

LAPINTERTRANSDIR="$DATADIRBASE/lapl_intertransmat"

mkdir -p $LAPINTERTRANSDIR

INTEGDIR="$DATADIRBASE/integralgrid"

mkdir -p $INTEGDIR

CLUSTSDIR="$DATADIRBASE/clusters"

mkdir -p $CLUSTSDIR

# name used in results filenames
NETNAME="micenet2017"

# laplacians and transition matrices resolution grid
SLICELENGTH="$((60*60))" # slice length = 1 hour

T_STOP="$((60*60*24*7))" # one week in seconds

# final grid resolution in unit of slices
INT_LENGTH=24 # 1 day

# waiting times (dynamic scales)
TAUWS=(1 60 "$((60*60))" "$((60*60*24))" "$((60*60*24*7))")

# number of times the clustering will be run (in order to compute the variation of information).
CLUSTNUMREPEAT=50

#compute laplacians and intertransition matrices
# python3 -u $RUNDIR/run_laplacians_transmats.py \
#         --datadir $DATADIRBASE \
#         --savedir $LAPINTERTRANSDIR \
#         --net_filename $NETFILENAME \
#         --net_name $NETNAME \
#         --not_lin_transmat \
#         --slice_length $SLICELENGTH \
#         --ncpu $NCPU \
#         --tau_w_list ${TAUWS[*]} \
#         --save_inter_T \
#         --compress_inter_T \
#         --use_dense_expm \
#         --t0 0 \
#         --tend $T_STOP \
#         > output_${JNAME}_lptm.txt 2> error_${JNAME}_lptm.txt

        
# compute covariance intervals 
# python3 -u $RUNDIR/run_cov_integrals.py \
#         --datadir $LAPINTERTRANSDIR \
#         --savedir $INTEGDIR \
#         --net_name $NETNAME \
#         --int_length $INT_LENGTH \
#         --ncpu $NCPU \
#         --only_expm_transmats \
#         --only_from_start_and_finish \
#         --verbose \
#         --print_mem_usage \
#         > output_${JNAME}_intg.txt 2> error_${JNAME}_intg.txt

# compute clustering of integral
python3 -u $RUNDIR/run_clusterings.py \
        --datadir $INTEGDIR \
        --savedir $CLUSTSDIR \
        --nproc_files $NCPU \
        --nproc_clustering 1 \
        --num_repeat $CLUSTNUMREPEAT \
        --net_name $NETNAME \
        --verbose 1 \
        --clust_verbose 1 \
        > output_${JNAME}_clust.txt 2> error_${JNAME}_clust.txt

