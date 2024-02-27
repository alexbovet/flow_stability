#
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

"""
Helper script to automatize and parallelize the computation of Laplacians, 
inter-event transition matrices and transition matrices for a temporal network.

Computes the inter event transition matrices. This is done by first splitting
the temporal network in a given number of slices (controlled by `num_slices` or `slice_length`) and spreading the computation
for each slice over `ncpu` parallel jobs.
The results for each slice and each value of the resolution (`tau_w_list`) are saved in separate files 
named {savedir}/{net_name}_tau_w{tau_w}_int{slice_number}__inter_trans_mat.{ext}.

"""
import sys
import os
import time
import glob
import traceback
import numpy as np
import pandas as pd

from multiprocessing import Pool
from argparse import ArgumentParser, SUPPRESS, ArgumentDefaultsHelpFormatter

from ..FlowStability import FlowIntegralClustering
# raise Exception

#%%

ap = ArgumentParser(prog='run_laplacians_transmats',
                    description=__doc__,
                    add_help=False,
                    formatter_class=ArgumentDefaultsHelpFormatter)


required = ap.add_argument_group('required arguments')
optional = ap.add_argument_group('optional arguments')

# Add back help 
optional.add_argument(
    '-h',
    '--help',
    action='help',
    default=SUPPRESS,
    help='show this help message and exit'
)


required.add_argument('--datadir', required=True, type=str,
                help="location where the temporal network is stored.")

required.add_argument('--net_filename', required=True, type=str,
                help="filename of the network to load.")

required.add_argument('--savedir', required=True, type=str,
                help="where the laplacians and transition matrices will be saved.")

optional.add_argument('--ncpu', default=4, type=int,
                      help="Size of the multiprocessing pool.")

optional.add_argument('--net_name', default='tempnet', type=str,
                help="Name that used when saving results,")

optional.add_argument('--t_s', default=10, type=float,
                help="time paramter of the linear approximation of the tran. mat.")

optional.add_argument('--tol', default=1e-8, type=float,
                help='Values smaller than tol are set to zero in sparse transition matrices.')

optional.add_argument('--num_slices', default=50, type=int,
                help="number of slices that will be used to parallelize and save the results")

optional.add_argument('--slice_length', default=None, type=float,
                help="Length of a single slice. Used to set the number of slices for parallelization instead of num_slices. If provided, will have priority over num_slices.")

optional.add_argument('--t0', default=None, type=float,
                help="time when to start the analysis. Default is the starting time of the first event.")

optional.add_argument('--tend', default=None, type=float,
                help="time when to stop the analysis. Default is the ending time of the last event.")

optional.add_argument('--verbose', action='store_true')

optional.add_argument('--batch_num', default=0, type=int,
                help="if the work is splitted in several batches (to split over several computers), batch numer")

optional.add_argument('--total_num_batches', default=1, type=int)

optional.add_argument('--tau_w_list', default=[1], type=float, nargs='+',
                help="list of waiting times to analyze. given as '(tau1 tau2 ...)'")

optional.add_argument('--not_expm_transmat', action='store_false',
                help="use to not compute the trans. mat. with expm, i.e. only with the linear approx.")

optional.add_argument('--not_lin_transmat', action='store_false',
                help="use to not compute the trans. mat. with linear approx ,i.e. only with the expm.")

optional.add_argument('--save_slice_trans', action='store_true',
                help="for each slice, saves the transition matrices T[t_start,t_end]")

optional.add_argument('--compress_slice_trans', action='store_true',
                help="compress the saved transition matrix of each slice.")

optional.add_argument('--force_csr', action='store_true',
                help="for each slice, the product of the inter_T are done in csr format")

optional.add_argument('--save_inter_T', action='store_true',
                help="use to save all interevent transition matrices T[t_k,t_k+1]" +\
                " for each slice. Can be used in combination with --compress_inter_T, " +\
                    " --save_delta_inter_T and --use_sparse_stoch.")

optional.add_argument('--compress_inter_T', action='store_true',
                help="Compresses the interevent transition matrices with gzip.")

optional.add_argument('--save_delta_inter_T', action='store_true',
                help="Saves the difference between consecutive inter_T. Only use if " +\
                    "inter_T have not been computed with use_sparse_stoch.")

optional.add_argument('--not_use_sparse_stoch', action='store_false',
                help="Use simple scipy sparse matrices, instead of the custom " +\
                    "stochastic sparse matrices, for computing and storing " +\
                        "inter transition matrices. Default is to use sparse_stoch.")

optional.add_argument('--use_dense_expm', action='store_true',
                help="Laplacians are converted to dense matrices before computing their " +\
                    " exponential. Can be faster if the matrices are not too big. " +\
                        "Need `--not_use_sparse_stoch`.")
    
optional.add_argument('--save_flow_int', action='store_true',
                help="computes and stores the integral of the autocov for each slice.")

optional.add_argument('--save_static_adjacencies', action='store_true',
                help="computes and saves static adjacency matrices for each slice.")

optional.add_argument('--time_slices_from_net_file', action='store_true',
                help="Uses the time slices saved with the TemporalNetwork file, in `net.time_slices_bounds`.")

optional.add_argument('--intervals_to_skip', default=[], type=int, nargs='+',
                help="list of intervals to skip. given as '(int1 int2 ...)'")


inargs = vars(ap.parse_args())
datadir = inargs['datadir']
savedir = inargs['savedir']
ncpu = inargs['ncpu']
net_name = inargs['net_name']
t_s = inargs['t_s']
tol = inargs['tol']
num_slices = inargs['num_slices']
slice_length = inargs['slice_length']
t0 = inargs['t0']
tend = inargs['tend']

net_filename = inargs['net_filename'] 

verbose = inargs['verbose']
batch_num = inargs['batch_num']
total_num_batches = inargs['total_num_batches']
tau_ws = np.array(inargs['tau_w_list'])
compute_expm_transmat = inargs['not_expm_transmat']
compute_lin_transmat = inargs['not_lin_transmat']
save_inter_T = inargs['save_inter_T']
compress_inter_T = inargs['compress_inter_T']
save_flow_int = inargs['save_flow_int']
save_slice_trans = inargs['save_slice_trans']
compress_slice_trans = inargs['compress_slice_trans']
force_csr = inargs['force_csr']
save_delta_inter_T = inargs['save_delta_inter_T']
use_sparse_stoch = inargs['not_use_sparse_stoch']
use_dense_expm = inargs['use_dense_expm']
time_slices_from_net_file = inargs['time_slices_from_net_file']
intervals_to_skip = np.array(inargs['intervals_to_skip'])


if save_flow_int:
    compute_intermediate = True
else:
    compute_intermediate = False

save_static_adjacencies = inargs['save_static_adjacencies']

if not compute_lin_transmat and not compute_expm_transmat:
    raise Exception('Nothing to compute.')
    

from TemporalNetwork import ContTempNetwork as NetClass

reverse_time_list = [False, True]

# arguments for compute_transition_matrices
args = {'verbose':verbose,
        'save_intermediate': compute_intermediate,
        'reverse_time' : None}    

#%%
if datadir == '':
    raise Exception('datadir must be given')

if savedir == '':
    raise Exception('savedir must be given')



attributes_list_to_save = ['_t_start_laplacians',
                        '_k_start_laplacians',
                        '_t_stop_laplacians',
                        '_k_stop_laplacians',
                        '_compute_times',
                        'num_nodes']


print('Arguments:')
for item in inargs.items():
    print(item)
#%%
net = NetClass.load(filename=os.path.join(datadir, net_filename),
                    attributes_list = ['events_table',
                          'times',
                          'time_grid',
                          'num_nodes',
                          'time_slices_bounds'])

net._compute_time_grid()

if time_slices_from_net_file:
    time_slices = net.time_slices_bounds
    
else:
    if t0 is None:
        t0 = net.times[0]
    
    if tend is None:
        tend = net.times[-1]
    
    if slice_length is None:
    
        slice_length = (tend-t0)/num_slices
    
    time_slices = [t0]
    t = t0
    while t < tend:
        t += slice_length
        time_slices.append(t)
    
t_starts = time_slices[:-1]
t_stops = time_slices[1:]


full_batch = list(enumerate(zip(t_starts, t_stops)))

batch_size = int(len(full_batch)/total_num_batches)

batch = full_batch[batch_num*batch_size:(batch_num+1)*batch_size]

if batch_num == total_num_batches-1:
    # last batch until the end
    batch = full_batch[batch_num*batch_size:]




#%%

def worker(net_start_stop):
    
    
    ev_table, t_start, t_stop, int_index = net_start_stop
    
    net = NetClass(events_table=ev_table)     # should trim ev_table with t_start and t_stop

    print(f'PID {os.getpid()}:', 'interval ', int_index)
    
        
    net_file0 = os.path.join(savedir, net_name + '_tau_w{tau_w:.3e}' + \
                    '_int{0:06d}_'.format(int_index)) 

    if save_static_adjacencies:
        
        adj_file = os.path.join(savedir, net_name + '_static_adjacency' + \
                                     '_int{0:06d}'.format(int_index) + '.pickle')
        if not os.path.exists(adj_file):
            
            t0 = time.time()
            A = net.compute_static_adjacency_matrix(start_time=t_start, end_time=t_stop)
            
            pd.to_pickle(A, adj_file)

            t1 = time.time()
            print(f'PID {os.getpid()}:', 
                  'computing and saving static adjacency matrix, took {0}'.format(t1-t0))
            
    # check if all files already exists
    all_lin_exists = True
    all_expm_exists = True
    all_inter_lin_exists = True
    all_inter_expm_exists = True
    
    for tau_w in tau_ws:
                
            
        net_file_forw = net_file0.format(tau_w=tau_w) 
        net_file_back = net_file0.format(tau_w=tau_w) + '_reversed'
        
        net_file_forw_expm = net_file_forw + '_trans_mat'
        net_file_back_expm = net_file_back + '_trans_mat'
        net_file_inter_expm = net_file_forw + '_inter_trans_mat'
      
        net_file_forw_lin = net_file_forw + '_lin_trans_mat'
        net_file_back_lin = net_file_back + '_lin_trans_mat'
        net_file_inter_lin = net_file_forw + '_lin_inter_trans_mat'
      
         
        if (not glob.glob(net_file_forw_lin + '*')) or \
           (not glob.glob(net_file_back_lin + '*')):
            all_lin_exists = False
        
        if (not glob.glob(net_file_inter_lin + '*')):
            all_inter_lin_exists = False
            
        if (not glob.glob(net_file_forw_expm + '*')) or \
            (not glob.glob(net_file_back_expm + '*')):
            all_expm_exists = False
            
        if (not glob.glob(net_file_inter_expm + '*')):
            all_inter_expm_exists = False            
    
    if (all_inter_expm_exists and all_inter_lin_exists and not save_slice_trans) or \
        (all_expm_exists and all_lin_exists):
        print(f'PID {os.getpid()}:', 'interval = ', int_index, 
                          ' all files already exists' )
    else:
            
        try:
            t0 = time.time()
            
        
            print(f'PID {os.getpid()}:', 
                  f' computing Laplacians t_start={t_start}, t_stop={t_stop}')
            net.compute_laplacian_matrices(t_start=t_start, t_stop=t_stop, verbose=verbose)
            
            print(f'PID {os.getpid()}:', 
                  f' k_start={net._k_start_laplacians}, k_stop={net._k_stop_laplacians}')
            
            t1 = time.time()
            print(f'PID {os.getpid()}:','finished in {:.2f}'.format(t1 - t0) )
            
            if compute_lin_transmat:
                print(f'PID {os.getpid()}:', ' computing stationary dists')
                net._compute_stationary_transition(verbose=verbose,
                                                   use_sparse_stoch=use_sparse_stoch)
                
                t2 = time.time()
                print(f'PID {os.getpid()}:','finished in {:.2f}'.format(t2 - t1 ))
            
                
        except Exception as e:
            print(f'PID {os.getpid()}:', '-+-+-+ Exception at interval=', int_index, 
                  file=sys.stdout)
            print(f'PID {os.getpid()}:', 'Exception: ', str(e), file=sys.stdout)
            
            print(f'PID {os.getpid()}:', '-+-+-+ Exception at interval=', int_index, 
                   file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

            
        for tau_w in tau_ws:
            
            try:
            
                lamda = 1/tau_w
                
                net_file = net_file0.format(tau_w=tau_w)
                
                # first compute inter trans matrices (does not depend on time direction)
                if compute_expm_transmat:
                    print(f'PID {os.getpid()}:', ' computing inter transition matrices')
                    net.compute_inter_transition_matrices(lamda=lamda,verbose=verbose,
                                                          use_sparse_stoch=use_sparse_stoch,
                                                          dense_expm=use_dense_expm)

                                    
                    if save_inter_T:
                        
                        if not glob.glob(net_file + '_inter_trans_mat*'):
                        
                            print(f'PID {os.getpid()}:', 'saving ', net_file + '_inter_trans_mat' )
                            
                            
                            net.save_inter_T(net_file + '_inter_trans_mat', 
                                                          lamda=lamda,
                                                          round_zeros=True, tol=tol,
                                                          compressed=compress_inter_T,
                                                          save_delta=save_delta_inter_T)
                                                        
                        else:
                            print(f'PID {os.getpid()}:', 'interval = ', int_index,  ' tau_w = ', tau_w, 
                                  '_inter_trans_mat already saved' )
                
                
                if compute_lin_transmat:
                    print(f'PID {os.getpid()}:', ' computing linear inter transition matrices')
                    net.compute_lin_inter_transition_matrices(lamda=lamda, 
                                                              verbose=verbose,
                                                              t_s=t_s,
                                                              use_sparse_stoch=use_sparse_stoch)
                    
                    if save_inter_T:
                        
                        if not glob.glob(net_file + '_lin_inter_trans_mat*'):
                            
                            print(f'PID {os.getpid()}:', 'saving ', net_file + '_lin_inter_trans_mat' )
                                        
                            net.save_inter_T_lin(net_file + '_lin_inter_trans_mat', 
                                                          lamda=lamda,
                                                          round_zeros=True, tol=tol,
                                                          compressed=compress_inter_T,
                                                          save_delta=save_delta_inter_T)
                        else:
                            print(f'PID {os.getpid()}:', 'interval = ', int_index,  ' tau_w = ', tau_w, 
                                      '_lin_inter_trans_mat already saved' )
                                 
                        
                # now compute and save the product of inter_trans_mat
                if save_slice_trans:
                    for reverse_time in reverse_time_list:
        
                        time_direction_str = ''
                        if reverse_time:
                            time_direction_str = '_reversed'
                            
                            
                        if 'reverse_time' in args.keys():
                            args['reverse_time'] = reverse_time
                            
                        net_file = net_file0.format(tau_w=tau_w) + time_direction_str
                        
                        print(f'PID {os.getpid()}:', 'tau_w', tau_w)

                        if compute_expm_transmat:
                            
                            # if this interval has not been treated yet
                            if not glob.glob(net_file + '_trans_mat*'):
                                print(f'PID {os.getpid()}:', 'computing trans_matrix', net_file)
                                t0 = time.time()
                                net.compute_transition_matrices(lamda=lamda, 
                                                                force_csr=force_csr,
                                                                tol=tol,
                                                                **args)
                                t1 = time.time()
                                print(f'PID {os.getpid()}:', 'finished in {:.2f}'.format(t1 - t0) )
                                
                                if save_flow_int:
                                    d = 1
                                    if reverse_time:
                                         d = -1
                                    I = FlowIntegralClustering(T_list=net.T[lamda],
                                          time_list=net.times[net._k_start_laplacians:\
                                                              net._k_stop_laplacians+1].values[::d]).I
                                    
                                    pd.to_pickle(I,net_file + '_flow_int.pickle')
                                
                                if isinstance(net.T[lamda], list):
                                    # we want flow_int but not save intermediate
                                    net.T[lamda] = net.T[lamda][-1]
                                    
                                print(f'PID {os.getpid()}:', 'saving ', net_file + '_trans_mat.pickle')

                                
                                net.save_T(net_file + '_trans_mat.pickle',
                                               round_zeros=True,
                                               tol=tol,
                                               compressed=compress_slice_trans)
                            
                                del net.T
    
                            else:
                                print(f'PID {os.getpid()}:', 'interval = ', int_index,  ' tau_w = ', tau_w, 
                                      ' expm T already treated, passing to next' )
                            
                        if compute_lin_transmat:
                            
                            if not glob.glob(net_file + '_lin_trans_mat*'):
                                print(f'PID {os.getpid()}:', 'computing lin trans matrix ', net_file )
                                t1 = time.time()
                                net.compute_lin_transition_matrices(lamda=lamda, 
                                                                    t_s=t_s, 
                                                                    **args)
                                t2 = time.time()
                                print(f'PID {os.getpid()}:', 'finished in {:.2f}'.format(t2 - t1 ))
                                
                                if save_flow_int:
                                    d = 1
                                    if reverse_time:
                                        d = -1
                                    I_lin = FlowIntegralClustering(T_list=net.T_lin[lamda][t_s],
                                              time_list=net.times[net._k_start_laplacians:\
                                                                  net._k_stop_laplacians+1].values[::d]).I
                                    
                                    pd.to_pickle(I_lin,net_file + '_lin_flow_int.pickle')
                                        
                                if isinstance(net.T_lin[lamda][t_s], list):
                                    # in the case we want flow_int but not save intermediate
                                    net.T_lin[lamda][t_s] = net.T_lin[lamda][t_s][-1]
                                        
                                        
                                print(f'PID {os.getpid()}:', 'saving ', net_file + '_lin_trans_mat.pickle' )

                                
                                net.save_T_lin(net_file + '_lin_trans_mat.pickle',
                                               round_zeros=True,
                                               tol=tol,
                                               compressed=compress_slice_trans,)
                                
        
        
                                del net.T_lin
        
                                
                            else:
                                print(f'PID {os.getpid()}:', 'interval = ', int_index,  ' tau_w = ', tau_w, 
                                      ' T_lin already treated, passing to next' )
                try:                        
                    del net.inter_T
                except:
                    pass
                try:
                    del net.inter_T_lin
                except: 
                    pass
                            
            except Exception:
                print(f'PID {os.getpid()}:', '-+-+-+ Exception at interval=', int_index, 
                      ', tau_w=', tau_w, file=sys.stdout)
                print(f'PID {os.getpid()}:', '-+-+-+ Exception at interval=', int_index, 
                      ', tau_w=', tau_w, file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    del net

#%%

if __name__ == '__main__':
    t00 = time.time()
    print('starting pool of {0} cpus'.format(ncpu))
    with Pool(ncpu) as p:
        work = p.map_async(worker, [(net.events_table, t_start, t_stop, int_index) for int_index, (t_start, t_stop) in \
                                        batch if int_index not in intervals_to_skip])
        data = work.get()
        
        
    print('***** Finished! in {:.2f}'.format(time.time()-t00))
