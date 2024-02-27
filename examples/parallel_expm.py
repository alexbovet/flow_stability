"""
#
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
import numpy as np
from scipy.sparse.linalg import expm_multiply, expm
from scipy.sparse import csc_matrix, vstack, csr_matrix, isspmatrix_csc
from scipy.sparse.csgraph import connected_components
from multiprocessing import Pool
from SparseStochMat import inplace_csr_row_normalize
from multiprocessing import RawArray
import time

# A global dictionary storing the variables passed from the initializer.
var_dict = {}


def _init_worker(data, indices, indptr, N):
    # reconstruct A from shared arrays
    var_dict['A'] = csc_matrix((np.frombuffer(data, dtype=np.float64),
                                  np.frombuffer(indices, dtype=np.int32),
                                  np.frombuffer(indptr, dtype=np.int32)),
                                 shape=(N,N))
    var_dict['N'] = N
    
def _worker(args):
    
    i, thresh_ratio = args 
    
    delta_i = np.zeros(var_dict['N'],
                       dtype=np.float64)
    delta_i[i] = 1.0
    
    Tcol_i = expm_multiply(var_dict['A'],
                         delta_i)
    
    if thresh_ratio is not None:
        
        Tcol_i[Tcol_i<Tcol_i.max()/thresh_ratio]=0
    
    return csc_matrix(Tcol_i)

def compute_parallel_expm(A, nproc=1, thresh_ratio=None, 
                          normalize_rows=True,
                          verbose=True):
    """
    
    computes the exponential matrix of A by computing each column separately
    exploiting the fact that the column i of expm(A) is expm_multiply(A,delta_i)
    where delta i is the vector with zeros everywhere except on i.
    This only works if A is equal to (minus) a Laplacian matrix
    
    
    Parameters
    ----------
    A : scipy csc sparse matrix
        Square csc sparse matrix representing (+ or -) a Laplacian.
        If A is not csc, it will be converted to csc format.
    nproc : int, optional
        number of parallel processes. The default is 1.
    thresh_ratio: float, optional.
        Threshold ratio used to trim negligible values in the resulting matrix.
        For each columns `c`, values smaller than `max(c)/thresh_ratio` are set to 
        zero. Default is None.
    normalize_rows: bool, optional.
        Whether rows of the resulting matrix are normalized to sum to 1.


    Returns
    -------
    expm(A).
        scipy csr sparse matrix

    """
    
    if not isspmatrix_csc(A):
        A = csc_matrix(A)
        
    N = A.shape[0]
    
    # create arrays to share A between processes
    indices = RawArray('i',A.indices)
    indptr = RawArray('i',A.indptr)
    data = RawArray('d', A.data)
    
    if verbose:
        print('PID ', os.getpid(), ' : ',f'compute_parallel_expm starting a pool of {nproc} workers')
    t0 = time.time()
    with Pool(nproc, initializer=_init_worker, 
              initargs=(data, indices, indptr, N)) as p:
        res = p.map(_worker, [(i,thresh_ratio) for i in range(N)])
        
    # delete shared arrays
    global var_dict
    var_dict = {}
    
    # seems faster than _stack_sparse_cols
    T = vstack(res).T.tocsr()
    
    if normalize_rows:
        inplace_csr_row_normalize(T)
        
    if verbose:
        print('PID ', os.getpid(), ' : ', f'compute_parallel_expm took {time.time()-t0:.3f}s, computed expm has {T.getnnz()} non-zeros.')
        
    return T

def _stack_sparse_cols(col_list):
    
    # create csc sparse matric from sparse column list
    data_size = sum(C.data.size for C in col_list)
    N = max(col_list[0].shape)
    
    data = np.zeros(data_size, dtype=np.float64)
    indices = np.zeros(data_size, dtype=np.int32)
    indptr = np.zeros(N+1, dtype=np.int32)
    
    ptr = 0
    for col, C in enumerate(col_list):
        nnz_col = C.data.size
        data[ptr:ptr+nnz_col] = C.T.tocsc().data
        indices[ptr:ptr+nnz_col] = C.T.tocsc().indices
        ptr = ptr + nnz_col
        indptr[col+1] = ptr
        
        
    return csc_matrix((data, indices, indptr), shape=(N,N))

def _expm_worker(cmp_ind):
    
    return expm(var_dict['A'][cmp_ind,:][:,cmp_ind]).toarray()
    

def compute_subspace_expm_parallel(A, n_comp=None, comp_labels=None, verbose=False,
                           nproc=1,
                           thresh_ratio=None, 
                           normalize_rows=True,):
    """ compute the exponential matrix of `A` by applying expm on each connected
        subgraphs defined by A and recomposing it to return expm(A).
        Small subgraphs are computed in parallel, each using scipy expm,
        and large subgraphs are computed with compute_parallel_expm.

        Parameters:
        -----------
        
        A : scipy.sparse.csc_matrix
        
        nproc : int, optional
            number of parallel processes. The default is 1.
        thresh_ratio: float, optional.
            Threshold ratio used to trim negligible values in the resulting matrix.
            Values smaller than `max(expm(A))/thresh_ratio` are set to 
            zero. Default is None.
        normalize_rows: bool, optional.
            Whether rows of the resulting matrix are normalized to sum to 1.
        

        Returns:
        --------
        
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
        
    if verbose:
        print('PID ', os.getpid(), f' : subspace_expm with {n_comp} components')
    
            
    large_comps, = np.nonzero(comp_sizes >= 1000)
    small_comps, = np.nonzero(comp_sizes < 1000)
        
    subnets_expms = dict()
    # first compute large comps with exmp_multiply
    for cmp in large_comps:
        cmp_ind = cmp_indices[cmp]
        if verbose:
            print('PID ', os.getpid(), f' : computing component {cmp} over {n_comp}, with size {cmp_ind.size} using expm_multiply')        
        subnets_expms[cmp] = compute_parallel_expm(A[cmp_ind,:][:,cmp_ind],
                                                   nproc=nproc, 
                                                   thresh_ratio=None, 
                                                   normalize_rows=False,
                                                   verbose=verbose
                                                   ).toarray()
        
    
    Aindices = RawArray('i',A.indices)
    Aindptr = RawArray('i',A.indptr)
    Adata = RawArray('d', A.data)
    
    # now compute small comps in parallel
    if verbose:
            print('PID ', os.getpid(), f' : computing {small_comps.size} small components with {nproc} workers')
    t0 = time.time()
    with Pool(nproc, initializer=_init_worker, 
          initargs=(Adata, Aindices, Aindptr, num_nodes)) as p:
        res = p.map(_expm_worker, [cmp_indices[c] for c in small_comps])
        
    # delete shared arrays
    global var_dict
    var_dict = {}
    
    if verbose:
            print('PID ', os.getpid(), f' : small components computation took {time.time()-t0:.3f}s')
    t0 = time.time()
    
    # organize results
    for c, sub_expm in zip(small_comps, res):
        subnets_expms[c] = sub_expm
    
    # constructors for sparse array
    data = np.zeros((comp_sizes**2).sum(), dtype=np.float64)
    indices = np.zeros((comp_sizes**2).sum(), dtype=np.int32)
    indptr = np.zeros(num_nodes+1, dtype=np.int32)
    
    # reconstruct csr sparse matrix
    if verbose:
            print('PID ', os.getpid(), ' : reconstructing expm mat')
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
    
    
