# distutils: language = c++
# cython: profile=False
# cython: linetrace=True

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


cimport cython
import array

import numpy as np
cimport numpy as np

from cpython cimport array
from libcpp cimport bool

from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.unordered_map cimport unordered_map as umap
from cython.operator cimport dereference as deref, preincrement as inc
from scipy.sparse._sparsetools import csr_diagonal, csc_matvec

from flowstab.sparse_accumulator cimport SPA


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cdef Py_ssize_t csr_csc_matmul_countnnz(double[:] Adata,
            int[:] Aindices,
            int[:] Aindptr,
            double[:] Bdata,
            int[:] Bindices,
            int[:] Bindptr):
    """ Count max number of non-zero elements to allocate for the 
        multiplication of a NxM csr matrix with a MxN csc matrix
    
        returns int
        """
    
    cdef Py_ssize_t size = Aindptr.shape[0]-1 # num row in A
    cdef Py_ssize_t i, j, k , l, m
    cdef Py_ssize_t nz = 0

        
    for i in range(size): # iterate thourgh rows
        
        if Aindptr[i] != Aindptr[i+1]: # if this row of A is not empty
            
            for j in range(size): # j is the column of C
                
                l = Bindptr[j] # iterator over B.data col elements
                if l != Bindptr[j+1]: # if this col of B is not empty
                    
                    m = Bindices[l] # iterator over B col elements
                    for k in range(Aindptr[i],Aindptr[i+1]): # A.indices[k] is the col in A
                        while m < Aindices[k] and l + 1 < Bindptr[j+1]: 
                            # advance in B col until we are at the same position than in A row
                            l += 1
                            m = Bindices[l]
                            
                        if m == Aindices[k]:
                            nz += 1

                            break
                
    return nz
    
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def stoch_mat_add(
            int size, # big matrix size
            double[:] Adata,
            int[:] Aindices,
            int[:] Aindptr,
            int[:] Anz_rowcols,
            double Adiag_val,
            double[:] Bdata,
            int[:] Bindices,
            int[:] Bindptr,
            int[:] Bnz_rowcols,
            double Bdiag_val):
    """ addition of square stochastic sparse matrices C = A + B
    """
    
    cdef Py_ssize_t Asize = Anz_rowcols.shape[0]
    cdef Py_ssize_t Bsize = Bnz_rowcols.shape[0]
    cdef Py_ssize_t k

    # compute overlap and non-overlaping subspaces
    Anz_set = set(Anz_rowcols)
    Bnz_set = set(Bnz_rowcols)
    Cnz_set = Anz_set.union(Bnz_set)
    
    interset = Anz_set.intersection(Bnz_set)
    Anz_only = Anz_set - interset
    Bnz_only = Bnz_set - interset
    
    Cnz_rowcols = sorted(list(Cnz_set))
    
    cdef Py_ssize_t small_size = len(Cnz_set)

    # define a Python array
    # cdef array.array atemplate = array.array('i', [])
    # Cnz_rowcols = array.clone(atemplate, small_size, zero=True)
    
    # cdef set[int].iterator c_col = Cnz_set.begin()
    # k = 0
    # while c_col != Cnz_set.end():
    #     Cnz_rowcols[k] = deref(c_col)
    #     k+=1
    #     inc(c_col)
    
    
    # cdef int[:] Cnz_rowcols_view = Cnz_rowcols
    # cdef Py_ssize_t small_size = Cnz_rowcols.shape[0] # size of C.T_small
    cdef double[:] Cdata = np.zeros(Adata.shape[0] + Bdata.shape[0] + \
                                    Aindptr.shape[0] + Bindptr.shape[0], dtype=np.float64)
    cdef int[:] Cindices = -1*np.ones(Adata.shape[0] + Bdata.shape[0] + \
                                    Aindptr.shape[0] + Bindptr.shape[0], dtype=np.int32)
    cdef int[:] Cindptr = -1*np.ones(small_size+1, dtype=np.int32)
    
    cdef Py_ssize_t kc = 0 # data/indices index
    cdef Py_ssize_t i
    cdef Py_ssize_t nzi
    cdef Py_ssize_t indnz

    cdef Py_ssize_t ia = 0 # row index of A.T_small
    cdef Py_ssize_t ib = 0 # row index of B.T_small
    cdef Py_ssize_t ic = 0 # row index of C.T_small
        
    cdef bool is_not_empty_rowcol_a
    cdef bool is_not_empty_rowcol_b
    
    
    # mapping between A/B column indices and C col indices
    cdef int[:] Acol_to_Ccol = np.zeros(Asize, dtype=np.int32)
    cdef int[:] Bcol_to_Ccol = np.zeros(Bsize, dtype=np.int32)
                
    cdef Py_ssize_t ic_col = 0
    for k in range(Asize): # map col in A to col in C
        while Anz_rowcols[k] != Cnz_rowcols[ic_col]:
            ic_col+=1
        Acol_to_Ccol[k] = ic_col
    
    ic_col = 0
    for k in range(Bsize): # map col in A to col in C
        while Bnz_rowcols[k] != Cnz_rowcols[ic_col]:
            ic_col+=1
        Bcol_to_Ccol[k] = ic_col

    # sparse accumulator
    spa = new SPA(small_size)
    
    Cindptr[0] = 0
    for i in Cnz_rowcols: # iterate thourgh rows
        spa.reset(i)

        if i in Anz_set:
            for k in range(Aindptr[ia],Aindptr[ia+1]):
                spa.scatter(Adata[k], Acol_to_Ccol[Aindices[k]])
            ia += 1
            is_not_empty_rowcol_a = True
            
        if i in Bnz_set:
            for k in range(Bindptr[ib],Bindptr[ib+1]):
                spa.scatter(Bdata[k], Bcol_to_Ccol[Bindices[k]])
            ib += 1
            is_not_empty_rowcol_b = True
            
        if i in Anz_only:
            # we need to add the diagonal term of B
            spa.scatter(Bdiag_val, ic)
        if i in Bnz_only:
            # we need to add the diagonal term of A
            spa.scatter(Adiag_val, ic)
            
        # set col indices and data for C
        nzi = 0 # num nonzero in row i of C
        for indnz in spa.LS:
            Cindices[kc] = indnz
            Cdata[kc] = spa.w[indnz]
            nzi += 1
            kc += 1
        

        Cindptr[ic+1] = Cindptr[ic] + nzi
            
        ic += 1

    
    return (size, Cdata, Cindices, Cindptr, Cnz_rowcols, Adiag_val + Bdiag_val)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def stoch_mat_sub(
            int size, # big matrix size
            double[:] Adata,
            int[:] Aindices,
            int[:] Aindptr,
            int[:] Anz_rowcols,
            double Adiag_val,
            double[:] Bdata,
            int[:] Bindices,
            int[:] Bindptr,
            int[:] Bnz_rowcols,
            double Bdiag_val):
    """ addition of square stochastic sparse matrices C = A + B
    """
    
    cdef Py_ssize_t Asize = Anz_rowcols.shape[0]
    cdef Py_ssize_t Bsize = Bnz_rowcols.shape[0]
    cdef Py_ssize_t k

    # compute overlap and non-overlaping subspaces
    Anz_set = set(Anz_rowcols)
    Bnz_set = set(Bnz_rowcols)
    Cnz_set = Anz_set.union(Bnz_set)
    
    interset = Anz_set.intersection(Bnz_set)
    Anz_only = Anz_set - interset
    Bnz_only = Bnz_set - interset
    
    Cnz_rowcols = sorted(list(Cnz_set))
    
    cdef Py_ssize_t small_size = len(Cnz_set)
    
    cdef double[:] Cdata = np.zeros(Adata.shape[0] + Bdata.shape[0] + \
                                    Aindptr.shape[0] + Bindptr.shape[0], dtype=np.float64)
    cdef int[:] Cindices = -1*np.ones(Adata.shape[0] + Bdata.shape[0] + \
                                    Aindptr.shape[0] + Bindptr.shape[0], dtype=np.int32)
    cdef int[:] Cindptr = -1*np.ones(small_size+1, dtype=np.int32)
    
    cdef Py_ssize_t kc = 0 # data/indices index
    cdef Py_ssize_t i
    cdef Py_ssize_t nzi
    cdef Py_ssize_t indnz

    cdef Py_ssize_t ia = 0 # row index of A.T_small
    cdef Py_ssize_t ib = 0 # row index of B.T_small
    cdef Py_ssize_t ic = 0 # row index of C.T_small
        
    cdef bool is_not_empty_rowcol_a
    cdef bool is_not_empty_rowcol_b
    
    
    # mapping between A/B column indices and C col indices
    cdef int[:] Acol_to_Ccol = np.zeros(Asize, dtype=np.int32)
    cdef int[:] Bcol_to_Ccol = np.zeros(Bsize, dtype=np.int32)
                
    cdef Py_ssize_t ic_col = 0
    for k in range(Asize): # map col in A to col in C
        while Anz_rowcols[k] != Cnz_rowcols[ic_col]:
            ic_col+=1
        Acol_to_Ccol[k] = ic_col
    
    ic_col = 0
    for k in range(Bsize): # map col in A to col in C
        while Bnz_rowcols[k] != Cnz_rowcols[ic_col]:
            ic_col+=1
        Bcol_to_Ccol[k] = ic_col

    # sparse accumulator
    spa = new SPA(small_size)
    
    Cindptr[0] = 0
    for i in Cnz_rowcols: # iterate thourgh rows
        spa.reset(i)

        if i in Anz_set:
            for k in range(Aindptr[ia],Aindptr[ia+1]):
                spa.scatter(Adata[k], Acol_to_Ccol[Aindices[k]])
            ia += 1
            is_not_empty_rowcol_a = True
            
        if i in Bnz_set:
            for k in range(Bindptr[ib],Bindptr[ib+1]):
                spa.scatter(-1*Bdata[k], Bcol_to_Ccol[Bindices[k]])
            ib += 1
            is_not_empty_rowcol_b = True
            
        if i in Anz_only:
            # we need to add the diagonal term of B
            spa.scatter(-1*Bdiag_val, ic)
        if i in Bnz_only:
            # we need to add the diagonal term of A
            spa.scatter(Adiag_val, ic)
            
        # set col indices and data for C
        nzi = 0 # num nonzero in row i of C
        for indnz in spa.LS:
            Cindices[kc] = indnz
            Cdata[kc] = spa.w[indnz]
            nzi += 1
            kc += 1
        

        Cindptr[ic+1] = Cindptr[ic] + nzi
            
        ic += 1

    
    return (size, Cdata, Cindices, Cindptr, Cnz_rowcols, Adiag_val - Bdiag_val)     
    
def test_set(int[:] arr):
    
    cdef cset[int] aset
    cdef Py_ssize_t k
    cdef Py_ssize_t size = arr.shape[0]
    
    for k in range(size):
        aset.insert(arr[k])
        
    
    # define a Python array
    cdef array.array atemplate = array.array('i', [])
    Cnz_rowcols = array.clone(atemplate, aset.size(), zero=True)
    
    cdef cset[int].iterator i = aset.begin()
    k = 0
    while i != aset.end():
        Cnz_rowcols[k] = deref(i)
        inc(i)
        k+=1

    return Cnz_rowcols

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing   
def rebuild_nnz_rowcol(double[:] T_data,
                              long long [:] T_indices,
                              long long [:] T_indptr,
                              long long [:] nonzero_indices,
                              long long size,
                              double diag_val):
    """ returns a CSR matrix (data,indices,rownnz, shape) built from the CSR 
        matrix T_small but with
        added row-colums at zero_indicies (with 1 on the diagonal)
        
        Call:
        -----
        (data, indices, indptr, n_rows) = cython_rebuild_nnz_rowcol(double[:] T_data,
                                                                   long long [:] T_indices,
                                                                   long long [:] T_indptr,
                                                                   long long [:] nonzero_indices,
                                                                   long long size,
                                                                   double diag_val)
                                    
    """
    
    cdef Py_ssize_t row_id_small_t = -1
    cdef Py_ssize_t row_id = 0
    cdef Py_ssize_t k = 0
    cdef Py_ssize_t i = 0   
    cdef Py_ssize_t data_ind = 0   
    cdef Py_ssize_t num_data_row, row_start, row_end
    
    cdef long long T_nnz = T_data.shape[0] # num nonzeros in T_small
    cdef long long Tbig_nnz = T_nnz + size - nonzero_indices.shape[0] # num nonzeros in full T
    cdef long long T_size = nonzero_indices.shape[0] # size of small T
    
    if diag_val == 0:
        Tbig_nnz = T_nnz
    
    cdef double[:] data = np.zeros(Tbig_nnz,
                                   dtype=np.float64)
    cdef long long[:] indices = np.zeros(Tbig_nnz,
                                   dtype=np.int64)
    cdef long long[:] indptr = np.zeros(size+1,
                                   dtype=np.int64)
    cdef long long[:] Ts_indices = np.zeros(T_nnz,
                                   dtype=np.int64)
    cdef cset[long long] nz_ind_set
    
    for k in range(T_size):
        nz_ind_set.insert(nonzero_indices[k])
    

    # map col indices to new positions
    for k in range(T_nnz):
        Ts_indices[k] = nonzero_indices[T_indices[k]]
    
    row_id_small_t = -1
    data_ind = 0
    for row_id in range(size):
        row_id_small_t +=1
        if not nz_ind_set.count(row_id):
            if diag_val == 0.0:
                # add an empty row
                indptr[row_id+1] = indptr[row_id]
            else:
                # add a row with just 1 on the diagonal
                data[data_ind] = diag_val
                indices[data_ind] = row_id
                indptr[row_id+1] = indptr[row_id]+1
                
                data_ind += 1                
            row_id_small_t -= 1
        else:
            row_start = T_indptr[row_id_small_t]    
            row_end = T_indptr[row_id_small_t+1]  
            
            num_data_row = row_end - row_start
            
            data[data_ind:(data_ind+num_data_row)] = T_data[row_start:row_end]
            indices[data_ind:data_ind+num_data_row] = Ts_indices[row_start:row_end]
            indptr[row_id+1] = indptr[row_id]+num_data_row
            
            data_ind += num_data_row
            

    return (data, indices, indptr, size)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
@cython.cdivision(True)
def inplace_csr_row_normalize(double[:] X_data,
                              long long[:] X_indptr,
                              Py_ssize_t n_row,
                              double row_sum=1.0):
    """ row normalize scipy sparse csr matrices inplace.
        inspired from sklearn sparsefuncs_fast.pyx.
        
        Assumes that X_data has only positive values
        
        Call:
        -----
        inplace_csr_row_normalize(double[:] X_data, long long [:] X_indptr, Py_ssize_t n_row, double row_sum)
        
    """
    
    cdef Py_ssize_t i, j
    cdef double sum_
    cdef bool row_0 = (row_sum == 0.0)

    # if row_sum == 0.0:
    #     raise ValueError('row_sum must be different than 0.0')
        
    for i in range(n_row):
        sum_ = 0.0

        for j in range(X_indptr[i], X_indptr[i + 1]):
            sum_ += X_data[j]

        if sum_ == 0.0:
            # do not normalize empty rows (can happen if CSR is not pruned
            # correctly)
            continue
        
        if row_0:
            for j in range(X_indptr[i], X_indptr[i + 1]):
                X_data[j] -= sum_/(X_indptr[i + 1]-X_indptr[i])
        else:
            for j in range(X_indptr[i], X_indptr[i + 1]):
                X_data[j] /= (sum_/row_sum)
                

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
@cython.cdivision(True)
def inplace_csr_row_normalize_array(double[:] X_data,
                                    long long[:] X_indptr,
                                    Py_ssize_t n_row,
                                    double[:] row_sum):
    """ row normalize scipy sparse csr matrices inplace.
        inspired from sklearn sparsefuncs_fast.pyx.
        
        Assumes that X_data has only positive values
        
        Call:
        -----
        inplace_csr_row_normalize_array(double[:] X_data, int [:] X_indptr, Py_ssize_t n_row, double row_sum)
        
    """
    
    cdef Py_ssize_t i, j
    cdef double sum_

        
    for i in range(n_row):
        sum_ = 0.0

        for j in range(X_indptr[i], X_indptr[i + 1]):
            sum_ += X_data[j]

        if sum_ == 0.0:
            # do not normalize empty rows (can happen if CSR is not pruned
            # correctly)
            continue

        if row_sum[i] == 0.0:
            for j in range(X_indptr[i], X_indptr[i + 1]):
                X_data[j] -= sum_/(X_indptr[i + 1]-X_indptr[i])
        else:
            for j in range(X_indptr[i], X_indptr[i + 1]):
                X_data[j] /= (sum_/row_sum[i])  
                
            
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
@cython.cdivision(True)
def cython_inplace_csr_row_normalize_triu(double[:] X_data,
                                     long long [:] X_indptr,
                                     long long [:] X_indices,
                                     Py_ssize_t n_row,
                                     Py_ssize_t n_col,
                                     double[:] row_sum):
    """ row normalize scipy sparse csr matrices inplace.
        This function normalizes the rows of a upper
        triangular matrices T such that T + T.T - diag(T) is
        row and columns normalized.
        
        
        Assumes that X_data has only positive values and that X is square.
        
        /!| Only works if there are no empty rows where row_sum != 0.
        
        Call:
        -----
        cython_inplace_csr_row_normalize_triu(double[:] X_data,
                                     int [:] X_indptr,
                                     int [:] X_indices,
                                     Py_ssize_t n_row,
                                     Py_ssize_t n_col,
                                     double[:] row_sum)
        
    """
    
    cdef Py_ssize_t i, j, k, l
    cdef double sum_, r
    cdef double row_sum_tmp
    cdef double[:] col_sum = np.zeros(n_row, dtype=np.float64)
    cdef double[:] diag = np.zeros(n_row, dtype=np.float64)
    cdef double[:] ones = np.ones(n_col, dtype=np.float64)
    
    assert n_row == n_col
    
        
    # we have to normalize rows a number of times equal to the number of
    # non-zero row_sum values
    
    for k in range(n_row):
        if row_sum[k] != 0.0:
            diag = np.zeros(n_row, dtype=np.float64)
            #update diag
            csr_diagonal(0, n_row, n_col, X_indptr, X_indices, X_data, diag)
            
            col_sum = np.zeros(n_row, dtype=np.float64)
            #update col_sum (X.T @ 1)
            csc_matvec(n_row, n_col, X_indptr, X_indices, X_data, ones, col_sum)
                        
            for i in range(k,n_row):

                row_sum_tmp = row_sum[i] + diag[i] - col_sum[i]
                sum_ = 0.0
        
                for j in range(X_indptr[i], X_indptr[i + 1]):
                    sum_ += X_data[j]
        
                if sum_ == 0.0:
                    # # the row is empty, so we have to normalize the col to row_sum[i]
                    # if col_sum[i] != 0.0:
                    #     for l in range(X_indices.shape[0]):
                    #         if X_indices[l] == i:
                    #             X_data[l] /= (col_sum[i]/row_sum[i])
                    continue
                    
                else:
                    for j in range(X_indptr[i], X_indptr[i + 1]):
                        X_data[j] /= (sum_/row_sum_tmp)
                                
                
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def sparse_stoch_from_full_csr(int[:] nz_rowcols,
                               double[:] Tf_data,
                               int[:] Tf_indices,
                               int[:] Tf_indptr,
                               double diag_val):
        """ initialize SparseStochMat from a full size row stochastic 
            csr_matrix 
        """
        
        cdef Py_ssize_t k, its, tsrow, tbrow, i, nzr        
        
        cdef Py_ssize_t Ts_nnz = Tf_data.shape[0] - Tf_indptr.shape[0] + 1 + nz_rowcols.shape[0]
        cdef Py_ssize_t Ts_size = nz_rowcols.shape[0]

        
        cdef double[:] Ts_data = np.zeros(Ts_nnz, dtype=np.float64)
        cdef int[:] Ts_indices = -1*np.ones(Ts_nnz, dtype=np.int32)
        cdef int[:] Ts_indptr = -1*np.ones(Ts_size+1, dtype=np.int32)
        
        #map indices from big to small T as cpp unordered map
        cdef umap[int,int] BtoS
        for k in range(Ts_size):
            BtoS[nz_rowcols[k]] = k
            
        its = 0
        Ts_indptr[0] = 0
        for tsrow in range(Ts_size):
            tbrow = nz_rowcols[tsrow]
            nzr = 0
            for k in range(Tf_indptr[tbrow],Tf_indptr[tbrow+1]):
                Ts_data[its] = Tf_data[k]
                Ts_indices[its] = BtoS[Tf_indices[k]]
                its += 1
                nzr += 1
            Ts_indptr[tsrow+1] = Ts_indptr[tsrow] + nzr
            
        return (Tf_indptr.shape[0] - 1, Ts_data, Ts_indices, Ts_indptr,
                   nz_rowcols, diag_val)
    
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cpdef double get_submat_sum(double[:] Adata,
                            int[:] Aindices,
                            int[:] Aindptr,
                            int[:] row_idx,
                            int[:] col_idx):
    """ return the sum of the elements in the submatrix defined by
        row_idx and col_idx, i.e A[row_ids,:][:,col_idx].sum()
    
    """

    cdef double s = 0.0
    cdef cset[int] col_set
    cdef Py_ssize_t i, j, row
    cdef Py_ssize_t num_row = row_idx.shape[0]
    cdef Py_ssize_t num_col = col_idx.shape[0]
    
    for i in range(num_col):
        col_set.insert(col_idx[i])
        
    for i in range(num_row):
        row = row_idx[i]
        for j in range(Aindptr[row],Aindptr[row+1]):
            if col_set.count(Aindices[j]):
                s += Adata[j]
                
    return s

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def aggregate_csr_mat(double[:] Adata,
                            int[:] Aindices,
                            int[:] Aindptr,
                            int[:] idxs_array,
                            int[:] idxptr):
    """
        Aggregates a csr matrix A 
        and returns a coo matrix B
        
        performs B.shape[0]**2 operations
    """
        
    cdef vector[double] Bdata
    cdef vector[int] Brows
    cdef vector[int] Bcols
    cdef Py_ssize_t row, col
    cdef int new_size = idxptr.shape[0]-1
    cdef double pt
    
    for row in range(new_size):
        for col in range(new_size):
            pt = get_submat_sum(Adata, Aindices, 
                                      Aindptr, 
                                      idxs_array[idxptr[row]:idxptr[row+1]],
                                      idxs_array[idxptr[col]:idxptr[col+1]])
            if pt > 0.0:
                Bdata.push_back(pt)
                Brows.push_back(row)
                Bcols.push_back(col)
                
    return Bdata, Brows, Bcols, new_size


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def aggregate_csr_mat_2(double[:] Adata,
                            int[:] Aindices,
                            int[:] Aindptr,
                            int[:] idxs_array,
                            int[:] idxptr):
    """
        Aggregates a csr matrix A 
        and returns a coo matrix B
        
        Performs Adata.size operations
    """
        
    cdef vector[double] Bdata
    cdef vector[int] Brows
    cdef vector[int] Bcols
    cdef Py_ssize_t row, col, kb, ka, k
    cdef int new_size = idxptr.shape[0]-1
    cdef double pt
    
    # map indices of A to indicies of B
    cdef umap[int,int] AtoB
    for kb in range(new_size):
        for ka in range(idxptr[kb],idxptr[kb+1]):
            AtoB[idxs_array[ka]] = kb
    
    # loop over values of A
    for row in range(Aindptr.shape[0]-1):
        for k in range(Aindptr[row],Aindptr[row+1]):
            col = Aindices[k]

            Bdata.push_back(Adata[k])
            Brows.push_back(AtoB[row])
            Bcols.push_back(AtoB[col])
                
    return Bdata, Brows, Bcols, new_size
    
    
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cpdef double compute_delta_PT_moveto(double[:] PTdata,
                            int[:] PTindices,
                            int[:] PTindptr,
                            double[:] PTcscdata,
                            int[:] PTcscindices,
                            int[:] PTcscindptr,
                            Py_ssize_t k,
                            int[:] idx):
    """ return the gain in stability obtained by moving node
        k into community defined by index list in idx. 
        
        PT is the PT matrix (in csr) and PTcsc is the same matrix
        in csc format
    """
    
    cdef double s = 0.0
    cdef Py_ssize_t i, j, l
    cdef Py_ssize_t num_idx = idx.shape[0]
    
    cdef cset[int] idx_set
    
    for i in range(num_idx):
        idx_set.insert(idx[i])
    

    for j in range(PTindptr[k],PTindptr[k+1]):
        if idx_set.count(PTindices[j]):
            s += PTdata[j]
    for j in range(PTcscindptr[k],PTcscindptr[k+1]):
        if idx_set.count(PTcscindices[j]):
            s += PTcscdata[j]
                    
    # add PT_kk
    j = PTindptr[k]
    l = PTindices[j]
    # check if PT[k,k] is nonzero
    while l < k and j + 1 < PTindptr[k+1]:
        j += 1
        l = PTindices[j]
    if l == k:
        s += PTdata[j]
        
    return s

    
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def compute_delta_S_moveto(double[:] PTdata,
                            int[:] PTindices,
                            int[:] PTindptr,
                            double[:] PTcscdata,
                            int[:] PTcscindices,
                            int[:] PTcscindptr,
                            Py_ssize_t k,
                            int[:] idx,
                            double[:] p1,
                            double[:] p2):
    """ return the gain in stability obtained by moving node
        k into community defined by index list in idx. 
        
        PT is the stability matrix (in csr) and PTcsc is the same matrix
        in csc format
    """
    
    cdef double s = 0.0
    cdef Py_ssize_t i
    cdef Py_ssize_t num_idx = idx.shape[0]
    

    s = compute_delta_PT_moveto(PTdata,
                                PTindices,
                                PTindptr,
                                PTcscdata,
                                PTcscindices,
                                PTcscindptr,
                                k,
                                idx)

        
    # now substract contribution from p1^T @ p2
    for i in range(num_idx):
        s -= p1[k]*p2[idx[i]]
        s -= p1[idx[i]]*p2[k]
    s -= p1[k]*p2[k]
    
    return s
    
    
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cpdef double compute_delta_PT_moveout(double[:] PTdata,
                            int[:] PTindices,
                            int[:] PTindptr,
                            double[:] PTcscdata,
                            int[:] PTcscindices,
                            int[:] PTcscindptr,
                            Py_ssize_t k,
                            int[:] idx):
    """ return the gain in stability obtained by moving node
        k outside of the community defined by index list in idx. 
        
        PT is the PT matrix (in csr) and PTcsc is the same matrix
        in csc format
    """
    
    cdef double s = 0.0
    cdef Py_ssize_t i, j, l
    cdef Py_ssize_t num_idx = idx.shape[0]
    
    cdef cset[int] idx_set
    
    for i in range(num_idx):
        idx_set.insert(idx[i])
    

    for j in range(PTindptr[k],PTindptr[k+1]):
        if idx_set.count(PTindices[j]):
            s -= PTdata[j]
    for j in range(PTcscindptr[k],PTcscindptr[k+1]):
        if idx_set.count(PTcscindices[j]):
            s -= PTcscdata[j]
                    
    # add PT_kk
    j = PTindptr[k]
    l = PTindices[j]
    # check if PT[k,k] is nonzero
    while l < k and j + 1 < PTindptr[k+1]:
        j += 1
        l = PTindices[j]
    if l == k:
        s += PTdata[j]
        
    return s
    
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def compute_delta_S_moveout(double[:] PTdata,
                            int[:] PTindices,
                            int[:] PTindptr,
                            double[:] PTcscdata,
                            int[:] PTcscindices,
                            int[:] PTcscindptr,
                            Py_ssize_t k,
                            int[:] idx,
                            double[:] p1,
                            double[:] p2):
    """ return the gain in stability obtained by moving node
        k outside of the community defined by index list in idx. 
        
        PT is the PT matrix (in csr) and PTcsc is the same matrix
        in csc format
    """
    
    cdef double s = 0.0
    cdef Py_ssize_t i
    cdef Py_ssize_t num_idx = idx.shape[0]
    

    s = compute_delta_PT_moveout(PTdata,
                                 PTindices,
                                 PTindptr,
                                 PTcscdata,
                                 PTcscindices,
                                 PTcscindptr,
                                 k,
                                 idx)

        
    # now substract contribution from p1^T @ p2
    for i in range(num_idx):
        s += p1[k]*p2[idx[i]]
        s += p1[idx[i]]*p2[k]
    s -= p1[k]*p2[k]
    
    return s
            
