"""
This module provides drop-in replacement functions in case cython is not installed.
"""
import numpy as np
from scipy.sparse import (
    csr_matrix,
    )

def sparse_stoch_from_full_csr(nz_rowcols, Tf_data, Tf_indices, Tf_indptr, diag_val):
    """Pure python implementation of the sparce_stoch_mat
    """
    T_s_nnz = Tf_data.shape[0] - Tf_indptr.shape[0] + 1 + nz_rowcols.shape[0]

    T_s_data = np.zeros(T_s_nnz, dtype=np.float64)
    T_s_indices = -1*np.ones(T_s_nnz, dtype=np.int32)
    T_s_indptr = -1*np.ones(nz_rowcols.size+1, dtype=np.int32)

    #map indices from big to small T
    BtoS = {v:k for k,v in enumerate(nz_rowcols)}

    its = 0
    T_s_indptr[0] = 0
    for tsrow, tbrow in enumerate(nz_rowcols):
        nzr = 0
        for k in range(Tf_indptr[tbrow],Tf_indptr[tbrow+1]):
            T_s_data[its] = Tf_data[k]
            T_s_indices[its] = BtoS[Tf_indices[k]]
            its += 1
            nzr += 1
        T_s_indptr[tsrow+1] = T_s_indptr[tsrow] + nzr
    return (Tf_indptr.shape[0] - 1,
            T_s_data,
            T_s_indices,
            T_s_indptr,
            nz_rowcols,
            diag_val)


def inplace_csr_row_normalize(Tf_data, Tf_indptr, n_row:int, row_sum:float):
    """Row normalize a scipy sparse csr matrix inplace
    """
    for i in range(n_row):
        row_sum_tmp = Tf_data[Tf_indptr[i]:Tf_indptr[i+1]].sum()
        if row_sum_tmp != 0:
            Tf_data[Tf_indptr[i]:Tf_indptr[i+1]] /= (row_sum_tmp/row_sum)

class SPA:
    """sparse accumulator
    with multiple switch technique
        
    from: Implementing Sparse Matrices for Graph Algorithms. 
    in Graph Algorithms in the Language of Linear Algebra 
    94720, 287–313 (2011).
    """

    def __init__(self, size, current_row=0):

        self.size = size
        # values
        self.w = np.zeros(size, dtype=np.float64)

        # switch: if == current row, position is occupied
        self.b = -1*np.ones(size, dtype=np.int32)
        self.LS = list()
        self.current_row = current_row


    def scatter(self, value, pos ):

        if self.b[pos] < self.current_row:
            self.w[pos] = value
            self.b[pos] = self.current_row
            self.LS.append(pos)
        else:
            self.w[pos] += value

    def reset(self, current_row):

        self.current_row = current_row
        self.LS = []

def stoch_mat_add(
    size:int, # big matrix size
    Adata,
    Aindices,
    Aindptr,
    Anz_rowcols,
    Adiag_val:float,
    Bdata,
    Bindices,
    Bindptr,
    Bnz_rowcols,
    Bdiag_val:float):
    """ subtraction of square stochastic sparse matrices C = A - B
    """
    Asize = Anz_rowcols.shape[0]
    Bsize = Bnz_rowcols.shape[0]
    Anz_set = set(Anz_rowcols)
    Bnz_set = set(Bnz_rowcols)
    Cnz_set = Anz_set.union(Bnz_set)

    interset = Anz_set.intersection(Bnz_set)
    Anz_only = Anz_set - interset
    Bnz_only = Bnz_set - interset

    # size of C.T_small
    small_size = len(Cnz_set)

    Cnz_rowcols = sorted(list(Cnz_set))

    spa = SPA(small_size)

    Cdata = np.zeros(Aindptr.shape[0] + Bindptr.shape[0] + \
                          Adata.shape[0] + Bdata.shape[0],
                     dtype=np.float64)
    Cindices = -1*np.ones(Aindptr.shape[0] + Bindptr.shape[0] + \
                          Adata.shape[0] + Bdata.shape[0],
                          dtype=np.int32)
    Cindptr = -1*np.ones(small_size+1, dtype=np.int32)


    kc = 0 # data/indices index

    ia = 0 # row index of A.T_small
    ib = 0 # row index of B.T_small
    ic = 0 # row index of C.T_small

    Acol_to_Ccol = np.zeros(Asize, dtype=np.int32)

    ia_col = 0
    ic_col = 0
    for ia_col in range(Asize): # map col in A to col in C
        while Anz_rowcols[ia_col] != Cnz_rowcols[ic_col]:
            ic_col+=1
        Acol_to_Ccol[ia_col] = ic_col

    Bcol_to_Ccol = np.zeros(Bsize, dtype=np.int32)

    ib_col = 0
    ic_col = 0
    for ib_col in range(Bsize): # map col in A to col in C
        while Bnz_rowcols[ib_col] != Cnz_rowcols[ic_col]:
            ic_col+=1
        Bcol_to_Ccol[ib_col] = ic_col

    Cindptr[0] = 0
    for i in Cnz_rowcols: # iterate thourgh rows
        spa.reset(current_row=i)

        if i in Anz_set:
            for val, pos in zip(Adata[Aindptr[ia]:Aindptr[ia+1]],
                                Aindices[Aindptr[ia]:Aindptr[ia+1]]):
                spa.scatter(val, Acol_to_Ccol[pos])
            ia += 1

        if i in Bnz_set:
            for val, pos in zip(Bdata[Bindptr[ib]:Bindptr[ib+1]],
                                Bindices[Bindptr[ib]:Bindptr[ib+1]]):
                spa.scatter(val, Bcol_to_Ccol[pos])
            ib += 1

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


        # set indptr for C
        Cindptr[ic+1] = Cindptr[ic] + nzi

        ic += 1
    return (size, Cdata, Cindices, Cindptr, Cnz_rowcols, Adiag_val + Bdiag_val)

def stoch_mat_sub(
    size:int, # big matrix size
    Adata,
    Aindices,
    Aindptr,
    Anz_rowcols,
    Adiag_val:float,
    Bdata,
    Bindices,
    Bindptr,
    Bnz_rowcols,
    Bdiag_val:float):
    """ addition of square stochastic sparse matrices C = A + B
    """
    Asize = Anz_rowcols.shape[0]
    Bsize = Bnz_rowcols.shape[0]
    Anz_set = set(Anz_rowcols)
    Bnz_set = set(Bnz_rowcols)
    Cnz_set = Anz_set.union(Bnz_set)

    interset = Anz_set.intersection(Bnz_set)
    Anz_only = Anz_set - interset
    Bnz_only = Bnz_set - interset

    # size of C.T_small
    small_size = len(Cnz_set)

    Cnz_rowcols = sorted(list(Cnz_set))

    spa = SPA(small_size)

    Cdata = np.zeros(Aindptr.shape[0] + Bindptr.shape[0] + \
                          Adata.shape[0] + Bdata.shape[0],
                     dtype=np.float64)

    Cindices = -1*np.ones(Aindptr.shape[0] + Bindptr.shape[0] + \
                          Adata.shape[0] + Bdata.shape[0],
                          dtype=np.int32)
    Cindptr = -1*np.ones(small_size+1, dtype=np.int32)

    kc = 0 # data/indices index

    ia = 0 # row index of A.T_small
    ib = 0 # row index of B.T_small
    ic = 0 # row index of C.T_small

    Acol_to_Ccol = np.zeros(Asize, dtype=np.int32)

    ia_col = 0
    ic_col = 0
    for ia_col in range(Asize): # map col in A to col in C
        while Anz_rowcols[ia_col] != Cnz_rowcols[ic_col]:
            ic_col+=1
        Acol_to_Ccol[ia_col] = ic_col

    Bcol_to_Ccol = np.zeros(Bnz_rowcols.size, dtype=np.int32)

    ib_col = 0
    ic_col = 0
    for ib_col in range(Bsize): # map col in A to col in C
        while Bnz_rowcols[ib_col] != Cnz_rowcols[ic_col]:
            ic_col+=1
        Bcol_to_Ccol[ib_col] = ic_col

    Cindptr[0] = 0
    for i in Cnz_rowcols: # iterate thourgh rows
        spa.reset(current_row=i)

        if i in Anz_set:
            for val, pos in zip(Adata[Aindptr[ia]:Aindptr[ia+1]],
                                Aindices[Aindptr[ia]:Aindptr[ia+1]]):
                spa.scatter(val, Acol_to_Ccol[pos])
            ia += 1

        if i in Bnz_set:
            for val, pos in zip(Bdata[Bindptr[ib]:Bindptr[ib+1]],
                                Bindices[Bindptr[ib]:Bindptr[ib+1]]):
                spa.scatter(-1*val, Bcol_to_Ccol[pos])
            ib += 1

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


        # set indptr for C
        Cindptr[ic+1] = Cindptr[ic] + nzi

        ic += 1
    return (size, Cdata, Cindices, Cindptr, Cnz_rowcols, Adiag_val - Bdiag_val)

def rebuild_nnz_rowcol(
    T_data,
    T_indices,
    T_indptr,
    nonzero_indices,
    size: int,
    diag_val:float):
    """ returns a CSR matrix (data,indices,rownnz, shape) built from the CSR 
        matrix T_small but with
        added row-colums at zero_indicies (with 1 on the diagonal)
        
        Call:
        -----
        (data, indices, indptr, n_rows) = rebuild_nnz_rowcol(T_data,
                                                             T_indices,
                                                             T_indptr,
                                                             nonzero_indices,
                                                             size,
                                                             diag_val)
                                    
    """
    data = []
    indices = []
    rownnz = [] # num nnz element per row

    Ts_indices = [nonzero_indices[i] for i in T_indices]
 
    row_id_small_t = -1
    for row_id in range(size):
        row_id_small_t +=1
        if row_id not in nonzero_indices:
            # add a row with just 1 on the diagonal
            if diag_val != 0:
                data.append(diag_val)
                indices.append(row_id)
                rownnz.append(1)
            else:
                rownnz.append(0)
 
            row_id_small_t -= 1
        else:
            row_start = T_indptr[row_id_small_t]
            row_end = T_indptr[row_id_small_t+1]
 
            data.extend(T_data[row_start:row_end])
            indices.extend(Ts_indices[row_start:row_end])
            rownnz.append(row_end-row_start) # nnz of the row
 
    indptr = np.append(0, np.cumsum(rownnz))

    return (data, indices, indptr, size)
