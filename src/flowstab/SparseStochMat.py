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
import importlib.util
import os
import time
from copy import copy
from functools import wraps

import numpy as np
from scipy.sparse import (
    coo_matrix,
    csr_matrix,
    eye,
    isspmatrix_csc,
    isspmatrix_csr,
    spmatrix,
)
from scipy.sparse._sparsetools import csr_scale_columns, csr_scale_rows

USE_CYTHON = True
if importlib.util.find_spec("cython") is not None:
    import _cython_sparse_stoch as _css
    from _cython_sparse_stoch import (
        cython_aggregate_csr_mat,
        cython_aggregate_csr_mat_2,
        cython_compute_delta_PT_moveout,
        cython_compute_delta_PT_moveto,
        cython_compute_delta_S_moveout,
        cython_compute_delta_S_moveto,
        cython_csr_add,
        cython_csr_csc_matmul,
        cython_csr_csrT_matmul,
        cython_csr_matmul,
        cython_get_submat_sum,
        cython_inplace_csr_row_normalize_array,
        cython_rebuild_nnz_rowcol,
    )

else:
    print("Could not load cython functions. Some functionality might be broken.")
    USE_CYTHON = False
    from . import _cython_subst as _css


USE_SPARSE_DOT_MKL = True
if importlib.util.find_spec("sparse_dot_mkl") is not None:
    from sparse_dot_mkl import dot_product_mkl, gram_matrix_mkl
    from sparse_dot_mkl._mkl_interface import MKL
    print("MKL_INT_NUMPY", MKL.MKL_INT_NUMPY)
else:
    USE_SPARSE_DOT_MKL = False
    print("Could not load sparse_dot_mkl. Will use scipy.sparse for matrix products.")

# timing decorator
def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        if kwargs.get("verbose"):
            log = f"PID  {os.getpid()} : function {f.__name__}" +\
                  f", elapsed time: {end-start:.4e} s"
            if "log_message" in kwargs and len(kwargs["log_message"])>0:
                log += ", " + kwargs["log_message"]

            print(log)
        return result
    return wrapper


class sparse_stoch_mat:
    """A sparse stochastic matrix representation.
        
    Represents a matrix as an array of indices corresponding to 
    row/colums that do not have only a 1 on the diagonal and a 
    smaller CSR scipy sparse matrix containing the these rows/colums.
        
    """

    def __init__(self, size, data, indices, indptr, nz_rowcols,
                 diag_val=1.0):

        self.size = size
        self.nz_rowcols = np.unique(np.array(nz_rowcols, dtype=np.int32)) #sorted unique
        self.T_small = csr_matrix((np.array(data, dtype=np.float64),
                                   np.array(indices, dtype=np.int32),
                                   np.array(indptr, dtype=np.int32)),
                                  shape=(len(nz_rowcols),
                                         len(nz_rowcols)))
        self.diag_val = diag_val
        self.shape = (size,size)


    @classmethod
    def from_small_csr_matrix(cls, size, T_small, nz_rowcols, diag_val=1.0):
        """Initialize sparse_stoch_mat from a small csr_matrix"""
        if not isspmatrix_csr(T_small):
            raise TypeError("T_small must be in CSR format.")

        return cls(size, T_small.data, T_small.indices, T_small.indptr,
                   nz_rowcols, diag_val=diag_val)

    @classmethod
    def from_full_csr_matrix(cls, Tcsr, nz_rowcols=None, diag_val=1.0):
        """Initialize sparse_stoch_mat from a full size row stochastic
        csr_matrix 
        """
        if not isspmatrix_csr(Tcsr):
            raise TypeError("T_small must be in CSR format.")

        if nz_rowcols is None:
            nz_rows, nz_cols = (Tcsr - diag_val * eye(Tcsr.shape[0], format="csr")).nonzero()
            nz_rowcols = np.union1d(nz_rows,nz_cols)

        res = _css.sparse_stoch_from_full_csr(
                np.array(nz_rowcols, dtype=np.int32),
                Tcsr.data,
                Tcsr.indices,
                Tcsr.indptr,
                diag_val)

        return cls(*res)

    @classmethod
    def create_diag(cls, size, diag_val=1.0):
        """Returns a diagonal matrix with an empty T_small.

        Parameters
        ----------
        size : int
            linear size of the matrix.
        diag_val : float, optional
            Value of the diagonal. The default is 1.0.

        """
        T_small = csr_matrix((0,0))

        return cls.from_small_csr_matrix(size, T_small, [], diag_val=diag_val)

    def inplace_row_normalize(self, row_sum=1.0):

        self.T_small.indptr = self.T_small.indptr.astype(np.int64, copy=False)
        self.T_small.indices = self.T_small.indices.astype(np.int64, copy=False)

        _css.inplace_csr_row_normalize(self.T_small.data, self.T_small.indptr,
                                            self.T_small.shape[0], row_sum)

        self.diag_val = row_sum

    def set_to_zeroes(self, tol=1e-8, relative=True, use_absolute_value=False):
        """In place replaces zeroes in the T_small sparse matrix that are,
        within the tolerence, close to zero with actual zeroes
        """
        if self.T_small.data.size > 0:
            if relative:
                tol = tol*np.abs([self.T_small.data.min(),self.T_small.data.max()]).max()

            if use_absolute_value:
                self.T_small.data[np.abs(self.T_small.data) <= tol] = 0
            else:
                self.T_small.data[self.T_small.data <= tol] = 0

            self.T_small.eliminate_zeros()


    def to_full_mat(self):
        """Returns a full size sparse matrix"""
        return rebuild_nnz_rowcol(self.T_small,
                                  self.nz_rowcols,
                                  self.size,
                                  self.diag_val)

    def tocsr(self):

        return self.to_full_mat()

    def toarray(self):

        return self.to_full_mat().toarray()

    def copy(self):

        return sparse_stoch_mat.from_small_csr_matrix(copy(self.size),
                                                      self.T_small.copy(),
                                                      self.nz_rowcols.copy(),
                                                      copy(self.diag_val))
    def to_dict(self):

        return {"size" : self.size,
                "data" : self.T_small.data,
                "indices" : self.T_small.indices,
                "indptr" : self.T_small.indptr,
                "nz_rowcols" : self.nz_rowcols,
                 "diag_val" : self.diag_val}

    def sub_diag(self, diag_val=1.0):
        """Returns a sparse_stoch_matrix results of
        self - diag(diag_val)
        """
        return sparse_stoch_mat.from_small_csr_matrix(self.size,
                                                      self.T_small - diag_val*eye(self.T_small.shape[0],
                                                                                  format="csr"),
                                                      self.nz_rowcols,
                                                      self.diag_val-diag_val)
    def add_diag(self, diag_val=1.0):
        """Returns a sparse_stoch_matrix results of
        self + diag(diag_val)
        """
        return sparse_stoch_mat.from_small_csr_matrix(self.size,
                                                      self.T_small + diag_val*eye(self.T_small.shape[0],
                                                                                  format="csr"),
                                                      self.nz_rowcols,
                                                      self.diag_val+diag_val)

    def transpose(self, copy=False):

        if copy:
            nz_rowcols = self.nz_rowcols.copy()
        else:
            nz_rowcols = self.nz_rowcols
        return sparse_stoch_mat.from_small_csr_matrix(self.size,
                                                      T_small=self.T_small.transpose(copy=copy).tocsr(),
                                                      nz_rowcols=nz_rowcols,
                                                      diag_val=self.diag_val)
    @property
    def T(self):
        return self.transpose()

    def check_nz_intersect_len(self, B):
        """Returns the length of the intersection of self.nz_rowcols and B.nz_rowcols"""
        if isinstance(B, sparse_stoch_mat):

            return len(set(self.nz_rowcols).intersection(set(B.nz_rowcols)))

        else:
            raise TypeError("B must be a sparse_stoch_mat")


    def __repr__(self):

        return f"{self.size}x{self.size} stochastic sparse matrix with T_small:\n" + \
              self.T_small.__repr__()

    def __add__(self, B):
        """Addition of two sparse stoch mat
        C = A + B 
            
        """
        if isinstance(B, sparse_stoch_mat):

            if not self.T_small.has_canonical_format:
                self.T_small.sort_indices()

            if not B.T_small.has_canonical_format:
                B.T_small.sort_indices()

            size, Cdata,Cindices,Cindptr, Cnz_rowcols, Cdiag_val = \
                _css.stoch_mat_add(self.size, # big matrix size
                                self.T_small.data,
                                self.T_small.indices,
                                self.T_small.indptr,
                                self.nz_rowcols,
                                self.diag_val,
                                B.T_small.data,
                                B.T_small.indices,
                                B.T_small.indptr,
                                B.nz_rowcols,
                                B.diag_val)

            return sparse_stoch_mat(size,Cdata,Cindices,Cindptr,
                                    Cnz_rowcols, Cdiag_val)


        elif isinstance(B, (spmatrix, np.ndarray)):
            return self.to_full_mat() + B
        else:
            raise NotImplementedError

    def __radd__(self, B):
        """Addition of two sparse stoch mat
        C = B + A = A + B 
            
        """
        return self.__add__(B)

    def __sub__(self, B):
        """Substraction of two sparse stoch mat.
        """
        if isinstance(B, sparse_stoch_mat):

            if not self.T_small.has_canonical_format:
                self.T_small.sort_indices()

            if not B.T_small.has_canonical_format:
                B.T_small.sort_indices()

            size, Cdata,Cindices,Cindptr, Cnz_rowcols, Cdiag_val = \
                _css.stoch_mat_sub(self.size, # big matrix size
                                self.T_small.data,
                                self.T_small.indices,
                                self.T_small.indptr,
                                self.nz_rowcols,
                                self.diag_val,
                                B.T_small.data,
                                B.T_small.indices,
                                B.T_small.indptr,
                                B.nz_rowcols,
                                B.diag_val)

            return sparse_stoch_mat(size,Cdata,Cindices,Cindptr,
                                    Cnz_rowcols, Cdiag_val)

        elif isinstance(B, (spmatrix, np.ndarray)):
            return self.to_full_mat() - B
        else:
            raise NotImplementedError

    def __rsub__(self, B):

        if isinstance(B, (spmatrix, np.ndarray)):
            return B - self.to_full_mat()
        else:
            raise NotImplementedError

    def __matmul__(self, B):
        """Matrix multiplication with self on the left and B on the right.
        
        C = A @ B
            
        """
        if isinstance(B, sparse_stoch_mat):
            # We split the problem in A @ B = (A-aI) @ (B-bI) + b*A + a*B - a*b*I
            # where a is the diag_val of A and b is the diag_val of B
            # Moreover, if intersection(A.nz_rowvals,B.nz_rowvals) = [],
            # (A-aI) @ (B-bI) = 0 and there is no multiplication to do

            if not self.T_small.has_canonical_format:
                self.T_small.sort_indices()

            if not B.T_small.has_canonical_format:
                B.T_small.sort_indices()

            Anz_set = set(self.nz_rowcols)
            Bnz_set = set(B.nz_rowcols)
            Cnz_set = Anz_set.union(Bnz_set)

            interset = Anz_set.intersection(Bnz_set)

            if self.diag_val == 1.0:
                aB = B # avoid making a copy of B
            else:
                aB = self.diag_val * B
            if B.diag_val == 1.0:
                bA = self
            else:
                bA = B.diag_val * self

            if len(interset) == 0:
                # no need to compute (A-aI) @ (B-bI)
                return (bA + aB).sub_diag(self.diag_val * B.diag_val)
            else:
                # we need to compute (A-aI) @ (B-bI)

                # size of C.T_small
                small_size = len(Cnz_set)

                Cnz_rowcols = sorted(list(Cnz_set))

                # A/B to C col mapping
                Acol_to_Ccol = np.zeros(self.nz_rowcols.size, dtype=np.int32)

                ia_col = 0
                ic_col = 0
                for ia_col in range(self.nz_rowcols.size): # map col in A to col in C
                    while self.nz_rowcols[ia_col] != Cnz_rowcols[ic_col]:
                        ic_col+=1
                    Acol_to_Ccol[ia_col] = ic_col

                Bcol_to_Ccol = np.zeros(B.nz_rowcols.size, dtype=np.int32)

                ib_col = 0
                ic_col = 0
                for ib_col in range(B.nz_rowcols.size): # map col in A to col in C
                    while B.nz_rowcols[ib_col] != Cnz_rowcols[ic_col]:
                        ic_col+=1
                    Bcol_to_Ccol[ib_col] = ic_col

                # we just have to compute Anz_rowcols * Bnz_rowcols values
                # C values will be on rows of Anz_rowcols and columns of Bnz_rowcols

                # build two small_size x small_size matrices for (A-aI) and
                # (B-bI) in the subspace of Csmall and take their product
                AmI_small = sparse_stoch_mat.from_small_csr_matrix(small_size,
                                                                self.T_small - \
                                                                    self.diag_val * eye(self.T_small.shape[0],
                                                                                        format="csr"),
                                                                Acol_to_Ccol,
                                                                diag_val=0.0)

                BmI_small = sparse_stoch_mat.from_small_csr_matrix(small_size,
                                                                B.T_small - \
                                                                    B.diag_val * eye(B.T_small.shape[0],
                                                                                        format="csr"),
                                                                Bcol_to_Ccol,
                                                                diag_val=0.0)

                CmI = sparse_stoch_mat.from_small_csr_matrix(self.size,
                                                             AmI_small.tocsr() @ BmI_small.tocsr(),
                                                             Cnz_rowcols,
                                                             diag_val=0.0)

                return CmI + (bA + aB).sub_diag(self.diag_val * B.diag_val)


        elif isinstance(B, (spmatrix, np.ndarray)):
            return self.to_full_mat() @ B

        else:
            raise NotImplementedError

    def __rmatmul__(self, s2):
        """Matrix multiplication with self on the right and s2 on the left.
        simply convert to full size sparse mat and 
        perform operation.
        """
        if isinstance(s2, (spmatrix, np.ndarray)):
            return s2 @ self.to_full_mat()
        else:
            raise NotImplementedError

    def __mul__(self, o):
        """Scalar multiplication with self on the right and o on the left.
        """
        if isinstance(o, (float,int)):

            return sparse_stoch_mat(self.size,
                                    self.T_small.data*o,
                                    self.T_small.indices.copy(),
                                    self.T_small.indptr.copy(),
                                    self.nz_rowcols.copy(),
                                    self.diag_val*o)
        else:
            raise NotImplementedError

    def __rmul__(self, o):
        """Scalar multiplication with self on the left and o on the right.

        """
        return self.__mul__(o)








def csr_add(A,B, use_cython=USE_CYTHON):
    """Addition of square csr matrix"""
    size = A.shape[0]

    if use_cython:

        Cdata,Cindices,Cindptr = cython_csr_add(A.data, A.indices, A.indptr,
                                                B.data, B.indices, B.indptr)

        return csr_matrix((Cdata,Cindices,Cindptr), shape=(size,size))

    else:



        spa = SPA(size)

        Cdata = np.zeros(A.nnz + B.nnz, dtype=A.data.dtype)
        Cindices = -1*np.ones(A.nnz + B.nnz, dtype=np.int32)
        Cindptr = -1*np.ones(size+1, dtype=np.int32)
        kc = 0 # data/indices index

        Cindptr[0] = 0
        for i in range(size): # iterate thourgh rows
            spa.reset(current_row=i)
            for val, pos in zip(A.data[A.indptr[i]:A.indptr[i+1]],
                                A.indices[A.indptr[i]:A.indptr[i+1]]):
                spa.scatter(val, pos)
            for val, pos in zip(B.data[B.indptr[i]:B.indptr[i+1]],
                                B.indices[B.indptr[i]:B.indptr[i+1]]):
                spa.scatter(val, pos)

            # set col indices and data for C
            nzi = 0 # num nonzero in row i of C
            for indnz in spa.LS:
                Cindices[kc] = indnz
                Cdata[kc] = spa.w[indnz]
                nzi += 1
                kc += 1

            # set indptr for C
            Cindptr[i+1] = Cindptr[i] + nzi

        return csr_matrix((Cdata,Cindices,Cindptr), shape=(size,size))




def csr_matmul(A,B, use_cython=USE_CYTHON):

    size = A.shape[0]

    if use_cython:
        Cdata,Cindices,Cindptr = cython_csr_matmul(A.data, A.indices, A.indptr,
                                                B.data, B.indices, B.indptr)

        return csr_matrix((Cdata,Cindices,Cindptr), shape=(size,size))
    else:

        spa = SPA(size)

        Cdata = []
        Cindices = []
        Cindptr = -1*np.ones(size+1, dtype=np.int32)
        kc = 0 # data/indices index

        Cindptr[0] = 0
        for i in range(size): # iterate thourgh rows
            spa.reset(current_row=i)
            for k in range(A.indptr[i],A.indptr[i+1]): # k is the col in A
                for j in range(B.indptr[A.indices[k]],B.indptr[A.indices[k]+1]): # for each COL of B, we add its contribution
                    spa.scatter(A.data[k] * B.data[j], B.indices[j])


            # set col indices and data for C
            nzi = 0 # num nonzero in row i of C
            for irnz in spa.LS:
                Cindices.append(irnz)
                Cdata.append(spa.w[irnz])
                nzi += 1
                kc += 1

            # set indptr for C
            Cindptr[i+1] = Cindptr[i] + nzi

        return csr_matrix((Cdata, np.array(Cindices, dtype=np.int32),Cindptr),
                          shape=(size,size))




def csr_csc_matmul(A,B):
    """Multiply a CSR matrix (A) with a CSC matrix (B) and return a CSR matrix (C)"""
    size = A.shape[0]

    if USE_CYTHON:
        Cdata,Cindices,Cindptr = cython_csr_csc_matmul(A.data, A.indices, A.indptr,
                                                B.data, B.indices, B.indptr)

        return csr_matrix((Cdata,Cindices,Cindptr), shape=(size,size))
    else:

        spa = SPA(size)

        Cdata = []
        Cindices = []
        Cindptr = -1*np.ones(size+1, dtype=np.int32)
        kc = 0 # data/indices index

        Cindptr[0] = 0
        for i in range(size): # iterate thourgh rows
            spa.reset(current_row=i)

            for j in range(size): # j is the column of C

                l = B.indptr[j] # iterator over B.data col elements
                m = B.indices[l] # iterator over B col elements
                for k in range(A.indptr[i],A.indptr[i+1]): # A.indices[k] is the col in A
                    while m < A.indices[k] and l + 1 < B.indptr[j+1]: # advance in B col until we are at the same position than in A row
                        l += 1
                        m = B.indices[l]

                    if m == A.indices[k]:
                        spa.scatter(A.data[k] * B.data[l], j)



            # set col indices and data for C
            nzi = 0 # num nonzero in row i of C
            for irnz in spa.LS:
                Cindices.append(irnz)
                Cdata.append(spa.w[irnz])
                nzi += 1
                kc += 1

            # set indptr for C
            Cindptr[i+1] = Cindptr[i] + nzi

        return csr_matrix((Cdata, np.array(Cindices, dtype=np.int32),Cindptr),
                          shape=(size,size))



def csr_csrT_matmul(A,B):
    """Multiply a CSR matrix (A) with a CSC matrix (B) which has the same
    sparsity pattern than A.T and return a CSR matrix (C)
    """
    size = A.shape[0]

    if USE_CYTHON:
        Cdata,Cindices,Cindptr = cython_csr_csrT_matmul(A.data, A.indices, A.indptr,
                                                B.data, B.indices, B.indptr)

        return csr_matrix((Cdata,Cindices,Cindptr), shape=(size,size))
    else:

        spa = SPA(size)

        Cdata = []
        Cindices = []
        Cindptr = -1*np.ones(size+1, dtype=np.int32)
        kc = 0 # data/indices index

        Cindptr[0] = 0
        for i in range(size): # iterate thourgh rows
            spa.reset(current_row=i)

            for j in range(i,size): # j is the column of C, Since C is symmetric,
                                    # we only compute half

                l = B.indptr[j] # iterator over B.data col elements
                m = B.indices[l] # iterator over B col elements
                for k in range(A.indptr[i],A.indptr[i+1]): # A.indices[k] is the col in A
                    while m < A.indices[k] and l + 1 < B.indptr[j+1]: # advance in B col until we are at the same position than in A row
                        l += 1
                        m = B.indices[l]

                    if m == A.indices[k]:
                        spa.scatter(A.data[k] * B.data[l], j)



            # set col indices and data for C
            nzi = 0 # num nonzero in row i of C
            for irnz in spa.LS:
                Cindices.append(irnz)
                Cdata.append(spa.w[irnz])
                nzi += 1
                kc += 1

            # set indptr for C
            Cindptr[i+1] = Cindptr[i] + nzi

        return csr_matrix((Cdata, np.array(Cindices, dtype=np.int32),Cindptr),
                          shape=(size,size))



def inplace_csr_row_normalize(X, row_sum=1.0):
    """Row normalize scipy sparse csr matrices inplace such that each row sum
    to `row_sum` (default is 1.0).
        
    If `row_sum=0` will remove the mean from the row, but only on nnz values.
    Skips rows that sum to 0.
        
    inspired from sklearn sparsefuncs_fast.pyx.
        
    Parameters
    ----------
        X : csr_matrix or sparse_stoch_mat
        Matrix to be row normalized
        
    row_sum : float or ndarray of same linear size than X (default is 1.0).
        Desired value of the sum of the rows
            
    Returns
    -------
        None : operates in place.

    """
    if isinstance(X, sparse_stoch_mat):
        X.inplace_row_normalize(row_sum=row_sum)


    elif isspmatrix_csr(X) or isspmatrix_csc(X):

        if isspmatrix_csc(X):
            print("Warning: row normalization on a CSC matrix will normalize columns.")

        # TODO: resolve this once both cython functions are called via _css
        if USE_CYTHON:
            X.indptr = X.indptr.astype(np.int64, copy=False)
            X.indices = X.indices.astype(np.int64, copy=False)
            if isinstance(row_sum, float):
                _css.inplace_csr_row_normalize(X.data, X.indptr,
                                                 X.shape[0], row_sum)

            elif isinstance(row_sum, np.ndarray) and row_sum.dtype == np.float64:
                assert row_sum.shape[0] == X.shape[0] and len(row_sum.shape) == 1

                cython_inplace_csr_row_normalize_array(X.data, X.indptr,
                                                       X.shape[0], row_sum)

            else:
                raise TypeError("row_sum must by float or ndarray of floats")


        else:

            if isinstance(row_sum, float):
                row_sum = np.ones(X.shape[0])*row_sum

            for i in range(X.shape[0]):
                row_sum_tmp = X.data[X.indptr[i]:X.indptr[i+1]].sum()
                if row_sum_tmp != 0:
                    if row_sum[i] == 0.0:
                        X.data[X.indptr[i]:X.indptr[i+1]] -= row_sum_tmp/(X.indptr[i+1]-X.indptr[i])
                    else:
                        X.data[X.indptr[i]:X.indptr[i+1]] /= (row_sum_tmp/row_sum[i])


    else:
        raise TypeError("X must be in ndarray, CSR or sparse_stoch_mat format.")

def rebuild_nnz_rowcol(T_small, nonzero_indices, size, diag_val=1.0):
    """Returns a CSR matrix built from the CSR matrix T_small with
    T_small values at row-colums corresponding to nonzero_indices 
    and 1 on the diagonal elsewhere.
        
    Returns
    -------
        T_full_size : scipy csr sparse matrix
        Full size transition matrix

    """
    if USE_CYTHON:
        (data, indices, indptr, n_rows) = \
                    cython_rebuild_nnz_rowcol(T_small.data,
                                              T_small.indices.astype(np.int64, copy=False),
                                              T_small.indptr.astype(np.int64, copy=False),
                                              nonzero_indices.astype(np.int64, copy=False),
                                              size,
                                              diag_val)
    else:

        data = []
        indices = []
        rownnz = [] # num nnz element per row

        # n_rows = T_small.shape[0] + zero_indices.shape[0]

        # zero_indices = set(zero_indices)

        # map col indices to new positions
        # new_col_inds = [i for i in range(size) if i not in zero_indices]

        Ts_indices = [nonzero_indices[i] for i in T_small.indices]

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
                row_start = T_small.indptr[row_id_small_t]
                row_end = T_small.indptr[row_id_small_t+1]

                data.extend(T_small.data[row_start:row_end])
                indices.extend(Ts_indices[row_start:row_end])
                rownnz.append(row_end-row_start) # nnz of the row

        indptr = np.append(0, np.cumsum(rownnz))

    return csr_matrix((data, indices, indptr),
                       shape=(size,size))


def inplace_csr_matmul_diag(A, diag_vec):
    """Inplace multiply a csr matrix A with a diag matrix D = diag(diag_vec):
    
    A = A @ D
        
    i.e. column i of A is scaled by diag_vec[i]
        
    """
    assert isinstance(diag_vec, np.ndarray)

    diag_vec = diag_vec.squeeze()

    assert diag_vec.shape[0] == diag_vec.size

    assert A.shape[1] == diag_vec.size, "Invalid array size"


    if isspmatrix_csr(A):

        csr_scale_columns(A.shape[0], A.shape[1], A.indptr,
                          A.indices, A.data, diag_vec)

    elif isspmatrix_csc(A):
        csr_scale_rows(A.shape[0], A.shape[1], A.indptr,
                          A.indices, A.data, diag_vec)

    else:
        raise ValueError("A must be a csr or csc matrix")



def inplace_diag_matmul_csr(A, diag_vec):
    """Inplace multiply a diag matrix D = diag(diag_vec) with a csr matrix A:
    
    A = D @ A
        
    i.e. row i of A is scaled by diag_vec[i]
        
    """
    assert isinstance(diag_vec, np.ndarray)

    diag_vec = diag_vec.squeeze()

    assert diag_vec.shape[0] == diag_vec.size

    assert A.shape[1] == diag_vec.size, "Invalid array size"


    if isspmatrix_csr(A):
        csr_scale_rows(A.shape[0], A.shape[1], A.indptr,
                       A.indices, A.data, diag_vec)

    elif isspmatrix_csc(A):
        csr_scale_columns(A.shape[0], A.shape[1], A.indptr,
                          A.indices, A.data, diag_vec)

    else:
        raise ValueError("A must be a csr or csc matrix")



class sparse_autocov_csr_mat:
    """Class for autocovariance matrix stored as a CSR matrix
        
            

            
    """

    def __init__(self, PT, S, symmetric=False):

        assert isspmatrix_csr(S)
        assert isspmatrix_csr(PT)

        self.PT = PT
        self.S = S
        self.size = S.shape[0]
        self.PT_symmetric = symmetric

        self.p1 = None
        self.p2 = None

        self.shape = (self.size,self.size)

        self.S.sort_indices()

        if not self.PT_symmetric:
            #store a version of PT as csc for fast access to columns
            self.Scsc = self.S.tocsc()
            self.PTcsc = self.PT.tocsc()
        else:
            self.Scsc = self.S
            self.PTcsc = self.PT

    def __repr__(self):

        if self.PT_symmetric:
            return f"{self.size}x{self.size} sparse autocovariance matrix with symmetric S:\n" + \
                  self.S.__repr__()
        else:
            return f"{self.size}x{self.size} sparse autocovariance matrix with S:\n" + \
                  self.S.__repr__()


    def copy(self):

        return self.__class__(self.PT.copy(),
                              self.S.copy(),
                              copy(self.PT_symmetric))

    def toarray(self):


        return self.S.toarray()


    def get_submat_sum(self, row_idx, col_idx):
        """Returns the sum of the elements of the autocov submatrix
        defined by the indices in idx, i.e. S[row_idx,col_idx].sum().
            
        """
        if USE_CYTHON:
            return cython_get_submat_sum(self.S.data, self.S.indices,
                                          self.PT.indptr,
                                          row_idx,
                                          col_idx)

        else:

            return self.S._major_index_fancy(row_idx)._minor_index_fancy(col_idx).sum()



    def get_element(self, i,j):
        """Returns element (i,j)"""
        # slightly more fast to directly compute location in csr data
        k, = np.where(self.S.indices[self.S.indptr[i]:self.S.indptr[i+1]] == j)
        if len(k) == 0:
            return 0
        else:
            return self.PT.data[self.S.indptr[i]+k[0]]

    def get_row_idx_sum(self, row, idx):
        """Return sum of elements at positions given by `idx` in row `row`.

        Parameters
        ----------
        row : int
            Index of row.
        idx : list
            List of indices along row `row`.

        Returns
        -------
        Autocov[row,idx].sum()

        """
        if USE_CYTHON:
            PTsum = cython_get_submat_sum(self.S.data, self.S.indices,
                                          self.S.indptr,
                                          np.array([row], dtype=np.int32),
                                          np.array(idx, dtype=np.int32))
        else:
            PTrow = self.PT._major_index_fancy(row)
            PTsum = PTrow.data[np.isin(PTrow.indices,idx)].sum()

        return  PTsum

    def get_col_idx_sum(self, col, idx):
        """Return sum of elements at positions given by `idx` in col `col`.
        
        Parameters
        ----------
        col : int
            Index of col.
        idx : list
            List of indices along col `col`.

        Returns
        -------
        Autocov[idx,col].sum()

        """
        if USE_CYTHON:
            PTsum = cython_get_submat_sum(self.Scsc.data, self.Scsc.indices,
                                          self.Scsc.indptr,
                                          np.array([col], dtype=np.int32),
                                          np.array(idx, dtype=np.int32))
        else:
            PTcol = self.Scsc._major_index_fancy(col)
            PTsum = PTcol.data[np.isin(PTcol.indices,idx)].sum()

        return  PTsum



    def aggregate(self, idx_list):
        """Returns a new sparse_autocol_mat where elements of
            the original mat have been aggregated according to 
            idx_list.
        
        Parameters
        ----------
        idx_list : list of lists of ints
            idx_list[i] and idx_list[j] contains the list of 
            row indices and col_indices to be aggregated to form S[i,j].

        Returns
        -------
        new aggregated sparse_autocov_mat

        """
        # convert idx_list to a single array of indices and an array of
        # pointers to start/stops for each cluster.
        idxs_array = np.array([i for idx in idx_list for i in idx], dtype=np.int32)
        idxptr = np.cumsum([0]+[len(idx) for idx in idx_list], dtype=np.int32)



        new_size = idxptr.size-1

        if USE_CYTHON:
            # choose the fastest version
            if new_size**2 < self.S.data.size:
                Sdata, Srows, Scols, new_size = cython_aggregate_csr_mat(self.S.data,
                                                                        self.S.indices,
                                                                        self.S.indptr,
                                                                        idxs_array,
                                                                        idxptr)
            else:
                Sdata, Srows, Scols, new_size = cython_aggregate_csr_mat_2(self.S.data,
                                                                        self.S.indices,
                                                                        self.S.indptr,
                                                                        idxs_array,
                                                                        idxptr)

            if new_size**2 < self.PT.data.size:
                PTdata, PTrows, PTcols, new_size = cython_aggregate_csr_mat(self.PT.data,
                                                                        self.PT.indices,
                                                                        self.PT.indptr,
                                                                        idxs_array,
                                                                        idxptr)
            else:
                PTdata, PTrows, PTcols, new_size = cython_aggregate_csr_mat_2(self.PT.data,
                                                                        self.PT.indices,
                                                                        self.PT.indptr,
                                                                        idxs_array,
                                                                        idxptr)
        else:
            # AtoB = {}
            # for kb in range(new_size):
            #     for ka in range(idxptr[kb],idxptr[kb+1]):
            #         AtoB[idxs_array[ka]] = kb

            # Bdata = []
            # Brows = []
            # Bcols = []
            # # loop over values of A
            # for row in range(self.S.indptr.shape[0]-1):
            #     for k in range(self.S.indptr[row],self.S.indptr[row+1]):
            #         col = self.S.indices[k]

            #         Bdata.append(self.S.data[k])
            #         Brows.append(AtoB[row])
            #         Bcols.append(AtoB[col])
            raise NotImplementedError

        newPT = coo_matrix((PTdata,(PTrows,PTcols)), shape=(new_size,new_size))
        newS = coo_matrix((Sdata,(Srows,Scols)), shape=(new_size,new_size))



        return self.__class__(newPT.tocsr(),
                              newS.tocsr(),
                              self.PT_symmetric)


    def is_all_zeros(self):
        """Returns True of all values are equal to zero.
        checks only nonzero values of self.PT 
        """
        self.S.eliminate_zeros()

        if self.S.getnnz() == 0:
            return True
        else:
            return False


    def _compute_delta_S_moveto(self, k, idx):
        """Return the gain in stability obtained by moving node
        k into the community defined by index list in idx.
            
        """
        if USE_CYTHON:


            return cython_compute_delta_PT_moveto(self.S.data,
                                                   self.S.indices,
                                                   self.S.indptr,
                                                   self.Scsc.data,
                                                   self.Scsc.indices,
                                                   self.Scsc.indptr,
                                                   k,
                                                   idx)




        else:
            return self.S.get_row_idx_sum(k,idx) \
                        + self.S.get_col_idx_sum(k,idx) \
                        + self.S.get_element(k,k)

    def _compute_delta_S_moveout(self, k, idx):
        """Return the gain in stability obtained by moving node
        k outside the community defined by index list in idx.
            
        """
        if USE_CYTHON:

            return cython_compute_delta_PT_moveout(self.S.data,
                                                   self.S.indices,
                                                   self.S.indptr,
                                                   self.Scsc.data,
                                                   self.Scsc.indices,
                                                   self.Scsc.indptr,
                                                   k,
                                                   np.array(idx, dtype=np.int32))



        else:
            return - self.S.get_row_idx_sum(k,idx) \
                        - self.S.get_col_idx_sum(k,idx) \
                        + self.S.get_element(k,k)




class sparse_autocov_mat:
    """Class for autocovariance matrix in the form:
        
        S = PT - P0
            
    where PT = diag(p) @ T, is a sparse csr matrix 
    and P0 = np.outer(p1,p2) is a dense matrix
        
    only PT, p1 and p2 are stored for memory efficiency
            
    """

    def __init__(self, PT, p1, p2, PT_symmetric=False):

        assert isspmatrix_csr(PT)
        if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
            assert not isinstance(p1, np.matrix)
            assert not isinstance(p2, np.matrix)
            assert len(p2.shape) == 1
            assert len(p1.shape) == 1
            assert PT.shape[0] == PT.shape[1] == p1.size == p2.size,\
                f"PT.shape[0]={PT.shape[0]}, PT.shape[1]={PT.shape[1]}, p1.size={p1.size}, p2.size={p2.size}"
            self.p_scalars = False
        elif isinstance(p1, (float,int)) and isinstance(p2, (float,int)):
            self.p_scalars = True
            self.p1p2 = p1*p2
        else:
            TypeError("p1 and p2 must be two 1D arrays or two scalar")


        self.PT = PT
        self.p1 = p1
        self.p2 = p2
        self.size = PT.shape[0]
        self.PT_symmetric = PT_symmetric

        self.shape = (self.size,self.size)

        self.PT.sort_indices()

        if not self.PT_symmetric:
            #store a version of PT as csc for fast access to columns
            self.PTcsc = self.PT.tocsc()
        else:
            self.PTcsc = self.PT

    def __repr__(self):

        if self.PT_symmetric:
            return f"{self.size}x{self.size} sparse autocovariance matrix with symmetric PT:\n" + \
                  self.PT.__repr__()
        else:
            return f"{self.size}x{self.size} sparse autocovariance matrix with PT:\n" + \
                  self.PT.__repr__()

    @classmethod
    def from_T(cls, T, p1=None, p2=None):
        """Generate autocovariance matrix from transition matrix T as
        S = diag(p1) @ T - p1^T @ p2.
        
        Parameters
        ----------
        T : NxN scipy csr matrix
            Transition matrix. T[i,j] is the prob to go from i to j between t1 and t2.
        p1 : numpy ndarray, optional
            Probability vector (size = N) at t1. Default is p1[i] = 1/N for all i.
        p2 : numpy ndarray, optional
            Probability vector (size = N) at t2. Default is p2 = p1 @ T.
            
        Returns
        -------
        sparse_autocov_mat

        """
        assert isspmatrix_csr(T)
        assert T.shape[0] == T.shape[1]

        if p1 is not None:
            assert isinstance(p1,np.ndarray)
            assert not isinstance(p1,np.matrix)
            assert len(p1.shape) == 1
            assert T.shape[0] == p1.size
        else:
            p1 = np.ones(T.shape[0])/T.shape[0]

        if p2 is not None:
            assert isinstance(p2,np.ndarray)
            assert not isinstance(p2,np.matrix)
            assert len(p2.shape) == 1
            assert T.shape[0] == p2.size
        else:
            p2 = p1 @ T

        PT = T.copy()
        inplace_diag_matmul_csr(PT, p1)

        return cls(PT=PT, p1=p1, p2=p2)

    @classmethod
    def from_T_forward(cls, T, p1=None, p2=None):
        """Generate the forward autocovariance matrix from transition matrix T as
        S = diag(p1) @ T @ diag(1/p2) @ T.T @ diag(p1) - p1.T @ p1.
        
        Parameters
        ----------
        T : NxN scipy csr matrix
            Transition matrix. T[i,j] is the prob to go from i to j between t1 and t2.
        p1 : numpy ndarray, optional
            Probability vector (size = N) at t1. Default is p1[i] = 1/N for all i.
        p2 : numpy ndarray, optional
            Probability vector (size = N) at t2. Default is p2 = p1 @ T.
            
        Returns
        -------
        sparse_autocov_mat

        """
        assert isspmatrix_csr(T)
        assert T.shape[0] == T.shape[1]

        if p1 is not None:
            assert isinstance(p1,np.ndarray)
            assert not isinstance(p1,np.matrix)
            assert len(p1.shape) == 1
            assert T.shape[0] == p1.size
            p1_scalar = False
        else:
            p1 = np.ones(T.shape[0])/T.shape[0]
            p1_scalar = True

        if p2 is not None:
            assert isinstance(p2,np.ndarray)
            assert not isinstance(p2,np.matrix)
            assert len(p2.shape) == 1
            assert T.shape[0] == p2.size
        else:
            p2 = p1 @ T

        p2m1 = p2.copy()
        p2m1[p2m1==0] = 1 # to avoid product of 0 * inf, which gives nan
        p2m1 = 1/p2m1

        PT = T.copy()
        # T @ diag(1/p2)
        inplace_csr_matmul_diag(PT,p2m1)
        PT = PT @ T.T
        inplace_diag_matmul_csr(PT, p1)
        inplace_csr_matmul_diag(PT, p1)

        if p1_scalar:
            return cls(PT=PT, p1=p1[0], p2=p1[0], PT_symmetric=True)

        else:
            return cls(PT=PT, p1=p1, p2=p1, PT_symmetric=True)

    def copy(self):

        return self.__class__(self.PT.copy(),
                              copy(self.p1),
                              copy(self.p2),
                              copy(self.PT_symmetric))

    def toarray(self):

        if self.p_scalars:
            return self.PT.toarray() - np.ones(self.shape)*self.p1p2

        else:
            return self.PT.toarray() - np.outer(self.p1,self.p2)


    def get_submat_sum(self, row_idx, col_idx):
        """Returns the sum of the elements of the autocov submatrix
        defined by the indices in idx, i.e. S[row_idx,col_idx].sum().
            
        """
        if self.p_scalars:
            p0_sum = len(row_idx)*len(col_idx)*self.p1p2
        else:

            # requires too much memory
            #p0_sum = np.outer(self.p1[row_idx],self.p2[col_idx]).sum()
            p0_sum = np.einsum("i,j->i",self.p1[row_idx],self.p2[col_idx]).sum()



        if USE_CYTHON:
            PTsum = cython_get_submat_sum(self.PT.data, self.PT.indices,
                                          self.PT.indptr,
                                          row_idx,
                                          col_idx)

        else:

            PTsum = self.PT._major_index_fancy(row_idx)._minor_index_fancy(col_idx).sum()

        return PTsum - p0_sum

    def get_element(self, i,j):
        """Returns element (i,j)"""
        if self.p_scalars:
            p0 = self.p1p2
        else:
            p0 = self.p1[i] * self.p2[j]

        # slightly more fast to directly compute location in csr data
        k, = np.where(self.PT.indices[self.PT.indptr[i]:self.PT.indptr[i+1]] == j)
        if len(k) == 0:
            return -1*p0
        else:
            return self.PT.data[self.PT.indptr[i]+k[0]] - p0

    def get_row_idx_sum(self, row, idx):
        """Return sum of elements at positions given by `idx` in row `row`.

        Parameters
        ----------
        row : int
            Index of row.
        idx : list
            List of indices along row `row`.

        Returns
        -------
        Autocov[row,idx].sum()

        """
        if self.p_scalars:
            p0 = len(idx)*self.p1p2
        else:
            p0 = (self.p1[row] * self.p2[idx]).sum()

        if USE_CYTHON:
            PTsum = cython_get_submat_sum(self.PT.data, self.PT.indices,
                                          self.PT.indptr,
                                          np.array([row], dtype=np.int32),
                                          np.array(idx, dtype=np.int32))
        else:
            PTrow = self.PT._major_index_fancy(row)
            PTsum = PTrow.data[np.isin(PTrow.indices,idx)].sum()

        return  PTsum - p0

    def get_col_idx_sum(self, col, idx):
        """Return sum of elements at positions given by `idx` in col `col`.
        
        Parameters
        ----------
        col : int
            Index of col.
        idx : list
            List of indices along col `col`.

        Returns
        -------
        Autocov[idx,col].sum()

        """
        if self.p_scalars:
            p0 = len(idx)*self.p1p2
        else:
            p0 = (self.p1[idx] * self.p2[col]).sum()


        if USE_CYTHON:
            PTsum = cython_get_submat_sum(self.PTcsc.data, self.PTcsc.indices,
                                          self.PTcsc.indptr,
                                          np.array([col], dtype=np.int32),
                                          np.array(idx, dtype=np.int32))
        else:
            PTcol = self.PTcsc._major_index_fancy(col)
            PTsum = PTcol.data[np.isin(PTcol.indices,idx)].sum()

        return  PTsum - p0



    def aggregate(self, idx_list):
        """Returns a new sparse_autocol_mat where elements of
            the original mat have been aggregated according to 
            idx_list.
        
        Parameters
        ----------
        idx_list : list of lists of ints
            idx_list[i] and idx_list[j] contains the list of 
            row indices and col_indices to be aggregated to form S[i,j].

        Returns
        -------
        new aggregated sparse_autocov_mat

        """
        # convert idx_list to a single array of indices and an array of
        # pointers to start/stops for each cluster.
        idxs_array = np.array([i for idx in idx_list for i in idx], dtype=np.int32)
        idxptr = np.cumsum([0]+[len(idx) for idx in idx_list], dtype=np.int32)



        new_size = idxptr.size-1

        if USE_CYTHON:
            # choose the fastest version
            if new_size**2 < self.PT.data.size:
                PTdata, PTrows, PTcols, new_size = cython_aggregate_csr_mat(self.PT.data,
                                                                        self.PT.indices,
                                                                        self.PT.indptr,
                                                                        idxs_array,
                                                                        idxptr)
            else:
                PTdata, PTrows, PTcols, new_size = cython_aggregate_csr_mat_2(self.PT.data,
                                                                        self.PT.indices,
                                                                        self.PT.indptr,
                                                                        idxs_array,
                                                                        idxptr)
        else:
            AtoB = {}
            for kb in range(new_size):
                for ka in range(idxptr[kb],idxptr[kb+1]):
                    AtoB[idxs_array[ka]] = kb

            Bdata = []
            Brows = []
            Bcols = []
            # loop over values of A
            for row in range(self.PT.indptr.shape[0]-1):
                for k in range(self.PT.indptr[row],self.PT.indptr[row+1]):
                    col = self.PT.indices[k]

                    Bdata.append(self.PT.data[k])
                    Brows.append(AtoB[row])
                    Bcols.append(AtoB[col])

        newPT = coo_matrix((PTdata,(PTrows,PTcols)), shape=(new_size,new_size))

        # the aggregated S will not have scalars p
        if self.p_scalars:
            oldp1 = np.ones(self.shape[0])/self.shape[0]
            oldp2 = oldp1
        else:
            oldp1 = self.p1
            oldp2 = self.p2

        newp1 = np.zeros(new_size, dtype=np.float64)
        newp2 = np.zeros(new_size, dtype=np.float64)
        for k, idx in enumerate(idx_list):
            newp1[k] = oldp1[idx].sum()
            newp2[k] = oldp2[idx].sum()

        #normalize p1 and p2 for rounding errors
        newp1 = newp1/newp1.sum()
        newp2 = newp2/newp2.sum()

        return self.__class__(newPT.tocsr(), newp1, newp2, self.PT_symmetric)


    def is_all_zeros(self):
        """Returns True of all values are equal to zero.
        checks only nonzero values of self.PT 
        """
        if self.p_scalars:
            for row in range(self.size):
                for k in range(self.PT.indptr[row],self.PT.indptr[row+1]):
                    col = self.PT.indices[k]
                    if not np.allclose(self.PT.data[k], self.p1p2):
                        return False
        else:
            for row in range(self.size):
                for k in range(self.PT.indptr[row],self.PT.indptr[row+1]):
                    col = self.PT.indices[k]
                    if not np.allclose(self.PT.data[k], self.p1[row]*self.p2[col]):
                        return False

        return True

    def _compute_delta_S_moveto(self, k, idx):
        """Return the gain in stability obtained by moving node
        k into the community defined by index list in idx.
            
        """
        if USE_CYTHON:

            if self.p_scalars:
                PTsum = cython_compute_delta_PT_moveto(self.PT.data,
                                                       self.PT.indices,
                                                       self.PT.indptr,
                                                       self.PTcsc.data,
                                                       self.PTcsc.indices,
                                                       self.PTcsc.indptr,
                                                       k,
                                                       idx)

                return PTsum - (2*len(idx)+1)*self.p1p2

            else:
                return cython_compute_delta_S_moveto(self.PT.data,
                                                       self.PT.indices,
                                                       self.PT.indptr,
                                                       self.PTcsc.data,
                                                       self.PTcsc.indices,
                                                       self.PTcsc.indptr,
                                                       k,
                                                       idx,
                                                       self.p1,
                                                       self.p2)

        else:
            return self._S.get_row_idx_sum(k,idx) \
                        + self._S.get_col_idx_sum(k,idx) \
                        + self._S.get_element(k,k)

    def _compute_delta_S_moveout(self, k, idx):
        """Return the gain in stability obtained by moving node
        k outside the community defined by index list in idx.
            
        """
        if USE_CYTHON:

            if self.p_scalars:
                PTsum = cython_compute_delta_PT_moveout(self.PT.data,
                                                       self.PT.indices,
                                                       self.PT.indptr,
                                                       self.PTcsc.data,
                                                       self.PTcsc.indices,
                                                       self.PTcsc.indptr,
                                                       k,
                                                       np.array(idx, dtype=np.int32))

                return PTsum + (2*len(idx)-1)*self.p1p2

            else:
                return cython_compute_delta_S_moveout(self.PT.data,
                                                       self.PT.indices,
                                                       self.PT.indptr,
                                                       self.PTcsc.data,
                                                       self.PTcsc.indices,
                                                       self.PTcsc.indptr,
                                                       k,
                                                       np.array(idx, dtype=np.int32),
                                                       self.p1,
                                                       self.p2)

        else:
            return - self._S.get_row_idx_sum(k,idx) \
                        - self._S.get_col_idx_sum(k,idx) \
                        + self._S.get_element(k,k)


@timing
def sparse_outer(p, use_mkl=True, triu=True, verbose=False, log_message=""):
    """Computes the sparse outer product p.T @ p

    Parameters
    ----------
    p : (1,N) csr sparse matrix
        Sparse array.
    use_mkl : bool
        Whether to use INTEL MKL for multithreading.
    triu : bool
        Whether to return an upper triangular matrix. Usually slower but 
        output requires less memory.

    Returns
    -------
    O : (N,N) csr sparse matrix.
    
    If USE_SPARSE_DOT_MKL and use_mkl returns a matrix with only the upper triangle filled.

    """
    assert isspmatrix_csr(p)
    assert p.shape[0] == 1

    p.eliminate_zeros()
    p.sort_indices()

    if USE_SPARSE_DOT_MKL and use_mkl:
        if triu:
            Odata = gram_matrix_mkl(p.data.reshape(1,p.data.size))[np.triu_indices(p.data.size)]
        else:
            Odata = gram_matrix_mkl(p.data.reshape(1,p.data.size)).reshape(p.data.size**2,1).squeeze()
    elif triu:
        Odata = np.outer(p.data,p.data)[np.triu_indices(p.data.size)]
    else:
        Odata = np.outer(p.data,p.data).reshape(p.data.size**2,1).squeeze()

    if triu:
        indices_list = [p.indices[p.indices >= r] for r in p.indices]
        row_len = iter([i.size for i in indices_list])
        num_per_row = [next(row_len) if r in p.indices else 0 for r in range(p.shape[1])]
    else:
        indices_list = [p.indices]*p.indices.size
        num_per_row = [p.indices.size if r in p.indices else 0 for r in range(p.shape[1])]

    Oindices = np.concatenate(indices_list)

    Oindptr = np.cumsum([0] + num_per_row)

    return csr_matrix((Odata, Oindices, Oindptr), shape=(p.shape[1], p.shape[1]))



@timing
def sparse_matmul(A,B, verbose=False, log_message=""):
    """Sparse matrix multiplication.
    Uses sparse_dot_mkl if available, otherwise scipy sparse
    """
    if USE_SPARSE_DOT_MKL:
        return dot_product_mkl(A,B)
    else:
        return A @ B

@timing
def sparse_gram_matrix(A, transpose, verbose=False, log_message=""):
    """If transpose is True, returns A @ A.T
    else, returns A.T @ A
        
    If USE_SPARSE_DOT_MKL returns a matrix with only the upper triangle filled.
        
    """
    if USE_SPARSE_DOT_MKL:
        return gram_matrix_mkl(A, transpose=transpose)
    elif transpose:
        return A @ A.T
    else:
        return A.T @ A
