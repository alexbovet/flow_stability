# distutils: language = c++
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
cimport numpy as np
from libc.math cimport log2, fmax
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort
from libcpp.unordered_set cimport unordered_set as cset
from cython.parallel import prange
from cython.view cimport array as cvarray
from cython.operator cimport dereference as deref, preincrement as inc

import numpy as np

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def sum_Sto(double[:, ::1] S , int k, list ix_cf):
    
    cdef double delta_r = 0
    cdef Py_ssize_t i
    cdef int[:] ix_cf_arr = np.array(ix_cf, dtype=np.int32)
    cdef Py_ssize_t ix_cf_size = ix_cf_arr.size
    
    for i in range(ix_cf_size):
        delta_r += S[k,ix_cf_arr[i]]
        delta_r += S[ix_cf_arr[i],k]
    delta_r += S[k,k]
        
    return delta_r

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def sum_Sout(double[:, ::1] S , int k, list ix_ci):
    
    cdef double delta_r = 0
    cdef Py_ssize_t i
    cdef int[:] ix_ci_arr = np.array(ix_ci, dtype=np.int32)
    cdef Py_ssize_t ix_ci_size = ix_ci_arr.size
    
    for i in range(ix_ci_size):
        delta_r -= S[k,ix_ci_arr[i]]
        delta_r -= S[ix_ci_arr[i],k]
    delta_r += S[k,k]
        
    return delta_r

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def compute_S(double[:] p1, double[:] p2, double[:,:] T):
    
    cdef Py_ssize_t imax = T.shape[0]
    cdef Py_ssize_t jmax = T.shape[1]
    
    S = np.zeros((imax,jmax), dtype=np.float64)
    cdef double[:,:] S_view = S
    
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    
    for i in range(imax):
        for j in range(jmax):
            S_view[i,j] = p1[i]*T[i,j] - p1[i]*p2[j]
    
    return S

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def compute_S_0t0(double[:] p0, double[:] pt, double[:,:] T):
    
    cdef Py_ssize_t imax = T.shape[0]
    
    S = np.zeros((imax,imax), dtype=np.float64)
    cdef double[:,:] S_view = S
    
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    
    for i in range(imax):
        for j in range(imax):            
            for k in range(imax):
                S_view[i,j] = S_view[i,j] + p0[i]*T[i,k]*(1/pt[k])*T[j,k]*p0[j]
            S_view[i,j] = S_view[i,j] - p0[i]*p0[j]
            
    return S


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing            
def cython_nmi(list clusters1, list clusters2, int N, int n1, int n2):
    """ Computes normalized mutual information
    
        Call:
        -----
        
        cython_nmi(list clusters1, list clusters2, int N, int n1, int n2)
    
    """
    # loop over pairs of clusters
    cdef double[:] p1 = np.zeros(n1, dtype=np.float64) # probs to belong to clust1
    cdef double[:] p2 = np.zeros(n2, dtype=np.float64)
    cdef double[:] p12 = np.zeros(n1*n2, dtype=np.float64) # probs to belong to clust1 & clust2
    cdef double H1 = 0.0
    cdef double H2 = 0.0
    cdef double H12 = 0.0
    cdef Py_ssize_t k = 0
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    cdef set clust1
    cdef set clust2
    
    
    for i,clust1 in enumerate(clusters1):
        p1[i] = len(clust1)/N
        for j, clust2 in enumerate(clusters2):
            p12[k] = len(clust1.intersection(clust2))/N
            k += 1
    for j, clust2 in enumerate(clusters2):
        p2[j] = len(clust2)/N
        
    # Shannon entropies
    for i in range(n1):
        if p1[i] != 0:
            H1 -= p1[i]*log2(p1[i])
    for j in range(n2):
        if p2[j] != 0:
            H2 -= p2[j]*log2(p2[j])
    for j in range(n1*n2):
        if p12[j] != 0:
            H12 -= p12[j]*log2(p12[j])


    return (H1 + H2 - H12)/fmax(H1,H2)



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing   
@cython.cdivision(True)
@cython.nonecheck(True)
def cython_nvi(list clusters1, list clusters2, int N):
    """ Computes normalized variation of information
    
        Call:
        -----
        
        cython_nvi(list clusters1, list clusters2, int N)
    
    """
    # loop over pairs of clusters
    cdef double ni 
    cdef double nj
    cdef double nij 
    cdef double VI = 0.0
    cdef set clust1
    cdef set clust2
    cdef Py_ssize_t n1 = len(clusters1)
    cdef Py_ssize_t n2 = len(clusters2)
    cdef Py_ssize_t i, j, n_inter, l_inter
    
    # version that requires much less memory
    for i in range(n1):
        clust1 = clusters1[i]
        ni = <double>len(clust1)
        n_inter = 0
        for j in range(n2):
            clust2 = clusters2[j]
            l_inter = len(clust1.intersection(clust2))
            nij = <double>l_inter
            n_inter += l_inter
            if nij > 0:
                nj = <double>len(clust2)
                VI -= nij*log2((nij*nij)/(ni * nj))
            
            if n_inter >= ni:
                # we have found all the possible intersections
                break
            
    return VI/(<double>N*log2(<double>N))

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing   
@cython.cdivision(True)
@cython.nonecheck(True)
def cython_nvi_parallel(list clusters1, list clusters2, int N, int num_threads):
    """ Computes normalized variation of information with a parallelized loop
    
        Call:
        -----
        
        cython_nvi(list clusters1, list clusters2, int N)
    
    """
    # loop over pairs of clusters
    cdef double ni # probs to belong to clust1
    cdef double nj
    cdef double nij # probs to belong to clust1 & clust2
    cdef double VI = 0.0
    cdef set clust1
    cdef set clust2
    cdef Py_ssize_t n1 = len(clusters1)
    cdef Py_ssize_t n2 = len(clusters2)
    cdef Py_ssize_t i, j
    cdef double[:] VI_res = cvarray(shape=(n1,),itemsize=sizeof(double),format='d')
    
    # version that requires much less memory
    for i in range(n1):
        clust1 = clusters1[i]
        ni = <double>len(clust1)
        
        for j in prange(n2, num_threads=num_threads, nogil=True, 
                        schedule='static'):
            with gil:
                clust2 = clusters2[j]
                nij = <double>len(clust1.intersection(clust2))
            if nij > 0:
                with gil:
                    nj = <double>len(clust2)
                VI_res[i] -= nij*log2((nij*nij)/(ni * nj))
                
    for i in prange(n1, num_threads=num_threads, nogil=True, 
                        schedule='static'):
        VI += VI_res[i]
        
    return VI/(<double>N*log2(<double>N))



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing   
@cython.cdivision(True)
@cython.nonecheck(False)
def cython_nvi_vectors(list clusters1, list clusters2, int N):
    """ Computes normalized variation of information
    
        Call:
        -----
        
        cython_nvi(list clusters1, list clusters2, int N)
    
    """
    cdef double ni 
    cdef double nj
    cdef double nij 
    cdef double VI = 0.0
    cdef int[:] clust1
    cdef int[:] clust2
    cdef vector[int] v1
    cdef vector[int] v2
    cdef Py_ssize_t n1 = len(clusters1)
    cdef Py_ssize_t n2 = len(clusters2)
    cdef Py_ssize_t i, j, k, cil, cjl
    cdef int node
    
    
    
    for i in range(n1):
        clust1 = clusters1[i]
        ni = 0.0
        cil = clust1.shape[0]
        for k in range(cil):
            node = clust1[k]
            v1.push_back(node)
            ni += 1.0
        sort(v1.begin(), v1.end())
        
        for j in range(n2):
            clust2 = clusters2[j]
            
            nj = 0.0
            cjl = clust2.shape[0]
            for k in range(cjl):
                node = clust2[k]
                v2.push_back(node)
                nj += 1.0
            sort(v2.begin(), v2.end())
            
            nij = <double>sorted_vectors_intersection_length(v1,v2)
            
            if nij > 0.0:
                VI -= nij*log2((nij**2)/(ni * nj))
                
            v2.clear()
            
        v1.clear()
            
    return VI/(<double>N*log2(<double>N))

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing   
@cython.cdivision(True)
@cython.nonecheck(False)
def cython_nvi_mat(list clusters1, list clusters2, int N):
    """ Computes normalized variation of information
    
        Call:
        -----
        
        cython_nvi(list clusters1, list clusters2, int N)
    
        # this is slower that cython_nvi
    """

    #creates csc sparse indicator matrices for clusters1 and 2
    cdef Py_ssize_t n1 = len(clusters1)
    cdef Py_ssize_t n2 = len(clusters2)
    
    # each column correspond to a cluster\
    cdef int[:] H1_indices = np.zeros(N, dtype=np.int32)
    cdef int[:] H1_indptr = np.zeros(n1+1, dtype=np.int32)
    cdef int[:] H2_indices = np.zeros(N, dtype=np.int32)
    cdef int[:] H2_indptr = np.zeros(n2+1, dtype=np.int32)
    
    cdef Py_ssize_t i, j, k
    cdef int nc
    cdef set c
    cdef list l
    
    for i in range(n1):
        c = clusters1[i]
        l = list(c)
        nc = len(l)
        H1_indptr[i+1] = H1_indptr[i] + nc
        k = 0
        for j in range(H1_indptr[i],H1_indptr[i+1]):
            H1_indices[j] = l[k]
            k += 1
        
    for i in range(n2):
        c = clusters2[i]
        l = list(c)
        nc = len(c)
        H2_indptr[i+1] = H2_indptr[i] + nc
        k = 0
        for j in range(H2_indptr[i],H2_indptr[i+1]):
            H2_indices[j] = l[k]
            k += 1
            
        
    cdef double VI = 0.0
    cdef double ni = 0.0
    cdef double nj = 0.0
    cdef double nij = 0.0

    for i in range(n1):
        # num nonzero elements in columns i of H1 = len(clust1)
        ni = <double>(H1_indptr[i+1] - H1_indptr[i])
        for j in range(n2):
            # scalar product of H1[:,i].T and H2[:,j]
            nij = <double>indices_intersection_length(H1_indices[H1_indptr[i]:H1_indptr[i+1]],
                                              H2_indices[H2_indptr[j]:H2_indptr[j+1]])
            if nij > 0.0:
                # num nonzero elements in columns j of H2 = len(clust2)
                nj = <double>(H2_indptr[j+1] - H2_indptr[j])
                VI -= nij*log2((nij**2)/(ni * nj))
    
    return VI/(<double>N*log2(<double>N))


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing   
@cython.cdivision(True)
@cython.nonecheck(False)
def cython_nvi_mat_test(list clusters1, list clusters2, int N):
    """ Computes normalized variation of information
    
        Call:
        -----
        
        cython_nvi(list clusters1, list clusters2, int N)
    
        # this is slower that cython_nvi
    """

    #creates csc sparse indicator matrices for clusters1 and 2
    cdef Py_ssize_t n1 = len(clusters1)
    cdef Py_ssize_t n2 = len(clusters2)
    
    # each column correspond to a cluster\
    cdef int[:] H1_indices = np.zeros(N, dtype=np.int32)
    cdef int[:] H1_indptr = np.zeros(n1+1, dtype=np.int32)
    cdef int[:] H2_indices = np.zeros(N, dtype=np.int32)
    cdef int[:] H2_indptr = np.zeros(n2+1, dtype=np.int32)
    
    cdef Py_ssize_t i, j, k
    cdef int nc
    cdef set c
    cdef list l
    
    for i in range(n1):
        c = clusters1[i]
        l = list(c)
        nc = len(l)
        H1_indptr[i+1] = H1_indptr[i] + nc
        k = 0
        for j in range(H1_indptr[i],H1_indptr[i+1]):
            H1_indices[j] = l[k]
            k += 1
        
    for i in range(n2):
        c = clusters2[i]
        l = list(c)
        nc = len(c)
        H2_indptr[i+1] = H2_indptr[i] + nc
        k = 0
        for j in range(H2_indptr[i],H2_indptr[i+1]):
            H2_indices[j] = l[k]
            k += 1
            
        

    
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing   
cdef int indices_intersection_length(int[:] indices1, int[:] indices2):
    
    # copy arrays for sorting
    cdef int[::1] sind1 = indices1.copy()
    cdef int[::1] sind2 = indices2.copy()
        
    # sort arrays in place, must be c continuous (enforced with [::1])
    sort(&sind1[0], (&sind1[0]) + sind1.shape[0])
    sort(&sind2[0], (&sind2[0]) + sind2.shape[0])
    
    cdef int length = 0
    
    cdef Py_ssize_t n1 = sind1.shape[0]
    cdef Py_ssize_t n2 = sind2.shape[0]
    cdef int ind1, ind2
    cdef Py_ssize_t i = 0, j = 0
    
    while i < n1 and j < n2:
        ind1 = sind1[i]
        ind2 = sind2[j]
        if ind1 == ind2:
            length += 1
            i += 1
            j += 1
        elif ind1 < ind2:
            i += 1
        else:
            j += 1
    
    return length

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing   
cdef int sorted_vectors_intersection_length(const vector[int] indices1, const vector[int] indices2):
    
   
    
    cdef int length = 0
    
    cdef Py_ssize_t n1 = indices1.size()
    cdef Py_ssize_t n2 = indices2.size()
    cdef int ind1, ind2
    cdef Py_ssize_t i = 0, j = 0
    
    while i < n1 and j < n2:
        ind1 = indices1[i]
        ind2 = indices2[j]
        if ind1 == ind2:
            length += 1
            i += 1
            j += 1
        elif ind1 < ind2:
            i += 1
        else:
            j += 1
    
    return length


cdef int cset_intersection_length(cset[int] set1, cset[int] set2):
    
    cdef int length = 0
    cdef Py_ssize_t l1 = set1.size()
    cdef Py_ssize_t l2 = set2.size()
    cdef int el
    cdef cset[int].iterator i
        
    if l1 <= l2:
        # iterate over elements of set1
        i = set1.begin()
        while i != set1.end():
            el = deref(i)
            if set2.count(el):
                length += 1
            inc(i)
    else:
        # iterate over elements of set2
        i = set2.begin()
        while i != set2.end():
            el = deref(i)
            if set1.count(el):
                length += 1
            inc(i)
                
    return length
                
    
def test():

    cdef cset[int] a    
    cdef cset[int] b
    
    a.insert(0)
    a.insert(2)
    b.insert(1)
    b.insert(2)
    
    print(cset_intersection_length(a,b))
    
    
    s = set([0,1,2,3])
    
    cdef cset[int] c = s
    
    cdef cset[int].iterator i
    cdef int v
    
    i = c.begin()
    while i != c.end():
        v = deref(i)
        print('cset', v)
        inc(i)
        
    print(s)
    
