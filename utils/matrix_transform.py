import numba as nb
import numpy as np
import scipy
from cvxopt import spmatrix
from numpy.linalg import eigh
from scipy import sparse


def null_dense(A, eps=1e-15):
    """
    Calculate the null space of matrix A
    eps: singular value eps, if None, use scipy nullspace
    """
    assert A is not None
    if eps is None:
        null_space = scipy.linalg.nullspace(A)
    else:
        u, s, vh = np.linalg.svd(A)
        null_space = vh[-1].T
        # null_space = np.compress(s <= eps, vh, axis=0).T
    return null_space

def null_dense_unitNorm(A, eps=1e-15):
    """
    Calculate the null space of matrix A
    eps: singular value eps, if None, use scipy nullspace
    """
    assert A is not None

    AA = np.dot(A.T, A)
    evals_small, evecs_small = eigh(AA)
    null_space = evecs_small[:, 0]
    return null_space


def scipy_sparse_to_spmatrix(A):
    coo = A.tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP


def null_sparse(A):
    """
    calculate the null space from sparse matrix
    """
    assert A is not None
    from scipy.sparse.linalg import eigsh
    AA = np.dot(A.T, A)
    evals_small, evecs_small = eigsh(AA, 1, sigma=0, which='LM')
    null_space = evecs_small.squeeze()

    return null_space


def null_sparse_matlab(A):
    """
    calculate the null space from sparse matrix
    """
    import matlab
    import matlab.engine
    eng = matlab.engine.start_matlab()
    from scipy.io import savemat
    import os
    # AA = np.dot(A.T, A)
    savemat('sparse_A.mat', {'A': A})
    vt = eng.solve_sparse('sparse_A.mat')
    null_space = np.asarray(vt).squeeze()

    return null_space





def solve_Axb_CVXOPT(A, b):
    """
    calculate the null space from sparse matrix
    """
    assert A is not None
    from cvxopt import solvers, matrix
    numx = A.shape[1]
    g = np.zeros(numx)
    g[2::3] = -1 # nz should be positive
    G = sparse.spdiags(g, 0, len(g), len(g))
    G_cvt = scipy_sparse_to_spmatrix(G)
    h = matrix(np.ones(numx) * 1e-6, tc='d')
    A_cvt = scipy_sparse_to_spmatrix(A)
    b_cvt = matrix(b)
    P = A_cvt.T * A_cvt
    q = -2 * A_cvt.T * b_cvt
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G_cvt, h)
    x = np.array(sol['x']).squeeze()

    return x

def null_dense_CVXOPT(A):
    """
    calculate the null space from sparse matrix
    """
    assert A is not None
    from cvxopt import solvers, matrix
    numx = A.shape[1]
    q = matrix(np.zeros(numx), tc='d')
    g = np.zeros(numx)
    g[2] = 1
    G_cvt = matrix(np.diag(g))
    h = matrix(np.ones(numx) * 1e-6, tc='d')
    A_cvt = matrix(A)
    P = A_cvt.T * A_cvt
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G_cvt, h)
    null_space = np.array(sol['x']).squeeze()

    return null_space



def extractUpperMatrix(A, include_diag=False):
    """
    extract the upper part of the matrix, default excluding the diagonal
    :param A: N*N
    :return: vector of the upper elements
    """

    assert A.shape[0] == A.shape[1]
    size = A.shape[0]
    if include_diag:
        return A[np.triu_indices(3)]
    else:
        return A[np.triu_indices(size, k = 1)]