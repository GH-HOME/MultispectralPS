import numpy as np
import tqdm
import scipy
import scipy.sparse as sp
import math
from scipy.optimize import nnls
from utils.matrix_transform import scipy_sparse_to_spmatrix, null_sparse
import logging

def ilrs_phi(e, sigma=5*math.pi/180):
    diagonal = sigma * sigma / (e * e + sigma * sigma)
    return scipy.sparse.diags([diagonal], [0], format='csr')


def run_IRLS(maxIter, tol, A, x0, p):
    """
    solve SCPS with L1 minimization
    :param maxIter: Max iteration number for the IRLS
    :param tol: stop threshold
    :param A: Sparse matrix for Ax=0
    :return: x
    """

    assert A is not None
    assert x0 is not None
    assert len(x0) == A.shape[1]

    x_prev = x0 * 0.0 + 231231.0
    x = x0
    it = 0
    while scipy.linalg.norm(x - x_prev) > tol and it < maxIter:
        logging.info("IRLS: Iteration %d of %d, Error: %f Tolerance: %f" % (
        it + 1, maxIter, scipy.linalg.norm(x - x_prev), tol))
        x_prev = x
        e = A.dot(x)
        if np.isnan(np.min(x)):
            logging.warning("NaNs were found. Returning the previous iterate.")
            return x

        phi_m = ilrs_phi(e)
        x = null_sparse(phi_m.dot(A))
        xt = np.array(x).transpose()

        if np.isnan(np.min(xt)):
            logging.warning("NaNs were found. Returning the previous iterate.")
            return x

        x = xt
        it = it + 1
    return x

def null_sparse_CVXOPT(A, p):
    """
    calculate the null space from sparse matrix
    """
    assert A is not None
    from cvxopt import solvers, matrix
    numx = A.shape[1]
    q = matrix(np.zeros(numx), tc='d')
    g = np.zeros(numx)
    g[2::3] = 1
    g[3 * p:] = 1
    G = sp.spdiags(g, 0, len(g), len(g))
    G_cvt = scipy_sparse_to_spmatrix(G)
    h = matrix(np.ones(numx) * 1e-6, tc='d')
    A_cvt = scipy_sparse_to_spmatrix(A)
    P = A_cvt.T * A_cvt
    solvers.options['show_progress'] = True
    sol = solvers.qp(P, q, G_cvt, h)
    null_space = np.array(sol['x']).squeeze()
    return null_space

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

def solveAM(img_set, mask, L, max_iter= 1000, tol= 1.0e-5, use_normalize = True):
    """
        Semi-calibrated photometric stereo
        solution method based on alternating minimization
    """

    [H, W, f] = img_set.shape
    M = img_set[mask]
    if use_normalize:
        M /= np.linalg.norm(M, axis=0, keepdims=True)  # normailize the image

    N = np.zeros((3, M.shape[0]))
    p, f = M.shape
    E = np.ones(f)
    N_old = np.zeros((3, M.shape[0]))
    N_old[2] = 1
    for iter in tqdm.trange(max_iter):
        # Step 1 : Solve for N
        N = np.linalg.lstsq(np.diag(E) @ L, M.T, rcond=None)[0]
        N = N / np.linalg.norm(N, axis=0, keepdims=True)
        # Step 2 : Solve for E
        LN = L @ N
        for i in range(f):
            # E[i] = (LN[i, :] @ M[:, i]) / (LN[i, :] @ LN[i, :])
            E[i] = nnls(np.reshape(LN[i, :], (p, 1)), M[:, i])[0]
        # normalize E
        E /= np.linalg.norm(E)
        if np.linalg.norm(N - N_old) < tol:
            break
        else:
            N_old = N
    N = N / np.linalg.norm(N, axis=0, keepdims=True)

    Normal = np.zeros([H, W, 3])
    Normal[mask] = N.T
    reflectance = np.zeros([H, W, f])
    reflectance[mask] = img_set[mask] / (Normal[mask] @ L.T)
    return [Normal, reflectance]


def solveLinear(img_set, mask, L, IRLS=False, MaxIter=100, tol=1e-5, use_normalize=True):
    """
    estimate surface normal from multi-channel spectral image with unifrom chromaticity assumption
    Ref: semi-calibrated photometric stereo. TPAMI2019

    :param img_set: [H, W, f] multispectral image
    :param mask: [H, W] bool mask for ROI
    :param L: [f, 3] light directions
    :return:
            surface normal     : Normal: [H, W, 3]
            reflectance:       : [H, W, f]
    """

    from scipy import sparse as sp

    [H, W, f_org] = img_set.shape
    M = img_set[mask].T
    if use_normalize:
        M /= np.linalg.norm(M, axis=0, keepdims=True)  # normailize the image

    illum_pixels_id = np.where(np.min(M, axis=0) > 0.0)[0]
    f, p = M.shape
    Dl = sp.kron(-sp.identity(p), L)
    Drt = sp.lil_matrix((f, p * f))
    for i in range(len(illum_pixels_id)):
        Drt.setdiag(M[:, illum_pixels_id[i]], k=i * f)
    D = sp.hstack([Dl, Drt.T])
    null_space = null_sparse_CVXOPT(D, p)
    # null_space = null_sparse(D)
    if IRLS:
        null_space = run_IRLS(MaxIter, tol, D, null_space, p)

    if np.abs(np.sum(null_space[3 * p:])) < 1e-5:
        return [None, None]

    E = np.diag(1.0 / null_space[3 * p:])
    if np.mean(E) < 0.0:  # flip if light intensities are negative
        E *= -1.0
    solution = np.linalg.lstsq(E @ L, M, rcond=None)
    Normal = np.zeros([H, W, 3])
    reflectance = np.zeros([H, W, f_org])
    N = solution[0].T
    N = N / np.linalg.norm(N, axis=1, keepdims=True)
    Normal[mask] = N

    reflectance[mask] = img_set[mask] / (Normal[mask] @ L.T)

    return [Normal, reflectance]


def solveFact(img_set, mask, L, use_normalize = True):
    """
    estimate surface normal from multi-channel spectral image with unifrom chromaticity assumption
    Ref: semi-calibrated photometric stereo. TPAMI2019

    remove part of image observations based on the sorted thres_mask
    :param img_set: [H, W, f] multispectral image
    :param mask: [H, W] bool mask for ROI
    :param L: [f, 3] light directions
    :return:
            surface normal     : Normal: [H, W, 3]
            albedo chromaticity: albedo_c: [H, W, f]
    """

    [height, width, f_org] = img_set.shape
    M = img_set[mask]
    if use_normalize:
        M /= np.linalg.norm(M, axis=0, keepdims=True) # normailize the image
    p, f = M.shape

    M = M.T
    # Look at pixels that are illuminated under ALL the illuminations
    # illum_pixels_id = np.where(np.min(M, axis=0) > 0.0)[0]
    illum_pixels_id = np.arange(p)
    # Step 1 factorize (uncalibrated photometric stereo step)
    f = M.shape[0]
    import scipy
    u, s, vt = scipy.linalg.svd(M[:, illum_pixels_id], full_matrices=False)
    u = u[:, :3]
    s = s[:3]
    S_hat = u @ np.diag(np.sqrt(s))
    # Step 2 solve for ambiguity H
    A = np.zeros((2 * f, 9))
    for i in range(f):
        s = S_hat[i, :]
        A[2 * i, :] = np.hstack([np.zeros(3), -L[i, 2] * s, L[i, 1] * s])
        A[2 * i + 1, :] = np.hstack([L[i, 2] * s, np.zeros(3), -L[i, 0] * s])
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    H = np.reshape(vt[-1, :], (3, 3)).T
    S_hat = S_hat @ H
    E = np.identity(f)
    for i in range(f):
        E[i, i] = np.linalg.norm(S_hat[i, :])
    solution = np.linalg.lstsq(E @ L, M, rcond=None)
    N = solution[0].T

    N = N / np.linalg.norm(N, axis=1, keepdims=True)
    Normal = np.zeros([height, width, 3])
    Normal[mask] = N
    reflectance = np.zeros([height, width, f_org])
    reflectance[mask] = img_set[mask] / (Normal[mask] @ L.T)

    return [Normal, reflectance]

def run_MPS_SCPS(img_set, mask, L, method = 'Linear', max_iter= 500, tol= 1.0e-5, use_normalize=True):

    if method == 'Linear':
        normal, reflectance = solveLinear(img_set, mask, L, use_normalize =use_normalize)
    elif method == 'Linear_L1':
        normal, reflectance = solveLinear(img_set, mask, L, True, max_iter, tol, use_normalize)
    elif method == 'Fact':
        normal, reflectance = solveFact(img_set, mask, L, use_normalize)
    elif method == 'AM':
        normal, reflectance = solveAM(img_set, mask, L, max_iter, tol, use_normalize)
    else:
        raise Exception('Unknown method name')

    return [normal, reflectance]
