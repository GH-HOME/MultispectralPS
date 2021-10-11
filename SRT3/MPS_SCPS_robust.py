import numpy as np
from utils.PSUtils.pre_processing import get_valid_mask_by_sort, get_valid_mask
import tqdm
import scipy.sparse as sp
from utils.matrix_transform import scipy_sparse_to_spmatrix

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
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G_cvt, h)
    null_space = np.array(sol['x']).squeeze()
    return null_space

def solveAM_robust(img_set, mask, L, t_low = 0.5, shadow_ratio = 1e-4, max_iter=1000, tol = 1e-4):
    """
    estimate surface normal from multi-channel spectral image with unifrom chromaticity assumption
    Ref: semi-calibrated photometric stereo. TPAMI2019

    :param img_set: [H, W, f] multispectral image
    :param mask: [H, W] bool mask for ROI
    :param L: [f, 3] light directions
    :return:
            surface normal     : Normal: [H, W, 3]
            albedo chromaticity: albedo_c: [H, W, 3]
            albedo norm        : albedo_n: [H, W]
            residue map        : residue: [H, W]
    """
    [H, W, f] = img_set.shape
    m = img_set[mask].T
    thres_mask = get_valid_mask(m, t_low, shadow_ratio)
    m = img_set[mask].T
    f, p = m.shape
    n = np.zeros((3, p))
    alpha = np.ones(f)
    N_old = np.zeros([3, p])
    N_old[2] = 1
    try:
        for i in tqdm.trange(max_iter):
            # solve for n
            EL = np.diag(alpha) @ L
            for j in range(p):
                n[:, j] = np.linalg.lstsq(EL[thres_mask[:, j]], m[thres_mask[:, j], j], rcond=None)[0]
            albedo_n_flat = np.linalg.norm(n, axis=0)
            n = n / albedo_n_flat[np.newaxis, :]
            LN = L @ n
            for j in range(f):
                # alpha[j] = scipy.optimize.nnls(np.reshape(LN[j, :], (p, 1)), m[j, :])[0]
                alpha[j] = np.linalg.norm(m.T[:, j])/ np.linalg.norm(LN[j])
            alpha = alpha / np.linalg.norm(alpha)
            iter_norm = np.linalg.norm(n - N_old)
            # print(iter_norm)
            if iter_norm < tol:
                break
            else:
                N_old = np.copy(n)
    except Exception as e:
        return [None, None]

    Normal = np.zeros([H, W, 3])
    Normal[mask] = n.T
    reflectance = np.zeros([H, W, f])
    reflectance[mask] = img_set[mask] / (Normal[mask] @ L.T)
    return [Normal, reflectance]

def solveLinear_robust(img_set, mask, L, t_low = 0.6, shadow_ratio = 0.2):
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
    m = img_set[mask].T

    thres_mask = get_valid_mask(m, t_low, shadow_ratio)

    m = img_set[mask].T
    f, p = m.shape
    n = np.zeros((3, p))

    # generate L list
    L_list = []
    diagM_list = []
    light_id = np.arange(f)
    print('Build matrix in solveLinear of thres_SCPS')
    for i in tqdm.trange(p):
        L_list.append(-L[thres_mask[:, i]])
        num_row = np.sum(thres_mask[:, i])
        m_diag = sp.coo_matrix((m[thres_mask[:, i], i], (np.arange(num_row), light_id[thres_mask[:, i]])), shape=(num_row, f))
        diagM_list.append(m_diag)
    Dl = sp.block_diag(L_list)
    Drt = sp.vstack(diagM_list)
    D = sp.hstack([Dl, Drt])

    null_space = null_sparse_CVXOPT(D, p)
    if np.abs(np.sum(null_space[3 * p:])) < 1e-5:
        return [None, None]
    E = np.diag(1.0 / null_space[3 * p:])
    if np.mean(E) < 0.0:  # flip if light intensities are negative
        E *= -1.0
        null_space *= -1

    n = null_space[:3 * p].reshape(p, 3)
    albedo_n_flat = np.linalg.norm(n, axis=1)
    n = n / albedo_n_flat[:, np.newaxis]

    Normal = np.zeros([H, W, 3])
    reflectance = np.zeros([H, W, f_org])
    Normal[mask] = n
    reflectance[mask] = img_set[mask] / (Normal[mask] @ L.T)

    return [Normal, reflectance]

def solveFact_robust(img_set, mask, L, t_low = 0.6, shadow_ratio = 0.25, remove_ratio=0.25):
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

    [H, W, f_org] = img_set.shape
    m = img_set[mask].T

    thres_mask = get_valid_mask(m, t_low, shadow_ratio)
    if f_org - f_org*remove_ratio > 3:
        mask_t_idx = np.argsort(-np.sum(thres_mask, axis=1))[:int(f_org - f_org*remove_ratio)]
        mask_t = np.zeros(f_org).astype(np.bool)
        mask_t[mask_t_idx] = True
    else:
        mask_t = np.ones(f_org).astype(np.bool)

    from SRT3.MPS_SCPS import solveFact
    Normal, reflectance = solveFact(img_set[:,:,mask_t], mask, L[mask_t])

    reflectance = np.zeros([H, W, f_org])
    reflectance[mask] = img_set[mask] / (Normal[mask] @ L.T)

    return [Normal, reflectance]

def run_MPS_SCPS_rob(img_set, mask, L, method= 'linear',
                     t_low = 0.8, shadow_ratio = 1e-5,
                     max_iter=1000, tol = 1e-4,
                     remove_ratio=0.7):
    if method == 'rob_Linear':
        normal, reflectance = solveLinear_robust(img_set, mask, L, t_low, shadow_ratio)
    elif method == 'rob_Fact':
        normal, reflectance = solveFact_robust(img_set, mask, L, t_low, shadow_ratio, remove_ratio)
    elif method == 'rob_AM':
        normal, reflectance = solveAM_robust(img_set, mask, L, t_low, shadow_ratio, max_iter, tol)
    else:
        raise Exception('Unknown method name')

    return [normal, reflectance]