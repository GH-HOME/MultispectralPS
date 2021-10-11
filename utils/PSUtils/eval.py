import logging
import numpy as np

def evalsurfaceNormal(N_gt, N_est, mask):
    """
    N_gt and N_est are both [H, W, 3] surface normal
    mask: [H, W] bool matrix to indicate forward mask
    """
    mask_nan = np.isnan(np.sum(N_est, axis=2))
    mask = np.logical_and(mask, ~mask_nan)
    gt = N_gt[mask, :]
    est = N_est[mask, :]
    ae, MAE = evaluate_angular_error(gt, est)
    error_map = np.zeros([N_gt.shape[0], N_gt.shape[1]], dtype=np.float64)
    error_map[mask] = ae
    return [error_map, MAE, np.median(ae)]


def eval_L2_error_map(GT, EST, mask):
    """
    N_gt and N_est are both [H, W, 3] surface normal
    mask: [H, W] bool matrix to indicate forward mask
    """
    mask_nan = np.isnan(np.sum(EST, axis=2))
    mask = np.logical_and(mask, ~mask_nan)
    gt = GT[mask, :]
    est = EST[mask, :]
    se, MSE = evaluate_L2_error(gt, est)
    error_map = np.zeros([GT.shape[0], GT.shape[1]], dtype=np.float64)
    error_map[mask] = se
    return [error_map, MSE, np.median(se)]


def evaluate_angular_error(gtnormal=None, normal=None, background=None):
    """
    gtnormal: [N, 3]
    normal: estimated normal [N, 3]
    background: bool index with length N
    return angular error with size N
    """
    if gtnormal is None or normal is None:
        logging.error('None EST or GT for normal angular error eval.')
        return [None, None]
    if background is not None:
        gtnormal[background] = 0
        normal[background] = 0
    gtnormal = gtnormal / np.maximum(np.linalg.norm(gtnormal, axis=1, keepdims=True), 1e-10)
    normal = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-10)

    ae = np.multiply(gtnormal, normal)
    aesum = np.sum(ae, axis=1)
    aesum[np.isnan(aesum)] = 0
    coord = np.where(aesum > 1.0)
    aesum[coord] = 1.0
    coord = np.where(aesum < -1.0)
    aesum[coord] = -1.0
    ae = np.arccos(aesum) * 180.0 / np.pi
    if background is not None:
        ae[background] = 0
    return ae, np.mean(ae)


def evaluate_L2_error(GT=None, EST=None):
    """
    gtnormal: [N, 1]
    normal: estimated normal [N, 1]
    return MAE with size N and mean MSE error
    """

    if GT is None or EST is None:
        logging.error('None EST or GT for L2 eval.')
        return [None, None]
    se = np.square(GT - EST)
    se = np.linalg.norm(GT - EST, axis=1)
    MSE = np.mean(se)
    return se, MSE