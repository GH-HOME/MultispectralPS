import logging
import numpy as np
import os
from sklearn import preprocessing
import tqdm

def fillHole(mask_valid, mask_full, input_with_hole, require_normalize=False):
    """
    Fill the hole in the input data
    Parameters
    ----------
    mask_valid: The mask where input with value
    mask: The full mask we want to recover [M, N]
    input_with_hole: The input with the hole [M, N, C]

    Returns input with hole filled
    -------

    """

    assert mask_full is not None
    assert input_with_hole is not None
    from scipy.interpolate import griddata
    height, width, channel = input_with_hole.shape
    coll, roww = np.meshgrid(np.arange(width), np.arange(height))
    valid_positions = np.dstack([roww[mask_valid], coll[mask_valid]]).squeeze()

    input_full = np.zeros_like(input_with_hole)
    for i in range(channel):
        value_c = griddata(valid_positions, input_with_hole[mask_valid, i], (roww, coll), method='nearest')
        input_full[:,:,i] = value_c
        input_full[~mask_full, i] = 0

    if require_normalize:
        mask_valid = np.linalg.norm(input_full, axis=2) > 1e-6
        input_full = input_full / np.maximum(np.linalg.norm(input_full, axis=2, keepdims=True), 1e-10)
        input_full[~mask_valid] = 0

    return input_full


def get_valid_mask_by_sort(M, t_h=0.8, shadow_low = 0.2):
    """
    generate a valid mask for low-frequency observation
    :param M:[f, p] image observations
    :param t_low: 20% to keep the observations
    :param t_shadow: 2% remove the shadow based on this ratio
    :return: thres_mask [f, p]
    """
    assert M is not None
    f, p = M.shape

    M_temp = np.copy(M)
    # step 1: sort the observations based on the observations
    ind = np.argsort(M_temp, axis=0)

    t_select_id_h = int(f * t_h)
    t_select_id_l = int(f * shadow_low)

    t_select_id = np.maximum(t_select_id_h - t_select_id_l, 4)
    ind_select = ind[t_select_id_l:t_select_id_l+t_select_id, :]
    thres_mask = np.zeros([f, p]).astype(np.bool)
    for i in range(p):
        thres_mask[ind_select[:, i], i] = True
    return thres_mask


def get_valid_mask(M, t_low, shadow_ratio = 1e-4):
    """
    generate a valid mask for low-frequency observation
    :param M:[f, p] image observations
    :param t_low: 20% to keep the observations
    :param t_shadow: 2% remove the shadow based on this ratio
    :return: thres_mask [f, p]
    """
    assert M is not None
    f, p = M.shape

    # shadow_thres = np.mean(M) * shadow_ratio
    M_temp = np.copy(M)
    M = M / np.mean(M, axis=1, keepdims=True)
    t_select_id = int(f * t_low)

    t_select_id = np.maximum(t_select_id, 4) # at lease select 4 lights
    thres_mask = np.zeros([f, p]).astype(np.bool) # the low-frequency observation mask for all the pixels
    shadow_mask = M < np.median(M) * shadow_ratio
    index_f = np.arange(f)
    for i in tqdm.trange(p):

        # step 1: sort the observations based on the observations
        non_shadow_id = index_f[~shadow_mask[:, i]]
        ind = np.argsort(M_temp[non_shadow_id, i])
        select_id = np.minimum(t_select_id, f - np.sum(shadow_mask[:, i]))
        ind_select = non_shadow_id[ind[:select_id]]
        thres_mask[ind_select, i] = True
    return thres_mask


def data_loader_img_mask_light(img_path, mask_path, L_dir_path):

    """load data for image observation under the light directions"""

    assert os.path.exists(img_path)
    assert os.path.exists(mask_path)
    assert os.path.exists(L_dir_path)

    L_dir = np.load(L_dir_path)
    mask = np.load(mask_path)
    img_set = np.load(img_path)

    h, w = mask.shape
    f = len(L_dir)
    assert img_set.shape[0] == h
    assert img_set.shape[1] == w
    assert img_set.shape[2] == f

    # normalize the light directions
    L_dir = L_dir / np.linalg.norm(L_dir, axis = 1, keepdims=True)

    return [img_set, mask, L_dir]


def data_loader_normal_mask_light(normal_path, mask_path, L_dir_path):

    """load data for image observation under the light directions"""

    assert os.path.exists(normal_path)
    assert os.path.exists(mask_path)
    assert os.path.exists(L_dir_path)

    L_dir = np.load(L_dir_path)
    mask = np.load(mask_path)
    N_gt = np.load(normal_path)


    # normalize the light directions
    L_dir = L_dir / np.linalg.norm(L_dir, axis = 1, keepdims=True)

    return [N_gt, mask, L_dir]