import numpy as np
import os


def generate_no_shadow_random_light_normal(p, f):
    '''
    p: number of pixels to genererate
    p: number of lights
    '''
    np.random.seed(0)
    nbands = f  # for data generation
    npixels = p  # for data generation
    n_gt = np.random.rand(3, npixels) - .5  #  surface normals
    for i in range(npixels):  # make sure n_z is upward
        if n_gt[2, i] < 0.:
            n_gt[2, i] *= -1.
    n_gt = n_gt / np.linalg.norm(n_gt, axis=0, keepdims=True)  # normalize surface normals
    L = np.zeros((nbands, 3))  # light matrix
    for i in range(nbands):
        while True:
            l = np.random.rand(3) - .5
            l = l / np.linalg.norm(l)  # normalize (not sure if this is needed)
            if l[2] < 0.:  # make sure l_z is upward
                l[2] *= -1.
            if np.min(l @ n_gt) < 0.:  # avoid any shadows
                pass  # redo the random generation of light vector
            else:
                L[i, :] = l
                break
    m = L @ n_gt  # generate measurements
    m[m < 0.] = 0.
    return [m, L, n_gt]


def render_Lambertian(N_gt, L_dir, mask, albedo_map, method_type='CPS', render_shadow = True):

    h, w = mask.shape
    f = len(L_dir)
    shading = np.zeros([h, w, f])
    shading[mask] = N_gt[mask] @ L_dir.T

    if len(albedo_map.shape) == 2 and method_type == 'CPS':
        # render conventional PS
        img_set = shading * albedo_map[:,:, np.newaxis]
    if len(albedo_map.shape) == 3 and method_type == 'MPS':
        assert albedo_map.shape[2] == f
        img_set = shading * albedo_map

    if render_shadow:
        img_set[img_set < 0.0 ] = 0.0
    return img_set


def render_example(data_folder):
    N_path = os.path.join(data_folder, 'normal.npy')
    mask_path = os.path.join(data_folder, 'mask.npy')
    L_path = os.path.join(data_folder, 'L_dir.npy')

    from utils.PSUtils.pre_processing import data_loader_normal_mask_light
    [N_gt, mask, L_dir] = data_loader_normal_mask_light(N_path, mask_path, L_path)
    albedo_map = np.ones_like(mask).astype(np.float64)
    img_set = render_Lambertian(N_gt, L_dir, mask, albedo_map, method_type='CPS')
    np.save(os.path.join(data_folder, 'imgs.npy'), img_set)

if __name__ == '__main__':
    render_example('sample/bunny')
    render_example('sample/sphere')



