import numpy as np
from matplotlib import pyplot as plt
import os

from SRT3 import MPS_SCPS, MPS_SCPS_robust
from utils.PSUtils.render import render_Lambertian
from utils.PSUtils.eval import evalsurfaceNormal

Data_folder = r'F:\PS_Library\utils\PSUtils\sample\sphere'
N_path = os.path.join(Data_folder, 'normal.npy')
L_path = os.path.join(Data_folder, 'L_dir.npy')
mask_path = os.path.join(Data_folder, 'mask.npy')

Data_folder = r'F:\synthetic_exp\MPS\isotropic_SBRDF\bunny\shading_wo_shadow\Imgs'
N_path = os.path.join(Data_folder, 'normal_gt.npy')
L_path = os.path.join(Data_folder, '../light_directions.npy')
mask_path = os.path.join(Data_folder, 'mask.npy')


L_dir = np.load(L_path)
mask = np.load(mask_path)
N_gt = np.load(N_path)

h, w = mask.shape
f = len(L_dir)

# set the
albedo_map = np.ones([h, w, f])
chrom = np.random.random(f)
albedo_map = albedo_map * chrom[np.newaxis, np.newaxis, :]

img_set = render_Lambertian(N_gt, L_dir, mask, albedo_map, method_type='MPS', render_shadow=True)

method_set = ['Linear', 'AM', 'Fact']
for method in method_set:
    [N_est, reflectance] = MPS_SCPS.run_MPS_SCPS(img_set, mask, L_dir, method)
    [error_map, MAE, MedianE] = evalsurfaceNormal(N_gt, N_est, mask)
    print(method, MAE)

method_set = ['rob_Linear', 'rob_AM', 'rob_Fact']
for method in method_set:
    [N_est, reflectance] = MPS_SCPS_robust.run_MPS_SCPS_rob(img_set, mask, L_dir, method)
    [error_map, MAE, MedianE] = evalsurfaceNormal(N_gt, N_est, mask)
    print(method, MAE)




