import os
import cv2
import numpy as np
import torch_scatter
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import functools

from einops import rearrange, repeat
from torch_scatter import segment_coo
from cuda_splatting import render_cuda, render_depth_cuda
from gaussians import build_covariance

DepthRenderingMode = ["depth", "disparity", "relative_disparity", "log"]



class Gaussians:
    means: torch.FloatTensor
    covariances: torch.FloatTensor
    scales: torch.FloatTensor
    rotations: torch.FloatTensor
    harmonics: torch.FloatTensor
    opacities: torch.FloatTensor

turbo_colormap_data = [[0.18995, 0.07176, 0.23217], [0.19483, 0.08339, 0.26149], [0.19956, 0.09498, 0.29024],
                       [0.20415, 0.10652, 0.31844], [0.20860, 0.11802, 0.34607], [0.21291, 0.12947, 0.37314],
                       [0.21708, 0.14087, 0.39964], [0.22111, 0.15223, 0.42558], [0.22500, 0.16354, 0.45096],
                       [0.22875, 0.17481, 0.47578], [0.23236, 0.18603, 0.50004], [0.23582, 0.19720, 0.52373],
                       [0.23915, 0.20833, 0.54686], [0.24234, 0.21941, 0.56942], [0.24539, 0.23044, 0.59142],
                       [0.24830, 0.24143, 0.61286], [0.25107, 0.25237, 0.63374], [0.25369, 0.26327, 0.65406],
                       [0.25618, 0.27412, 0.67381], [0.25853, 0.28492, 0.69300], [0.26074, 0.29568, 0.71162],
                       [0.26280, 0.30639, 0.72968], [0.26473, 0.31706, 0.74718], [0.26652, 0.32768, 0.76412],
                       [0.26816, 0.33825, 0.78050], [0.26967, 0.34878, 0.79631], [0.27103, 0.35926, 0.81156],
                       [0.27226, 0.36970, 0.82624], [0.27334, 0.38008, 0.84037], [0.27429, 0.39043, 0.85393],
                       [0.27509, 0.40072, 0.86692], [0.27576, 0.41097, 0.87936], [0.27628, 0.42118, 0.89123],
                       [0.27667, 0.43134, 0.90254], [0.27691, 0.44145, 0.91328], [0.27701, 0.45152, 0.92347],
                       [0.27698, 0.46153, 0.93309], [0.27680, 0.47151, 0.94214], [0.27648, 0.48144, 0.95064],
                       [0.27603, 0.49132, 0.95857], [0.27543, 0.50115, 0.96594], [0.27469, 0.51094, 0.97275],
                       [0.27381, 0.52069, 0.97899], [0.27273, 0.53040, 0.98461], [0.27106, 0.54015, 0.98930],
                       [0.26878, 0.54995, 0.99303], [0.26592, 0.55979, 0.99583], [0.26252, 0.56967, 0.99773],
                       [0.25862, 0.57958, 0.99876], [0.25425, 0.58950, 0.99896], [0.24946, 0.59943, 0.99835],
                       [0.24427, 0.60937, 0.99697], [0.23874, 0.61931, 0.99485], [0.23288, 0.62923, 0.99202],
                       [0.22676, 0.63913, 0.98851], [0.22039, 0.64901, 0.98436], [0.21382, 0.65886, 0.97959],
                       [0.20708, 0.66866, 0.97423], [0.20021, 0.67842, 0.96833], [0.19326, 0.68812, 0.96190],
                       [0.18625, 0.69775, 0.95498], [0.17923, 0.70732, 0.94761], [0.17223, 0.71680, 0.93981],
                       [0.16529, 0.72620, 0.93161], [0.15844, 0.73551, 0.92305], [0.15173, 0.74472, 0.91416],
                       [0.14519, 0.75381, 0.90496], [0.13886, 0.76279, 0.89550], [0.13278, 0.77165, 0.88580],
                       [0.12698, 0.78037, 0.87590], [0.12151, 0.78896, 0.86581], [0.11639, 0.79740, 0.85559],
                       [0.11167, 0.80569, 0.84525], [0.10738, 0.81381, 0.83484], [0.10357, 0.82177, 0.82437],
                       [0.10026, 0.82955, 0.81389], [0.09750, 0.83714, 0.80342], [0.09532, 0.84455, 0.79299],
                       [0.09377, 0.85175, 0.78264], [0.09287, 0.85875, 0.77240], [0.09267, 0.86554, 0.76230],
                       [0.09320, 0.87211, 0.75237], [0.09451, 0.87844, 0.74265], [0.09662, 0.88454, 0.73316],
                       [0.09958, 0.89040, 0.72393], [0.10342, 0.89600, 0.71500], [0.10815, 0.90142, 0.70599],
                       [0.11374, 0.90673, 0.69651], [0.12014, 0.91193, 0.68660], [0.12733, 0.91701, 0.67627],
                       [0.13526, 0.92197, 0.66556], [0.14391, 0.92680, 0.65448], [0.15323, 0.93151, 0.64308],
                       [0.16319, 0.93609, 0.63137], [0.17377, 0.94053, 0.61938], [0.18491, 0.94484, 0.60713],
                       [0.19659, 0.94901, 0.59466], [0.20877, 0.95304, 0.58199], [0.22142, 0.95692, 0.56914],
                       [0.23449, 0.96065, 0.55614], [0.24797, 0.96423, 0.54303], [0.26180, 0.96765, 0.52981],
                       [0.27597, 0.97092, 0.51653], [0.29042, 0.97403, 0.50321], [0.30513, 0.97697, 0.48987],
                       [0.32006, 0.97974, 0.47654], [0.33517, 0.98234, 0.46325], [0.35043, 0.98477, 0.45002],
                       [0.36581, 0.98702, 0.43688], [0.38127, 0.98909, 0.42386], [0.39678, 0.99098, 0.41098],
                       [0.41229, 0.99268, 0.39826], [0.42778, 0.99419, 0.38575], [0.44321, 0.99551, 0.37345],
                       [0.45854, 0.99663, 0.36140], [0.47375, 0.99755, 0.34963], [0.48879, 0.99828, 0.33816],
                       [0.50362, 0.99879, 0.32701], [0.51822, 0.99910, 0.31622], [0.53255, 0.99919, 0.30581],
                       [0.54658, 0.99907, 0.29581], [0.56026, 0.99873, 0.28623], [0.57357, 0.99817, 0.27712],
                       [0.58646, 0.99739, 0.26849], [0.59891, 0.99638, 0.26038], [0.61088, 0.99514, 0.25280],
                       [0.62233, 0.99366, 0.24579], [0.63323, 0.99195, 0.23937], [0.64362, 0.98999, 0.23356],
                       [0.65394, 0.98775, 0.22835], [0.66428, 0.98524, 0.22370], [0.67462, 0.98246, 0.21960],
                       [0.68494, 0.97941, 0.21602], [0.69525, 0.97610, 0.21294], [0.70553, 0.97255, 0.21032],
                       [0.71577, 0.96875, 0.20815], [0.72596, 0.96470, 0.20640], [0.73610, 0.96043, 0.20504],
                       [0.74617, 0.95593, 0.20406], [0.75617, 0.95121, 0.20343], [0.76608, 0.94627, 0.20311],
                       [0.77591, 0.94113, 0.20310], [0.78563, 0.93579, 0.20336], [0.79524, 0.93025, 0.20386],
                       [0.80473, 0.92452, 0.20459], [0.81410, 0.91861, 0.20552], [0.82333, 0.91253, 0.20663],
                       [0.83241, 0.90627, 0.20788], [0.84133, 0.89986, 0.20926], [0.85010, 0.89328, 0.21074],
                       [0.85868, 0.88655, 0.21230], [0.86709, 0.87968, 0.21391], [0.87530, 0.87267, 0.21555],
                       [0.88331, 0.86553, 0.21719], [0.89112, 0.85826, 0.21880], [0.89870, 0.85087, 0.22038],
                       [0.90605, 0.84337, 0.22188], [0.91317, 0.83576, 0.22328], [0.92004, 0.82806, 0.22456],
                       [0.92666, 0.82025, 0.22570], [0.93301, 0.81236, 0.22667], [0.93909, 0.80439, 0.22744],
                       [0.94489, 0.79634, 0.22800], [0.95039, 0.78823, 0.22831], [0.95560, 0.78005, 0.22836],
                       [0.96049, 0.77181, 0.22811], [0.96507, 0.76352, 0.22754], [0.96931, 0.75519, 0.22663],
                       [0.97323, 0.74682, 0.22536], [0.97679, 0.73842, 0.22369], [0.98000, 0.73000, 0.22161],
                       [0.98289, 0.72140, 0.21918], [0.98549, 0.71250, 0.21650], [0.98781, 0.70330, 0.21358],
                       [0.98986, 0.69382, 0.21043], [0.99163, 0.68408, 0.20706], [0.99314, 0.67408, 0.20348],
                       [0.99438, 0.66386, 0.19971], [0.99535, 0.65341, 0.19577], [0.99607, 0.64277, 0.19165],
                       [0.99654, 0.63193, 0.18738], [0.99675, 0.62093, 0.18297], [0.99672, 0.60977, 0.17842],
                       [0.99644, 0.59846, 0.17376], [0.99593, 0.58703, 0.16899], [0.99517, 0.57549, 0.16412],
                       [0.99419, 0.56386, 0.15918], [0.99297, 0.55214, 0.15417], [0.99153, 0.54036, 0.14910],
                       [0.98987, 0.52854, 0.14398], [0.98799, 0.51667, 0.13883], [0.98590, 0.50479, 0.13367],
                       [0.98360, 0.49291, 0.12849], [0.98108, 0.48104, 0.12332], [0.97837, 0.46920, 0.11817],
                       [0.97545, 0.45740, 0.11305], [0.97234, 0.44565, 0.10797], [0.96904, 0.43399, 0.10294],
                       [0.96555, 0.42241, 0.09798], [0.96187, 0.41093, 0.09310], [0.95801, 0.39958, 0.08831],
                       [0.95398, 0.38836, 0.08362], [0.94977, 0.37729, 0.07905], [0.94538, 0.36638, 0.07461],
                       [0.94084, 0.35566, 0.07031], [0.93612, 0.34513, 0.06616], [0.93125, 0.33482, 0.06218],
                       [0.92623, 0.32473, 0.05837], [0.92105, 0.31489, 0.05475], [0.91572, 0.30530, 0.05134],
                       [0.91024, 0.29599, 0.04814], [0.90463, 0.28696, 0.04516], [0.89888, 0.27824, 0.04243],
                       [0.89298, 0.26981, 0.03993], [0.88691, 0.26152, 0.03753], [0.88066, 0.25334, 0.03521],
                       [0.87422, 0.24526, 0.03297], [0.86760, 0.23730, 0.03082], [0.86079, 0.22945, 0.02875],
                       [0.85380, 0.22170, 0.02677], [0.84662, 0.21407, 0.02487], [0.83926, 0.20654, 0.02305],
                       [0.83172, 0.19912, 0.02131], [0.82399, 0.19182, 0.01966], [0.81608, 0.18462, 0.01809],
                       [0.80799, 0.17753, 0.01660], [0.79971, 0.17055, 0.01520], [0.79125, 0.16368, 0.01387],
                       [0.78260, 0.15693, 0.01264], [0.77377, 0.15028, 0.01148], [0.76476, 0.14374, 0.01041],
                       [0.75556, 0.13731, 0.00942], [0.74617, 0.13098, 0.00851], [0.73661, 0.12477, 0.00769],
                       [0.72686, 0.11867, 0.00695], [0.71692, 0.11268, 0.00629], [0.70680, 0.10680, 0.00571],
                       [0.69650, 0.10102, 0.00522], [0.68602, 0.09536, 0.00481], [0.67535, 0.08980, 0.00449],
                       [0.66449, 0.08436, 0.00424], [0.65345, 0.07902, 0.00408], [0.64223, 0.07380, 0.00401],
                       [0.63082, 0.06868, 0.00401], [0.61923, 0.06367, 0.00410], [0.60746, 0.05878, 0.00427],
                       [0.59550, 0.05399, 0.00453], [0.58336, 0.04931, 0.00486], [0.57103, 0.04474, 0.00529],
                       [0.55852, 0.04028, 0.00579], [0.54583, 0.03593, 0.00638], [0.53295, 0.03169, 0.00705],
                       [0.51989, 0.02756, 0.00780], [0.50664, 0.02354, 0.00863], [0.49321, 0.01963, 0.00955],
                       [0.47960, 0.01583, 0.01055]]

def interpolate(colormap, x):
    x = max(0.0, min(1.0, x))
    a = int(x * 255.0)
    b = min(255, a + 1)
    f = x * 255.0 - a
    return [colormap[a][0] + (colormap[b][0] - colormap[a][0]) * f,
            colormap[a][1] + (colormap[b][1] - colormap[a][1]) * f,
            colormap[a][2] + (colormap[b][2] - colormap[a][2]) * f]


def interpolate_or_clip(colormap, x):
    if x < 0.0:
        return [0.0, 0.0, 0.0]
    elif x > 1.0:
        return [1.0, 1.0, 1.0]
    else:
        return interpolate(colormap, x)

def normalize_depth(depth, d_min, d_max):
    # normalize linearly between d_min and d_max
    data = np.clip(depth, d_min, d_max)
    return (data - d_min) / (d_max - d_min)

def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=c2w.device),
        torch.linspace(0, H - 1, H, device=c2w.device)
    )
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i + 0.5, j + 0.5
    elif mode == 'random':
        i = i + torch.rand_like(i)
        j = j + torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    batchsize = K.shape[0]
    rays_o_all = torch.zeros(batchsize, H, W, 3).to(K.device)
    rays_d_all = torch.zeros(batchsize, H, W, 3).to(K.device)

    for i in range(batchsize):
        rays_o, rays_d = get_rays(
            H, W, K[i, ...], c2w[i, ...],
            inverse_y=inverse_y,
            flip_x=flip_x,
            flip_y=flip_y,
            mode=mode
        )
        rays_o_all[i, ...] = rays_o
        rays_d_all[i, ...] = rays_d

    return rays_o_all, rays_d_all

def visualize_depth(depth, mask=None, depth_min=None, depth_max=None, direct=False):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    if not direct:
        depth = 1.0 / (depth + 1e-6)
    invalid_mask = np.logical_or(np.isnan(depth), np.logical_not(np.isfinite(depth)))
    if mask is not None:
        invalid_mask += np.logical_not(mask)
    if depth_min is None:
        depth_min = np.percentile(depth[np.logical_not(invalid_mask)], 5)
    if depth_max is None:
        depth_max = np.percentile(depth[np.logical_not(invalid_mask)], 95)
    depth[depth < depth_min] = depth_min
    depth[depth > depth_max] = depth_max
    depth[invalid_mask] = depth_max

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled_uint8 = np.uint8(depth_scaled * 255)
    depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_MAGMA)
    depth_color[invalid_mask, :] = 0

    return depth_color


@functools.lru_cache(maxsize=128)
def create_full_step_id(shape):
    ray_id = torch.arange(shape[0]).view(-1,1).expand(shape).flatten()
    step_id = torch.arange(shape[1]).view(1,-1).expand(shape).flatten()
    return ray_id, step_id

class GaussianSplattingDecoder(nn.Module):
    def __init__(self, cfg):
        super(GaussianSplattingDecoder, self).__init__()
        self.render_h, self.render_w = [88, 160]
        self.clip_range = False #True
        self.depth_range = [0.1, 100]
        self.use_lidar_depth = False
        self.mask_render = False
        self.semantic_head = True
        self.img_recon_head = False
        self.nonlinear_sample = False
        self.min_depth, self.max_depth = [0.1, 100]
        self.mode = 'bilinear'
        self.render_type = 'density'
        self.stepsize = 0.5
        self.semantic_dim = 6
        voxels_size = [576, 128, 16]
        self.max_ray_number = -1
        real_size = [-19.8, 95.4, -12.8, 12.8, -0.5, 3.5]
        self.real_size = real_size

        self.register_buffer('xyz_min', torch.from_numpy(np.array([self.real_size[0], self.real_size[2], self.real_size[4]])))
        self.register_buffer('xyz_max', torch.from_numpy(np.array([self.real_size[1], self.real_size[3], self.real_size[5]])))
        self.num_voxels = voxels_size[0] * voxels_size[1] * voxels_size[2]
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels).pow(1 / 3)
        ratio_x = self.real_size[1] / (self.real_size[1] - self.real_size[0])
        ratio_y = self.real_size[3] / (self.real_size[3] - self.real_size[2])
        ratio_z = self.real_size[5] / (self.real_size[5] - self.real_size[4])
        N_samples = int(np.linalg.norm(np.array(
            [voxels_size[0] * ratio_x, voxels_size[1] * ratio_y, voxels_size[2] * ratio_z]) + 1) / self.stepsize) + 1

        self.rng = torch.arange(N_samples)[None].float().cuda()

    def grid_sampler(self, xyz, grid, align_corners=True, mode='bilinear'):
        '''Wrapper for the interp operation'''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        grid = grid.unsqueeze(0)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1  # XYZ
        ret_lst = F.grid_sample(grid, ind_norm.float(), mode=mode, align_corners=align_corners)
        ret_lst = ret_lst.reshape(grid.shape[1], -1).T.reshape(*shape, grid.shape[1]).squeeze()
        return ret_lst

    def sample_ray(self, rays_o, rays_d, is_train, nonlinear_sample=False):
        """ Sample query points on rays """
        z_vals = self.stepsize * self.voxel_size * self.rng
        z_vals = z_vals.repeat(rays_d.shape[-2], 1)

        if is_train:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand

        if nonlinear_sample:
            t_to_s, s_to_t = self.construct_ray_warps(
               'piecewise', torch.ones_like(z_vals) * z_vals.min(), torch.ones_like(z_vals) * z_vals.max())
            tdist = s_to_t(z_vals / z_vals.max())
            z_vals = tdist

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [
        # N_rays, N_samples, 3]

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e10],
                          -1)  # [N_rays, N_samples]
        intervals = dists * torch.norm(rays_d[..., None, :], dim=-1)

        if self.clip_range:
            mask_outbbox = ((self.xyz_min > rays_pts) | (
                        rays_pts > self.xyz_max)).any(dim=-1)
        else:
            mask_outbbox = ((-1000 > rays_pts) | (rays_pts > 1000)).any(dim=-1)

        return rays_pts, mask_outbbox, z_vals, intervals

    @staticmethod
    def construct_ray_warps(fn, t_near, t_far):
        """Construct a bijection between metric distances and normalized distances.

        See the text around Equation 11 in https://arxiv.org/abs/2111.12077 for a
        detailed explanation.

        Args:
          fn: the function to ray distances.
          t_near: a tensor of near-plane distances.
          t_far: a tensor of far-plane distances.

        Returns:
          t_to_s: a function that maps distances to normalized distances in [0, 1].
          s_to_t: the inverse of t_to_s.
        """
        if fn is None:
            fn_fwd = lambda x: x
            fn_inv = lambda x: x
        else:
            inv_mapping = {
                'reciprocal': torch.reciprocal,
                'log': torch.exp,
                'exp': torch.log,
                'sqrt': torch.square,
                'square': torch.sqrt
            }
            fn_fwd = fn
            fn_inv = inv_mapping[fn.__name__]

        s_near, s_far = [fn_fwd(x) for x in (t_near, t_far)]
        t_to_s = lambda t: (fn_fwd(t) - s_near) / (s_far - s_near)
        s_to_t = lambda s: fn_inv(s * s_far + (1 - s) * s_near)
        return t_to_s, s_to_t

    def get_density(self, rays_o, rays_d, voxel, rgb_recon, semantic_recon, is_train, mode, nonlinear_sample=False, render_mask=None):
        with torch.no_grad():
            if render_mask is None:
                rays_o_i = rays_o.flatten(0, 2)  # H,W,3
                rays_d_i = rays_d.flatten(0, 2)  # H,W,3
            else:
                rays_o_i = rays_o[render_mask]
                rays_d_i = rays_d[render_mask]
            rays_pts, mask_outbbox, z_vals, intervals = self.sample_ray(
                rays_o_i, rays_d_i, is_train=is_train,
                nonlinear_sample=nonlinear_sample)

        mask_rays_pts = rays_pts[~mask_outbbox]
        density = self.grid_sampler(mask_rays_pts, voxel, mode=mode)
        internal_values = {}
        if self.render_type == 'prob':
            probs = torch.zeros_like(rays_pts[..., 0])
            probs[:, -1] = 1
            density = torch.sigmoid(density) # already use sigmoid
            probs[~mask_outbbox] = density
            probs = probs.cumsum(dim=1).clamp(max=1)
            weights = probs.diff(dim=1, prepend=torch.zeros((rays_pts.shape[:1])).unsqueeze(1).to('cuda'))
            depth = (weights * z_vals).sum(-1)

            if self.img_recon_head:
                rgb = self.grid_sampler(mask_rays_pts, rgb_recon)
                rgb_cache = torch.zeros_like(rays_pts)
                rgb_cache[~mask_outbbox] = rgb
                rgb_marched = torch.sum(probs[..., None] * rgb_cache, -2)
            else:
                rgb_marched = depth

            if self.semantic_head:
                semantic = self.grid_sampler(mask_rays_pts, semantic_recon)
                B, N = rays_pts.shape[:2]
                semantic_cache = torch.zeros((B, N, self.semantic_dim)).to(rays_pts.device)
                semantic_cache[~mask_outbbox] = semantic  # 473088, 287, 3
                semantic_marched = torch.sum(probs[..., None] * semantic_cache, -2)
            else:
                semantic_marched = depth

        elif self.render_type == 'DVGO':
            from ..nerf_head.utils import Raw2Alpha, Alphas2Weights, ub360_utils_cuda, silog_loss
            probs = torch.zeros_like(rays_pts[..., 0])
            probs[:, -1] = 1
            probs[~mask_outbbox] = density

            alpha = Raw2Alpha.apply(probs.flatten(), 0, 0.5)
            ray_id = torch.arange(rays_pts.shape[:2][0]).view(-1, 1).expand(rays_pts.shape[:2]).flatten().to(alpha.device)
            weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id.to(alpha.device), len(rays_pts))

            depth = segment_coo(
                src=(weights.reshape(probs.shape) * z_vals).reshape(-1),
                index=ray_id,
                out=torch.zeros([len(rays_pts)]).to(weights.device),
                reduce='sum') + 1e-7

            if self.semantic_head:
                semantic = self.grid_sampler(mask_rays_pts, semantic_recon)
                B, N = rays_pts.shape[:2]
                semantic_cache = torch.zeros((B, N, self.semantic_dim)).to(rays_pts.device)
                semantic_cache[~mask_outbbox] = semantic
                semantic_marched = segment_coo(
                    src=(weights.reshape(probs.shape).unsqueeze(-1) * semantic_cache).reshape((-1, semantic.shape[-1])),
                    index=ray_id,
                    out=torch.zeros([len(rays_pts), semantic.shape[-1]]).to(weights.device),
                    reduce='sum')
            else:
                semantic_marched = depth

            if self.img_recon_head:
                rgb = self.grid_sampler(mask_rays_pts, rgb_recon)
                rgb_cache = torch.zeros_like(rays_pts)
                rgb_cache[~mask_outbbox] = rgb
                rgb_marched = segment_coo(
                    src=(weights.reshape(probs.shape).unsqueeze(-1) * rgb_cache).reshape((-1, 3)),
                    index=ray_id,
                    out=torch.zeros([len(rays_pts), 3]).to(weights.device),
                    reduce='sum')
            else:
                rgb_marched = depth
            internal_values['z_vals'] = z_vals
            interval_list = z_vals[..., 1:] - z_vals[..., :-1]
            internal_values['dists'] = interval_list
            internal_values['weights'] = weights

        elif self.render_type == 'density':
            alpha = torch.zeros_like(rays_pts[..., 0])
            alpha[~mask_outbbox] = 1 - torch.exp(-F.softplus(density) * intervals[~mask_outbbox])
            alphainv_cum = torch.cat([torch.ones_like((1 - alpha)[..., [0]]), (1 - alpha).clamp_min(1e-10).cumprod(-1)], -1)  # accumulated transmittance
            weights = alpha * alphainv_cum[..., :-1]  # alpha * accumulated transmittance = weights
            depth = (weights * z_vals).sum(-1)

            # if self.pdf_sample:
            #     z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            #     z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance)
            #     z_samples = z_samples.detach()
            #     z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            #     dists = z_vals[..., 1:] - z_vals[..., :-1]
            #     dists = torch.cat(
            #         [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)
            #     intervals = dists * torch.norm(rays_d[..., None, :], dim=-1)
            #     pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            #     mask_rays_pts = rays_pts[~pts]
            #     density = self.grid_sampler(mask_rays_pts, voxel, mode=mode)
            #     alpha = torch.zeros_like(rays_pts[..., 0])
            #     alpha[~mask_outbbox] = 1 - torch.exp(
            #         -F.softplus(density) * intervals[~mask_outbbox])
            #     alphainv_cum = torch.cat(
            #         [torch.ones_like((1 - alpha)[..., [0]]),
            #          (1 - alpha).clamp_min(1e-10).cumprod(-1)],
            #         -1)  # accumulated transmittance
            #     weights = alpha * alphainv_cum[...,
            #                       :-1]  # alpha * accumulated transmittance = weights
            #     depth = (weights * z_vals).sum(-1)

            # save interval values
            internal_values['z_vals'] = z_vals
            internal_values['dists'] = intervals
            internal_values['weights'] = weights
            if self.img_recon_head:
                rgb = self.grid_sampler(mask_rays_pts, rgb_recon)
                rgb_cache = torch.zeros_like(rays_pts)
                rgb_cache[~mask_outbbox] = rgb
                rgb_marched = torch.sum(weights[..., None] * rgb_cache, -2)
            else:
                rgb_marched = depth

            if self.semantic_head:
                semantic = self.grid_sampler(mask_rays_pts, semantic_recon)
                B, N = rays_pts.shape[:2]
                semantic_cache = torch.zeros((B, N, self.semantic_dim)).to(rays_pts.device)
                semantic_cache[~mask_outbbox] = semantic  # 473088, 287, 3
                semantic_marched = torch.sum(weights[..., None] * semantic_cache, -2)
            else:
                semantic_marched = depth
        else:
            raise NotImplementedError

        return depth, rgb_marched, semantic_marched, internal_values

    def compute_depth_loss(self, depth_est, depth_gt, mask, interval_list=None):
        '''
        Args:
            mask: depth_gt > 0
        '''
        if self.depth_loss_type == 'silog':
            d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
            loss = torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2))
        elif self.depth_loss_type == 'sigma':
            batchsize = len(interval_list)
            loss = 0
            offset = 0
            err = 1
            for b in range(batchsize):
                weights = interval_list[b]['weights']
                z_vals = interval_list[b]['z_vals']
                dists = interval_list[b]['dists']
                num = dists.shape[0]
                depth = depth_gt[offset:offset+num]
                # filter outbox points
                weight_mask = (weights < 1e-3)
                l = -torch.log(weights + 1e-5)
                l[weight_mask] = 0
                dists[..., -1] = 0.
                l = l * torch.exp(-(z_vals - depth[:, None]) ** 2 / (2 * err)) * dists
                #l = -torch.log(weights + 1e-5) * torch.exp(-(z_vals - depth[:, None]) ** 2 / (2 * err)) * dists
                loss = loss + torch.mean(torch.sum(l, dim=1)) / batchsize
                offset += num
        elif self.depth_loss_type == 'l1':
            loss = F.l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
        elif self.depth_loss_type == 'rl1':
            depth_est = (1 / depth_est) * self.max_depth
            depth_gt = (1 / depth_gt) * self.max_depth
            loss = F.l1_loss(depth_est[mask], depth_gt[mask], size_average=True)

        elif self.depth_loss_type == 'sml1':
            loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
        else:
            raise NotImplementedError()

        return loss

    def compute_image_loss(self, image_est, image_gt):
        loss = F.smooth_l1_loss(image_est, image_gt, size_average=True)
        return loss

    def compute_semantic_loss_flatten(self, sem_est, sem_gt):
        '''
        Args:
            sem_est: N, C
            sem_gt: N
        '''
        if self.contrastive:
            sem_est = torch_scatter.scatter_mean(sem_est, sem_gt, 0)
            sem_gt = torch_scatter.scatter_mean(sem_gt, sem_gt, 0)
            loss = F.cross_entropy(sem_est, sem_gt.long())
        else:
            loss = F.cross_entropy(sem_est, sem_gt.long(), ignore_index=-100)

        return loss

    def compute_semantic_loss(self, sem_est, sem_gt):
        '''
        Args:
            sem_est: B, N, C, H, W, predicted unnormalized logits
            sem_gt: B, N, H, W
        '''
        B, N, C, H, W = sem_est.shape
        sem_est = sem_est.view(B*N, -1, H, W)
        sem_gt = sem_gt.view(B*N, H, W)
        loss = F.cross_entropy(sem_est, sem_gt.long(), ignore_index=-100)

        return loss

    def volume_rendering(self, density_prob, rgb_recon, semantic_pred, intricics, pose_spatial, is_train=True, render_mask=None):
        '''
        B: batchsize, N: num_view, H, W: img_size
        Args:
            density_prob (B, 1, X, Y, Z)
            rgb_recon (B, 3, X, Y, Z)
            semantic_pred (B, C, X, Y, Z)
            intricics (B, N, 4, 4)
            pose_spatial: camera-to-ego (B, N, 4, 4)
        Returns:
            depth (B, N_view, H, W)
            rgb (B, N_view, C, H, W)
            semantic (B, N_view, C, H, W)
        '''
        batch_size, num_camera = intricics.shape[:2]
        intricics = intricics.view(-1, 4, 4)
        pose_spatial = pose_spatial.view(-1, 4, 4)

        with torch.no_grad():  # do not affect speed and memory
            rays_o, rays_d = get_rays_of_a_view(
                H=self.render_h,
                W=self.render_w,
                K=intricics,
                c2w=pose_spatial,
                inverse_y=True,
                flip_x=False,
                flip_y=False,
                mode='center'
            )
        rays_o = rays_o.view(batch_size, num_camera, self.render_h, self.render_w, 3)  # rays_o (B, N, H, W, 3)
        rays_d = rays_d.view(batch_size, num_camera, self.render_h, self.render_w, 3)  # rays_d (B, N, H, W, 3)

        batch_depth = []
        batch_rgb = []
        batch_semantic = []
        interval_list = []
        for b in range(batch_size):
            rmask = render_mask[b] if self.mask_render else None
            depth, rgb_marched, semantic, inter = self.get_density(
                rays_o[b], rays_d[b], density_prob[b], rgb_recon[b], semantic_pred[b], is_train,
                mode=self.mode, nonlinear_sample=self.nonlinear_sample, render_mask=rmask
            )

            if not self.mask_render:
                depth = depth.reshape(num_camera, self.render_h, self.render_w)
                depth = depth.clamp(self.min_depth, self.max_depth)
                rgb_marched = rgb_marched.reshape(num_camera, self.render_h, self.render_w, -1)
                semantic = semantic.reshape(num_camera, self.render_h, self.render_w, -1)

            batch_depth.append(depth)
            batch_rgb.append(rgb_marched)
            batch_semantic.append(semantic)
            interval_list.append(inter)

        if self.mask_render:
            batch_depth = torch.cat(batch_depth, 0)
            batch_rgb = torch.cat(batch_rgb, 0)
            batch_semantic = torch.cat(batch_semantic, 0)
        else:
            batch_depth = torch.stack(batch_depth)
            batch_rgb = torch.stack(batch_rgb).permute(0, 1, 4, 2, 3)
            batch_semantic = torch.stack(batch_semantic).permute(0, 1, 4, 2, 3)

        return batch_depth, batch_rgb, batch_semantic, interval_list

    def depth2ego_points(self, depth, camK, cam2veh):
        '''
        Args:
            depth: B x H x W
            camK: B x 3 x 3
            cam2veh: B x 4 x 4
        Returns:
            ego_points: B x H x W x 3
        '''
        B, H, W = depth.shape
        pt0, pt1 = torch.meshgrid(
            torch.linspace(0, W - 1, W, device=depth.device),
            torch.linspace(0, H - 1, H, device=depth.device)
        )
        pt0 = pt0.t().float().unsqueeze(0).tile(B, 1, 1) + 0.5
        pt1 = pt1.t().float().unsqueeze(0).tile(B, 1, 1) + 0.5
        pt0 = (pt0 - camK[:, 0:1, 2:3]) * depth / (camK[:, 0:1, 0:1] + 1e-6)
        pt1 = (pt1 - camK[:, 1:2, 2:3]) * depth / (camK[:, 1:2, 1:2] + 1e-6)
        cam_points = torch.stack([pt0, pt1, depth, torch.ones_like(depth)], dim=-1).reshape(B, H*W, 4)
        ego_points = torch.matmul(cam_points, cam2veh.permute(0, 2, 1))[:, :, :3]
        ego_points = ego_points.reshape(B, H, W, 3)
        return ego_points

    @staticmethod
    def gen_point_range_mask(pts, coors_range_xyz=None):
        if coors_range_xyz is None:
            return torch.ones_like(pts).bool()
        mask11 = pts[..., 0] < coors_range_xyz[1]
        mask12 = pts[..., 0] > coors_range_xyz[0]
        mask21 = pts[..., 1] < coors_range_xyz[3]
        mask22 = pts[..., 1] > coors_range_xyz[2]
        mask31 = pts[..., 2] < coors_range_xyz[5]
        mask32 = pts[..., 2] > coors_range_xyz[4]

        return mask11 & mask12 & mask21 & mask22 & mask31 & mask32

    @staticmethod
    def limit_true_count(tensor, max_true_count):
        flattened = tensor.view(-1)
        true_indices = torch.nonzero(flattened).squeeze()
        if true_indices.numel() <= max_true_count:
            return tensor
        selected_indices = true_indices[torch.randperm(true_indices.numel())[:int(max_true_count)]]
        flattened.zero_()
        flattened[selected_indices] = 1
        return tensor.view(tensor.size())

    def gaussian_rasterization(self, density_prob, rgb_recon, semantic_pred, intrinsics, extrinsics, is_train=True, render_mask=None):
        near = torch.ones(1, 7).cuda() * 1
        far = torch.ones(1, 7).cuda() * 100
        background_color = torch.zeros((3), dtype=torch.float32).cuda()
        b, v = intrinsics.shape[:2]
        intrinsics = intrinsics[..., :3, :3]
        intrinsic[..., 0, :] /= 100
        intrinsic[...,1, :] /= 100

        transform = torch.Tensor(np.diag([1, 1, 1, 1])).cuda()
        extrinsics = transform.unsqueeze(0).unsqueeze(0) @ extrinsics

        device = density_prob.device
        xs = torch.range(
            self.real_size[0], self.real_size[1],
            (self.real_size[1] - self.real_size[0]) / density_prob.shape[2], device=device)[:-1]
        ys = torch.range(
            self.real_size[2], self.real_size[3],
            (self.real_size[3] - self.real_size[2]) / density_prob.shape[3], device=device)[:-1]
        zs = torch.range(
            self.real_size[4], self.real_size[5],
            (self.real_size[5] - self.real_size[4]) / density_prob.shape[4], device=device)[:-1]
        W, H, D = len(xs), len(ys), len(zs)
        xyzs = torch.stack([
            xs[None, :, None].expand(H, W, D),
            ys[:, None, None].expand(H, W, D),
            zs[None, None, :].expand(H, W, D)
        ], dim=-1).flatten(0, 2) # max-min [95.2000, 12.6000,  3.2500, -19.8000, -12.8000,  -0.5000]
        density_prob = density_prob.flatten()

        mask = (density_prob > 0) #& (semantic_pred.flatten()==3)
        xyzs = xyzs[mask]

        UNOCCUPANCY = [0, 0, 0]
        ROAD = [128, 64, 128]
        OTHERS = [128, 0, 0]
        BUSH = [128, 128, 0]
        DYNAMIC = [64, 0, 128]
        CURB = [64, 64, 128]
        COLOR_DICT = np.array([ROAD, OTHERS, BUSH, DYNAMIC, CURB, UNOCCUPANCY])[..., [2, 1, 0]]

        semantic_pred[semantic_pred == -1] = 5
        COLOR_DICT = torch.Tensor(COLOR_DICT).cuda()
        harmonics = COLOR_DICT[semantic_pred.long().flatten()]
        harmonics = harmonics[mask]

        density_prob = density_prob[mask]
        density_prob = density_prob.unsqueeze(0)
        xyzs = xyzs.unsqueeze(0)

        g = xyzs.shape[1]

        gaussians = Gaussians
        gaussians.means = xyzs  ######## Gaussian center ########
        gaussians.opacities = torch.where(density_prob>0, 1., 0.) ######## Gaussian opacities ########

        scales = torch.ones(3).unsqueeze(0).to(device) * 0.05
        rotations = torch.Tensor([1, 0, 0, 0]).unsqueeze(0).to(device)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        gaussians.covariances = covariances ######## Gaussian covariances ########

        harmonics = harmonics.unsqueeze(-1).unsqueeze(0)
        # harmonics = torch.ones_like(xyzs).unsqueeze(-1)
        gaussians.harmonics = harmonics ######## Gaussian harmonics ########

        color = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (self.render_h, self.render_w),
            repeat(background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b v i j -> (b v) g i j", g=g),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=False,
            use_sh=False,
        )

        depth = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (self.render_h, self.render_w),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            mode="disparity",
            scale_invariant=False,

        )
        return color, depth

    def forward(self,
                density_prob,
                occ_semantic,
                intrinsic,
                pose_spatial,
                render_img_gt,
                depths,
                semantic_gt,
                ):
        with torch.no_grad():
            if False: ## NeRF
                render_depth, rgb_pred, semantic_pred, interval_list = self.volume_rendering(
                    density_prob,
                    density_prob.tile(1, 3, 1, 1, 1),  # pseudo color
                    occ_semantic,
                    intrinsic,
                    pose_spatial,
                    is_train=self.training
                )
            else: ### 3DGS
                print('density_prob', density_prob.shape)
                print('occ_semantic', occ_semantic.shape)
                semantic_pred, render_depth = self.gaussian_rasterization(
                    density_prob.flip(2).flip(3),
                    density_prob.tile(1, 3, 1, 1, 1),  # pseudo color
                    occ_semantic.flip(1).flip(2),
                    intrinsic,
                    pose_spatial,
                    is_train=self.training
                )
                render_depth = render_depth.unsqueeze(0).clone()
                semantic_pred = semantic_pred.unsqueeze(0).clone().clamp(0, 1)

            print('pred depth:', render_depth.shape)
            print('pred sem:', semantic_pred.shape)

            # print('pred semantic range:', semantic_pred.shape, semantic_pred.min(), semantic_pred.max())
            # print('pred depth range:', render_depth.shape, render_depth.min(), render_depth.max())
            # print('GT depth range:', depths.shape, depths.min(), depths.max())
            outpath = 'debug_3dgs/'
            os.makedirs(outpath, exist_ok=True)
            for batch_idx in range(render_depth.shape[0]):
                with torch.no_grad():
                    self.visualize_image_depth_pair(render_img_gt[batch_idx].tile(3, 1, 1, 1).cpu().numpy(),
                                                    depths[batch_idx, :, :, :].tile(3, 1, 1),
                                                    render_depth[batch_idx, :, :, :].tile(3, 1, 1), batch_idx,
                                                    outpath)
                    if self.semantic_head:
                        self.visualize_image_semantic_pair(render_img_gt[batch_idx].cpu().numpy(),
                                                           semantic_gt[batch_idx, :, :, :],
                                                           semantic_pred[batch_idx],
                                                           batch_idx,
                                                           outpath)
    def visualize_image_depth_pair(self, images, depth, render, batch_idx=0, path='.'):
            '''
            This is a debug function!!
            Args:
                images: num_camera, 3, H, W
                depth: num_camera, H, W
                render: num_camera, H, W
            '''
            concated_render_list = []
            concated_image_list = []
            depth = depth.cpu().numpy()
            render = render.cpu().numpy()

            for b in range(len(images)):
                visual_img = cv2.resize(images[b].transpose((1, 2, 0)), (depth.shape[-1], depth.shape[-2]))
                img_mean = np.array([0.485, 0.456, 0.406])[None, None, :]
                img_std = np.array([0.229, 0.224, 0.225])[None, None, :]
                visual_img = np.ascontiguousarray((visual_img * img_std + img_mean) * 255, dtype=np.uint8)
                concated_image_list.append(visual_img)
                pred_depth_color = visualize_depth(render[b])
                pred_depth_color = pred_depth_color[..., [2, 1, 0]]
                concated_render_list.append(cv2.resize(pred_depth_color.copy(), (depth.shape[-1], depth.shape[-2])))

            normalized_voxel_depth = normalize_depth(depth, d_min=self.min_depth, d_max=self.max_depth)
            fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(6, 6))
            ij = [[i, j] for i in range(2) for j in range(3)]
            for i in range(len(ij)):
                colors_voxel = []
                for depth_val in normalized_voxel_depth[i][normalized_voxel_depth[i] > 0].reshape(-1):
                    colors_voxel.append(interpolate_or_clip(colormap=turbo_colormap_data, x=depth_val))
                ax[ij[i][0], ij[i][1]].imshow(concated_image_list[i])
                ax[ij[i][0] + 2, ij[i][1]].imshow(np.ones_like(concated_render_list[i]) * 255)
                # ax[ij[i][0] + 2, ij[i][1]].scatter(normalized_voxel_depth[i].nonzero()[1],
                #                                    normalized_voxel_depth[i].nonzero()[0], c=colors_voxel, alpha=0.5, s=0.5)
                ax[ij[i][0] + 2, ij[i][1]].imshow(concated_render_list[i])
                ax[ij[i][0] + 4, ij[i][1]].imshow(concated_render_list[i])

                for j in range(3):
                    ax[i, j].axis('off')

            plt.subplots_adjust(wspace=0.01, hspace=0.01)
            plt.show()
            # plt.savefig('{}/output_depth{}.png'.format(path, batch_idx))

    def visualize_image_semantic_pair(self, images, seg_gt, seg_render_color, batch_idx=0, path='.'):
            '''
            This is a debug function!!
            Args:
                images: num_camera, 3, H, W
                seg_gt: num_camera, H, W
                seg_render: num_camera, 6, H, W
            '''
            image_list = []
            seg_gt_list = []
            seg_render_list = []
            seg_gt = seg_gt.cpu().numpy()
            seg_gt[seg_gt == -100] = 5  # map ignore index to 5 for visualization
            # seg_render = np.argmax(seg_render.cpu().numpy(), axis=1)
            seg_render_color = seg_render_color.permute(0, 2, 3, 1).cpu().numpy()

            UNOCCUPANCY = [0, 0, 0]
            ROAD = [128, 64, 128]
            OTHERS = [128, 0, 0]
            BUSH = [128, 128, 0]
            DYNAMIC = [64, 0, 128]
            CURB = [64, 64, 128]
            COLOR_DICT = np.array([ROAD, OTHERS, BUSH, DYNAMIC, CURB, UNOCCUPANCY])

            for b in range(len(images)):
                # visual_img = cv2.resize(images[b].transpose((1, 2, 0)), (seg_render.shape[-1], seg_render.shape[-2]))
                visual_img = cv2.resize(images[b].transpose((1, 2, 0)), (seg_render_color.shape[2], seg_render_color.shape[1]))
                img_mean = np.array([0.485, 0.456, 0.406])[None, None, :]
                img_std = np.array([0.229, 0.224, 0.225])[None, None, :]
                visual_img = np.ascontiguousarray((visual_img * img_std + img_mean) * 255, dtype=np.uint8)
                image_list.append(visual_img)

                seg_gt_color = np.zeros((seg_gt.shape[-2], seg_gt.shape[-1], 3), dtype=np.uint8)
                for i in range(6):
                    seg_gt_color[seg_gt[b] == i, :] = COLOR_DICT[i]
                seg_gt_color = seg_gt_color[..., [2, 1, 0]]
                # seg_gt_list.append(cv2.resize(seg_gt_color.copy(), (seg_render.shape[-1], seg_render.shape[-2]),
                #                               interpolation=cv2.INTER_NEAREST))
                seg_gt_list.append(cv2.resize(seg_gt_color.copy(), (seg_render_color.shape[2], seg_render_color.shape[1]),
                                              interpolation=cv2.INTER_NEAREST))
                # seg_render_color = np.zeros((seg_render.shape[-2], seg_render.shape[-1], 3), dtype=np.uint8)
                # for i in range(6):
                #     seg_render_color[seg_render[b] == i, :] = COLOR_DICT[i]
                # seg_render_color = seg_render_color[..., [2, 1, 0]]
                # seg_render_list.append(cv2.resize(seg_gt_color.copy(), (seg_render_color.shape[2], seg_render_color.shape[1]),
                #                               interpolation=cv2.INTER_NEAREST))
                seg_render_list.append(seg_render_color[b])
                # seg_render_list.append(seg_render_color.copy())

            fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(6, 6))
            ij = [[i, j] for i in range(2) for j in range(3)]
            for i in range(len(ij)):
                ax[ij[i][0], ij[i][1]].imshow(image_list[i])
                ax[ij[i][0] + 2, ij[i][1]].imshow(seg_gt_list[i])
                ax[ij[i][0] + 4, ij[i][1]].imshow(seg_render_list[i])

                for j in range(3):
                    ax[i, j].axis('off')

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.show()
            plt.close()
            # plt.savefig('{}/output_sem{}.png'.format(path, batch_idx))


if __name__ == '__main__':
    nerf = GaussianSplattingDecoder({}).cuda()

    density_prob = torch.load('debug_render/density_prob.pth').detach().cuda()
    occ_semantic = torch.load('debug_render/gt.pth').detach().cuda()
    intrinsic = torch.load('debug_render/intrinsic.pth').detach().cuda()
    pose_spatial = torch.load('debug_render/pose_spatial.pth').detach().cuda()
    render_img_gt = torch.load('debug_render/render_img_gt.pth').detach().cuda()
    depths = torch.load('debug_render/depths.pth').detach().cuda()
    semantic_gt = torch.load('debug_render/semantic_gt.pth').detach().cuda()

    nerf(
        density_prob,
        occ_semantic,
        intrinsic,
        pose_spatial,
        render_img_gt,
        depths,
        semantic_gt,
    )