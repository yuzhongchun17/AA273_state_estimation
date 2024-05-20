import torch
import numpy as np
import torch.nn.functional as F
from model_helpers import sample_pdf

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# convention helper
def opengl_to_opencv(pose_gl):
    transform = np.array([[1,0,0],
                        [0,-1,0],
                        [0,0,-1]])
    Rc_cv = pose_gl[:3,:3] @ transform
    center = pose_gl[:3,3].reshape(3,1)
    pose_cv = np.hstack([Rc_cv, center])# the translation stays the same, since both opengl and opencv are represented in world frame
    return pose_cv

def get_rays(H, W, K, c2w):
    """Takes camera calibration and pose to output ray origins and directions.
    Args:
        H (int): Number of pixels in height.
        W (int): Number of pixels in width.
        K (torch.Tensor): Camera calibration matrix of shape [3, 3].
        c2w (torch.Tensor): Camera to world transformation matrix of shape [3, 4].

    Returns:
        torch.Tensor: Ray origins of shape [H, W, 3].
        torch.Tensor: Ray directions of shape [H, W, 3].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure K and c2w are torch.Tensors and move them to GPU
    K = K if isinstance(K, torch.Tensor) else torch.from_numpy(K).float()
    c2w = c2w if isinstance(c2w, torch.Tensor) else torch.from_numpy(c2w).float()
    K = K.to(device)
    c2w = c2w.to(device)

    # opengl to opencv (why?)
    c2w_cv = torch.from_numpy(opengl_to_opencv(c2w.cpu().numpy())).float().to(device)

    # Extract camera origin (c) and rotation (Rc) from the camera-to-world transformation matrix (camera pose)
    center_cam = c2w_cv[:3, 3].view(3, 1).to(device)  # Camera origin in world frame, shape (3, 1)
    Rc = c2w_cv[:3, :3].to(device)  # Camera rotation in world frame, shape (3, 3)

    # Obtain uv coordinates for the entire image (homogeneous)
    u, v = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')  # H x W
    u = u.flatten().float()  # (HxW,)
    v = v.flatten().float()  # (HxW,)
    ones = torch.ones_like(u, device=device)  # Ensure ones are on the same device
    uv_coord = torch.stack([u, v, ones], dim=1).to(device)  # (HxW, 3)

    # Inverse of the camera intrinsic matrix
    K_inv = torch.inverse(K).to(device)
    uv_cam = (K_inv @ uv_coord.T).T  # (HxW, 3) # pixel 2d to camera 3d
    rays_d = (Rc @ uv_cam.T).T  # (HxW, 3) rotate ray direction: camera 3d -> world 3d 
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True) # normalize if needed for better result
    
    # braodcasting the camera origin for each ray
    rays_o = center_cam.T.expand(H * W, -1)  # (HxW, 3)

    # Reshape to image dimensions
    rays_o = rays_o.view(H, W, 3)
    rays_d = rays_d.view(H, W, 3)

    return rays_o, rays_d

def raw2outputs(raw, z_vals, rays_d):
    """
    Transforms model's predictions to semantically meaningful values.

    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model. RGB color and density.
        z_vals: [num_rays, num_samples along ray]. Sample points along ray.
        rays_d: [num_rays, 3]. Direction of each ray.

    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    
    # Function to calculate alpha from density channel of neural network. alpha = 1 - exp(-density * dist)
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw) * dists)
    
    # Extract RGB values and apply sigmoid activation
    rgb = torch.sigmoid(raw[..., :3])  # [num_rays, num_samples, 3]

    # Calculate the distance between sample points (add ver large num at the end of the ray)
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[:, :1])], dim=-1)  # [N_rays, N_samples]

    # Multiply the distance by the norm of the ray direction to get the real distance in the world space
    ray_d_norms = torch.norm(rays_d, dim=-1, keepdim=True)  # [num_rays, 1]
    dists = dists * ray_d_norms  # [num_rays, num_samples]

    # Extract density (sigma) from raw predictions 
    density = raw[..., 3]  # [num_rays, num_samples]

    # Calculate the alpha value for each sample point
    alpha = raw2alpha(density, dists)  # [num_rays, num_samples]

    # Compute weights
    T = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10], dim=-1), dim=-1)[:, :-1] # cumulative product
    weights = alpha * T  # [num_rays, num_samples]

    # Calculate the expected color for each ray (rgb map)
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [num_rays, 3]

    # Calculate the expected depth for each ray (depth map)
    depth_map = torch.sum(weights * z_vals, dim=-1)  # [num_rays]

    return rgb_map, depth_map, weights

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d

def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, depth_map, weights = raw2outputs(raw, z_vals, rays_d)

    if N_importance > 0:

        rgb_map_0, depth_map_0 = rgb_map, depth_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, depth_map, weights = raw2outputs(raw, z_vals, rays_d)

    ret = {'rgb_map' : rgb_map, 'depth_map': depth_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['depth0'] = depth_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    return ret
