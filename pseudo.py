import numpy as np
from plyfile import PlyData
import torch.nn.functional as F
import math
import torch

def load_ply_xyz_rgb(ply_path):
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    if "red" in v.data.dtype.names:
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32) / 255.0
    else:
        rgb = np.ones_like(xyz, dtype=np.float32) * 0.5
    return xyz, rgb


@torch.no_grad()
def point_render_one_view(
    pseudo_cam,
    xyz: np.ndarray,          # [N,3] float32
    rgb: np.ndarray,          # [N,3] float32 in [0,1]
    device: str = "cuda",
    radius: int = 2,          # 邻域半径，2 => 5x5
    sigma: float = 1.0,       # 邻域高斯权重
    z_weight: float = 0.8,    # 深度权重的衰减系数(越小越偏近处)
    max_points: int = 200000, # 预渲染时随机下采样，防止太慢
    w_min: float = 1e-4       # coverage 阈值
):
    """
    Weighted point blending rasterization (LMGS-like):
    - 每个点投影到像素邻域 (2r+1)^2
    - 权重 = exp(-dxy^2/(2*sigma^2)) * exp(-(z-zmin)/z_weight)
    - 像素端做加权平均，返回:
        pr_rgb : [3,H,W] float
        pr_mask: [1,H,W] {0,1}
        pr_depth:[1,H,W] float
        pr_cov : [1,H,W] float (sum of weights)
    """
    H, W = pseudo_cam.image_height, pseudo_cam.image_width

    # --- optional random subsample for speed ---
    N = xyz.shape[0]
    if max_points is not None and N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]

    xyz_t = torch.from_numpy(xyz).to(device)  # [N,3]
    rgb_t = torch.from_numpy(rgb).to(device)  # [N,3]

    # world -> camera
    w2c = pseudo_cam.world_view_transform.t()  # 4x4
    ones = torch.ones((xyz_t.shape[0], 1), device=device)
    pts_h = torch.cat([xyz_t, ones], dim=1)               # [N,4]
    pts_c = (pts_h @ w2c)[:, :3]                          # [N,3]

    # filter points behind camera
    z = pts_c[:, 2]
    valid = z > 1e-6
    pts_c = pts_c[valid]
    rgb_t = rgb_t[valid]
    z = z[valid]
    if pts_c.shape[0] == 0:
        pr_rgb = torch.zeros((3, H, W), device=device)
        pr_mask = torch.zeros((1, H, W), device=device)
        pr_depth = torch.zeros((1, H, W), device=device)
        pr_cov = torch.zeros((1, H, W), device=device)
        return pr_rgb, pr_mask, pr_depth, pr_cov

    # projection: camera -> clip -> ndc
    P = pseudo_cam.full_proj_transform.t()                # 4x4
    pts_h2 = torch.cat([pts_c, torch.ones((pts_c.shape[0], 1), device=device)], dim=1)  # [M,4]
    clip = pts_h2 @ P                                     # [M,4]
    ndc = clip[:, :3] / (clip[:, 3:4] + 1e-8)             # [M,3]

    x_ndc, y_ndc = ndc[:, 0], ndc[:, 1]
    u = ((x_ndc + 1.0) * 0.5) * (W - 1)                   # [M]
    v = ((1.0 - (y_ndc + 1.0) * 0.5)) * (H - 1)           # [M]

    # keep points inside image (with margin for radius)
    margin = radius + 1
    inside = (u >= -margin) & (u <= (W - 1 + margin)) & (v >= -margin) & (v <= (H - 1 + margin))
    u, v, z, rgb_t = u[inside], v[inside], z[inside], rgb_t[inside]
    if u.numel() == 0:
        pr_rgb = torch.zeros((3, H, W), device=device)
        pr_mask = torch.zeros((1, H, W), device=device)
        pr_depth = torch.zeros((1, H, W), device=device)
        pr_cov = torch.zeros((1, H, W), device=device)
        return pr_rgb, pr_mask, pr_depth, pr_cov

    # depth weight (nearer higher)
    z_min = torch.min(z)
    wz = torch.exp(-(z - z_min) / max(z_weight, 1e-6))    # [M]

    # prepare accumulators (flattened HW)
    HW = H * W
    acc_w = torch.zeros((HW,), device=device)
    acc_rgb = torch.zeros((HW, 3), device=device)
    acc_d = torch.zeros((HW,), device=device)

    # neighbor offsets
    # precompute gaussian for integer offsets
    offsets = []
    gauss = []
    inv_2sig2 = 1.0 / max(2.0 * sigma * sigma, 1e-8)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            offsets.append((dx, dy))
            gauss.append(math.exp(-(dx * dx + dy * dy) * inv_2sig2))
    offsets = torch.tensor(offsets, device=device, dtype=torch.long)   # [K,2]
    gauss = torch.tensor(gauss, device=device, dtype=torch.float32)    # [K]
    K = offsets.shape[0]

    # base pixel (round) and neighbor pixels
    ui0 = torch.round(u).long()  # [M]
    vi0 = torch.round(v).long()  # [M]

    # expand to [M*K]
    ui = ui0[:, None] + offsets[None, :, 0]   # [M,K]
    vi = vi0[:, None] + offsets[None, :, 1]   # [M,K]

    # valid neighbor pixels
    inside2 = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)  # [M,K]
    if inside2.sum() == 0:
        pr_rgb = torch.zeros((3, H, W), device=device)
        pr_mask = torch.zeros((1, H, W), device=device)
        pr_depth = torch.zeros((1, H, W), device=device)
        pr_cov = torch.zeros((1, H, W), device=device)
        return pr_rgb, pr_mask, pr_depth, pr_cov

    # weights: w = wz * gauss(offset)
    w = (wz[:, None] * gauss[None, :])                      # [M,K]
    w = w[inside2]                                          # [T]
    ui = ui[inside2]                                        # [T]
    vi = vi[inside2]                                        # [T]

    pix = (vi * W + ui).long()                              # [T]

    # expand rgb/z to match [T]
    # We need to align each neighbor with its source point:
    # inside2 flattens in row-major order, so repeat point values K times then mask
    z_rep = z[:, None].expand(-1, K)[inside2]               # [T]
    rgb_rep = rgb_t[:, None, :].expand(-1, K, -1)[inside2]  # [T,3]

    # scatter add
    acc_w.scatter_add_(0, pix, w)
    acc_d.scatter_add_(0, pix, w * z_rep)
    acc_rgb.scatter_add_(0, pix[:, None].expand(-1, 3), w[:, None] * rgb_rep)

    # normalize
    acc_w_safe = acc_w.clamp_min(1e-8)
    out_rgb = (acc_rgb / acc_w_safe[:, None]).view(H, W, 3)
    out_d = (acc_d / acc_w_safe).view(H, W)
    out_cov = acc_w.view(H, W)

    mask = (out_cov > w_min).float()

    pr_rgb = out_rgb.permute(2, 0, 1).contiguous()          # [3,H,W]
    pr_mask = mask.unsqueeze(0).contiguous()                # [1,H,W]
    pr_depth = out_d.unsqueeze(0).contiguous()              # [1,H,W]
    pr_cov = out_cov.unsqueeze(0).contiguous()              # [1,H,W]
    return pr_rgb, pr_mask, pr_depth, pr_cov


def intrinsics_from_fov(W, H, FoVx, FoVy, device):
    # FoVx/FoVy 是弧度（你 Camera 里一般就是）
    fx = 0.5 * W / math.tan(0.5 * FoVx)
    fy = 0.5 * H / math.tan(0.5 * FoVy)
    cx = 0.5 * (W - 1)
    cy = 0.5 * (H - 1)
    return (torch.tensor(fx, device=device),
            torch.tensor(fy, device=device),
            torch.tensor(cx, device=device),
            torch.tensor(cy, device=device))

@torch.no_grad()
def normals_from_depth(depth, fx, fy, cx, cy, eps=1e-6):
    """
    depth: [H,W] torch tensor
    return: normals [3,H,W], valid_mask [1,H,W]
    """
    assert depth.dim() == 2
    H, W = depth.shape
    device = depth.device

    # valid depth
    valid = torch.isfinite(depth) & (depth > eps)

    # pixel grid
    u = torch.arange(W, device=device).float().view(1, W).expand(H, W)
    v = torch.arange(H, device=device).float().view(H, 1).expand(H, W)

    # backproject to camera space points
    Z = depth
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    P = torch.stack([X, Y, Z], dim=0)  # [3,H,W]

    # finite differences: dx along width, dy along height
    # dx: [3,H,W-1], dy: [3,H-1,W]
    dx = P[:, :, 1:] - P[:, :, :-1]
    dy = P[:, 1:, :] - P[:, :-1, :]

    # make same size by cropping to (H-1, W-1)
    dx = dx[:, :-1, :]     # [3,H-1,W-1]
    dy = dy[:, :, :-1]     # [3,H-1,W-1]

    # normal = cross(dy, dx)  (方向不重要，因为我们用 cos 相似度)
    n = torch.cross(dy.permute(1,2,0), dx.permute(1,2,0), dim=-1)  # [H-1,W-1,3]
    n = n.permute(2,0,1).contiguous()  # [3,H-1,W-1]

    n = F.normalize(n, dim=0, eps=eps)

    # pad back to [3,H,W]
    n = F.pad(n, (0,1,0,1), mode="replicate")  # right & bottom pad

    valid = valid.unsqueeze(0).float()  # [1,H,W]
    return n, valid

def masked_normal_cosine_loss(n1, n2, mask, eps=1e-6):
    """
    n1,n2: [3,H,W] normalized (or nearly)
    mask:  [1,H,W] in {0,1}
    """
    n1 = F.normalize(n1, dim=0, eps=eps)
    n2 = F.normalize(n2, dim=0, eps=eps)
    cos = (n1 * n2).sum(dim=0, keepdim=True).clamp(-1.0, 1.0)  # [1,H,W]
    loss_map = (1.0 - cos) * mask
    denom = mask.sum().clamp_min(1.0)
    return loss_map.sum() / denom