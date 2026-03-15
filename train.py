#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import cv2
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.depth_utils import estimate_depth, load_depth_model
import torch.nn.functional as F
from pseudo import load_ply_xyz_rgb, point_render_one_view, intrinsics_from_fov, normals_from_depth, \
    masked_normal_cosine_loss
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

depth_model = load_depth_model('vitl')


def safe_pearson(x, y, eps=1e-6):
    x = x - x.mean()
    y = y - y.mean()
    vx = torch.sum(x * x)
    vy = torch.sum(y * y)
    if vx < eps or vy < eps:
        return torch.tensor(0.0, device=x.device)
    return torch.sum(x * y) / (torch.sqrt(vx * vy) + eps)


def load_preprocessed_masks(scene_name, preprocessed_mask_dir, resolution_factor, mask_param=None):
    mask_cache = {}

    if mask_param is not None:
        mask_dir_name = f"preprocessed_masks_{mask_param}"
    else:
        mask_dir_name = "preprocessed_masks"

    base_mask_dir = os.path.join(os.path.dirname(preprocessed_mask_dir), mask_dir_name)
    res_dir = os.path.join(base_mask_dir, scene_name, f"r{resolution_factor}")
    mask_files = [f for f in os.listdir(res_dir) if f.endswith('.pt')]
    print(f"Loading {len(mask_files)} preprocessed masks from {res_dir}")

    for mask_file in mask_files:
        image_name = mask_file.replace('.pt', '')
        mask_path = os.path.join(res_dir, mask_file)
        mask_tensor = torch.load(mask_path, map_location='cuda')
        mask_cache[image_name] = mask_tensor

    return mask_cache


def apply_mask_to_image(image, mask):
    mask_expanded = mask.unsqueeze(0).expand_as(image)
    masked_image = image * mask_expanded
    return masked_image


def masked_l1(render, gt, mask):
    """
    render, gt: [C,H,W] 或 [1,C,H,W]
    mask: [H,W] / [1,H,W] / [1,1,H,W]，值域0/1
    返回按mask归一化后的L1标量
    """
    if render.dim() == 3:
        render = render.unsqueeze(0)
    if gt.dim() == 3:
        gt = gt.unsqueeze(0)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif mask.dim() == 3:  # [1,H,W]
        mask = mask.unsqueeze(1)  # [1,1,H,W]

    if mask.shape[1] == 1 and render.shape[1] == 3:
        mask3 = mask.repeat(1, 3, 1, 1)
    else:
        mask3 = mask

    diff = (render - gt).abs() * mask3
    denom = mask3.sum().clamp_min(1.0)
    return diff.sum() / denom


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, depth_model):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, depth_model=depth_model)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack, pseudo_stack = None, None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    scene_name = os.path.basename(dataset.source_path)
    mask_cache = {}

    if opt.lambda_far > 0:
        preprocessed_mask_dir = "./preprocessed_masks"
        resolution_factor = dataset.resolution
        mask_param = getattr(args, 'mask_param', None) if 'args' in globals() else None
        mask_cache = load_preprocessed_masks(scene_name, preprocessed_mask_dir, resolution_factor, mask_param)
    #     if opt.lambda_far > 0:
    #         preprocessed_mask_dir = "./preprocessed_masks"

    #         # ===== 从 args 中识别 images_2 / images_4 =====
    #         images_dir = getattr(args, "images", None)
    #         if images_dir is None:
    #             images_dir = getattr(args, "image_dir", "")

    #         if images_dir is None:
    #             images_dir = ""

    #         if "images_4" in images_dir:
    #             resolution_factor = 4
    #         elif "images_2" in images_dir:
    #             resolution_factor = 2
    #         else:
    #             resolution_factor = 1

    #         mask_param = getattr(args, 'mask_param', None)

    #         mask_cache = load_preprocessed_masks(
    #             scene_name,
    #             preprocessed_mask_dir,
    #             resolution_factor,
    #             mask_param
    #         )

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # pseudo cameras
    pseudo_cams = scene.getPseudoCameras().copy()  # 你 Scene 里就是这么取的 :contentReference[oaicite:6]{index=6}

    # 点云路径：和 dataset_readers 一致 :contentReference[oaicite:7]{index=7}
    ply_path = os.path.join(args.source_path, f"{args.n_views}_views/dense/fused.ply")
    xyz, rgb = load_ply_xyz_rgb(ply_path)

    # 预渲染 K 张 point render（建议先 K=32 或更小）
    K = min(len(pseudo_cams), getattr(args, "point_pseudo_num", 32))
    pseudo_cams = pseudo_cams[:K]

    point_render_maps = []
    point_render_masks = []
    point_render_depths = []
    point_render_covs = []

    for cam in pseudo_cams:
        pr, m, pd, cov = point_render_one_view(
            cam, xyz, rgb, device="cuda",
            radius=getattr(args, "point_radius", 2),
            sigma=getattr(args, "point_sigma", 1.0),
            z_weight=getattr(args, "point_z_weight", 0.8),
            max_points=getattr(args, "point_max_points", 200000),
        )

        point_render_maps.append(pr)  # [3,H,W]
        point_render_masks.append(m)  # [1,H,W]
        point_render_depths.append(pd)  # [1,H,W]
        point_render_covs.append(cov)  # [1,H,W]

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if iteration == 1 or iteration % 500 == 0:
            gaussians.update_density_score()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, is_train=True, iteration=iteration,
                            drop_min=opt.drop_min, drop_max=opt.drop_max)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

        with torch.no_grad():
            N = visibility_filter.shape[0]
            if gaussians.visibility_counter.shape[0] < N:
                new_counter = torch.zeros(N, device="cuda")
                new_counter[:gaussians.visibility_counter.shape[0]] = gaussians.visibility_counter
                gaussians.visibility_counter = new_counter
            gaussians.visibility_counter[:visibility_filter.shape[0]] += visibility_filter.float()

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        rendered_depth = render_pkg["depth"][0]
        midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda()
        rendered_depth = rendered_depth.reshape(-1, 1)
        midas_depth = midas_depth.reshape(-1, 1)

        depth_loss = torch.min(
            1 - safe_pearson(-midas_depth, rendered_depth),
            1 - safe_pearson(1 / (midas_depth + 200.), rendered_depth)
        )

        far_mask = None
        if opt.lambda_far > 0 and viewpoint_cam.image_name in mask_cache:
            far_mask = mask_cache[viewpoint_cam.image_name].to(gt_image.device).float()
        far_loss = 0.0
        if far_mask is not None and opt.lambda_far > 0:
            gt_far = apply_mask_to_image(gt_image, far_mask)
            render_far = apply_mask_to_image(image, far_mask)
            far_loss = l1_loss(render_far, gt_far)

        loss += args.depth_weight * depth_loss + opt.lambda_far * far_loss
        # loss += args.depth_weight * depth_loss
        # loss += opt.lambda_far * far_loss

        if iteration % args.sample_pseudo_interval == 0 and iteration > args.start_sample_pseudo and iteration < args.end_sample_pseudo:
            if not pseudo_stack:
                # pseudo_stack = scene.getPseudoCameras().copy()
                pseudo_stack = list(enumerate(pseudo_cams))
            # pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
            pseudo_idx, pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))

            render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
            rendered_depth_pseudo = render_pkg_pseudo["depth"][0]
            midas_depth_pseudo = estimate_depth(render_pkg_pseudo["render"], depth_model, mode='train')

            rendered_depth_pseudo = rendered_depth_pseudo.reshape(-1, 1)
            midas_depth_pseudo = midas_depth_pseudo.reshape(-1, 1)
            depth_loss_pseudo = (1 - safe_pearson(rendered_depth_pseudo, -midas_depth_pseudo)).mean()

            if torch.isnan(depth_loss_pseudo).sum() == 0:
                loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
                loss += loss_scale * args.depth_pseudo_weight * depth_loss_pseudo

            loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
            pseudo_rgb = render_pkg_pseudo["render"]  # [3,H,W]
            point_rgb = point_render_maps[pseudo_idx]  # [3,H,W]
            mask = point_render_masks[pseudo_idx]  # [1,H,W]
            point_d = point_render_depths[pseudo_idx]  # [1,H,W]

            # gaussian depth map (把你已有的 [H,W] reshape回来)
            gs_d = render_pkg_pseudo["depth"][0].unsqueeze(0)  # [1,H,W]

            # 深度门控：只在两者接近的地方做点一致性（阈值用相对值更稳）
            tau = getattr(args, "point_depth_gate", 0.05)  # 5% 相对阈值（可调）
            ref = gs_d.median().clamp_min(1e-6)
            gate = (torch.abs(gs_d - point_d) < tau * ref).float()

            final_mask = mask * gate

            diff = (pseudo_rgb - point_rgb) * final_mask
            l1_pr = diff.abs().sum() / (final_mask.sum() * 3.0 + 1e-6)

            loss += loss_scale * args.point_pseudo_weight * l1_pr

            # ============================
            # Save pseudo-view debug images
            # ============================
            if getattr(args, "save_pseudo_debug", False) and (
                    iteration % getattr(args, "pseudo_debug_interval", 200) == 0):

                out_dir = os.path.join(args.model_path, "pseudo_debug")
                os.makedirs(out_dir, exist_ok=True)

                def _to_bgr_uint8(img3chw):
                    # img: [3,H,W], range ~ [0,1]
                    x = img3chw.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    x = (x * 255.0).astype(np.uint8)
                    return x[:, :, ::-1]  # RGB -> BGR

                def _mask_to_gray(mask1hw):
                    # mask: [1,H,W] in {0,1} or [0,1]
                    m = mask1hw.detach().clamp(0, 1)[0].cpu().numpy()
                    return (m * 255.0).astype(np.uint8)

                def _depth_to_colormap(depth1hw):
                    d = depth1hw.detach().cpu().numpy()[0]
                    valid = d[np.isfinite(d)]
                    if valid.size < 10:
                        d_norm = np.zeros_like(d, dtype=np.uint8)
                    else:
                        lo = np.percentile(valid, 1)
                        hi = np.percentile(valid, 99)
                        d = np.clip((d - lo) / (hi - lo + 1e-12), 0, 1)
                        d_norm = (d * 255).astype(np.uint8)
                    return cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)

                # 1) save RGBs
                cv2.imwrite(os.path.join(out_dir, f"{iteration:05d}_pseudo_student.jpg"), _to_bgr_uint8(pseudo_rgb))
                cv2.imwrite(os.path.join(out_dir, f"{iteration:05d}_pseudo_teacher.jpg"), _to_bgr_uint8(point_rgb))

                # 2) save masks (raw + colored)
                mask_gray = _mask_to_gray(mask)
                gate_gray = _mask_to_gray(gate)
                final_gray = _mask_to_gray(final_mask)

                cv2.imwrite(os.path.join(out_dir, f"{iteration:05d}_mask_raw.png"), mask_gray)
                cv2.imwrite(os.path.join(out_dir, f"{iteration:05d}_mask_gate.png"), gate_gray)
                cv2.imwrite(os.path.join(out_dir, f"{iteration:05d}_mask_final.png"), final_gray)

                cv2.imwrite(os.path.join(out_dir, f"{iteration:05d}_mask_final_jet.png"),
                            cv2.applyColorMap(final_gray, cv2.COLORMAP_JET))

                # 3) save depth visualization
                cv2.imwrite(os.path.join(out_dir, f"{iteration:05d}_depth_gs.jpg"), _depth_to_colormap(gs_d))
                cv2.imwrite(os.path.join(out_dir, f"{iteration:05d}_depth_point.jpg"), _depth_to_colormap(point_d))

                # 4) save a combined grid (very useful for paper/debug)
                row0 = np.concatenate([_to_bgr_uint8(pseudo_rgb), _to_bgr_uint8(point_rgb)], axis=1)
                row1 = np.concatenate([
                    cv2.applyColorMap(final_gray, cv2.COLORMAP_JET),
                    _depth_to_colormap(gs_d)
                ], axis=1)
                grid = np.concatenate([row0, row1], axis=0)

                cv2.imwrite(os.path.join(out_dir, f"{iteration:05d}_grid.jpg"), grid)
            # ---------- Pseudo-Real Reprojection Consistency (Geometry-backed) ----------
            if getattr(args, "prrc_weight", 0.0) > 0:

                device = render_pkg["render"].device
                H, W = viewpoint_cam.image_height, viewpoint_cam.image_width

                # Real view (student render & GT)
                I_gt = gt_image  # [3,H,W]
                D_real = render_pkg["depth"][0]  # [H,W]

                # Pseudo side teacher: point render (geometry-backed)
                I_pseudo_teacher = point_render_maps[pseudo_idx].to(device)  # [3,H,W]
                M_pseudo_teacher = point_render_masks[pseudo_idx].to(device)  # [1,H,W]

                # --- 1) sample valid pixels on real view ---
                # valid depth mask
                valid = (D_real > 1e-6).view(-1)
                valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
                if valid_idx.numel() > 0:

                    N = int(getattr(args, "prrc_sample_num", 4096))
                    if valid_idx.numel() > N:
                        perm = torch.randperm(valid_idx.numel(), device=device)[:N]
                        sample_idx = valid_idx[perm]
                    else:
                        sample_idx = valid_idx

                    # pixel (u,v)
                    u = (sample_idx % W).float()
                    v = (sample_idx // W).float()
                    z = D_real.view(-1)[sample_idx]  # [N]

                    # --- 2) backproject to real camera coordinates using FoV intrinsics ---
                    # fx, fy from FoV (pinhole)
                    fx = 0.5 * W / torch.tan(torch.tensor(viewpoint_cam.FoVx * 0.5, device=device))
                    fy = 0.5 * H / torch.tan(torch.tensor(viewpoint_cam.FoVy * 0.5, device=device))
                    cx = (W - 1) * 0.5
                    cy = (H - 1) * 0.5

                    x_cam = (u - cx) / fx * z
                    y_cam = -(v - cy) / fy * z  # NOTE: minus to match OpenGL-style y-up
                    pts_cam = torch.stack([x_cam, y_cam, z], dim=-1)  # [N,3]

                    # cam -> world  (your code uses row-vector convention)
                    ones = torch.ones((pts_cam.shape[0], 1), device=device)
                    pts_cam_h = torch.cat([pts_cam, ones], dim=-1)  # [N,4]
                    w2c_real = viewpoint_cam.world_view_transform.t()  # [4,4]
                    c2w_real = torch.linalg.inv(w2c_real)  # [4,4]
                    pts_world_h = pts_cam_h @ c2w_real
                    pts_world = pts_world_h[:, :3]

                    # --- 3) project world points into pseudo view using full_proj_transform ---
                    P_pseudo = pseudo_cam.full_proj_transform.t()  # [4,4]
                    pts_world_h = torch.cat([pts_world, ones], dim=-1)  # [N,4]
                    clip = pts_world_h @ P_pseudo  # [N,4]
                    ndc = clip[:, :3] / (clip[:, 3:4] + 1e-6)  # [N,3]

                    u2 = (ndc[:, 0] + 1) * 0.5 * (W - 1)
                    v2 = (1 - ndc[:, 1]) * 0.5 * (H - 1)

                    # --- 4) grid_sample teacher on pseudo view ---
                    gx = (u2 / (W - 1)) * 2 - 1
                    gy = (v2 / (H - 1)) * 2 - 1
                    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)  # [1,N,1,2]

                    pseudo_col = torch.nn.functional.grid_sample(
                        I_pseudo_teacher.unsqueeze(0), grid, align_corners=False
                    )[0].squeeze(-1)  # [3,N]

                    pseudo_msk = torch.nn.functional.grid_sample(
                        M_pseudo_teacher.unsqueeze(0), grid, align_corners=False
                    )[0].squeeze(-1)  # [1,N]

                    # --- 5) fetch GT colors on real view (same sampled pixels) ---
                    gt_col = I_gt.view(3, -1)[:, sample_idx]  # [3,N]

                    # valid mask: inside image + positive depth + teacher coverage
                    in_img = (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)
                    cov = (pseudo_msk[0] > 0.5)
                    good = in_img & cov

                    if good.any():
                        prrc_loss = (gt_col[:, good] - pseudo_col[:, good]).abs().mean()
                        # warmup scale (keep consistent with your other pseudo losses)
                        loss_scale = min((iteration - args.start_sample_pseudo) / 500.0, 1.0)
                        loss += loss_scale * args.prrc_weight * prrc_loss

        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss

        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold,
                                                opt.opacity_cull, scene.cameras_extent,
                                                size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                                   0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name),
                                             depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name),
                                                 rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name),
                                                 surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name),
                                                 rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name),
                                                 rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2000, 5000, 7_000, 10000, 15000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--save_pseudo_debug", action="store_true")
    parser.add_argument("--pseudo_debug_interval", type=int, default=2000)

    parser.add_argument("--point_pseudo_weight", type=float, default=0.1)
    parser.add_argument("--point_pseudo_num", type=int, default=32)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--point_sigma", type=float, default=1.0)
    parser.add_argument("--point_z_weight", type=float, default=0.8)
    parser.add_argument("--point_max_points", type=int, default=200000)
    parser.add_argument("--point_depth_gate", type=float, default=0.15)
    parser.add_argument("--normal_pseudo_weight", type=float, default=0.02)
    parser.add_argument("--normal_teacher_beta", type=float, default=0.7)  # teacher_d = beta*pt + (1-beta)*mono

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, depth_model=depth_model)

    # All done
    print("\nTraining complete.")
