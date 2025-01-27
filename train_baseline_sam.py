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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModelBaseline
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
def initialize_sam_model():
    """Initialize and return a SAM model on a specified GPU."""
    checkpoint = "/media/staging2/dhwang/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg = "sam2.1_hiera_b+.yaml"
    device = "cuda"
    
    sam_model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam_model, points_per_side=8,points_per_batch=128)
    return mask_generator
import torch

def compute_mask_regularization(viewpoint_cam, d_xyz, d_rotation, d_scaling, current_sam_mask, gaussians, max_points=5000):
    """
    Compute a mask-based regularization by projecting gaussians into the camera image space and 
    applying a penalty on deformation parameters (d_xyz, d_rotation, d_scaling) for gaussians 
    that fall into one or more SAM masks (each corresponding to a different object).
    
    current_sam_mask: List[Tensor], each entry is a binary mask for an object: shape (H, W)
    gaussians: object with gaussians.get_xyz attribute (N,3)
    d_xyz, d_rotation, d_scaling: deformation outputs, shape (N, ...)
    """

    device = viewpoint_cam.full_proj_transform.device

    # current_sam_mask is a list of masks: each mask is HxW
    mask_list = torch.from_numpy(current_sam_mask).cuda()

    xyz = gaussians.get_xyz  # shape: N x 3
    N = xyz.shape[0]

    # Optionally limit number of points for performance
    if N > max_points:
        idx = torch.randperm(N, device=device)[:max_points]
        xyz_subset = xyz[idx]
        d_xyz_subset = d_xyz[idx]
        d_rotation_subset = d_rotation[idx]
        d_scaling_subset = d_scaling[idx]
    else:
        idx = torch.arange(N, device=device)
        xyz_subset = xyz
        d_xyz_subset = d_xyz
        d_rotation_subset = d_rotation
        d_scaling_subset = d_scaling

    # Project points into image space
    ones = torch.ones((xyz_subset.shape[0], 1), device=device)
    xyz_h = torch.cat([xyz_subset, ones], dim=1)  # Nx4

    p = xyz_h @ viewpoint_cam.full_proj_transform  # Nx4
    valid_mask = p[:, 3] > 0
    p = p[valid_mask]
    d_xyz_valid = d_xyz_subset[valid_mask]
    d_rotation_valid = d_rotation_subset[valid_mask]
    d_scaling_valid = d_scaling_subset[valid_mask]

    if p.shape[0] == 0:
        # No valid points after projection
        return torch.tensor(0.0, device=device)

    px = (p[:, 0] / p[:, 3]) * 0.5 + 0.5
    py = (p[:, 1] / p[:, 3]) * 0.5 + 0.5

    w = viewpoint_cam.image_width
    h = viewpoint_cam.image_height
    px = (px * w).long()
    py = (py * h).long()

    in_bounds = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    px = px[in_bounds]
    py = py[in_bounds]

    d_xyz_in = d_xyz_valid[in_bounds]
    d_rotation_in = d_rotation_valid[in_bounds]
    d_scaling_in = d_scaling_valid[in_bounds]

    if px.numel() == 0:
        # No points inside the image
        return torch.tensor(0.0, device=device)

    # For each mask, compute the regularization for the points that lie inside it
    total_reg_loss = torch.tensor(0.0, device=device)
    for mask in mask_list:
        # Sample the mask at the projected pixel locations
        sampled_mask_vals = mask[py, px]

        # Gaussians inside this mask
        masked_gaussians = sampled_mask_vals > 0.5

        d_xyz_masked = d_xyz_in[masked_gaussians]
        d_rotation_masked = d_rotation_in[masked_gaussians]
        d_scaling_masked = d_scaling_in[masked_gaussians]

        # Compute deviations from mean to encourage consistency
        # For xyz
        if d_xyz_masked.numel() > 0:
            mean_xyz = d_xyz_masked.mean(dim=0, keepdim=True)
            xyz_variance = ((d_xyz_masked - mean_xyz) ** 2).mean()
        else:
            xyz_variance = torch.tensor(0.0, device=device)

        # For rotation
        if d_rotation_masked.numel() > 0:
            mean_rot = d_rotation_masked.mean(dim=0, keepdim=True)
            rot_variance = ((d_rotation_masked - mean_rot) ** 2).mean()
        else:
            rot_variance = torch.tensor(0.0, device=device)

        # For scaling
        if d_scaling_masked.numel() > 0:
            mean_scl = d_scaling_masked.mean(dim=0, keepdim=True)
            scl_variance = ((d_scaling_masked - mean_scl) ** 2).mean()
        else:
            scl_variance = torch.tensor(0.0, device=device)

        # Aggregate the variance-based penalties
        mask_reg_loss = xyz_variance + rot_variance + scl_variance
        total_reg_loss += mask_reg_loss

    return total_reg_loss

def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModelBaseline(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    print(f"training on sequence_length: {opt.sequence_length}")
    viewpoint_stack = None
    sam_mask_stack = None  # Initialize SAM mask stack
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

    # initialize sam model and check for cached masks
    sam_model = initialize_sam_model()
    cache_dir = os.path.join(dataset.source_path, "sam_masks_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # make a list of sam_masks for each viewpoint
    sam_masks = []
    for viewpoint in tqdm(scene.getTrainCameras(), desc="Loading/Generating SAM masks"):
        cache_path = os.path.join(cache_dir, f"{viewpoint.image_name}_mask.npy")
        
        if os.path.exists(cache_path):
            # Load cached mask if it exists
            mask = np.load(cache_path, allow_pickle=True)
            sam_masks.append(mask)
        else:
            # Generate and cache mask if not found
            image = viewpoint.original_image.detach().cpu().numpy()
            image = image.transpose(1, 2, 0)
            mask = sam_model.generate(image)
            mask = [m['segmentation'] for m in mask]
            np.save(cache_path, mask)
            sam_masks.append(mask)

    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Reset both viewpoint and mask stacks when empty
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            sam_mask_stack = sam_masks.copy()
            # Get sorted indices based on fid
            sorted_indices = sorted(range(len(viewpoint_stack)), key=lambda i: viewpoint_stack[i].fid)
            # Sort both stacks using the indices
            viewpoint_stack = [viewpoint_stack[i] for i in sorted_indices]
            sam_mask_stack = [sam_masks[i] for i in sorted_indices]

            total_viewpoints = len(viewpoint_stack)
            step = (total_viewpoints - 1) / (opt.sequence_length - 1)
            
            # Select uniformly spread out viewpoints
            selected_indices = [int(round(i * step)) for i in range(opt.sequence_length)]
            viewpoint_stack = [viewpoint_stack[i] for i in selected_indices]
            sam_mask_stack = [sam_masks[i] for i in selected_indices]

        # Pop both viewpoint and corresponding mask
        idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(idx)
        current_sam_mask = sam_mask_stack.pop(idx)  # Pop the corresponding mask

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            # Deform is called to calculate gaussian parameters at time t according to the current viewpoint camera. We need to change this so 
            # A series of t from a series of viewpoints are used to calculate the gaussian parameters at different time frames
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)
        #print('d_xyz:', d_xyz.shape)
        # Render
        
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        # depth = render_pkg_re["depth"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        regular_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        mask_reg_weight = 0.5 # TODO: Adjust this weight as needed
        if iteration < opt.warm_up:
            mask_reg_loss = 0.0
        else:
            # Add mask regularization loss
            mask_reg_loss = compute_mask_regularization(
                viewpoint_cam, 
                d_xyz, 
                d_rotation, 
                d_scaling, 
                current_sam_mask,  # Use the popped mask
                gaussians
            )
        loss = regular_loss + mask_reg_weight * mask_reg_loss
        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
                gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


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


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40_000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")
