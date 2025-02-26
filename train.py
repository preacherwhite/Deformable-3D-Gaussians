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
from scene import Scene, GaussianModel, DeformModel, DeformModelODE, DeformModelTORCHODEStart, DeformModelBaseline
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import random
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, base_model_path):
    # tune parameters to fit in different sequence lengths
    opt.iterations = opt.iterations
    opt.deform_lr_max_steps = opt.iterations

    opt.use_iterative_update = getattr(opt, 'use_iterative_update', False)
    opt.iterative_update_decay = getattr(opt, 'iterative_update_decay', 0.9)
    opt.iterative_update_interval = getattr(opt, 'iterative_update_interval', 1000)
    opt.max_training_switches = getattr(opt, 'max_training_switches', 6)

    print("Training for {} iterations".format(opt.iterations))
    print("Densifying until iteration {}".format(opt.densify_until_iter))
    print("Position learning rate max steps: {}".format(opt.position_lr_max_steps))
    print("Deform learning rate max steps: {}".format(opt.deform_lr_max_steps))
    print(f"Max training switches before simultaneous optimization: {opt.max_training_switches}")

    #start training
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)    

    if dataset.is_ode:
        if dataset.use_torch_ode:
            print("Using TORCH ODE for deformation")
            deform = DeformModelTORCHODEStart(dataset.is_blender, dataset.is_6dof, D=dataset.D, W=dataset.W, input_ch=dataset.input_ch, output_ch=dataset.output_ch,
                                 multires=dataset.multires) 
        else:
            print("Using ODE for deformation")
            deform = DeformModelODE(dataset.is_blender, dataset.is_6dof, D=dataset.D, W=dataset.W, input_ch=dataset.input_ch, output_ch=dataset.output_ch,
                                 multires=dataset.multires, scale_lr = opt.scale_lr, use_linear=dataset.use_linear, use_emb=dataset.use_emb, rtol=opt.rtol, atol=opt.atol, output_scale = dataset.output_scale) 
    else:
        print("Using DeformModelBaseline")
        deform = DeformModelBaseline(dataset.is_blender, dataset.is_6dof, D=dataset.D, W=dataset.W, input_ch=dataset.input_ch, output_ch=dataset.output_ch, multires=dataset.multires)
    if opt.use_iterative_update:
        print("Using iterative update with decay factor: {}".format(opt.iterative_update_decay))
        print("Iterative update interval: {} iterations".format(opt.iterative_update_interval))
    deform.train_setting(opt)
    
    if(base_model_path != ""):
        print("Loading base model from: {}".format(base_model_path))
        temp_model_path = dataset.model_path
        dataset.model_path = base_model_path
        scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
        dataset.model_path = temp_model_path
    else:
        scene = Scene(dataset, gaussians)
        gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0


    batch_size = opt.num_cams_per_iter
    non_warmup_iterations = opt.iterations - opt.warm_up
    iterations = opt.warm_up + (non_warmup_iterations // batch_size)
    progress_bar = tqdm(range(iterations), desc="Training progress")
    saving_iterations.append(iterations)
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

    update_gaussians = True
    update_deform = True
    next_switch_iteration = opt.warm_up + opt.iterative_update_interval
    current_interval = opt.iterative_update_interval
    switch_count = 0
    simultaneous_optimization = False
    for iteration in range(1, (iterations) + 1):
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack or iteration == opt.warm_up:
            viewpoint_stack = scene.getTrainCameras().copy()
            # Print all fids in the viewpoint stack
            if iteration >= opt.warm_up:
                # Sort the viewpoints by fid
                # TODO: Remove the first viewpoint from the stack due to repeated fit, check later
                viewpoint_stack = sorted(viewpoint_stack, key=lambda x: x.fid)[1:]
                # Calculate the total number of viewpoints
                total_viewpoints = len(viewpoint_stack)
                
                # Calculate the step size for uniform distribution
                step = (total_viewpoints - 1) / (opt.sequence_length - 1)
                
                # Select uniformly spread out viewpoints
                if opt.spread_out_sequence:
                    selected_indices = [int(round(i * step)) for i in range(opt.sequence_length)]
                    viewpoint_stack = [viewpoint_stack[i] for i in selected_indices]
                else:
                    viewpoint_stack = viewpoint_stack[:opt.sequence_length]
                    

        total_frame = len(viewpoint_stack)

        k = min(opt.num_cams_per_iter, len(viewpoint_stack))
        if iteration < opt.warm_up:
            sampled_indices = random.sample(range(len(viewpoint_stack)), 1)
            sampled_indices.sort()
            sampled_cams =  [viewpoint_stack[i] for i in sampled_indices]
            viewpoint_stack = [cam for i, cam in enumerate(viewpoint_stack) if i not in sampled_indices]
            d_xyz_list = [0.0 for _ in sampled_cams]
            d_rotation_list = [0.0 for _ in sampled_cams]
            d_scaling_list = [0.0 for _ in sampled_cams]
        else:
            sampled_indices = random.sample(range(len(viewpoint_stack)), k)
            sampled_indices.sort()  # Sort to maintain temporal order
            sampled_cams = [viewpoint_stack[i] for i in sampled_indices]
            
            # Remove the sampled cameras from the stack
            viewpoint_stack = [cam for i, cam in enumerate(viewpoint_stack) if i not in sampled_indices]
            sampled_fids = [viewpoint_cam.fid for viewpoint_cam in sampled_cams]
            # N = gaussians.get_xyz.shape[0]
            # time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = 0 #if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)

            # Deform is called to calculate gaussian parameters at time t according to the current viewpoint camera. We need to change this so 
            # A series of t from a series of viewpoints are used to calculate the gaussian parameters at different time frames

            #TODO: Why the detach? Need to experiment without it. (Note: 4D Gaussian paper does not have the stop gradient operation)
            
            max_batch_gaussians = opt.max_batch_gaussians 
            if max_batch_gaussians > 0:
                num_gaussians = gaussians.get_xyz.shape[0]
                num_batches = (num_gaussians + max_batch_gaussians - 1) // max_batch_gaussians # Ceiling division
                
                if dataset.use_torch_ode:
                    #can't use detach here because we won't be using gaussians later in training
                    sampled_fids = torch.Tensor(sampled_fids).repeat(max_batch_gaussians,1).to(gaussians.get_xyz.device)
                # Initialize lists to store results
                d_xyz_list_batched = []
                d_rotation_list_batched = []
                d_scaling_list_batched = []
                
                # Process gaussians in batches
                for i in range(num_batches):
                    start_idx = i * max_batch_gaussians
                    end_idx = min((i + 1) * max_batch_gaussians, num_gaussians)
                    
                    batch_xyz = gaussians.get_xyz[start_idx:end_idx]
                    
                    if dataset.use_torch_ode:
                        d_xyz, d_rotation, d_scaling = deform.step(batch_xyz, sampled_fids, batch_xyz)
                    else:
                        d_xyz, d_rotation, d_scaling = deform.step(batch_xyz, sampled_fids)
                    
                    # Append batch results
                    d_xyz_list_batched.append(d_xyz)
                    d_rotation_list_batched.append(d_rotation)
                    d_scaling_list_batched.append(d_scaling)
                
                # Concatenate results from all batches
                d_xyz_list = [torch.cat([batch[i] for batch in d_xyz_list_batched], dim=0) for i in range(len(d_xyz_list_batched[0]))]
                d_rotation_list = [torch.cat([batch[i] for batch in d_rotation_list_batched], dim=0) for i in range(len(d_rotation_list_batched[0]))]
                d_scaling_list = [torch.cat([batch[i] for batch in d_scaling_list_batched], dim=0) for i in range(len(d_scaling_list_batched[0]))]
            else:
                if dataset.use_torch_ode:
                    d_xyz_list, d_rotation_list, d_scaling_list = deform.step(gaussians.get_xyz, sampled_fids, gaussians.get_xyz)
                elif dataset.is_ode:
                    d_xyz_list, d_rotation_list, d_scaling_list = deform.step(gaussians.get_xyz.detach(), sampled_fids)
                else:
                    # expand the sampled_fids to match the shape of gaussians.get_xyz
                    # for baseline we will always use batch size 1
                    sampled_fids = sampled_fids[0].unsqueeze(0).expand(gaussians.get_xyz.shape[0], -1)
                    d_xyz_list, d_rotation_list, d_scaling_list = deform.step(gaussians.get_xyz.detach(), sampled_fids)
                    d_xyz_list = [d_xyz_list[0]]
                    d_rotation_list = [d_rotation_list[0]]
                    d_scaling_list = [d_scaling_list[0]]
                    # print(f"gaussians.get_xyz: {gaussians.get_xyz.shape}")
                    # print(f"d_xyz_list: {d_xyz_list.shape}, d_rotation_list: {d_rotation_list.shape}, d_scaling_list: {d_scaling_list.shape}")
        
        
        loss = 0.0
        Ll1 = 0.0
        visibility_filter_list = []
        render_pkg_re_list = []
        
        if len(sampled_cams) != len(d_xyz_list):
            print(f"Length mismatch: sampled_cams ({len(sampled_cams)}) != d_xyz_list ({len(d_xyz_list)})")
            assert False
        for viewpoint_cam, d_xyz, d_rotation, d_scaling in zip(sampled_cams, d_xyz_list, d_rotation_list, d_scaling_list):
            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device()
            fid = viewpoint_cam.fid

            # Render
            if iteration < opt.warm_up: 
                render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
            else:
                render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof, direct_compute = opt.direct_compute)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
                "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
            # depth = render_pkg_re["depth"]
            if base_model_path == "":
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                    radii[visibility_filter])
                visibility_filter_list.append(visibility_filter)
                render_pkg_re_list.append(render_pkg_re)

        # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 += l1_loss(image, gt_image)
            # accumulate loss for each viewpoint
            loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
        iter_end.record()
        loss /= len(sampled_cams)
        Ll1 /= len(sampled_cams)
        loss.backward()
        
        for viewpoint_cam in sampled_cams:
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
            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof, seq_length=opt.sequence_length, is_ode=dataset.is_ode)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(dataset.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter and base_model_path == "":
                for visibility_filter, render_pkg_re in zip(visibility_filter_list, render_pkg_re_list):
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
                if opt.use_iterative_update and iteration >= opt.warm_up and not simultaneous_optimization:
                    if iteration >= next_switch_iteration:
                        update_gaussians = not update_gaussians
                        update_deform = not update_deform
                        current_interval = max(int(current_interval * opt.iterative_update_decay), 5)
                        next_switch_iteration = iteration + current_interval
                        switch_count += 1
                        print(f"Switching update mode at iteration {iteration}. Next switch at {next_switch_iteration}")
                        
                        if switch_count >= opt.max_training_switches or current_interval <= 5:
                            simultaneous_optimization = True
                            print(f"Reached {opt.max_training_switches} switches. Switching to simultaneous optimization.")

                    if update_gaussians:
                        gaussians.optimizer.step()
                        gaussians.update_learning_rate(iteration)
                    if update_deform:
                        deform.optimizer.step()
                        deform.update_learning_rate(iteration)
                else:
                    # Simultaneous optimization or original update logic
                    if base_model_path == "":
                        gaussians.optimizer.step()
                        gaussians.update_learning_rate(iteration)
                    deform.optimizer.step()
                    deform.update_learning_rate(iteration)

                # Zero gradients
                if base_model_path == "":
                    gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()


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
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False, seq_length=150, is_ode=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()[:seq_length]},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras()[:seq_length])] for idx in
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
                    #time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)

                    if is_ode:
                        d_xyz, d_rotation, d_scaling = deform.step(xyz, [fid])

                        image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof, direct_compute=is_ode)["render"],
                        0.0, 1.0)
                    else:
                        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                        d_xyz, d_rotation, d_scaling = deform.step(xyz, time_input)
                        image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz[0], d_rotation[0], d_scaling[0], is_6dof, direct_compute=is_ode)["render"],
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
                        default=[3005,7000, 8000, 9000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--base_model_path", type=str, default="")
    args = parser.parse_args(sys.argv[1:])
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    print("also saving at itration :{}".format(args.iterations))
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.base_model_path)

    # All done
    print("\nTraining complete.")
