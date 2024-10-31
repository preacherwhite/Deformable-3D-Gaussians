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
from utils.loss_utils import l1_loss, ssim, kl_divergence, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel, DeformModelODE, DeformModelTORCHODE
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import random
import numpy as np
import time
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, base_model_path,gau_index):
    # tune parameters to fit in different sequence lengths
    full_iteration = opt.iterations
    opt.warm_up = 0
    opt.iterations = round((full_iteration - opt.warm_up) * (opt.sequence_length / 150) + opt.warm_up)
    opt.deform_lr_max_steps = opt.iterations
    opt.position_lr_max_steps = round((opt.position_lr_max_steps - opt.warm_up) * (opt.sequence_length / 150) + opt.warm_up)
    opt.densify_until_iter = round((opt.densify_until_iter - opt.warm_up) * (opt.sequence_length / 150) + opt.warm_up)

    print("Training for {} iterations".format(opt.iterations))
    print("Densifying until iteration {}".format(opt.densify_until_iter))
    print("Position learning rate max steps: {}".format(opt.position_lr_max_steps))
    print("Deform learning rate max steps: {}".format(opt.deform_lr_max_steps))

    #start training
    tb_writer = prepare_output_and_logger(dataset)
    #gaussians = GaussianModel(dataset.sh_degree)
    gaussians = GaussianModel(dataset.sh_degree)
    
    print("Using ODE for deformation")
    if dataset.use_torch_ode:
        deform = DeformModelTORCHODE(dataset.is_blender, dataset.is_6dof, D=dataset.D, W=dataset.W, input_ch=dataset.input_ch, output_ch=dataset.output_ch, multires=dataset.multires)
    elif dataset.is_ode:
        deform = DeformModelODE(dataset.is_blender, dataset.is_6dof, D=dataset.D, W=dataset.W, input_ch=dataset.input_ch, output_ch=dataset.output_ch, multires=dataset.multires, use_emb=dataset.use_emb, skips = None)
    else:
        deform = DeformModel(dataset.is_blender, dataset.is_6dof, D=dataset.D, W=dataset.W, input_ch=dataset.input_ch, output_ch=dataset.output_ch, multires=dataset.multires)
    print("Using DeformModel")
    deform_baseline = DeformModel(dataset.is_blender, dataset.is_6dof, D=dataset.D, W=dataset.W, input_ch=dataset.input_ch, output_ch=dataset.output_ch, multires=dataset.multires)
    deform.train_setting(opt)

    temp_model_path = dataset.model_path
    dataset.model_path = base_model_path

    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    dataset.model_path = temp_model_path
    scene.model_path = temp_model_path
    #gaussians.training_setup(opt)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras()
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0


    batch_size = opt.num_cams_per_iter
    non_warmup_iterations = opt.iterations - opt.warm_up
    iterations = opt.warm_up + (non_warmup_iterations // batch_size)
    progress_bar = tqdm(range(iterations), desc="Training progress")

    testing_iterations = [opt.warm_up + ((it - opt.warm_up) // batch_size)for it in testing_iterations]
    
    saving_iterations = [opt.warm_up + ((it - opt.warm_up)  // batch_size) for it in saving_iterations]
    
    #smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)


    print("\n[ITER {}] Saving Gaussians".format(0))
    scene.save(0)
    deform.save_weights(dataset.model_path, 0)
    
        # get the min and max fid for all cameras
    min_fid = min([viewpoint_cam.fid for viewpoint_cam in viewpoint_stack])
    max_fid = max([viewpoint_cam.fid for viewpoint_cam in viewpoint_stack])

    for iteration in range(1, (iterations) + 1):
        
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Randomly sample values of fid (float) witin the range of min_fid and max_fid
        # sampled_fids = [random.uniform(min_fid, max_fid) for _ in range(batch_size)]
        # sampled_fids.sort()

        sampled_fids = torch.linspace(min_fid.item(), max_fid.item(), steps=batch_size, device="cuda")
        
        if gau_index != -1:
            # Select only one Gaussian if train_single_gaussian is True
            if iteration == 1: 
                print(f"Training on single Gaussian at index: {gau_index}")
            
            d_xyz_list, d_rotation_list, d_scaling_list = deform.step(gaussians.get_xyz[gau_index:gau_index+1], sampled_fids)   

            with torch.no_grad():
                d_xyz_list_baseline, d_rotation_list_baseline, d_scaling_list_baseline = deform_baseline.step(gaussians.get_xyz[gau_index:gau_index+1], sampled_fids)

        else:
            # Original behavior for all Gaussians
            d_xyz_list, d_rotation_list, d_scaling_list = deform.step(gaussians.get_xyz, sampled_fids)
            with torch.no_grad():
                d_xyz_list_baseline, d_rotation_list_baseline, d_scaling_list_baseline = deform_baseline.step(gaussians.get_xyz, sampled_fids)
        
        loss = 0.0

        if gau_index != -1:
            gau_mean = gaussians.get_xyz[gau_index:gau_index+1]
        else:
            gau_mean = gaussians.get_xyz


        d_xyz_baseline = torch.stack(d_xyz_list_baseline, dim=0)
        
        if dataset.is_ode:
            extended_gau_mean = gau_mean.repeat(d_xyz_list.shape[0], 1, 1)
            loss = l1_loss( d_xyz_list,d_xyz_baseline.detach().clone() + extended_gau_mean)
        else:
            d_xyz_list = torch.stack(d_xyz_list, dim=0)
            extended_gau_mean = gau_mean.repeat(d_xyz_list.shape[0], 1, 1)
            loss = l1_loss( d_xyz_list+ extended_gau_mean ,d_xyz_baseline.detach().clone() + extended_gau_mean)

        loss.backward()
        # Optimizer step
        deform.optimizer.step()
        deform.optimizer.zero_grad()
        deform.update_learning_rate(iteration)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            
            cur_psnr = training_report(tb_writer, iteration, loss, loss, l2_loss, iter_start.elapsed_time(iter_end),
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
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), [fid])
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
                        default=list(range(1000, 150001, 2000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=list(range(1000, 150001, 2000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--base_model_path", type=str, default="")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--gau_index", type=int, default = -1)
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
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.base_model_path, args.gau_index)

    # All done
    print("\nTraining complete.")
