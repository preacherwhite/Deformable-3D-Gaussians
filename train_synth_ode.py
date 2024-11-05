import os
import torch
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from utils.loss_utils import l1_loss
from scene import DeformModelODE, DeformModelTORCHODE
import random
import numpy as np
import sys
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
def create_linear_trajectory(start_point, end_point, num_points):
    trajectory = torch.linspace(0, 1, num_points).unsqueeze(1).repeat(1, 3)
    trajectory = start_point.to("cuda") + (end_point.to("cuda") - start_point.to("cuda")) * trajectory.to("cuda")
    return trajectory

def create_trajectory_with_dimension_waves(start_point, end_point, num_points):
    t = torch.linspace(0, 1, num_points, device="cuda").unsqueeze(1)
    trajectory = start_point.to("cuda") + (end_point.to("cuda") - start_point.to("cuda")) * t

    # Apply different sine waves to each dimension
    sine_wave_x = 0.1 * torch.sin(2 * np.pi * 5 * t)  # Different frequency for x
    sine_wave_y = 0.05 * torch.sin(2 * np.pi * 5 * t) # Different frequency for y
    sine_wave_z = 0.02 * torch.sin(2 * np.pi * 5 * t) # Different frequency for z

    # Stack sine waves together and add to the trajectory
    sine_wave = torch.cat((sine_wave_x, sine_wave_y, sine_wave_z), dim=1)
    trajectory += sine_wave
    return trajectory

def create_quadratic_trajectory(start_point, end_point, num_points):
    t = torch.linspace(0, 1, num_points, device="cuda").unsqueeze(1)
    trajectory = start_point.to("cuda") + (end_point.to("cuda") - start_point.to("cuda")) * t**2
    return trajectory


def training(dataset, opt, saving_iterations, plot_interval, full_calculation):
    # Create output directory
    os.makedirs(dataset.model_path, exist_ok=True)

    # Initialize tensorboard writer
    tb_writer = SummaryWriter(dataset.model_path)

    # Create artificial 3D linear trajectory
    start_point = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    end_point = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    trajectory = create_trajectory_with_dimension_waves(start_point, end_point, opt.sequence_length)

    print(trajectory.shape)

    deform = DeformModelODE(dataset.is_blender, dataset.is_6dof, D=dataset.D, W=dataset.W, input_ch=dataset.input_ch, output_ch=dataset.output_ch, multires=dataset.multires, use_emb=dataset.use_emb, skips = None)
    #deform = DeformModelTORCHODE(dataset.is_blender, dataset.is_6dof, D=dataset.D, W=dataset.W, input_ch=dataset.input_ch, output_ch=dataset.output_ch, multires=dataset.multires)
    deform.train_setting(opt)

    def get_batch(trajectory, sequence_length, batch_size, device):
        # Select random starting indices for the batch
        if sequence_length - opt.num_cams_per_iter > 0:
            s = torch.from_numpy(np.random.choice(np.arange(sequence_length - opt.num_cams_per_iter, dtype=np.int64), batch_size, replace=False))
        else:
            raise ValueError("'a' cannot be empty unless no samples are taken")
        
        # Get the initial values for the batch
        batch_y0 = trajectory[s]  # (M, D)
        
        # Get the time steps for the batch, starting from each sequence location
        batch_t = (s.unsqueeze(-1).to(device) + torch.arange(opt.num_cams_per_iter, device=device)).float()  # (M, T)
        
        # Get the true values for the batch over the time steps
        batch_y = torch.stack([trajectory[s + i] for i in range(opt.num_cams_per_iter)], dim=0)  # (T, M, D)
        
        return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

    # Training loop
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1):
        # Sample random points along the trajectory using get_batch
        initial_point, sampled_fids, sampled_points = get_batch(trajectory, opt.sequence_length, 40, "cuda")
        
        d_xyz, _, _ = deform.step(initial_point, sampled_fids.float())
        # Compute loss (L1 distance from the true trajectory)
        loss = l1_loss(d_xyz, sampled_points)
        # Backward pass and optimization step
        loss.backward()
        deform.optimizer.step()
        deform.optimizer.zero_grad()
        deform.update_learning_rate(iteration)


        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)

        # Update progress bar
        progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
        progress_bar.update(1)

        # Plot 3D curves for d_xyz and sampled_points every k iterations
        if iteration % plot_interval == 0:
            # Generate full trajectory prediction in batches
            # Generate full trajectory prediction
            with torch.no_grad():
                if full_calculation:
                    # Start from initial point (frame 0)
                    initial_point = trajectory[0:1].to('cuda')
                    
                    # Generate time steps for full sequence
                    t = torch.arange(opt.sequence_length, device='cuda').float()
                    
                    # Get full trajectory prediction
                    predicted_xyz, _, _ = deform.step(initial_point, t)
                    
                    # Convert to numpy for plotting
                    predicted_trajectory = predicted_xyz.squeeze().cpu().numpy()
                    actual_trajectory = trajectory.cpu().numpy()
                else:
                    # Calculate number of bins needed
                    num_bins = opt.sequence_length // opt.num_cams_per_iter
                    if opt.sequence_length % opt.num_cams_per_iter != 0:
                        num_bins += 1

                    # Initialize list to store predicted trajectories
                    predicted_trajectories = []
                    
                    # Start from initial point
                    current_point = trajectory[0:1].to('cuda')
                    t = torch.arange(opt.num_cams_per_iter, device='cuda').float()
                    # Process each bin sequentially
                    for bin_idx in range(num_bins):
                        # Get predicted trajectory for this bin
                        predicted_xyz, _, _ = deform.step(current_point, t)
                        
                        # Store the predictions
                        predicted_trajectories.append(predicted_xyz.squeeze())
                        
                        # Update current_point to last predicted point for next iteration
                        current_point = predicted_xyz[-1:] # Keep batch dimension
                    
                    # Concatenate all predictions
                    predicted_trajectory = torch.cat(predicted_trajectories, dim=0)
                    predicted_trajectory = predicted_trajectory[:opt.sequence_length].cpu().numpy()
                    
                    # Get actual trajectory for comparison
                    actual_trajectory = trajectory.cpu().numpy()
            # Create figure for predicted trajectory
            fig_pred = plt.figure(figsize=(10, 8))
            ax_pred = fig_pred.add_subplot(111, projection='3d')

            # Plot predicted trajectory
            ax_pred.plot(predicted_trajectory[:, 0], 
                        predicted_trajectory[:, 1], 
                        predicted_trajectory[:, 2], 
                        label='Predicted', color='red')

            # Set labels and title for predicted trajectory
            ax_pred.set_xlabel('X')
            ax_pred.set_ylabel('Y')
            ax_pred.set_zlabel('Z')
            ax_pred.set_title(f'Predicted Trajectory at Iteration {iteration}')
            ax_pred.legend()

            # Save the predicted trajectory plot
            pred_plot_path = os.path.join(dataset.model_path, f"predicted_trajectory_{iteration}.png")
            plt.savefig(pred_plot_path)
            plt.close(fig_pred)

            # Create figure for ground truth trajectory
            fig_gt = plt.figure(figsize=(10, 8))
            ax_gt = fig_gt.add_subplot(111, projection='3d')

            # Plot ground truth trajectory
            ax_gt.plot(actual_trajectory[:, 0],
                      actual_trajectory[:, 1],
                      actual_trajectory[:, 2],
                      label='Ground Truth', color='blue')

            # Set labels and title for ground truth trajectory
            ax_gt.set_xlabel('X')
            ax_gt.set_ylabel('Y')
            ax_gt.set_zlabel('Z')
            ax_gt.set_title(f'Ground Truth Trajectory at Iteration {iteration}')
            ax_gt.legend()

            # Save the ground truth trajectory plot
            gt_plot_path = os.path.join(dataset.model_path, f"ground_truth_trajectory_{iteration}.png")
            plt.savefig(gt_plot_path)
            plt.close(fig_gt)

        # Save model
        if iteration in saving_iterations:
            print("\n[ITER {}] Saving".format(iteration))
            deform.save_weights(dataset.model_path, iteration)

    progress_bar.close()
    
    # Add final complete trajectory plot
    with torch.no_grad():
        # Get complete trajectory prediction
        initial_point = trajectory[0].unsqueeze(0)  # Get first point
        t = torch.linspace(0, 1, opt.sequence_length, device="cuda")
        predicted_trajectory, _, _ = deform.step(initial_point, t)
        
        # Convert to numpy for plotting
        predicted_trajectory = predicted_trajectory.squeeze(1).cpu().numpy()
        actual_trajectory = trajectory.cpu().numpy()
        
        # Create final comparison plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot predicted trajectory
        ax.plot(predicted_trajectory[:, 0], 
                predicted_trajectory[:, 1], 
                predicted_trajectory[:, 2], 
                label='Predicted', color='red')
        
        # Plot actual trajectory
        ax.plot(actual_trajectory[:, 0], 
                actual_trajectory[:, 1], 
                actual_trajectory[:, 2], 
                label='Ground Truth', color='blue')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Complete Trajectory Comparison')
        ax.legend()
        
        # Save the final plot
        final_plot_path = os.path.join(dataset.model_path, "final_trajectory_comparison.png")
        plt.savefig(final_plot_path)
        plt.close(fig)

    print("Training complete.")

if __name__ == "__main__":
     # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--save_iterations", nargs="+", type=int, default=list(range(1000, 150001, 2000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--plot_interval", type=int, default=20)
    parser.add_argument("--full_sequence",type=bool, default=True)
    args = parser.parse_args(sys.argv[1:])
 
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    print("also saving at itration :{}".format(args.iterations))
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), args.save_iterations, args.plot_interval,args.full_sequence)


