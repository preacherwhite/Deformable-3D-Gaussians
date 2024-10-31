import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scene import DeformModelODE, DeformModelTORCHODE
from train_synth_ode import create_linear_trajectory, create_trajectory_with_dimension_waves, create_quadratic_trajectory

def render_trajectories(model_path, iteration, sequence_length):
    # Create synthetic trajectory
    start_point = torch.tensor([0.2, 0.2, 0.2], device="cuda")
    end_point = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    #synthetic_trajectory = create_linear_trajectory(start_point, end_point, sequence_length)
    synthetic_trajectory = create_trajectory_with_dimension_waves(start_point, end_point, sequence_length)
    # Load trained ODE model
    deform = DeformModelODE(is_blender=True, is_6dof=False, D=8, W=256, input_ch=3, output_ch=59, multires=10, use_emb=True, skips=None)
    #deform = DeformModelTORCHODE(is_blender=True, is_6dof=False, D=8, W=256, input_ch=3, output_ch=59, multires=10)
    deform.load_weights(model_path, iteration)

    # Generate learned trajectory
    initial_point = synthetic_trajectory[0].unsqueeze(0)
    fids = torch.linspace(0, 1, sequence_length, device="cuda")
    learned_trajectory, _, _ = deform.step(initial_point, fids)
    print("First value of learned trajectory:", learned_trajectory[0])
    print("Last value of learned trajectory:", learned_trajectory[-1])
    # Convert trajectories to numpy for plotting
    synthetic_trajectory = synthetic_trajectory.cpu().numpy()
    learned_trajectory = learned_trajectory.squeeze().detach().cpu().numpy()

    # Create 3D plot for synthetic trajectory
    fig_synthetic = plt.figure(figsize=(10, 8))
    ax_synthetic = fig_synthetic.add_subplot(111, projection='3d')

    # Plot synthetic trajectory
    ax_synthetic.plot(synthetic_trajectory[:, 0], synthetic_trajectory[:, 1], synthetic_trajectory[:, 2], 
                      label='Synthetic', color='blue')

    # Set labels and title for synthetic trajectory
    ax_synthetic.set_xlabel('X')
    ax_synthetic.set_ylabel('Y')
    ax_synthetic.set_zlabel('Z')
    ax_synthetic.set_title(f'Synthetic Trajectory (Iteration {iteration})')
    ax_synthetic.legend()

    # Save the synthetic trajectory plot
    output_dir = os.path.join(model_path, f"renders_{iteration}")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'synthetic_trajectory.png'))
    plt.close(fig_synthetic)

    # Create 3D plot for learned trajectory
    fig_learned = plt.figure(figsize=(10, 8))
    ax_learned = fig_learned.add_subplot(111, projection='3d')

    # Plot learned trajectory
    ax_learned.plot(learned_trajectory[:, 0], learned_trajectory[:, 1], learned_trajectory[:, 2], 
                    label='Learned ODE', color='red')

    # Set labels and title for learned trajectory
    ax_learned.set_xlabel('X')
    ax_learned.set_ylabel('Y')
    ax_learned.set_zlabel('Z')
    ax_learned.set_title(f'Learned Trajectory (Iteration {iteration})')
    ax_learned.legend()

    # Save the learned trajectory plot
    plt.savefig(os.path.join(output_dir, 'learned_trajectory.png'))
    plt.close(fig_learned)

    print(f"Synthetic trajectory plot saved in {output_dir}")
    print(f"Learned trajectory plot saved in {output_dir}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Render script parameters")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number of the save")
    parser.add_argument("--sequence_length", type=int, default=1000, help="Number of points in the trajectory")
    args = parser.parse_args()

    render_trajectories(args.model_path, args.iteration, args.sequence_length)
