import os
import torch
import numpy as np
from scene import Scene, GaussianModel, DeformModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_gaussian_model(dataset, model_path, iteration=-1):
    """Load a trained Gaussian model."""
    # # Create a basic dataset object with minimal attributes needed
    # class DummyDataset:
    #     def __init__(self, path):
    #         self.model_path = path
    #         self.sh_degree = 3  # Adjust if needed
            
    # dataset = DummyDataset(model_path)
    dataset.model_path = model_path
    print(dataset.model_path)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    return scene, gaussians

def sample_trajectories(gaussians, deform_model, num_time_steps=150, min_t=0.0, max_t=1.0):
    """Sample trajectories for all Gaussians over time."""
    # Create time points
    t = torch.linspace(min_t, max_t, num_time_steps).cuda()
    
    # Get initial positions
    xyz = gaussians.get_xyz
    num_gaussians = xyz.shape[0]
    
    # Prepare time input for all Gaussians
    time_input = t.unsqueeze(0).expand(num_gaussians, -1)
    
    # Get trajectories
    with torch.no_grad():
        d_xyz, d_rotation, d_scaling = deform_model.step(xyz.detach(), time_input)
        trajectories = d_xyz + xyz.unsqueeze(0)
    
    return trajectories, t

def visualize_trajectories(trajectories, t, save_dir):
    """Visualize the sampled trajectories."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy for plotting
    trajectories_np = trajectories.cpu().numpy()
    t_np = t.cpu().numpy()
    
    # Create 3D trajectory plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each Gaussian's trajectory
    for i in tqdm(range(trajectories_np.shape[1]), desc="Plotting trajectories"):
        traj = trajectories_np[:, i, :]
        ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Gaussian Trajectories')
    
    plt.savefig(os.path.join(save_dir, 'trajectories_3d.png'))
    plt.close()
    
    # Save trajectories as numpy array
    np.save(os.path.join(save_dir, 'trajectories.npy'), trajectories_np)
    np.save(os.path.join(save_dir, 'timestamps.npy'), t_np)

def main():
    parser = ArgumentParser(description="Sample Gaussian trajectories from trained model")
    lp = ModelParams(parser)
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Which iteration to load (-1 for latest)")
    parser.add_argument("--num_time_steps", type=int, default=150,
                        help="Number of time steps to sample")
    parser.add_argument("--output_dir", type=str, default="sampled_trajectories",
                        help="Directory to save the sampled trajectories")
    parser.add_argument("--base_model_path", type=str, default="")
    args = parser.parse_args()

    

    # Load the trained model
    print("Loading Gaussian model...")
    scene, gaussians = load_gaussian_model(lp.extract(args), args.base_model_path, args.iteration)
    
    # Load the deformation model
    print("Loading deformation model...")
    deform_model = DeformModel(is_blender=True, is_6dof=False, 
                              D=8, W=256, input_ch=1, output_ch=3, multires=10)
    deform_model.load_weights(args.base_model_path, iteration=args.iteration)
    
    # Sample trajectories
    print("Sampling trajectories...")
    trajectories, timestamps = sample_trajectories(gaussians, deform_model, 
                                                 num_time_steps=args.num_time_steps)
    
    # Visualize and save results
    print("Visualizing trajectories...")
    visualize_trajectories(trajectories, timestamps, args.output_dir)
    
    print(f"Results saved to {args.output_dir}")
    print(f"Trajectory shape: {trajectories.shape}")

if __name__ == "__main__":
    main() 