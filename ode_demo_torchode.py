import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchode as to
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)


class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y**3, true_A)


# Set up torchode solver components
term = to.ODETerm(Lambda())
step_method = to.Dopri5(term=term)
step_size_controller = to.IntegralController(atol=1e-9, rtol=1e-7, term=term)
solver = to.AutoDiffAdjoint(step_method, step_size_controller)

# Solve for true trajectory
with torch.no_grad():
    problem = to.InitialValueProblem(y0=true_y0, t_eval=t.unsqueeze(0))
    solution = solver.solve(problem)
    true_y = solution.ys.squeeze(0)


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = (s.unsqueeze(-1).to(device) + torch.arange(args.batch_time, device=device)).float() * (25.0 / (args.data_size - 1))
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)



    


def visualize(true_y, pred_y, odefunc, itr, location):

    if args.viz:
        fig = plt.figure(figsize=(16, 6), facecolor='white')
        ax_traj = fig.add_subplot(121, frameon=False)
        ax_phase = fig.add_subplot(122, frameon=False)
        pred_y = pred_y.squeeze()
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0], true_y.cpu().numpy()[:, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0], pred_y.cpu().numpy()[:, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        # Sample points for vector field
        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        points = torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)

        # Create a figure with 10 rows of subplots, 2 columns each
        vec_fig, vec_axes = plt.subplots(10, 2, figsize=(10, 50), facecolor='white')

        # Draw 10 vector fields at different time points
        time_points = np.linspace(0, 999, 10)
        for t_idx, (t_val, axes_row) in enumerate(zip(time_points, vec_axes)):
            # Left subplot - Phase portrait with current trajectory point
            axes_row[0].set_title(f'Phase Portrait (t = {t_val:.1f})')
            axes_row[0].set_xlabel('x')
            axes_row[0].set_ylabel('y')
            
            # Plot full trajectory
            axes_row[0].plot(true_y.cpu().numpy()[:, 0], true_y.cpu().numpy()[:, 1], 'g-', alpha=0.3)
            
            # Find closest time point in true_y and mark it
            t_idx_true = round(t_val)
            current_point = true_y[t_idx_true].cpu().numpy()
            axes_row[0].plot(current_point[0], current_point[1], 'ro')
            
            axes_row[0].set_xlim(-2, 2)
            axes_row[0].set_ylim(-2, 2)

            # Right subplot - Vector field
            axes_row[1].set_title(f'Vector Field (t = {t_val:.1f})')
            axes_row[1].set_xlabel('x')
            axes_row[1].set_ylabel('y')

            # Get vector field at current time
            t_tensor = torch.full((points.shape[0], 1), t_val, device=device)
            dydt = odefunc(t_tensor, points).cpu().detach().numpy()
            mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
            dydt = (dydt / mag)
            dydt = dydt.reshape(21, 21, 2)

            # Plot streamlines
            axes_row[1].streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
            axes_row[1].set_xlim(-2, 2)
            axes_row[1].set_ylim(-2, 2)

        vec_fig.tight_layout()
        plt.savefig(f"{location}/vector_fields_{itr}.png")
        plt.close(vec_fig)

        fig.tight_layout()
        plt.savefig(f"{location}/{itr}.png")
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net_y = nn.Sequential(
            nn.Linear(2, 50),  # Changed input dim to 3 to include time
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
        )

        self.net_t = nn.Sequential(
            nn.Linear(1, 50),  # Changed input dim to 3 to include time
            nn.Tanh(),
            nn.Linear(50, 50),  # Changed input dim to 3 to include time
            nn.Tanh(),
        )
        self.net_out = nn.Sequential(
            nn.Linear(50, 50),  # Changed input dim to 3 to include time
            nn.Tanh(),
            nn.Linear(50, 2),  # Changed input dim to 3 to include time
        )
        for m in self.net_y.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        for m in self.net_t.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)

        for m in self.net_out.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)


    def forward(self, t, y):
        # Reshape t only if it's a tensor and not already in correct shape
        if t.ndim != 2:
            t = t.unsqueeze(1)
        t_enc = self.net_t(t)
        y_enc = self.net_y(y)
        latent = t_enc + y_enc
        out = self.net_out(latent)
        #t_y = torch.cat([y**3, t], dim=1)  # Concatenate time with state
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    location = 'png_4'
    ii = 0
    if args.viz:
        makedirs(location)
        
    func = ODEFunc().to(device)
    
    # Set up torchode solver with the learned function
    term = to.ODETerm(func)
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=1e-9, rtol=1e-7, term=term)
    solver = to.AutoDiffAdjoint(step_method, step_size_controller)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-4)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        
        # Solve using torchode
        #problem = to.InitialValueProblem(y0=batch_y0, t_eval=batch_t.unsqueeze(0).repeat(batch_y0.shape[0], 1))
        problem = to.InitialValueProblem(y0=batch_y0, t_eval=batch_t)
        solution = solver.solve(problem)
        pred_y = solution.ys.transpose(0, 1)  # Adjust dimensions to match original
        
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                # t_plot = t.unsqueeze(0)
                plot_y = true_y0
                plot_t = t.unsqueeze(0)
                # Create new solver for evaluation
                eval_term = to.ODETerm(func)
                eval_step_method = to.Dopri5(term=eval_term)
                eval_step_size_controller = to.IntegralController(atol=1e-9, rtol=1e-7, term=eval_term)
                eval_solver = to.AutoDiffAdjoint(eval_step_method, eval_step_size_controller)
                
                problem = to.InitialValueProblem(y0=true_y0, t_eval=plot_t)
                solution = eval_solver.solve(problem)
                pred_y = solution.ys.transpose(0, 1)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii, location)
                ii += 1

        end = time.time()