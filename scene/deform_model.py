import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetworkODE, DeformNetwork
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func

from torchdiffeq import odeint_adjoint
class DeformModelODE:
    def __init__(self, is_blender=False, is_6dof=False, D = 8, W = 256, input_ch = 3, output_ch = 59, multires = 10, scale_lr = False):
        self.deform = DeformNetworkODE(is_blender=is_blender, is_6dof=is_6dof, D = D, W = W, input_ch = input_ch, output_ch = output_ch, multires = multires).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5
        self.scale_lr = scale_lr

    def step(self, xyz, time_emb):  
        # xyz: N x 3
        # time_emb: N x 1
        #initial_state = torch.cat([xyz, initial_rotation, initial_scaling], dim=-1)
        #print('step_sizes=', xyz.shape,time_emb.shape)
        
        # t is the same for the entire gaussian set, so we can use the same t for all gaussians
        is_val = len(time_emb) == 1
        zero_start = time_emb[0] == 0
        if zero_start.item() and is_val:
            return [xyz] , [torch.zeros([xyz.shape[0], 4]).to(xyz.device)], [torch.zeros([xyz.shape[0],3]).to(xyz.device)]
        rtol = 0.001
        atol = 0.0001
        if not zero_start:
            time_emb = [0.0] + time_emb

        t_interval = torch.Tensor(time_emb).to(xyz.device)
        #print('t_interval=', t_interval.shape, 'xyz=', xyz.shape)
        #ode_value = odeint_adjoint(self.deform, xyz,t_interval, rtol= rtol, atol=atol,method='rk4', options={'step_size': 0.0025})
        # check if t_interval is strictly increasing
  
        ode_value = odeint_adjoint(self.deform, xyz,t_interval, rtol= rtol, atol=atol)
        xyz_new = torch.squeeze(ode_value)
        #print(xyz_new.shape)
        if is_val:
            if not zero_start:
                xyz_new = xyz_new[1].unsqueeze(0)
        elif not zero_start:
            xyz_new = xyz_new[1:]
        rotation_placeholder = torch.zeros([xyz_new.shape[0],xyz_new.shape[1], 4]).to(xyz.device)
        scale_placeholder = torch.zeros([xyz_new.shape[0],xyz_new.shape[1],3]).to(xyz.device)
        #print(xyz_new.shape, rotation_placeholder.shape, scale_placeholder.shape)
        return xyz_new, rotation_placeholder , scale_placeholder
        #print(xyz.shape, time_emb.shape)
        #return self.deform(xyz, time_emb)
        #return [0.0], [0.0], [0.0]

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        if self.scale_lr: 
            lr_init = training_args.position_lr_init * self.spatial_lr_scale  * training_args.num_cams_per_iter
            lr_final = training_args.position_lr_final * self.spatial_lr_scale  * training_args.num_cams_per_iter
        else:
            lr_init = training_args.position_lr_init * self.spatial_lr_scale
            lr_final = training_args.position_lr_final
        lr_delay_mult = training_args.position_lr_delay_mult
        max_steps = training_args.position_lr_max_steps
        self.deform_scheduler_args = get_expon_lr_func(lr_init=lr_init,
                                                    lr_final=lr_final,
                                                    lr_delay_mult=lr_delay_mult,
                                                    max_steps=max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

class DeformModel:
    def __init__(self, is_blender=False, is_6dof=False, D = 8, W = 256, input_ch = 3, output_ch = 59, multires = 10, scale_lr = False):
        self.deform = DeformNetwork(is_blender=is_blender, is_6dof=is_6dof, D = D, W = W, input_ch = input_ch, output_ch = output_ch, multires = multires).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5
        self.scale_lr = scale_lr

    def step(self, xyz, time_emb):  
        # xyz: N x 3
        # time_emb: N x 1
        d_xyz_list = []
        d_rotation_list = []
        d_scaling_list = []
        for t in time_emb:
            # expand t to match size of xyz
            t = t.expand(xyz.shape[0], 1)
            d_xyz, d_rotation, d_scaling = self.deform(xyz, t)
            d_xyz_list.append(d_xyz)
            d_rotation_list.append(d_rotation)
            d_scaling_list.append(d_scaling)
        return d_xyz_list, d_rotation_list, d_scaling_list

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        if self.scale_lr:
            lr_init = training_args.position_lr_init * self.spatial_lr_scale * training_args.num_cams_per_iter
            lr_final = training_args.position_lr_final * training_args.num_cams_per_iter
        else:
            lr_init = training_args.position_lr_init * self.spatial_lr_scale
            lr_final = training_args.position_lr_final
        lr_delay_mult = training_args.position_lr_delay_mult
        max_steps = training_args.position_lr_max_steps
        self.deform_scheduler_args = get_expon_lr_func(lr_init=lr_init,
                                                    lr_final=lr_final,
                                                    lr_delay_mult=lr_delay_mult,
                                                    max_steps=max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
