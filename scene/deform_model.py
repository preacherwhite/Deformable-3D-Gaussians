import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetworkODE, DeformNetwork, DeformNetworkSimple, DeformNetworkSimpleStart, DeformNetworkBaseline
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func

from torchdiffeq import odeint_adjoint
import torchode as to
class DeformModelTORCHODE:
    def __init__(self, is_blender=False, is_6dof=False, D = 8, W = 256, input_ch = 3, output_ch = 59, multires = 10, scale_lr = False, use_emb=True, skips = None):
        #self.deform = DeformNetworkODE(is_blender=is_blender, is_6dof=is_6dof, D = D, W = W, input_ch = input_ch, output_ch = output_ch, multires = multires, use_emb=use_emb, skips = skips).cuda()
        self.deform = DeformNetworkSimple().cuda()
        self.optimizer = None

    def step(self, xyz, time_emb):  
        # xyz: N x 3
        # time_emb: N x 1
        # t is the same for the entire gaussian set, so we can use the same t for all gaussians
        is_val = len(time_emb) == 1

        t_interval = torch.Tensor(time_emb).to(xyz.device)

        term = to.ODETerm(self.deform)
        step_method = to.Dopri5(term=term)
        step_size_controller = to.IntegralController(atol=1e-9, rtol=1e-7, term=term)
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)

        sol = solver.solve(to.InitialValueProblem(y0=xyz, t_eval=t_interval))
        xyz_new = torch.squeeze(sol.ys)

        rotation_placeholder = torch.zeros([xyz_new.shape[0],xyz_new.shape[1], 4]).to(xyz.device)
        scale_placeholder = torch.zeros([xyz_new.shape[0],xyz_new.shape[1],3]).to(xyz.device)

        return xyz_new, rotation_placeholder , scale_placeholder


    def train_setting(self, training_args):
        self.optimizer = torch.optim.Adam(self.deform.parameters(), lr=training_args.position_lr_init)

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

class DeformModelTORCHODEStart:
    def __init__(self, is_blender=False, is_6dof=False, D = 8, W = 256, input_ch = 3, output_ch = 59, multires = 10, scale_lr = False, use_emb=True, skips = None):
        #self.deform = DeformNetworkODE(is_blender=is_blender, is_6dof=is_6dof, D = D, W = W, input_ch = input_ch, output_ch = output_ch, multires = multires, use_emb=use_emb, skips = skips).cuda()
        self.deform = DeformNetworkSimpleStart().cuda()
        self.optimizer = None

    def step(self, xyz, time_emb, y_start):  
        # xyz: B x 3
        # time_emb: B x 1
        # y_start: B x 3
        t_interval = torch.Tensor(time_emb).to(xyz.device)

        term = to.ODETerm(self.deform, with_args=True)
        step_method = to.Dopri5(term=term)
        step_size_controller = to.IntegralController(atol=1e-9, rtol=1e-7, term=term)
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        #print(xyz.shape,t_interval.shape,y_start.shape)
        sol = solver.solve(to.InitialValueProblem(y0=xyz, t_eval=t_interval), args=y_start)
        xyz_new = torch.squeeze(sol.ys)

        rotation_placeholder = torch.zeros([xyz_new.shape[0],xyz_new.shape[1], 4]).to(xyz.device)
        scale_placeholder = torch.zeros([xyz_new.shape[0],xyz_new.shape[1],3]).to(xyz.device)

        return xyz_new, rotation_placeholder , scale_placeholder


    def train_setting(self, training_args):
        self.optimizer = torch.optim.Adam(self.deform.parameters(), lr=training_args.position_lr_init)

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
        pass

class DeformModelODESimpleStart:
    def __init__(self, is_blender=False, is_6dof=False, D = 8, W = 256, input_ch = 3, output_ch = 59, multires = 10, 
                 scale_lr = False, use_linear=0, use_emb=True, rtol = 0.001, atol = 0.0001, output_scale = 1, skips = [4]):
        self.deform = DeformNetworkSimpleStart().cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5
        self.scale_lr = scale_lr
        self.rtol = rtol
        self.atol = atol

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
        t_interval = torch.Tensor(time_emb).to(xyz.device)
        ode_value = odeint_adjoint(self.deform, xyz,t_interval, rtol= self.rtol, atol=self.atol)
        xyz_new = torch.squeeze(ode_value)
        if is_val:
            if not zero_start:
                xyz_new = xyz_new[1].unsqueeze(0)

        rotation_placeholder = torch.zeros([xyz_new.shape[0],xyz_new.shape[1], 4]).to(xyz.device)
        scale_placeholder = torch.zeros([xyz_new.shape[0],xyz_new.shape[1],3]).to(xyz.device)
        return xyz_new, rotation_placeholder , scale_placeholder

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, weight_decay=training_args.weight_decay)
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
        print("saving weights to ", out_weights_path)
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

class DeformModelODE:
    def __init__(self, is_blender=False, is_6dof=False, D = 8, W = 256, input_ch = 3, output_ch = 59, multires = 10, 
                 scale_lr = False, use_linear=0, use_emb=True, rtol = 0.001, atol = 0.0001, output_scale = 1, skips = [4]):
        self.deform = DeformNetworkODE(is_blender=is_blender, is_6dof=is_6dof, D = D, W = W, input_ch = input_ch, output_ch = output_ch, 
                                       multires = multires, use_linear=use_linear, use_emb=use_emb, output_scale=output_scale, skips=skips).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5
        self.scale_lr = scale_lr
        self.rtol = rtol
        self.atol = atol

    def step(self, xyz, time_emb):  
        # xyz: N x 3
        # time_emb: N x 1
        #initial_state = torch.cat([xyz, initial_rotation, initial_scaling], dim=-1)
        
        # t is the same for the entire gaussian set, so we can use the same t for all gaussians
        is_val = len(time_emb) == 1
        zero_start = time_emb[0] == 0
        if zero_start.item() and is_val:
            return xyz, torch.zeros([xyz.shape[0], 4]).to(xyz.device), torch.zeros([xyz.shape[0],3]).to(xyz.device)
        # if not zero_start:
        #     time_emb = [0.0] + time_emb

        t_interval = torch.Tensor(time_emb).to(xyz.device)
        ode_value = odeint_adjoint(self.deform, xyz,t_interval, rtol= self.rtol, atol=self.atol)
        xyz_new = torch.squeeze(ode_value)

        # if is_val:
        #     if not zero_start:
        #         xyz_new = xyz_new[1].unsqueeze(0)
        # elif not zero_start:
        #     xyz_new = xyz_new[1:]
        rotation = torch.zeros([xyz_new.shape[0],xyz_new.shape[1], 4]).to(xyz.device)
        scale = torch.zeros([xyz_new.shape[0],xyz_new.shape[1], 3]).to(xyz.device)
        return xyz_new, rotation, scale

        #return self.deform(xyz, time_emb)
        #return [0.0], [0.0], [0.0]

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, weight_decay=training_args.weight_decay)
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
        print("saving weights to ", out_weights_path)
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
        # time_emb: N x T
        d_xyz_list = []
        d_rotation_list = []
        d_scaling_list = []
        for i in range(time_emb.shape[1]):
            # expand t to match size of xyz
            # t = t.expand(xyz.shape[0], 1)
            d_xyz, d_rotation, d_scaling = self.deform(xyz, time_emb[:,i].unsqueeze(1))
            d_xyz_list.append(d_xyz)
            d_rotation_list.append(d_rotation)
            d_scaling_list.append(d_scaling)
        return torch.stack(d_xyz_list, dim=0), d_rotation_list, d_scaling_list

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, weight_decay=training_args.weight_decay)

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
            
class DeformModelBaseline:
    def __init__(self, is_blender=False, is_6dof=False, D = 8, W = 256, input_ch = 3, output_ch = 59, multires = 10):
        self.deform = DeformNetworkBaseline(is_blender=is_blender, is_6dof=is_6dof, D = D, W = W, input_ch = input_ch, output_ch = output_ch, multires = multires).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, xyz, time_emb):
        return self.deform(xyz, time_emb)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

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