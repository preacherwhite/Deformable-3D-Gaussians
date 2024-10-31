import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, multires=10, is_blender=False, is_6dof=False):
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch
        if is_blender:
            # Better for D-NeRF Dataset
            self.time_out = 30

            self.timenet = nn.Sequential(
                nn.Linear(time_input_ch, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.time_out))

            self.linear = nn.ModuleList(
                [nn.Linear(xyz_input_ch + self.time_out, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch + self.time_out, W)
                    for i in range(D - 1)]
            )

        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )

        self.is_blender = is_blender
        self.is_6dof = is_6dof


        # Defining branches for deformation calculation
        if is_6dof:
            self.branch_w = nn.Linear(W, 3)
            self.branch_v = nn.Linear(W, 3)
        else:
            self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, t):
        # print('t_shape',t.shape)
        # print('x_shape', x.shape)
        t_emb = self.embed_time_fn(t)
        if self.is_blender:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)

        if self.is_6dof:
            w = self.branch_w(h)
            v = self.branch_v(h)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / theta + 1e-5
            v = v / theta + 1e-5
            screw_axis = torch.cat([w, v], dim=-1)
            d_xyz = exp_se3(screw_axis, theta)
        else:
            d_xyz = self.gaussian_warp(h)
        scaling = 0
        rotation = 0
        # scaling = self.gaussian_scaling(h)
        # rotation = self.gaussian_rotation(h)


        return d_xyz , rotation, scaling

class DeformNetworkODE(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, multires=10, is_blender=False, is_6dof=False, use_linear=0, use_emb=True, output_scale=1, skips=[4]):
        super(DeformNetworkODE, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6 if is_blender else 10
        self.skips = skips if skips is not None else []
        self.output_scale = output_scale
        self.is_blender = is_blender
        self.is_6dof = is_6dof
        self.use_linear = use_linear
        self.use_emb = use_emb

        if use_emb:
            self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
            self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        else:
            time_input_ch = 1
            xyz_input_ch = 3

        self.time_input_ch = time_input_ch
        self.xyz_input_ch = xyz_input_ch
        self.total_input_ch = xyz_input_ch + time_input_ch

        if use_linear == 1:
            self.linear_layer = nn.Linear(self.total_input_ch, 3)
        elif use_linear == 2:
            self.A_t_net = nn.Linear(time_input_ch, xyz_input_ch*xyz_input_ch)
            self.b_t_net = nn.Linear(time_input_ch, xyz_input_ch)
        elif use_linear == 3:
            # Linear transform of only xyz, not including t
            self.xyz_linear = nn.Linear(xyz_input_ch, 3)
        elif use_linear == 4:
            # Linear transform of only z-coordinate
            self.z_linear = nn.Linear(1, 1)
        else:
            if is_blender:
                self.time_out = 30
                self.timenet = nn.Sequential(
                    nn.Linear(time_input_ch, 256), nn.ReLU(inplace=True),
                    nn.Linear(256, self.time_out))

                self.linear = nn.ModuleList(
                    [nn.Linear(xyz_input_ch + self.time_out, W)] + [
                        nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch + self.time_out, W)
                        for i in range(D - 1)]
                )
            else:
                self.linear = nn.ModuleList(
                    [nn.Linear(self.total_input_ch, W)] + [
                        nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.total_input_ch, W)
                        for i in range(D - 1)]
                )

            if is_6dof:
                self.branch_w = nn.Linear(W, 3)
                self.branch_v = nn.Linear(W, 3)
            else:
                self.gaussian_warp = nn.Linear(W, 3)

    def forward(self, t, x):
        if self.use_emb:
            t_emb = t.repeat(x.shape[0], 1)
            t_emb = self.embed_time_fn(t_emb)
            x_emb = self.embed_fn(x)
        else:
            t_emb = t.repeat(x.shape[0], 1)
            x_emb = x

        if self.use_linear == 1:
            h = torch.cat([x_emb, t_emb], dim=-1)
            return self.linear_layer(h) * self.output_scale
        elif self.use_linear == 2:
            A_t = self.A_t_net(t_emb).view(-1, self.xyz_input_ch, self.xyz_input_ch)
            b_t = self.b_t_net(t_emb)
            return (torch.bmm(A_t, x_emb.unsqueeze(-1)).squeeze(-1) + b_t) * self.output_scale
        elif self.use_linear == 3:
            # Linear transform of only xyz, not including t
            return self.xyz_linear(x_emb) * self.output_scale
        elif self.use_linear == 4:
            # Transform only the z-coordinate
            z = x_emb[:, 2:3]  # Extract z-coordinate
            z_transformed = self.z_linear(z)
            return torch.cat([torch.zeros_like(x_emb[:,:2]), z_transformed], dim=1) * self.output_scale  # Concatenate x, y (unchanged) with transformed z
        else:
            if self.is_blender:
                t_emb = self.timenet(t_emb)

            h = torch.cat([x_emb, t_emb], dim=-1)
            for i, l in enumerate(self.linear):
                h = self.linear[i](h)
                h = F.relu(h)
                if i in self.skips:
                    h = torch.cat([x_emb, t_emb, h], -1)

            if self.is_6dof:
                w = self.branch_w(h)
                v = self.branch_v(h)
                theta = torch.norm(w, dim=-1, keepdim=True)
                w = w / (theta + 1e-5)
                v = v / (theta + 1e-5)
                screw_axis = torch.cat([w, v], dim=-1)
                d_xyz = exp_se3(screw_axis, theta)
            else:
                d_xyz = self.gaussian_warp(h)

            return d_xyz * self.output_scale