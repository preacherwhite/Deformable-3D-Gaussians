ModelParams = dict(
    is_ode = True,
    D = 8,
    W = 256,
    max_gaussians = 10000,
    use_linear = 0,
    use_emb = True,
)

OptimizationParams = dict(
    scale_lr = False,
    direct_compute = True,
    sequence_length = 30,
    num_cams_per_iter = 10,
    spread_out_sequence = True,
    position_lr_init = 0.0001,
    position_lr_final = 0.0000001,
    rtol = 1e-4,
    atol = 1e-5,
    freeze_gaussians = True,
    densify_until_iter = 3000,
)