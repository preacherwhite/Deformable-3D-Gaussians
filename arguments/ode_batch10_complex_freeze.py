ModelParams = dict(
    is_ode = True,
    D = 8,
    W = 256,
    max_gaussians = 10000
)

OptimizationParams = dict(
    scale_lr = False,
    direct_compute = True,
    sequence_length = 30,
    num_cams_per_iter = 10,
    spread_out_sequence = True,
    position_lr_init = 0.00016,
    position_lr_final = 0.0000016,
    freeze_gaussians = True,
    densify_from_iter = 500,
    densify_until_iter = 3000
)