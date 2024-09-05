ModelParams = dict(
    is_ode = False,
    D = 8,
    W = 256,
    max_gaussians = 10000
)

OptimizationParams = dict(
    scale_lr = False,
    direct_compute = False,
    sequence_length = 30,
    num_cams_per_iter = 1,
    spread_out_sequence = False
)