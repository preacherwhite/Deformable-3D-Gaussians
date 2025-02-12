ModelParams = dict(
    is_ode = False,
    D = 8,
    W = 256,
    max_gaussians = 10000
)

OptimizationParams = dict(
    scale_lr = False,
    direct_compute = False,
    sequence_length = 5,
    num_cams_per_iter = 1,
    spread_out_sequence = False,
    #iterations = 12000,
    #position_lr_max_steps = 6000,
    #deform_lr_max_steps = 12000,
    #densify_unti_iter = 5000
    warm_up = 3000
)