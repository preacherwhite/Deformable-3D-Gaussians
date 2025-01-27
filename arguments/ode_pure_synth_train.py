ModelParams = dict(
    is_ode = True,
    D = 4,
    W = 128,
    use_linear = 0,
    use_emb = False,
    # output_scale = 1
    is_blender = False,
)

OptimizationParams = dict(
    scale_lr = False,
    direct_compute = True,
    sequence_length = 1000,
    num_cams_per_iter = 20,
    spread_out_sequence = True,
    position_lr_init = 1e-4,
    position_lr_final = 1e-6,
    rtol = 1e-4,
    atol = 1e-5,
    # freeze_gaussians = False,
    # use_iterative_update = True,
    # iterative_update_decay = 0.8,
    # iterative_update_interval = 200,
    # max_training_switches = 15,
    iterations = 2000,
    warm_up = 0,
    #weight_decay = 0.0001
    #densify_until_iter = 3000,
)