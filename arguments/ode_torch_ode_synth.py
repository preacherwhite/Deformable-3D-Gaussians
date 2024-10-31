ModelParams = dict(
    is_ode = True,
    D = 5,
    W = 128,
    use_linear = 0,
    use_emb = True,
    # output_scale = 1
    use_torch_ode = True,
)

OptimizationParams = dict(
    scale_lr = False,
    direct_compute = True,
    sequence_length = 150,
    num_cams_per_iter = 10,
    spread_out_sequence = True,
    position_lr_init = 0.0001,
    position_lr_final = 0.000001,
    rtol = 1e-3,
    atol = 1e-4,
    # freeze_gaussians = False,
    # use_iterative_update = True,
    # iterative_update_decay = 0.8,
    # iterative_update_interval = 200,
    # max_training_switches = 15,
    iterations = 150000,
    warm_up = 0,

    #weight_decay = 0.0001
    #densify_until_iter = 3000,
)