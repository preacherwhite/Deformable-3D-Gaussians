ModelParams = dict(
    is_ode = True,
    D = 4,
    W = 128,
    max_gaussians = 10000,
    use_linear = 0,
    use_emb = True,
    output_scale = 1e-5
)

OptimizationParams = dict(
    scale_lr = False,
    direct_compute = True,
    sequence_length = 30,
    num_cams_per_iter = 15,
    spread_out_sequence = True,
    position_lr_init = 0.0001,
    position_lr_final = 0.0000001,
    rtol = 1e-4,
    atol = 1e-5,
    freeze_gaussians = False,
    iterations = 40_000,
    #weight_decay = 0.0001
    #densify_until_iter = 3000,
)