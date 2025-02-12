ModelParams = dict(
    is_ode = True,
    D = 8,
    W = 256,
    use_linear = 0,
    use_emb = True,
    use_torch_ode = False,
)

OptimizationParams = dict(
    scale_lr = False,
    direct_compute = True,
    sequence_length = 30,
    num_cams_per_iter = 10,
    spread_out_sequence = True,
    position_lr_init = 2e-5,
    position_lr_final = 0.0000001,
    rtol = 1e-4,
    atol = 1e-5,
    freeze_gaussians = False,
    warm_up = 3000,
    #densify_until_iter = 3000,
    max_batch_gaussians = -1,
    iterations = 20000
)