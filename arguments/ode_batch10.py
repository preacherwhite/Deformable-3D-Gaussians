from arguments import ModelParams, OptimizationParams, PipelineParams, add_base_args
from argparse import ArgumentParser

class CustomModelParams(ModelParams):
    def __init__(self):
        super().__init__()
        # Override specific attributes
        self.is_ode = True
        self.max_gaussians = 10000
        self.D = 8
        self.W = 256

class CustomOptimizationParams(OptimizationParams):
    def __init__(self):
        super().__init__()
        # Override specific attributes
        self.scale_lr = False
        self.direct_compute = True
        self.sequence_legth = 30
        self.num_cams_per_iter = 10
        self.spread_out_sequence = True

class CustomPipelineParams(PipelineParams):
    def __init__(self):
        super().__init__()

def add_custom_args(parser):
    add_base_args(parser)
    # Add any additional custom arguments here
    parser.add_argument("--custom_flag", action="store_true", help="A custom flag for this parameter set")

def get_custom_args():
    parser = ArgumentParser(description="Custom training script parameters")
    add_custom_args(parser)
    return parser.parse_args()