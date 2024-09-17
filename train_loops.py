import os
import torch
from random import randint
import sys
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from train import training
import mmcv 
from utils.params_utils import merge_hparams
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[7000, 8000, 9000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str, default="")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    if args.configs:
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Set up autograd anomaly detection
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Extract parameters
    lp_args = lp.extract(args)
    op_args = op.extract(args)
    pp_args = pp.extract(args)
    original_model_path = lp_args.model_path
    # Run training for different sequence lengths
    sequence_lengths = [30, 40, 50,60]
    for seq_length in sequence_lengths:
        print(f"\nTraining with sequence length: {seq_length}")
        
        # reset args
        lp_args = lp.extract(args)
        op_args = op.extract(args)
        pp_args = pp.extract(args)

        # Update sequence length
        op_args.sequence_length = seq_length
        
        # Update model path
        new_model_path = os.path.join(original_model_path, f"seq_length_{seq_length}")
        lp_args.model_path = new_model_path
        
        # Create the new directory if it doesn't exist
        os.makedirs(new_model_path, exist_ok=True)
        
        print(f"Model will be saved in: {new_model_path}")
        
        # Run training
        training(lp_args, op_args, pp_args, args.test_iterations, args.save_iterations)

    # All done
    print("\nAll training iterations complete.")