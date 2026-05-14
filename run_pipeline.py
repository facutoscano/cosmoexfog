#%% Imports
import argparse
import yaml
import sys
import os

from src.modes import spatial_correlation
#from src.modes import multipole_cuts
#from src.modes import discrepancy_sims

def main():
    # Console Args
    parser = argparse.ArgumentParser(description="CosmoExFog: CMB Extragalactic Foregrounds Parameter Pipeline")
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Analysis mode to run')

    # Sub-parser for MODE 1
    parser_sc = subparsers.add_parser('spatial_corr', help='Run spatial correlation analysis (Voids/Filaments vs CMB)')

    # Sub-parser for MODE 2 (Placeholder for when we develop it)
    parser_mc = subparsers.add_parser('multipole_cuts', help='Run parameter estimation analysis with multipole cuts')
    parser_mc.add_argument('--sampler', choices=['iminuit', 'cobaya'], default='iminuit', help='Inference engine to use')

    # Sub-parser for MODE 3 (Placeholder for when we develop it)
    parser_disc = subparsers.add_parser('discrepancy', help='Search for discrepancies between simulations and masked data')

    args = parser.parse_args()

    # 2. YAML Config Loading
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)
        
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 3. Routing to the Corresponding Mode
    if args.mode == 'spatial_corr':
        spatial_correlation.run(args, config)
        
    elif args.mode == 'multipole_cuts':
        print(">> MODO 2: Multipole Cuts & Parameter Shifts (En desarrollo...)")
        # multipole_cuts.run(args, config)
        
    elif args.mode == 'discrepancy':
        print(">> MODO 3: Discrepancy & Simulation Analysis (En desarrollo...)")
        # discrepancy_sims.run(args, config)

if __name__ == '__main__':
    main()