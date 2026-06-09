"""
CosmoExFog — Main pipeline entry point
=======================================
Cosmological Parameters derived from CMB Extragalactic Foregrounds.

Modes:
  spatial_corr    Spatial correlation: voids/filaments vs CMB parameter maps
  multipole_cuts  Parameter shifts sweeping lmin, lmax, and mask variants
  discrepancy     Systematic anomaly search across simulations + mask re-eval

Examples:
  python run_pipeline.py --config config.yaml spatial_corr
  python run_pipeline.py multipole_cuts --sampler iminuit
  python run_pipeline.py multipole_cuts --sampler cobaya
  python run_pipeline.py discrepancy
"""

import argparse
import yaml
import sys
import os

from src.modes import spatial_correlation
from src.modes import multipole_cuts
from src.modes import discrepancy_sims


def main():
    # ── Argument parsing ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="CosmoExFog: CMB Extragalactic Foregrounds Parameter Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to YAML config file (default: config.yaml)')

    subparsers = parser.add_subparsers(dest='mode', required=True,
                                        help='Analysis mode to run')

    # Mode 1: Spatial Correlation
    subparsers.add_parser(
        'spatial_corr',
        help='Spatial correlation: voids/filaments vs CMB parameter maps')

    # Mode 2: Multipole Cuts
    parser_mc = subparsers.add_parser(
        'multipole_cuts',
        help='Parameter estimation sweeping lmin, lmax cuts and mask variants')
    parser_mc.add_argument(
        '--sampler', choices=['iminuit', 'cobaya'], default='iminuit',
        help='Inference engine: iminuit (fast) or cobaya (full MCMC)')

    # Mode 3: Discrepancy
    subparsers.add_parser(
        'discrepancy',
        help='Search for anomalous simulations and re-evaluate with masks')

    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    if not os.path.exists(args.config):
        print(f"Error: config file not found at '{args.config}'")
        sys.exit(1)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # ── Route to mode ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CosmoExFog  |  mode: {args.mode}")
    print(f"  Config: {args.config}")
    print(f"{'='*60}\n")

    if args.mode == 'spatial_corr':
        spatial_correlation.run(args, config)

    elif args.mode == 'multipole_cuts':
        multipole_cuts.run(args, config)

    elif args.mode == 'discrepancy':
        discrepancy_sims.run(args, config)


if __name__ == '__main__':
    main()