#!/usr/bin/env python3
"""
Convenience wrapper for running Quantum Seeker 2.0 locally.
"""
import argparse
import os

from quantum_seeker_v2 import execute_quantum_seeker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Quantum Seeker 2.0 analysis.")
    parser.add_argument(
        "--config",
        dest="config_path",
        help="Path to config JSON file (optional).",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Override output directory for results.",
        default=None,
    )
    parser.add_argument(
        "--seed",
        dest="random_seed",
        type=int,
        help="Override random seed for reproducibility.",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.random_seed is not None:
        overrides["random_seed"] = args.random_seed

    if args.config_path and not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    execute_quantum_seeker(config_path=args.config_path, config_override=overrides)


if __name__ == "__main__":
    main()
