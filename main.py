#!/usr/bin/env python3
"""
main.py - Entry point for the distributed phased array FSR simulation (v5).

Target moves in x-direction with constant z_T offset.
Nodes uniformly spaced at sqrt(R)/2 (half Fresnel zone radius).
Gradient descent with TV denoising for shadow profile retrieval.

Usage:
    python main.py                       # Default parameters
    python main.py --shape triangle      # Triangular target
    python main.py --K 6 --N 16         # 6 nodes
    python main.py --snr 30 --z_T 20    # Higher SNR, larger z_T
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.geometry import create_system
from src.simulation import run_simulation
from src.visualisation import plot_all


def main():
    parser = argparse.ArgumentParser(
        description="Distributed Phased Array FSR Simulation (v5, x-motion)"
    )
    parser.add_argument('--K', type=int, default=4,
                        help='Number of nodes (default: 4)')
    parser.add_argument('--N', type=int, default=16,
                        help='Elements per node (default: 16)')
    parser.add_argument('--R', type=float, default=1e4,
                        help='Target-to-receiver distance in wavelengths '
                             '(default: 1e4)')
    parser.add_argument('--x0', type=float, default=2000.0,
                        help='Target initial x-coordinate (default: 2000)')
    parser.add_argument('--vx', type=float, default=-1.0,
                        help='Target velocity in x-direction (default: -1.0)')
    parser.add_argument('--z_T', type=float, default=10.0,
                        help='Constant z-offset (default: 10)')
    parser.add_argument('--shape', type=str, default='rectangle',
                        choices=['rectangle', 'triangle', 'circle'],
                        help='Target shape (default: rectangle)')
    parser.add_argument('--width', type=float, default=100.0,
                        help='Target width in wavelengths (default: 100)')
    parser.add_argument('--height', type=float, default=80.0,
                        help='Target height in wavelengths (default: 80)')
    parser.add_argument('--snr', type=float, default=20.0,
                        help='SNR in dB (default: 20)')
    parser.add_argument('--xi_th', type=float, default=2.0,
                        help='Zone transition threshold (default: 2.0)')
    parser.add_argument('--doa', type=str, default='fft',
                        choices=['fft', 'music', 'esprit'],
                        help='DOA method for Zone B (default: fft)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for figures (default: results)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')

    args = parser.parse_args()

    config = create_system(
        K=args.K,
        N=args.N,
        R=args.R,
        x_0=args.x0,
        v_x=args.vx,
        z_T=args.z_T,
        target_shape=args.shape,
        target_width=args.width,
        target_height=args.height,
        snr_db=args.snr,
        xi_threshold=args.xi_th,
    )

    results = run_simulation(config, doa_method=args.doa, verbose=True)

    if not args.no_plot:
        plot_all(results, save_dir=args.output)

    return results


if __name__ == "__main__":
    main()
