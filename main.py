#!/usr/bin/env python3
"""
main.py — Entry point for the distributed phased array FSR simulation (v9).

A 3-D target moves obliquely (heading angle θ) across a baseline of K planar
N×N receive arrays.  Zone B estimates v_x, v_y, R_0, x_0, z_T via 2-D DOA,
STFT Doppler, and NLS.  Zone A retrieves the shadow profile via gradient
descent with TV denoising, jointly refining all five parameters.

Usage
-----
    python main.py                          # all defaults (Table 1)
    python main.py --shape triangle         # triangular target
    python main.py --K 6 --N 8             # 6 nodes, 8×8 arrays
    python main.py --theta 5               # 5° heading angle
    python main.py --snr 30 --N_S 100      # higher SNR, fewer iters
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.geometry import create_system
from src.pipeline import run_simulation
from src.visualisation import plot_all


def main():
    p = argparse.ArgumentParser(
        description="Distributed Phased Array FSR Simulation (v9)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # System parameters
    p.add_argument('--K', type=int, default=4,
                   help='Number of receiver nodes')
    p.add_argument('--N', type=int, default=16,
                   help='Elements per array dimension (N×N total)')
    p.add_argument('--R_0', type=float, default=1e4,
                   help='Nominal propagation distance (λ)')
    p.add_argument('--delta_node', type=float, default=None,
                   help='Inter-node spacing (λ).  Default: √R_0/2')

    # Target parameters
    p.add_argument('--x0', type=float, default=2000.0,
                   help='Initial target x-coordinate (λ)')
    p.add_argument('--v', type=float, default=1.0,
                   help='Target speed (λ/time)')
    p.add_argument('--theta', type=float, default=0.0,
                   help='Heading angle (degrees)')
    p.add_argument('--z_T', type=float, default=10.0,
                   help='Target altitude (λ)')
    p.add_argument('--dt', type=float, default=1.0,
                   help='Snapshot interval Δt')

    # Target shape
    p.add_argument('--shape', type=str, default='rectangle',
                   choices=['rectangle', 'triangle', 'circle'],
                   help='Target shadow shape')
    p.add_argument('--width', type=float, default=100.0,
                   help='Target width (λ)')
    p.add_argument('--height', type=float, default=80.0,
                   help='Target height (λ)')

    # Grid
    p.add_argument('--M_z', type=int, default=20,
                   help='Pixel rows')
    p.add_argument('--M_x', type=int, default=20,
                   help='Pixel columns')
    p.add_argument('--pixel_size', type=float, default=10.0,
                   help='Pixel side 2δ (λ)')

    # Noise / threshold
    p.add_argument('--snr', type=float, default=20.0,
                   help='Per-element SNR (dB)')
    p.add_argument('--xi_th', type=float, default=2.0,
                   help='Zone threshold ξ_th')

    # Algorithm
    p.add_argument('--gamma', type=float, default=5000.0,
                   help='Profile step size γ')
    p.add_argument('--tau', type=float, default=1e-6,
                   help='TV regularisation weight τ')
    p.add_argument('--N_S', type=int, default=200,
                   help='Profile iterations')
    p.add_argument('--N_D', type=int, default=20,
                   help='TV denoising iterations')
    p.add_argument('--N_V', type=int, default=20,
                   help='Outer parameter iterations (Alg 3)')

    # DOA method
    p.add_argument('--doa', type=str, default='music',
                   choices=['music', 'esprit'],
                   help='DOA estimation method')

    # Output
    p.add_argument('--output', type=str, default='results',
                   help='Output directory for figures')
    p.add_argument('--no-plot', action='store_true',
                   help='Skip figure generation')

    args = p.parse_args()

    config = create_system(
        K=args.K,
        N=args.N,
        R_0=args.R_0,
        delta_node=args.delta_node,
        x_0=args.x0,
        v=args.v,
        theta=float(args.theta) * (3.141592653589793 / 180.0),
        z_T=args.z_T,
        dt=args.dt,
        target_shape=args.shape,
        target_width=args.width,
        target_height=args.height,
        M_z=args.M_z,
        M_x=args.M_x,
        pixel_size=args.pixel_size,
        snr_db=args.snr,
        xi_threshold=args.xi_th,
        gamma_profile=args.gamma,
        tau=args.tau,
        N_S=args.N_S,
        N_D=args.N_D,
        N_V=args.N_V,
    )

    results = run_simulation(config, doa_method=args.doa, verbose=True)

    if not args.no_plot:
        plot_all(results, save_dir=args.output)

    return results


if __name__ == "__main__":
    main()
