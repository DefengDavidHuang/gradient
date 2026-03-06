"""
simulation.py - Main Simulation Pipeline (v5)
===============================================

Target moves in x-direction with constant z_T.
Nodes uniformly spaced at sqrt(R)/2.
Zone B: DOA + Doppler → v_x, R, x_0.
Zone A: Gradient descent + TV denoising → shadow profile + refined params.
"""

import numpy as np
import time as timer
from typing import Dict, Any

from .geometry import SystemConfig, create_system, ArrayNode, Target
from .fresnel import compute_fssr_direct, add_noise_to_fssr
from .doa import estimate_xT_music, estimate_xT_esprit
from .zone_b import (
    generate_received_signal_zone_b,
    estimate_parameters_zone_b,
    estimate_ell_x_single,
    beamform,
)
from .zone_a import (
    retrieve_shadow_profile_gradient,
    retrieve_with_velocity_estimation,
    compute_retrieval_accuracy,
)


def run_simulation(
    config: SystemConfig = None,
    doa_method: str = "fft",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete Zone B → Zone A simulation.

    Parameters
    ----------
    config : SystemConfig, optional
        If None, creates default system.
    doa_method : str
        "fft", "music", or "esprit" for DOA estimation in Zone B.
    verbose : bool

    Returns
    -------
    dict with results.
    """
    if config is None:
        config = create_system()

    results = {'config': config, 'timing': {}, 'doa_method': doa_method}
    target = config.target
    nodes = config.nodes
    K = config.K
    R = nodes[0].R
    r_F = np.sqrt(R)

    # Time axis
    T_total = abs(2 * target.x_0 / target.v_x)
    M_total = getattr(config, '_M_total', 4000)
    dt = T_total / M_total
    t_values = np.arange(M_total) * dt

    # Target trajectory: x_T(t) for each time step
    x_T_values = np.array([target.x_T(t) for t in t_values])

    # Per-node zone classification: ξ_k(t) = |x_T(t) - X_k| / sqrt(R)
    xi_per_node = {}
    zone_A_mask_per_node = {}
    zone_B_mask_per_node = {}

    for k, node in enumerate(nodes):
        xi_k = np.abs(x_T_values - node.x_centre) / r_F
        xi_per_node[k] = xi_k
        zone_A_mask_per_node[k] = xi_k <= config.xi_threshold
        zone_B_mask_per_node[k] = xi_k > config.xi_threshold

    # Global zone masks (any node in Zone A / all nodes in Zone B)
    any_zone_A = np.zeros(M_total, dtype=bool)
    all_zone_B = np.ones(M_total, dtype=bool)
    for k in range(K):
        any_zone_A |= zone_A_mask_per_node[k]
        all_zone_B &= zone_B_mask_per_node[k]

    M_A_total = int(np.sum(any_zone_A))

    if verbose:
        print("=" * 60)
        print(f"Distributed Phased Array FSR Simulation (v5, x-motion)")
        print("=" * 60)
        print(f"f_c={config.fc_ghz}GHz, λ={config.wavelength_m*100:.2f}cm, "
              f"R={R:.0e}λ ({R*config.wavelength_m:.0f}m)")
        print(f"r_F = √R = {r_F:.0f}λ, node spacing = {r_F/2:.0f}λ")
        print(f"K={K} nodes at: {[f'{n.x_centre:.0f}' for n in nodes]}")
        print(f"Target: x_0={target.x_0}, v_x={target.v_x}, z_T={target.z_T}")
        print(f"Pixels: {len(target.pixel_centres_x)}×{len(target.pixel_centres_z)}")
        print(f"M_total={M_total}, dt={dt:.4f}")
        for k in range(K):
            M_A_k = int(np.sum(zone_A_mask_per_node[k]))
            print(f"  Node {k} (X={nodes[k].x_centre:.0f}): "
                  f"Zone A = {M_A_k} snapshots")
        print()

    # ================================================================
    # FSSR (all nodes, all times)
    # ================================================================
    t0 = timer.time()
    fssr_clean, fssr_noisy = {}, {}
    for k, node in enumerate(nodes):
        fssr_clean[k] = compute_fssr_direct(node, target, x_T_values)
        fssr_noisy[k] = add_noise_to_fssr(fssr_clean[k], config.snr_db)

    results['fssr'] = {
        'clean': fssr_clean, 'noisy': fssr_noisy,
        'x_T': x_T_values, 't': t_values,
        'xi_per_node': xi_per_node,
        'zone_A_mask_per_node': zone_A_mask_per_node,
        'zone_B_mask_per_node': zone_B_mask_per_node,
        'any_zone_A': any_zone_A,
        'all_zone_B': all_zone_B,
    }
    results['timing']['fssr'] = timer.time() - t0
    if verbose:
        print(f"FSSR computation: {results['timing']['fssr']:.2f}s")
        for k in range(K):
            fc = fssr_clean[k]
            dev = max(abs(fc.min() - 1), abs(fc.max() - 1))
            print(f"  Node {k}: range=[{fc.min():.4f}, {fc.max():.4f}], "
                  f"max deviation={dev:.4f}")

    # ================================================================
    # Zone B: parameter estimation
    # ================================================================
    t0 = timer.time()
    if verbose:
        print(f"\n--- Zone B ---")

    # Use first contiguous Zone B segment (target approaching)
    zone_B_indices = np.where(all_zone_B)[0]
    if len(zone_B_indices) > 1:
        gaps = np.where(np.diff(zone_B_indices) > 1)[0]
        if len(gaps) > 0:
            zone_B_seg1 = zone_B_indices[:gaps[0] + 1]
        else:
            zone_B_seg1 = zone_B_indices
    else:
        zone_B_seg1 = zone_B_indices

    t_zone_B = t_values[zone_B_seg1]

    if verbose:
        print(f"  Zone B segment 1: {len(t_zone_B)} snapshots "
              f"(x_T: {x_T_values[zone_B_seg1[0]]:.0f} → "
              f"{x_T_values[zone_B_seg1[-1]]:.0f})")

    # Generate Zone B signals at all nodes
    zone_b_signals = []
    for node in nodes:
        X_k = generate_received_signal_zone_b(
            node, target, t_zone_B, snr_db=config.snr_db
        )
        zone_b_signals.append(X_k)

    # Estimate parameters
    zone_b_params = estimate_parameters_zone_b(
        nodes, zone_b_signals, dt, R_nominal=R, verbose=verbose
    )

    v_x_est = zone_b_params['v_x']
    R_est = zone_b_params['R']
    x_0_est = zone_b_params['x_0']

    results['zone_b'] = zone_b_params
    results['timing']['zone_b'] = timer.time() - t0

    if verbose:
        print(f"\n  Zone B estimates vs true:")
        print(f"    v_x: est={v_x_est:.4f}, true={target.v_x}")
        print(f"    R:   est={R_est:.1f}, true={R}")
        print(f"    x_0: est={x_0_est:.1f}, true={target.x_0}")

    # ================================================================
    # Zone A: gradient descent retrieval with ESTIMATED params
    # ================================================================
    t0 = timer.time()
    if verbose:
        print(f"\n--- Zone A: Gradient Descent (estimated params) ---")

    # Collect Zone A FSSR samples per node
    fssr_zone_A_per_node = []
    t_zone_A_per_node = []
    x_T_zone_A_per_node_est = []

    for k in range(K):
        mask_k = zone_A_mask_per_node[k]
        if np.sum(mask_k) > 0:
            fssr_zone_A_per_node.append(fssr_noisy[k][mask_k])
            t_k = t_values[mask_k]
            t_zone_A_per_node.append(t_k)
            # Estimated x_T trajectory for this node's Zone A
            x_T_est_k = x_0_est + v_x_est * t_k
            x_T_zone_A_per_node_est.append(x_T_est_k)

    # Joint retrieval with velocity and z_T estimation
    z_T_init = 0.0  # z_T unknown from Zone B; start with 0

    retrieval_result = retrieve_with_velocity_estimation(
        fssr_observations=fssr_zone_A_per_node,
        nodes=[nodes[k] for k in range(K) if np.sum(zone_A_mask_per_node[k]) > 0],
        target=target,
        t_values_per_node=t_zone_A_per_node,
        v_x_init=v_x_est,
        x_0_est=x_0_est,
        R_est=R_est,
        z_T_init=z_T_init,
        gamma_profile=5000.0,
        tau=1e-6,
        N_S_inner=100,
        N_V=20,
        gamma_v=0.5,
        dv=0.01,
        estimate_z_T=True,
        gamma_z=0.5,
        dz=0.5,
        estimate_R=True,
        gamma_R=10.0,
        dR=10.0,
        N_D=20,
        verbose=verbose,
    )

    P_estimated = retrieval_result['P_binary']
    acc_est = compute_retrieval_accuracy(P_estimated, target.silhouette)

    results['zone_a_estimated'] = {
        'P_retrieved': P_estimated,
        'P_continuous': retrieval_result['P_continuous'],
        'accuracy': acc_est,
        'v_x_refined': retrieval_result['v_x'],
        'z_T_refined': retrieval_result['z_T'],
        'R_refined': retrieval_result['R'],
        'costs': retrieval_result['costs'],
    }
    results['timing']['zone_a_estimated'] = timer.time() - t0

    # ================================================================
    # Zone A: retrieval with TRUE params (oracle)
    # ================================================================
    t0 = timer.time()
    if verbose:
        print(f"\n--- Zone A: Oracle (true params) ---")

    # Oracle x_T trajectories
    x_T_zone_A_per_node_true = []
    fssr_zone_A_oracle = []
    nodes_with_zone_A = []

    for k in range(K):
        mask_k = zone_A_mask_per_node[k]
        if np.sum(mask_k) > 0:
            fssr_zone_A_oracle.append(fssr_noisy[k][mask_k])
            x_T_zone_A_per_node_true.append(x_T_values[mask_k])
            nodes_with_zone_A.append(nodes[k])

    P_oracle, oracle_cost = retrieve_shadow_profile_gradient(
        fssr_observations=fssr_zone_A_oracle,
        nodes=nodes_with_zone_A,
        target=target,
        x_T_values_per_node=x_T_zone_A_per_node_true,
        R_est=R,
        z_T_est=target.z_T,
        gamma=5000.0,
        tau=1e-6,
        N_S=200,
        N_D=20,
        verbose=verbose,
    )

    P_oracle_binary = (P_oracle >= 0.5).astype(float)
    acc_ora = compute_retrieval_accuracy(P_oracle_binary, target.silhouette)

    results['zone_a_oracle'] = {
        'P_retrieved': P_oracle_binary,
        'P_continuous': P_oracle,
        'accuracy': acc_ora,
    }
    results['timing']['zone_a_oracle'] = timer.time() - t0
    results['zone_a'] = results['zone_a_estimated']

    # ================================================================
    # Summary
    # ================================================================
    results['timing']['total'] = sum(results['timing'].values())
    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS (v5, x-motion)")
        print(f"{'='*60}")
        print(f"Zone B: v_x err={abs(v_x_est-target.v_x):.4f}, "
              f"R err={abs(R_est-R):.1f}, "
              f"x_0 err={abs(x_0_est-target.x_0):.1f}")
        ra = results['zone_a_estimated']
        print(f"Zone A refined: v_x={ra['v_x_refined']:.4f} "
              f"(true={target.v_x}), "
              f"z_T={ra['z_T_refined']:.2f} (true={target.z_T})")
        if 'R_refined' in ra:
            print(f"  R={ra['R_refined']:.1f} (true={R})")
        print(f"\n{'Metric':<25} {'Estimated':>10} {'Oracle':>10}")
        print("-" * 45)
        for m in ['pixel_accuracy', 'iou', 'true_positive_rate',
                   'false_positive_rate']:
            print(f"{m:<25} {acc_est[m]:>10.4f} {acc_ora[m]:>10.4f}")
        print(f"\nTotal: {results['timing']['total']:.1f}s")

    return results


if __name__ == "__main__":
    results = run_simulation()
