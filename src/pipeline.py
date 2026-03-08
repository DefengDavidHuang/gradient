"""
pipeline.py — End-to-End Simulation Pipeline (v9)
===================================================

Orchestrates the full processing chain (Section 8 of the v9 document):

1. Signal generation → FSSR at all nodes / all times.
2. Zone classification per node per time step.
3. Zone B → 2-D DOA + Doppler → closed-form init → NLS → (v_x, v_y, R_0, x_0, z_T).
4. Zone A → Algorithm 3 joint profile + parameter estimation.
5. Output: P̂ ∈ {0,1}^{M_z×M_x} and refined parameters.
"""

from __future__ import annotations

import time as _timer
import numpy as np
from typing import Dict, Any

from .geometry import SystemConfig, create_system, node_positions
from .fresnel import compute_fssr_for_node, add_fssr_noise
from .signals import generate_zone_b_signal, measure_fssr_from_signal
from .zone_b import estimate_parameters_zone_b
from .zone_a import (
    retrieve_shadow_profile, joint_estimation,
    compute_retrieval_accuracy,
)


def run_simulation(
    config: SystemConfig | None = None,
    doa_method: str = "music",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the complete v9 simulation.

    Parameters
    ----------
    config : SystemConfig or None
        If None, default parameters are used.
    doa_method : str
        ``"music"`` or ``"esprit"`` for Zone B DOA.
    verbose : bool

    Returns
    -------
    dict — results including FSSR, Zone B estimates, Zone A retrieval.
    """
    if config is None:
        config = create_system()

    results: Dict[str, Any] = {'config': config, 'timing': {}}
    target = config.target
    nodes = config.nodes
    K = config.K
    N = config.N
    R_0 = config.R_0
    dt = config.dt
    r_F = np.sqrt(R_0)

    # ------------------------------------------------------------------
    # Time axis — centre around the node-crossing epoch
    # ------------------------------------------------------------------
    v_x = target.v_x
    v_y = target.v_y
    if abs(v_x) < 1e-12:
        v_x_eff = 1e-6 * np.sign(v_x) if v_x != 0 else 1e-6
    else:
        v_x_eff = v_x

    # Time when target centre crosses x = 0 (middle of node array)
    t_cross = -target.x_0 / v_x_eff
    # Half-span: ensure we see target approach and depart
    half_span = abs(target.x_0 / v_x_eff)
    t_start = t_cross - half_span
    t_end = t_cross + half_span
    T_total = t_end - t_start
    M_total = max(int(T_total / dt), 2000)
    t_values = t_start + np.arange(M_total) * dt

    # Target kinematics
    x_T = target.x_T(t_values)
    R_t = target.R(t_values, R_0)

    # ------------------------------------------------------------------
    # Zone classification per node (eq. 11)
    # ------------------------------------------------------------------
    xi_per_node = {}
    zA_mask = {}
    zB_mask = {}
    for k, nd in enumerate(nodes):
        xi_k = np.abs(x_T - nd.x_centre) / np.sqrt(np.maximum(R_t, 1.0))
        xi_per_node[k] = xi_k
        zA_mask[k] = xi_k <= config.xi_threshold
        zB_mask[k] = xi_k > config.xi_threshold

    any_zA = np.zeros(M_total, dtype=bool)
    all_zB = np.ones(M_total, dtype=bool)
    for k in range(K):
        any_zA |= zA_mask[k]
        all_zB &= zB_mask[k]

    if verbose:
        print("=" * 60)
        print("Distributed Phased Array FSR Simulation (v9)")
        print("=" * 60)
        print(f"K={K}, N={N} (N²={N*N}), R_0={R_0:.0e}, "
              f"Δ_node={config.delta_node:.1f}")
        print(f"Nodes at: {[f'{n.x_centre:.1f}' for n in nodes]}")
        print(f"Target: x_0={target.x_0}, v={target.v}, "
              f"θ={np.degrees(target.theta):.1f}°, z_T={target.z_T}")
        print(f"  v_x={target.v_x:.4f}, v_y={target.v_y:.6f}")
        print(f"Grid: {config.M_z}×{config.M_x}, 2δ={target.pixel_size}")
        print(f"M_total={M_total}, dt={dt}")
        for k in range(K):
            print(f"  Node {k} (X̄={nodes[k].x_centre:.1f}): "
                  f"Zone A = {int(np.sum(zA_mask[k]))} snapshots")
        print()

    # ------------------------------------------------------------------
    # FSSR (Fresnel model — all nodes, all times)
    # ------------------------------------------------------------------
    t0 = _timer.time()
    fssr_clean, fssr_noisy = {}, {}
    for k, nd in enumerate(nodes):
        fssr_clean[k] = compute_fssr_for_node(nd, target, t_values, R_0)
        fssr_noisy[k] = add_fssr_noise(fssr_clean[k], config.snr_db)

    results['fssr'] = {
        'clean': fssr_clean, 'noisy': fssr_noisy,
        'x_T': x_T, 't': t_values, 'R_t': R_t,
        'xi_per_node': xi_per_node,
        'zone_A_mask': zA_mask, 'zone_B_mask': zB_mask,
        'any_zone_A': any_zA, 'all_zone_B': all_zB,
    }
    results['timing']['fssr'] = _timer.time() - t0
    if verbose:
        print(f"FSSR computation: {results['timing']['fssr']:.2f}s")
        for k in range(K):
            fc = fssr_clean[k]
            print(f"  Node {k}: range=[{fc.min():.4f}, {fc.max():.4f}]")

    # ------------------------------------------------------------------
    # Zone B
    # ------------------------------------------------------------------
    t0 = _timer.time()
    if verbose:
        print(f"\n--- Zone B ---")

    # First contiguous all-Zone-B segment
    zB_idx = np.where(all_zB)[0]
    if len(zB_idx) > 1:
        gaps = np.where(np.diff(zB_idx) > 1)[0]
        seg = zB_idx[:gaps[0] + 1] if len(gaps) > 0 else zB_idx
    else:
        seg = zB_idx
    t_zoneB = t_values[seg]

    if verbose:
        print(f"  Zone B segment: {len(t_zoneB)} snapshots "
              f"(x_T: {x_T[seg[0]]:.0f} → {x_T[seg[-1]]:.0f})")

    # Generate N²-element signals at each node
    zb_signals = []
    for nd in nodes:
        X_k = generate_zone_b_signal(nd, target, t_zoneB, R_0, config.snr_db)
        zb_signals.append(X_k)

    # Estimate parameters
    zb_params = estimate_parameters_zone_b(
        nodes, zb_signals, dt, R_0_nominal=R_0,
        t_actual=t_zoneB,
        doa_method=doa_method, verbose=verbose,
    )

    results['zone_b'] = zb_params
    results['timing']['zone_b'] = _timer.time() - t0

    if verbose:
        print(f"\n  Zone B estimates vs true:")
        print(f"    v_x:  est={zb_params['v_x']:.4f}, true={target.v_x:.4f}")
        print(f"    v_y:  est={zb_params['v_y']:.6f}, true={target.v_y:.6f}")
        print(f"    R_0:  est={zb_params['R_0']:.1f}, true={R_0}")
        print(f"    x_0:  est={zb_params['x_0']:.1f}, true={target.x_0}")
        print(f"    z_T:  est={zb_params['z_T']:.2f}, true={target.z_T}")

    # ------------------------------------------------------------------
    # Zone A — estimated parameters
    # ------------------------------------------------------------------
    t0 = _timer.time()
    if verbose:
        print(f"\n--- Zone A: Joint Estimation (estimated params) ---")

    fssr_zA_per_node = []
    t_zA_per_node = []
    nodes_zA = []

    for k in range(K):
        mask = zA_mask[k]
        if np.sum(mask) > 0:
            fssr_zA_per_node.append(fssr_noisy[k][mask])
            t_zA_per_node.append(t_values[mask])
            nodes_zA.append(nodes[k])

    if len(nodes_zA) > 0:
        za_result = joint_estimation(
            fssr_obs=fssr_zA_per_node,
            nodes=nodes_zA,
            target=target,
            t_per_node=t_zA_per_node,
            v_x_init=zb_params['v_x'],
            v_y_init=zb_params['v_y'],
            z_T_init=zb_params['z_T'],
            R_0_init=zb_params['R_0'],
            x_0_init=zb_params['x_0'],
            gamma_profile=config.gamma_profile,
            tau=config.tau,
            N_S=config.N_S,
            N_D=config.N_D,
            N_V=config.N_V,
            verbose=verbose,
        )

        P_est = za_result['P_binary']
        acc_est = compute_retrieval_accuracy(P_est, target.silhouette)
        za_result['accuracy'] = acc_est
    else:
        za_result = {'P_binary': np.zeros_like(target.silhouette),
                     'accuracy': {}, 'costs': []}

    results['zone_a_estimated'] = za_result
    results['timing']['zone_a_estimated'] = _timer.time() - t0

    # ------------------------------------------------------------------
    # Zone A — oracle (true params)
    # ------------------------------------------------------------------
    t0 = _timer.time()
    if verbose:
        print(f"\n--- Zone A: Oracle (true params) ---")

    fssr_zA_oracle = []
    t_zA_oracle = []
    nodes_zA_oracle = []

    for k in range(K):
        mask = zA_mask[k]
        if np.sum(mask) > 0:
            fssr_zA_oracle.append(fssr_noisy[k][mask])
            t_zA_oracle.append(t_values[mask])
            nodes_zA_oracle.append(nodes[k])

    if len(nodes_zA_oracle) > 0:
        P_oracle, oracle_cost = retrieve_shadow_profile(
            fssr_obs=fssr_zA_oracle,
            nodes=nodes_zA_oracle,
            target=target,
            t_per_node=t_zA_oracle,
            R_0=R_0, v_x=target.v_x, v_y=target.v_y,
            z_T=target.z_T, x_0=target.x_0,
            gamma=config.gamma_profile, tau=config.tau,
            N_S=config.N_S, N_D=config.N_D,
            verbose=verbose,
        )
        P_oracle_bin = (P_oracle >= 0.5).astype(float)
        acc_ora = compute_retrieval_accuracy(P_oracle_bin, target.silhouette)
    else:
        P_oracle = np.zeros_like(target.silhouette)
        P_oracle_bin = P_oracle
        acc_ora = {}

    results['zone_a_oracle'] = {
        'P_continuous': P_oracle,
        'P_retrieved': P_oracle_bin,
        'accuracy': acc_ora,
    }
    results['timing']['zone_a_oracle'] = _timer.time() - t0
    results['zone_a'] = results['zone_a_estimated']

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    results['timing']['total'] = sum(results['timing'].values())

    if verbose:
        print(f"\n{'=' * 60}")
        print("RESULTS (v9)")
        print(f"{'=' * 60}")
        zb = zb_params
        print(f"Zone B: v_x err={abs(zb['v_x'] - target.v_x):.4f}, "
              f"v_y err={abs(zb['v_y'] - target.v_y):.6f}, "
              f"R_0 err={abs(zb['R_0'] - R_0):.1f}, "
              f"x_0 err={abs(zb['x_0'] - target.x_0):.1f}, "
              f"z_T err={abs(zb['z_T'] - target.z_T):.2f}")

        if 'v_x' in za_result:
            print(f"Zone A refined: v_x={za_result['v_x']:.4f}, "
                  f"v_y={za_result['v_y']:.6f}, "
                  f"z_T={za_result['z_T']:.2f}, "
                  f"R_0={za_result['R_0']:.1f}, "
                  f"x_0={za_result['x_0']:.1f}")

        a_e = za_result.get('accuracy', {})
        a_o = results['zone_a_oracle'].get('accuracy', {})
        if a_e and a_o:
            print(f"\n{'Metric':<25} {'Estimated':>10} {'Oracle':>10}")
            print("-" * 45)
            for m in ['pixel_accuracy', 'iou', 'true_positive_rate',
                       'false_positive_rate']:
                v_e = a_e.get(m, 0)
                v_o = a_o.get(m, 0)
                print(f"{m:<25} {v_e:>10.4f} {v_o:>10.4f}")

        print(f"\nTotal time: {results['timing']['total']:.1f}s")

    return results
