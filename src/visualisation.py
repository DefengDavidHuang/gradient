"""
visualisation.py — Plotting and Result Visualisation (v9)
==========================================================

Generates publication-quality figures for the v9 simulation results.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def plot_all(results: dict, save_dir: str = "results"):
    """Generate and save all standard figures."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    plot_fssr_curves(results, save_path)
    plot_zone_classification(results, save_path)
    plot_shadow_comparison(results, save_path)
    plot_cost_convergence(results, save_path)
    plot_summary(results, save_path)

    print(f"\nAll figures saved to {save_path}/")


def plot_fssr_curves(results: dict, save_path: Path):
    """FSSR time-series per node."""
    fssr = results['fssr']
    K = len(fssr['clean'])
    fig, axes = plt.subplots(K, 1, figsize=(10, 3 * K), sharex=True)
    if K == 1:
        axes = [axes]

    x_T = fssr['x_T']
    for k in range(K):
        ax = axes[k]
        ax.plot(x_T, fssr['clean'][k], 'b-', lw=1.2, label='Clean')
        ax.plot(x_T, fssr['noisy'][k], 'r.', ms=0.5, alpha=0.3, label='Noisy')
        ax.axhline(1.0, color='gray', ls='--', alpha=0.5)

        mask_A = fssr['zone_A_mask'][k]
        if np.any(mask_A):
            ax.axvspan(x_T[mask_A].min(), x_T[mask_A].max(),
                       alpha=0.1, color='green', label='Zone A')

        nd_x = results['config'].nodes[k].x_centre
        ax.set_ylabel(f'ε (Node {k}, X̄={nd_x:.0f})')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('x_T (λ)')
    fig.suptitle('FSSR vs Target x-Position', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path / 'fssr_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_zone_classification(results: dict, save_path: Path):
    """Trajectory and ξ_k(t) per node."""
    fssr = results['fssr']
    cfg = results['config']
    K = cfg.K

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = fssr['t']
    x_T = fssr['x_T']

    ax1 = axes[0]
    ax1.plot(t, x_T, 'b-', lw=1.5)
    for k, nd in enumerate(cfg.nodes):
        ax1.axhline(nd.x_centre, color=f'C{k}', ls=':', alpha=0.5,
                     label=f'Node {k} (X̄={nd.x_centre:.0f})')
    ax1.set_ylabel('x_T (λ)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Target x-Position and Node Positions')

    ax2 = axes[1]
    for k in range(K):
        ax2.semilogy(t, fssr['xi_per_node'][k], f'C{k}-', lw=1,
                     label=f'Node {k}')
    ax2.axhline(cfg.xi_threshold, color='r', ls='--',
                label=f'ξ_th={cfg.xi_threshold}')
    ax2.set_ylabel('ξ_k')
    ax2.set_xlabel('Time')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0.01)

    fig.tight_layout()
    fig.savefig(save_path / 'zone_classification.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_shadow_comparison(results: dict, save_path: Path):
    """True vs oracle vs estimated shadow profiles."""
    P_true = results['config'].target.silhouette
    P_oracle = results['zone_a_oracle'].get('P_retrieved',
                                             np.zeros_like(P_true))
    za_est = results['zone_a_estimated']
    P_est = za_est.get('P_binary', np.zeros_like(P_true))
    acc_o = results['zone_a_oracle'].get('accuracy', {})
    acc_e = za_est.get('accuracy', {})

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].imshow(P_true, cmap='YlOrBr', origin='lower', aspect='equal')
    axes[0, 0].set_title('True Profile')

    axes[0, 1].imshow(P_oracle, cmap='YlOrBr', origin='lower', aspect='equal')
    axes[0, 1].set_title(f"Oracle (Acc={acc_o.get('pixel_accuracy', 0):.3f},"
                         f" IoU={acc_o.get('iou', 0):.3f})")

    axes[0, 2].imshow(P_est, cmap='YlOrBr', origin='lower', aspect='equal')
    axes[0, 2].set_title(f"Estimated (Acc={acc_e.get('pixel_accuracy', 0):.3f},"
                         f" IoU={acc_e.get('iou', 0):.3f})")

    P_cont = za_est.get('P_continuous', P_est)
    axes[1, 0].imshow(P_cont, cmap='viridis', origin='lower',
                       aspect='equal', vmin=0, vmax=1)
    axes[1, 0].set_title('Estimated (continuous)')

    diff_o = np.abs(P_true - P_oracle)
    axes[1, 1].imshow(diff_o, cmap='Reds', origin='lower',
                       aspect='equal', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Oracle Error ({int(diff_o.sum())} px)')

    diff_e = np.abs(P_true - P_est)
    axes[1, 2].imshow(diff_e, cmap='Reds', origin='lower',
                       aspect='equal', vmin=0, vmax=1)
    axes[1, 2].set_title(f'Estimated Error ({int(diff_e.sum())} px)')

    fig.suptitle('Shadow Profile Comparison', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path / 'shadow_comparison.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_cost_convergence(results: dict, save_path: Path):
    """Cost convergence of Algorithm 3."""
    costs = results.get('zone_a_estimated', {}).get('costs', [])
    if not costs:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(1, len(costs) + 1), costs, 'b-o', ms=4)
    ax.set_xlabel('Outer Iteration (Algorithm 3)')
    ax.set_ylabel('Cost')
    ax.set_title('Joint Parameter Estimation: Cost Convergence')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path / 'cost_convergence.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_summary(results: dict, save_path: Path):
    """Text summary figure."""
    cfg = results['config']
    tgt = cfg.target
    zb = results.get('zone_b', {})
    za = results.get('zone_a_estimated', {})
    acc_o = results.get('zone_a_oracle', {}).get('accuracy', {})
    acc_e = za.get('accuracy', {})

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    text = (
        f"SIMULATION SUMMARY (v9)\n"
        f"{'=' * 55}\n\n"
        f"K={cfg.K}, N={cfg.N} (N²={cfg.N**2}), R_0={cfg.R_0:.0e}, "
        f"Δ_node={cfg.delta_node:.1f}\n"
        f"SNR={cfg.snr_db}dB, ξ_th={cfg.xi_threshold}\n"
        f"Target: x_0={tgt.x_0}, v={tgt.v}, θ={np.degrees(tgt.theta):.1f}°, "
        f"z_T={tgt.z_T}\n\n"
        f"Zone B Estimates:\n"
        f"  v_x: {zb.get('v_x', 0):.4f} (true: {tgt.v_x:.4f})\n"
        f"  v_y: {zb.get('v_y', 0):.6f} (true: {tgt.v_y:.6f})\n"
        f"  R_0: {zb.get('R_0', 0):.1f} (true: {cfg.R_0})\n"
        f"  x_0: {zb.get('x_0', 0):.1f} (true: {tgt.x_0})\n"
        f"  z_T: {zb.get('z_T', 0):.2f} (true: {tgt.z_T})\n\n"
        f"Zone A Refined:\n"
        f"  v_x={za.get('v_x', 'N/A')}, v_y={za.get('v_y', 'N/A')}\n"
        f"  z_T={za.get('z_T', 'N/A')}, R_0={za.get('R_0', 'N/A')}\n"
        f"  x_0={za.get('x_0', 'N/A')}\n\n"
        f"Shadow Profile:\n"
        f"  {'Metric':<25} {'Est':>8} {'Oracle':>8}\n"
        f"  {'-' * 41}\n"
    )
    for m in ['pixel_accuracy', 'iou', 'true_positive_rate',
               'false_positive_rate']:
        text += (f"  {m:<25} "
                 f"{acc_e.get(m, 0):>8.4f} {acc_o.get(m, 0):>8.4f}\n")

    ax.text(0.05, 0.95, text, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.savefig(save_path / 'summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
