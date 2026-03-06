"""
visualisation.py - Plotting and Result Visualisation (v5)
==========================================================

Updated for x-motion target model with per-node zone classification.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_all(results: dict, save_dir: str = "results"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    plot_fssr_curves(results, save_path)
    plot_zone_classification(results, save_path)
    plot_shadow_comparison(results, save_path)
    plot_cost_convergence(results, save_path)
    plot_summary(results, save_path)

    print(f"\nAll figures saved to {save_path}/")


def plot_fssr_curves(results: dict, save_path: Path):
    fssr_data = results['fssr']
    K = len(fssr_data['clean'])

    fig, axes = plt.subplots(K, 1, figsize=(10, 3 * K), sharex=True)
    if K == 1:
        axes = [axes]

    x_T = fssr_data['x_T']

    for k in range(K):
        ax = axes[k]
        ax.plot(x_T, fssr_data['clean'][k], 'b-', lw=1.5, label='Clean')
        ax.plot(x_T, fssr_data['noisy'][k], 'r.', ms=0.5, alpha=0.3,
                label='Noisy')
        ax.axhline(y=1.0, color='gray', ls='--', alpha=0.5)

        mask_A = fssr_data['zone_A_mask_per_node'][k]
        if np.any(mask_A):
            x_min, x_max = x_T[mask_A].min(), x_T[mask_A].max()
            ax.axvspan(x_min, x_max, alpha=0.1, color='green',
                       label=f'Zone A (Node {k})')

        node_x = results['config'].nodes[k].x_centre
        ax.set_ylabel(f'FSSR (Node {k}, X={node_x:.0f})')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('x_T (wavelengths)')
    fig.suptitle('FSSR vs Target x-Position', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path / 'fssr_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_zone_classification(results: dict, save_path: Path):
    fssr_data = results['fssr']
    config = results['config']
    K = config.K

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    t = fssr_data['t']
    x_T = fssr_data['x_T']

    # Plot 1: x_T trajectory
    ax1 = axes[0]
    ax1.plot(t, x_T, 'b-', lw=1.5)
    for k, node in enumerate(config.nodes):
        ax1.axhline(y=node.x_centre, color=f'C{k}', ls=':', alpha=0.5,
                     label=f'Node {k} (X={node.x_centre:.0f})')
    ax1.set_ylabel('x_T (λ)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Target x-Position and Node Positions')

    # Plot 2: ξ per node
    ax2 = axes[1]
    for k in range(K):
        xi_k = fssr_data['xi_per_node'][k]
        ax2.semilogy(t, xi_k, f'C{k}-', lw=1, label=f'Node {k}')

    ax2.axhline(y=config.xi_threshold, color='r', ls='--',
                label=f'ξ_th = {config.xi_threshold}')
    ax2.set_ylabel('ξ_k = |x_T - X_k| / √R')
    ax2.set_xlabel('Time')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0.01)

    fig.tight_layout()
    fig.savefig(save_path / 'zone_classification.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_shadow_comparison(results: dict, save_path: Path):
    """Plot true, oracle, and estimated shadow profiles."""
    P_true = results['config'].target.silhouette
    P_oracle = results['zone_a_oracle']['P_retrieved']
    P_estimated = results['zone_a_estimated']['P_retrieved']
    acc_o = results['zone_a_oracle']['accuracy']
    acc_e = results['zone_a_estimated']['accuracy']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Row 0: profiles
    axes[0, 0].imshow(P_true, cmap='YlOrBr', origin='lower', aspect='equal')
    axes[0, 0].set_title('True Profile')

    axes[0, 1].imshow(P_oracle, cmap='YlOrBr', origin='lower',
                       aspect='equal')
    axes[0, 1].set_title(f'Oracle (Acc: {acc_o["pixel_accuracy"]:.3f}, '
                          f'IoU: {acc_o["iou"]:.3f})')

    axes[0, 2].imshow(P_estimated, cmap='YlOrBr', origin='lower',
                       aspect='equal')
    axes[0, 2].set_title(f'Estimated (Acc: {acc_e["pixel_accuracy"]:.3f}, '
                          f'IoU: {acc_e["iou"]:.3f})')

    # Row 1: continuous profiles and errors
    P_cont_est = results['zone_a_estimated'].get('P_continuous', P_estimated)
    axes[1, 0].imshow(P_cont_est, cmap='viridis', origin='lower',
                       aspect='equal', vmin=0, vmax=1)
    axes[1, 0].set_title('Estimated (continuous)')

    diff_o = np.abs(P_true - P_oracle)
    axes[1, 1].imshow(diff_o, cmap='Reds', origin='lower', aspect='equal',
                       vmin=0, vmax=1)
    axes[1, 1].set_title(f'Oracle Error ({int(diff_o.sum())} px)')

    diff_e = np.abs(P_true - P_estimated)
    axes[1, 2].imshow(diff_e, cmap='Reds', origin='lower', aspect='equal',
                       vmin=0, vmax=1)
    axes[1, 2].set_title(f'Estimated Error ({int(diff_e.sum())} px)')

    fig.suptitle('Shadow Profile: Oracle vs Estimated (Gradient Descent + TV)',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path / 'shadow_comparison.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_cost_convergence(results: dict, save_path: Path):
    """Plot cost function convergence for the joint estimation."""
    costs = results['zone_a_estimated'].get('costs', [])
    if not costs:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(1, len(costs) + 1), costs, 'b-o', ms=4)
    ax.set_xlabel('Outer Iteration')
    ax.set_ylabel('Cost')
    ax.set_title('Joint Parameter Estimation: Cost Convergence')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path / 'cost_convergence.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_summary(results: dict, save_path: Path):
    config = results['config']
    zone_b = results['zone_b']
    acc_o = results['zone_a_oracle']['accuracy']
    acc_e = results['zone_a_estimated']['accuracy']
    ra = results['zone_a_estimated']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    text = (
        f"SIMULATION SUMMARY (v5, x-motion)\n"
        f"{'='*55}\n\n"
        f"System: K={config.K}, N={config.nodes[0].N}, "
        f"f_c={config.fc_ghz}GHz, SNR={config.snr_db}dB\n"
        f"R = {config.nodes[0].R:.0e}λ, "
        f"spacing = √R/2 = {np.sqrt(config.nodes[0].R)/2:.0f}λ\n"
        f"Nodes at: {[f'{n.x_centre:.0f}' for n in config.nodes]}\n"
        f"Target: x_0={config.target.x_0}, v_x={config.target.v_x}, "
        f"z_T={config.target.z_T}\n\n"
        f"Zone B Estimates:\n"
        f"  v_x: {zone_b['v_x']:.4f} (true: {config.target.v_x})\n"
        f"  R:   {zone_b['R']:.1f} (true: {config.nodes[0].R})\n"
        f"  x_0: {zone_b['x_0']:.1f} (true: {config.target.x_0})\n\n"
        f"Zone A Refined:\n"
        f"  v_x: {ra['v_x_refined']:.4f}\n"
        f"  z_T: {ra['z_T_refined']:.2f} (true: {config.target.z_T})\n"
        f"  R:   {ra.get('R_refined', 'N/A')}\n\n"
        f"Shadow Profile Retrieval:\n"
        f"  {'Metric':<25} {'Estimated':>10} {'Oracle':>10}\n"
        f"  {'-'*45}\n"
        f"  {'Pixel accuracy':<25} {acc_e['pixel_accuracy']:>10.4f}"
        f" {acc_o['pixel_accuracy']:>10.4f}\n"
        f"  {'IoU':<25} {acc_e['iou']:>10.4f}"
        f" {acc_o['iou']:>10.4f}\n"
        f"  {'TPR':<25} {acc_e['true_positive_rate']:>10.4f}"
        f" {acc_o['true_positive_rate']:>10.4f}\n"
        f"  {'FPR':<25} {acc_e['false_positive_rate']:>10.4f}"
        f" {acc_o['false_positive_rate']:>10.4f}\n"
    )

    ax.text(0.05, 0.95, text, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.savefig(save_path / 'summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
