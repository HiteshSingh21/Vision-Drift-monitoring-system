"""
KDE Distribution Plots for drift analysis.

Generates before/after retraining distribution plots and a combined
side-by-side report using Seaborn KDE over the most drifted feature dims.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# Plot styling
STYLE_CONFIG = {
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#b0b0b0',
    'ytick.color': '#b0b0b0',
    'grid.color': '#2a2a4a',
    'grid.alpha': 0.4,
    'font.family': 'sans-serif',
    'font.size': 11,
}

BASELINE_COLOR = '#00d2ff'
DRIFTED_COLOR = '#ff6b6b'
RECOVERED_COLOR = '#51cf66'


def _select_top_drifted_dims(baseline_emb, compare_emb, top_k=4):
    """Pick the top-K feature dimensions with the largest mean shift."""
    baseline_mean = baseline_emb.mean(axis=0)
    compare_mean = compare_emb.mean(axis=0)
    shift = np.abs(compare_mean - baseline_mean)
    return np.argsort(shift)[-top_k:][::-1]


def _to_numpy(tensor):
    """Convert tensor or array to numpy."""
    if hasattr(tensor, 'cpu'):
        return tensor.cpu().detach().numpy()
    return np.asarray(tensor)


def plot_drift_distributions(baseline_emb, drifted_emb, save_path='drift_before_training.png', top_k=4):
    """
    Plot KDE curves comparing baseline vs drifted distributions
    for the top-K most shifted feature dimensions.
    """
    baseline_np = _to_numpy(baseline_emb)
    drifted_np = _to_numpy(drifted_emb)

    dims = _select_top_drifted_dims(baseline_np, drifted_np, top_k=top_k)

    with plt.rc_context(STYLE_CONFIG):
        fig, axes = plt.subplots(1, top_k, figsize=(5 * top_k, 5))
        if top_k == 1:
            axes = [axes]

        fig.suptitle('DRIFT DETECTED - Feature Distribution Shift (Before Retraining)',
                     fontsize=16, fontweight='bold', color='#ff6b6b', y=1.02)

        for idx, dim in enumerate(dims):
            ax = axes[idx]

            sns.kdeplot(baseline_np[:, dim], ax=ax, color=BASELINE_COLOR,
                        fill=True, alpha=0.3, linewidth=2, label='Baseline')
            sns.kdeplot(drifted_np[:, dim], ax=ax, color=DRIFTED_COLOR,
                        fill=True, alpha=0.3, linewidth=2, label='Drifted')

            b_mu = baseline_np[:, dim].mean()
            b_var = baseline_np[:, dim].var()
            d_mu = drifted_np[:, dim].mean()
            d_var = drifted_np[:, dim].var()

            stats_text = (f'Baseline: mu={b_mu:.3f}, var={b_var:.4f}\n'
                          f'Drifted:  mu={d_mu:.3f}, var={d_var:.4f}')
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f3460', alpha=0.8),
                    family='monospace', color='#e0e0e0')

            ax.set_title(f'Feature Dim {dim}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Activation Value')
            ax.set_ylabel('Density')
            ax.legend(loc='upper right', fontsize=9, framealpha=0.7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        print(f"[Visualizer] Before-training plot saved: {save_path}")


def plot_recovery_distributions(baseline_emb, recovered_emb, save_path='drift_after_training.png', top_k=4):
    """
    Plot KDE curves comparing baseline vs recovered distributions
    after simulated retraining.
    """
    baseline_np = _to_numpy(baseline_emb)
    recovered_np = _to_numpy(recovered_emb)

    dims = _select_top_drifted_dims(baseline_np, recovered_np, top_k=top_k)

    with plt.rc_context(STYLE_CONFIG):
        fig, axes = plt.subplots(1, top_k, figsize=(5 * top_k, 5))
        if top_k == 1:
            axes = [axes]

        fig.suptitle('RECOVERY - Feature Distribution After Retraining',
                     fontsize=16, fontweight='bold', color='#51cf66', y=1.02)

        for idx, dim in enumerate(dims):
            ax = axes[idx]

            sns.kdeplot(baseline_np[:, dim], ax=ax, color=BASELINE_COLOR,
                        fill=True, alpha=0.3, linewidth=2, label='Baseline')
            sns.kdeplot(recovered_np[:, dim], ax=ax, color=RECOVERED_COLOR,
                        fill=True, alpha=0.3, linewidth=2, label='Recovered')

            b_mu = baseline_np[:, dim].mean()
            b_var = baseline_np[:, dim].var()
            r_mu = recovered_np[:, dim].mean()
            r_var = recovered_np[:, dim].var()

            stats_text = (f'Baseline:  mu={b_mu:.3f}, var={b_var:.4f}\n'
                          f'Recovered: mu={r_mu:.3f}, var={r_var:.4f}')
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f3460', alpha=0.8),
                    family='monospace', color='#e0e0e0')

            ax.set_title(f'Feature Dim {dim}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Activation Value')
            ax.set_ylabel('Density')
            ax.legend(loc='upper right', fontsize=9, framealpha=0.7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        print(f"[Visualizer] After-training plot saved: {save_path}")


def plot_combined_report(baseline_emb, drifted_emb, recovered_emb,
                         save_path='drift_combined_report.png', top_k=3):
    """
    Side-by-side comparison: before retraining (left) vs after (right)
    for the top-K most drifted dimensions.
    """
    baseline_np = _to_numpy(baseline_emb)
    drifted_np = _to_numpy(drifted_emb)
    recovered_np = _to_numpy(recovered_emb)

    dims = _select_top_drifted_dims(baseline_np, drifted_np, top_k=top_k)

    with plt.rc_context(STYLE_CONFIG):
        fig, axes = plt.subplots(top_k, 2, figsize=(14, 4.5 * top_k))
        if top_k == 1:
            axes = axes.reshape(1, 2)

        fig.suptitle('Drift Report - Before vs After Retraining',
                     fontsize=18, fontweight='bold', color='#e0e0e0', y=1.01)

        for row, dim in enumerate(dims):
            # Left: before retraining
            ax_before = axes[row, 0]
            sns.kdeplot(baseline_np[:, dim], ax=ax_before, color=BASELINE_COLOR,
                        fill=True, alpha=0.3, linewidth=2, label='Baseline')
            sns.kdeplot(drifted_np[:, dim], ax=ax_before, color=DRIFTED_COLOR,
                        fill=True, alpha=0.3, linewidth=2, label='Drifted')
            ax_before.set_title(f'Before Retraining - Dim {dim}',
                                fontsize=12, fontweight='bold', color='#ff6b6b')
            ax_before.set_xlabel('Activation Value')
            ax_before.set_ylabel('Density')
            ax_before.legend(fontsize=9, framealpha=0.7)
            ax_before.grid(True, alpha=0.3)

            # Right: after retraining
            ax_after = axes[row, 1]
            sns.kdeplot(baseline_np[:, dim], ax=ax_after, color=BASELINE_COLOR,
                        fill=True, alpha=0.3, linewidth=2, label='Baseline')
            sns.kdeplot(recovered_np[:, dim], ax=ax_after, color=RECOVERED_COLOR,
                        fill=True, alpha=0.3, linewidth=2, label='Recovered')
            ax_after.set_title(f'After Retraining - Dim {dim}',
                               fontsize=12, fontweight='bold', color='#51cf66')
            ax_after.set_xlabel('Activation Value')
            ax_after.set_ylabel('Density')
            ax_after.legend(fontsize=9, framealpha=0.7)
            ax_after.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        print(f"[Visualizer] Combined report saved: {save_path}")
