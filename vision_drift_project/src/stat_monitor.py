"""
Statistical Drift Monitor

Tracks mean and variance profiles of baseline feature embeddings and
compares incoming production windows against them. Drift is flagged
when the combined score (weighted mean deviation + variance expansion)
exceeds a configurable threshold.
"""

import torch
import numpy as np


class StatisticalDriftMonitor:
    """
    Detects distribution drift by comparing first/second-moment statistics
    (mean, variance) of production embeddings against a calibrated baseline.
    """

    def __init__(self, threshold=2.0, alpha=0.6):
        self.threshold = threshold
        self.alpha = alpha

        self._baseline_mean = None
        self._baseline_var = None
        self._baseline_embeddings = None
        self._is_fitted = False

    def fit(self, baseline_embeddings):
        """Compute and store baseline mean/variance profiles."""
        if isinstance(baseline_embeddings, np.ndarray):
            baseline_embeddings = torch.from_numpy(baseline_embeddings).float()

        self._baseline_embeddings = baseline_embeddings.cpu()
        self._baseline_mean = baseline_embeddings.mean(dim=0).cpu()
        self._baseline_var = baseline_embeddings.var(dim=0).cpu()
        self._is_fitted = True

        n, d = baseline_embeddings.shape
        print(f"[StatMonitor] Calibrated: {n} samples, dim={d}")
        print(f"[StatMonitor] Baseline mean norm={self._baseline_mean.norm():.4f}, "
              f"var mean={self._baseline_var.mean():.6f}")

    def check_drift(self, production_embeddings):
        """
        Check whether production embeddings have drifted from baseline.

        Returns:
            (is_drifted, drift_score, details_dict)
        """
        if not self._is_fitted:
            raise RuntimeError("Monitor not calibrated. Call fit() first.")

        if isinstance(production_embeddings, np.ndarray):
            production_embeddings = torch.from_numpy(production_embeddings).float()

        production_embeddings = production_embeddings.cpu()

        prod_mean = production_embeddings.mean(dim=0)
        prod_var = production_embeddings.var(dim=0)

        eps = 1e-8
        baseline_std = torch.sqrt(self._baseline_var + eps)

        # Normalized mean shift
        mean_shift = (prod_mean - self._baseline_mean) / baseline_std
        mean_deviation = mean_shift.norm().item()

        # Variance ratio deviation
        variance_ratio = prod_var / (self._baseline_var + eps)
        variance_expansion = (variance_ratio - 1.0).abs().mean().item()

        # Weighted combination
        drift_score = self.alpha * mean_deviation + (1.0 - self.alpha) * variance_expansion
        is_drifted = drift_score > self.threshold

        details = {
            'mean_deviation': mean_deviation,
            'variance_expansion': variance_expansion,
            'prod_mean_norm': prod_mean.norm().item(),
            'prod_var_mean': prod_var.mean().item(),
            'baseline_mean_norm': self._baseline_mean.norm().item(),
            'baseline_var_mean': self._baseline_var.mean().item(),
            'threshold': self.threshold,
        }

        return is_drifted, drift_score, details

    def get_baseline_stats(self):
        """Return stored baseline mean, variance, and raw embeddings."""
        if not self._is_fitted:
            raise RuntimeError("Monitor not calibrated. Call fit() first.")

        return {
            'mean': self._baseline_mean,
            'var': self._baseline_var,
            'embeddings': self._baseline_embeddings,
        }
