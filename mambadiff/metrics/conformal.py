"""Conformal prediction wrapper for calibrated trajectory uncertainty regions.

Reference: Split conformal prediction (Vovk et al.) applied to multi-modal
trajectory prediction. Given K predicted trajectories, constructs uncertainty
regions with guarantee P(GT within region) >= 1-alpha.

Supports two modes:
- per_step: independent quantile per timestep (original)
- max_over_time: single quantile on max-over-time error (proper full-trajectory coverage)
"""

import torch
import numpy as np


class ConformalTrajectoryCalibrator:
    def __init__(self, alpha=0.1, mode='max_over_time'):
        self.alpha = alpha
        self.mode = mode
        self.quantiles = None

    def calibrate(self, val_preds, val_gt):
        """Compute calibration quantiles from validation set.

        Args:
            val_preds: [N, K, T, 2] (nm)
            val_gt: [N, T, 2] (nm)
        """
        errors = (val_gt.unsqueeze(1) - val_preds).norm(dim=-1)
        min_errors = errors.min(dim=1)[0]  # [N, T]

        n = min_errors.shape[0]
        q = min((n + 1.0) * (1 - self.alpha) / n, 1.0)

        if self.mode == 'max_over_time':
            max_errors = min_errors.max(dim=1)[0]  # [N]
            self.quantiles = torch.quantile(max_errors, q)  # scalar
        else:
            self.quantiles = torch.quantile(min_errors, q, dim=0)  # [T]

    def evaluate(self, test_preds, test_gt):
        """Evaluate coverage and region size on test set."""
        assert self.quantiles is not None, "Must call calibrate() first"

        errors = (test_gt.unsqueeze(1) - test_preds).norm(dim=-1)
        min_errors = errors.min(dim=1)[0]  # [N, T]

        if self.mode == 'max_over_time':
            max_errors = min_errors.max(dim=1)[0]
            covered = max_errors <= self.quantiles
            per_step_coverage = (min_errors <= self.quantiles).float().mean(dim=0)
            return {
                'coverage': covered.float().mean().item(),
                'per_step_coverage': per_step_coverage.tolist(),
                'mean_radius_nm': self.quantiles.item(),
                'max_radius_nm': self.quantiles.item(),
                'radius_per_step': [self.quantiles.item()] * min_errors.shape[1],
            }
        else:
            per_step_covered = min_errors <= self.quantiles.unsqueeze(0)
            all_steps_covered = per_step_covered.all(dim=1)
            per_step_coverage = per_step_covered.float().mean(dim=0)
            return {
                'coverage': all_steps_covered.float().mean().item(),
                'per_step_coverage': per_step_coverage.tolist(),
                'mean_radius_nm': self.quantiles.mean().item(),
                'max_radius_nm': self.quantiles.max().item(),
                'radius_per_step': self.quantiles.tolist(),
            }

    def state_dict(self):
        return {'alpha': self.alpha, 'mode': self.mode, 'quantiles': self.quantiles}

    def load_state_dict(self, state):
        self.alpha = state['alpha']
        self.mode = state.get('mode', 'max_over_time')
        self.quantiles = state['quantiles']