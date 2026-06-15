"""Stratified evaluation metrics for ship trajectory prediction.

Reports ADE/FDE broken down by:
- Encounter type (head-on, crossing, overtaking, safe)
- Prediction difficulty (easy/medium/hard based on constant-velocity baseline)
"""

import torch
import numpy as np


def compute_cv_baseline_ade(obs, pred_gt, dt=20.0):
    """Constant-velocity baseline ADE for difficulty classification.

    Args:
        obs: [B, T_obs, 7] target ship observation (denormalized nm/knots)
            Features: (lat_nm, lon_nm, sog_knots, sin_cog, cos_cog, sin_hdg, cos_hdg)
        pred_gt: [B, T_pred, 2] target ship ground truth prediction (nm)
        dt: sampling interval in seconds

    Returns:
        cv_ade: [B] per-sample CV baseline ADE (nm)
    """
    last_pos = obs[:, -1, :2]
    sin_cog = obs[:, -1, 3]
    cos_cog = obs[:, -1, 4]
    last_sog = obs[:, -1, 2]

    cog_rad = torch.atan2(sin_cog, cos_cog)
    T = pred_gt.shape[1]
    steps = torch.arange(1, T + 1, device=obs.device, dtype=obs.dtype)

    speed_nm_per_step = last_sog * (dt / 3600.0)
    vx = speed_nm_per_step * torch.sin(cog_rad)
    vy = speed_nm_per_step * torch.cos(cog_rad)

    cv_lat = last_pos[:, 0:1] + vy.unsqueeze(1) * steps.unsqueeze(0)
    cv_lon = last_pos[:, 1:2] + vx.unsqueeze(1) * steps.unsqueeze(0)
    cv_pred = torch.stack([cv_lat, cv_lon], dim=-1)

    errors = (cv_pred - pred_gt).norm(dim=-1)
    return errors.mean(dim=-1)


def stratified_evaluate(pred_traj, target_gt, encounter_type, target_idx, mask, obs_target):
    """Compute stratified metrics.

    Args:
        pred_traj: [B, K, T, 2] predicted trajectories (nm)
        target_gt: [B, T, 2] ground truth (nm)
        encounter_type: [B, N, N] encounter matrix
        target_idx: [B] target ship index
        mask: [B, N]
        obs_target: [B, T_obs, 7] target ship obs (denormalized nm/knots)

    Returns:
        dict with per-category ADE/FDE and counts
    """
    B, K, T, _ = pred_traj.shape
    batch_idx = torch.arange(B, device=pred_traj.device)

    gt_expand = target_gt.unsqueeze(1).expand_as(pred_traj)
    distances = (pred_traj - gt_expand).norm(dim=-1)
    min_ade = distances.mean(dim=-1).min(dim=1)[0]
    min_fde = distances[:, :, -1].min(dim=1)[0]

    target_enc = encounter_type[batch_idx, target_idx]
    non_safe = (target_enc > 0) & mask
    dominant_type = torch.zeros(B, dtype=torch.long, device=pred_traj.device)
    for b in range(B):
        types = target_enc[b][non_safe[b]]
        if len(types) > 0:
            dominant_type[b] = types.mode()[0]

    cv_ade = compute_cv_baseline_ade(obs_target, target_gt)

    results = {
        'overall': {'ADE': min_ade.mean().item(), 'FDE': min_fde.mean().item(), 'n': B},
    }

    type_names = {0: 'safe', 1: 'head_on', 2: 'crossing_gw', 3: 'crossing_so',
                  4: 'overtaking', 5: 'being_overtaken'}
    for enc_type, name in type_names.items():
        type_mask = dominant_type == enc_type
        n = type_mask.sum().item()
        if n > 0:
            results[f'enc_{name}'] = {
                'ADE': min_ade[type_mask].mean().item(),
                'FDE': min_fde[type_mask].mean().item(),
                'n': n,
            }

    easy_mask = cv_ade < 0.05
    medium_mask = (cv_ade >= 0.05) & (cv_ade < 0.2)
    hard_mask = cv_ade >= 0.2

    for name, diff_mask in [('easy', easy_mask), ('medium', medium_mask), ('hard', hard_mask)]:
        n = diff_mask.sum().item()
        if n > 0:
            results[f'diff_{name}'] = {
                'ADE': min_ade[diff_mask].mean().item(),
                'FDE': min_fde[diff_mask].mean().item(),
                'n': n,
            }

    return results


def format_stratified_results(results):
    """Format stratified results for logging."""
    lines = []
    lines.append(f"  Overall: ADE={results['overall']['ADE']:.4f} FDE={results['overall']['FDE']:.4f} (n={results['overall']['n']})")

    lines.append("  By encounter type:")
    for key in sorted(results.keys()):
        if key.startswith('enc_'):
            r = results[key]
            lines.append(f"    {key[4:]}: ADE={r['ADE']:.4f} FDE={r['FDE']:.4f} (n={r['n']})")

    lines.append("  By difficulty:")
    for key in ['diff_easy', 'diff_medium', 'diff_hard']:
        if key in results:
            r = results[key]
            lines.append(f"    {key[5:]}: ADE={r['ADE']:.4f} FDE={r['FDE']:.4f} (n={r['n']})")

    return '\n'.join(lines)