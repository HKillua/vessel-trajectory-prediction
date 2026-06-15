"""COLREGs-aware evaluation metrics for ship trajectory prediction.

CCR (COLREGs Compliance Rate):
    For head-on and crossing-give-way encounters, COLREGs Rule 14/15
    require the give-way vessel to alter course to starboard (right).
    CCR measures the fraction of such encounters where the predicted
    trajectory shows a net rightward course change.

Predicted DCPA:
    Distance at Closest Point of Approach computed from predicted target
    trajectory and ground truth neighbor trajectories. Compared against
    ground truth DCPA to assess whether predictions preserve safe spatial
    relationships.
"""

import math
import torch


def compute_ccr(pred_best, encounter_type, target_idx, mask, thresholds_deg=None):
    """COLREGs Compliance Rate at multiple thresholds.

    Args:
        pred_best: [B, T, 2] best predicted trajectory (denormalized, nm)
        encounter_type: [B, N, N] encounter type matrix
            (0=safe, 1=head_on, 2=crossing_gw, 3=crossing_so, 4=overtaking)
        target_idx: [B] target ship index
        mask: [B, N] valid ship mask
        thresholds_deg: list of degree thresholds (default [5, 15, 30])

    Returns:
        dict: {threshold_deg: (n_compliant, n_applicable)}
    """
    if thresholds_deg is None:
        thresholds_deg = [5.0, 15.0, 30.0]

    B, T, _ = pred_best.shape
    batch_idx = torch.arange(B, device=pred_best.device)

    target_enc = encounter_type[batch_idx, target_idx]  # [B, N]
    target_enc = target_enc * mask.long()

    applicable = ((target_enc == 1) | (target_enc == 2)).any(dim=1)  # [B]
    n_applicable = applicable.sum().item()

    if n_applicable == 0:
        return {th: (0, 0) for th in thresholds_deg}

    delta = pred_best[:, 1:] - pred_best[:, :-1]  # [B, T-1, 2]
    speed_sq = delta[..., 0] ** 2 + delta[..., 1] ** 2
    is_moving = speed_sq > 1e-6
    raw_headings = torch.atan2(delta[..., 1], delta[..., 0])  # [B, T-1]

    headings = raw_headings.clone()
    for t in range(1, headings.shape[1]):
        stale = ~is_moving[:, t]
        headings[:, t] = torch.where(stale, headings[:, t - 1], headings[:, t])

    n_seg = headings.shape[1]
    q = max(1, n_seg // 4)

    heading_start = torch.atan2(
        headings[:, :q].sin().mean(dim=1),
        headings[:, :q].cos().mean(dim=1),
    )
    heading_end = torch.atan2(
        headings[:, -q:].sin().mean(dim=1),
        headings[:, -q:].cos().mean(dim=1),
    )

    dh = heading_end - heading_start
    course_change = torch.atan2(torch.sin(dh), torch.cos(dh))

    results = {}
    for th in thresholds_deg:
        threshold = math.radians(th)
        compliant = course_change > threshold
        results[th] = ((compliant & applicable).sum().item(), n_applicable)

    return results


def compute_predicted_dcpa(pred_best, pred_gt_all, target_idx, mask, encounter_type):
    """Predicted DCPA vs ground truth DCPA for encounter pairs.

    Args:
        pred_best: [B, T, 2] best predicted trajectory for target (nm)
        pred_gt_all: [B, N, T, 2] ground truth trajectories, all ships (nm)
        target_idx: [B] target ship index
        mask: [B, N] valid ship mask
        encounter_type: [B, N, N]

    Returns:
        dcpa_error_sum: float, sum of |pred_dcpa - gt_dcpa|
        pred_dcpa_sum: float, sum of predicted DCPAs
        gt_dcpa_sum: float, sum of ground truth DCPAs
        n_pairs: int, number of encounter pairs
    """
    B, N, T, _ = pred_gt_all.shape
    batch_idx = torch.arange(B, device=pred_best.device)

    target_gt = pred_gt_all[batch_idx, target_idx]  # [B, T, 2]
    target_enc = encounter_type[batch_idx, target_idx]  # [B, N]

    self_mask = torch.zeros(B, N, dtype=torch.bool, device=mask.device)
    self_mask[batch_idx, target_idx] = True
    valid = mask & ~self_mask & (target_enc > 0)

    if valid.sum() == 0:
        return 0.0, 0.0, 0.0, 0

    # Vectorized DCPA: broadcast [B,1,T,2] vs [B,N,T,2]
    pred_dcpa = (pred_best.unsqueeze(1) - pred_gt_all).norm(p=2, dim=-1).min(dim=2)[0]
    gt_dcpa = (target_gt.unsqueeze(1) - pred_gt_all).norm(p=2, dim=-1).min(dim=2)[0]

    v = valid.float()
    n_pairs = valid.sum().item()

    return (
        ((pred_dcpa - gt_dcpa).abs() * v).sum().item(),
        (pred_dcpa * v).sum().item(),
        (gt_dcpa * v).sum().item(),
        n_pairs,
    )