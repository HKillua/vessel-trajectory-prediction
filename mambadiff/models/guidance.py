import math
import torch
import torch.nn as nn


class COLREGsEnergyVectorized(nn.Module):
    """Two-tier hierarchical COLREGs-inspired energy for diffusion guidance.

    Tier 1 — E_proximity (universal):
        Quadratic penalty when predicted min pairwise distance falls below
        safe_dcpa. Valid for ALL encounter types since ships always avoid
        collision regardless of COLREGs compliance.

    Tier 2 — E_direction (initial maneuver direction):
        For give-way vessels (head_on, crossing_give_way, overtaking), checks
        the initial maneuver direction (first 1/2 of prediction horizon).
        Penalizes port (left) turns when COLREGs requires starboard (right).
        Head-on (Rule 14) requires BOTH vessels to turn starboard.
        Only judges the initial phase to avoid penalizing correct late-stage
        course corrections after passing astern.

    Both terms are CRI-gated so guidance is silent for low-risk pairs.
    """

    def __init__(self, safe_dcpa=0.5, maneuver_threshold_deg=5.0,
                 w_proximity=1.0, w_direction=0.1):
        super().__init__()
        self.safe_dcpa = safe_dcpa
        self.maneuver_threshold = math.radians(maneuver_threshold_deg)
        self.w_proximity = w_proximity
        self.w_direction = w_direction
        self.register_buffer('pos_std', torch.ones(2))

    def set_pos_std(self, std):
        self.pos_std.copy_(std)

    def forward(self, pred, encounter_type, cri, mask):
        B, N, T, _ = pred.shape

        lat = pred[:, :, :, 0] * self.pos_std[0]
        lon = pred[:, :, :, 1] * self.pos_std[1]

        # ---- Tier 1: E_proximity (universal collision avoidance) ----
        dx = lon.unsqueeze(1) - lon.unsqueeze(2)
        dy = lat.unsqueeze(1) - lat.unsqueeze(2)
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        min_dist = dist.min(dim=-1)[0]
        e_proximity = torch.relu(self.safe_dcpa - min_dist) ** 2

        # ---- Tier 2: E_direction (initial maneuver direction check) ----
        # Judge first 1/2 of prediction horizon (~5 min for 30-step/10-min window).
        # Large ships need Nomoto T≈60-180s to execute turns, so 1/3 was too short.
        # Still avoids penalizing late-stage course corrections after passing.
        ship_dx = lon[:, :, 1:] - lon[:, :, :-1]
        ship_dy = lat[:, :, 1:] - lat[:, :, :-1]
        speed_sq = ship_dx ** 2 + ship_dy ** 2
        is_moving = (speed_sq > 1e-6).float()
        course = torch.atan2(ship_dx, ship_dy)

        d_course = course[:, :, 1:] - course[:, :, :-1]
        d_course = torch.atan2(torch.sin(d_course), torch.cos(d_course))

        n_early = max(1, d_course.shape[2] // 2)
        early_d_course = d_course[:, :, :n_early]
        early_moving = is_moving[:, :, 1:n_early + 1].mean(dim=-1).clamp(min=0.1)

        net_turn_early = early_d_course.sum(dim=-1) * early_moving

        is_maneuvering = (net_turn_early.abs() > self.maneuver_threshold).float()

        wrong_turn = torch.relu(-net_turn_early) * is_maneuvering

        # Rules 14/15: head-on and crossing-give-way must turn starboard.
        # Rule 13 (overtaking): may pass on either side — no direction constraint.
        give_way_mask = ((encounter_type == 1)
                         | (encounter_type == 2)).float()
        e_direction = wrong_turn.unsqueeze(2) * give_way_mask

        # ---- Combine with CRI gating and masks ----
        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
        diag_mask = 1 - torch.eye(N, device=pred.device).unsqueeze(0)
        valid = pair_mask * diag_mask * (encounter_type > 0).float()

        energy = (self.w_proximity * e_proximity
                  + self.w_direction * e_direction) * cri * valid

        n_pairs = valid.sum(dim=(1, 2)).clamp(min=1.0)
        return energy.sum(dim=(1, 2)) / n_pairs


class DynamicGuidanceSchedule:
    """w(t) = w_max * schedule(t/T): weak at high noise, strong at low noise."""

    def __init__(self, schedule='cosine', w_max=1.0):
        self.schedule = schedule
        self.w_max = w_max

    def weight(self, t, T):
        frac = 1.0 - t / T
        if self.schedule == 'cosine':
            return self.w_max * (1 - math.cos(math.pi * frac)) / 2
        elif self.schedule == 'linear':
            return self.w_max * frac
        return self.w_max


def _stable_headings_from_positions(full, pos_std):
    """Compute headings from consecutive positions with numerical stability.

    For near-zero displacements, uses the heading from the previous valid step
    instead of computing atan2(~0, ~0) which produces noise.

    Args:
        full: [B, T+1, 2] positions in normalized space (lat, lon)
        pos_std: [2] position standard deviations for denormalization

    Returns:
        headings: [B, T] headings in radians
    """
    dx_phys = (full[:, 1:, 1] - full[:, :-1, 1]) * pos_std[1]
    dy_phys = (full[:, 1:, 0] - full[:, :-1, 0]) * pos_std[0]

    speed_sq = dx_phys ** 2 + dy_phys ** 2
    is_moving = speed_sq > 1e-6  # ~0.001 nm ≈ 1.8 meters

    raw_headings = torch.atan2(dx_phys, dy_phys)

    # Vectorized forward-fill: at each position, find the latest moving step
    # and copy its heading. Uses cummax on the moving mask's index.
    headings = raw_headings.clone()
    if headings.shape[1] > 1:
        # Build index of last valid (moving) step for each position
        moving_float = is_moving.float()
        # Create step indices weighted by movement; cummax gives the latest valid index
        steps = torch.arange(headings.shape[1], device=headings.device).unsqueeze(0)
        # -1 for non-moving steps so cummax picks the latest moving step
        weighted = steps * moving_float + (-1) * (1 - moving_float)
        latest_valid, _ = weighted.cummax(dim=1)
        latest_valid = latest_valid.long().clamp(min=0)
        # Gather headings from the latest valid step
        headings = torch.gather(raw_headings, 1, latest_valid)

    return headings, dx_phys, dy_phys, speed_sq


class NomotoProjection(nn.Module):
    """Project trajectory to satisfy Nomoto ship turning dynamics.

    Turn rate constraint is speed-adaptive: high-speed vessels have tighter
    constraints (large ships at speed cannot turn fast), while slow vessels
    have relaxed constraints (small ships maneuvering at low speed).
    """

    def __init__(self, max_turn_rate_deg=3.0, dt=20.0, speed_adaptive=True,
                 low_speed_kn=5.0, high_speed_kn=15.0, low_speed_rate_mult=2.5):
        super().__init__()
        self.max_turn_rate = math.radians(max_turn_rate_deg)
        self.dt = dt
        self.speed_adaptive = speed_adaptive
        self.low_speed_kn = low_speed_kn
        self.high_speed_kn = high_speed_kn
        self.low_speed_rate_mult = low_speed_rate_mult
        self.register_buffer('pos_std', torch.ones(2))
        self.register_buffer('sog_std', torch.tensor(1.0))

    def set_pos_std(self, std):
        self.pos_std.copy_(std)

    def set_sog_std(self, std):
        self.sog_std.copy_(std)

    def _get_max_dh(self, speeds_nm_per_step):
        """Speed-adaptive max heading change per step.

        At high speed (≥15kn): base max_turn_rate × dt
        At low speed (≤5kn): base × low_speed_rate_mult (relaxed)
        Linear interpolation in between.
        """
        if not self.speed_adaptive:
            return self.max_turn_rate * self.dt

        speed_kn = speeds_nm_per_step / (self.dt / 3600.0)
        # Clamp to [low, high] range for interpolation
        frac = ((speed_kn - self.low_speed_kn) /
                max(self.high_speed_kn - self.low_speed_kn, 1e-6)).clamp(0, 1)
        # Low speed → high multiplier, high speed → 1.0
        mult = self.low_speed_rate_mult * (1 - frac) + 1.0 * frac
        return self.max_turn_rate * self.dt * mult

    def forward(self, pred, obs_last):
        """
        pred: [B, T, 2] (lat, lon) in normalized space
        obs_last: [B, 2] last observed position in normalized space
        return: projected pred [B, T, 2] in normalized space
        """
        B, T, _ = pred.shape
        full = torch.cat([obs_last.unsqueeze(1), pred], dim=1)

        headings, dx_phys, dy_phys, speed_sq = _stable_headings_from_positions(
            full, self.pos_std
        )
        speeds = torch.sqrt(speed_sq + 1e-8)

        dh = headings[:, 1:] - headings[:, :-1]
        dh = torch.atan2(torch.sin(dh), torch.cos(dh))

        max_dh = self._get_max_dh(speeds[:, 1:])
        dh_clamped = torch.clamp(dh, -max_dh, max_dh)

        cumulative_dh = torch.cumsum(dh_clamped, dim=1)
        new_headings = torch.cat([
            headings[:, :1],
            headings[:, :1] + cumulative_dh,
        ], dim=1)

        new_dx_norm = speeds * torch.sin(new_headings) / self.pos_std[1]
        new_dy_norm = speeds * torch.cos(new_headings) / self.pos_std[0]

        new_pred = torch.stack([
            obs_last[:, 0:1] + torch.cumsum(new_dy_norm, dim=1),
            obs_last[:, 1:2] + torch.cumsum(new_dx_norm, dim=1),
        ], dim=-1)

        return new_pred


class NomotoLoss(nn.Module):
    """Penalize physically infeasible turn rates in predicted trajectories.

    Speed-adaptive: low-speed vessels get relaxed turn rate limits.
    """

    def __init__(self, max_turn_rate_deg=3.0, dt=20.0, speed_adaptive=True,
                 low_speed_kn=5.0, high_speed_kn=15.0, low_speed_rate_mult=2.5):
        super().__init__()
        self.max_turn_rate = math.radians(max_turn_rate_deg)
        self.dt = dt
        self.max_dh = self.max_turn_rate * dt
        self.speed_adaptive = speed_adaptive
        self.low_speed_kn = low_speed_kn
        self.high_speed_kn = high_speed_kn
        self.low_speed_rate_mult = low_speed_rate_mult
        self.register_buffer('pos_std', torch.ones(2))

    def set_pos_std(self, std):
        self.pos_std.copy_(std)

    def forward(self, pred, obs_last):
        """
        pred: [B, T, 2] predicted trajectory in normalized space
        obs_last: [B, 2] last observed position in normalized space
        return: scalar loss (mean violation magnitude)
        """
        full = torch.cat([obs_last.unsqueeze(1), pred], dim=1)

        headings, dx_phys, dy_phys, speed_sq = _stable_headings_from_positions(full, self.pos_std)
        speeds = torch.sqrt(speed_sq + 1e-8)

        dh = headings[:, 1:] - headings[:, :-1]
        dh = torch.atan2(torch.sin(dh), torch.cos(dh))

        if self.speed_adaptive:
            speed_kn = speeds[:, 1:] / (self.dt / 3600.0)
            frac = ((speed_kn - self.low_speed_kn) /
                    max(self.high_speed_kn - self.low_speed_kn, 1e-6)).clamp(0, 1)
            mult = self.low_speed_rate_mult * (1 - frac) + 1.0 * frac
            max_dh_adaptive = self.max_turn_rate * self.dt * mult
            violation = torch.relu(dh.abs() - max_dh_adaptive)
        else:
            violation = torch.relu(dh.abs() - self.max_dh)

        return violation.mean()