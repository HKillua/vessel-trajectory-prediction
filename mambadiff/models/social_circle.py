import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SocialCircleEncoder(nn.Module):
    """COLREGs-aware polar encoding of ship interactions.

    Maps each neighbor into angular bins aligned with COLREGs sectors:
    - 0° (ahead) → head-on zone
    - 90° (starboard) → crossing give-way zone
    - 180° (astern) → overtaking zone
    - 270° (port) → crossing stand-on zone

    Reference: SocialCircle (CVPR 2024), adapted from pedestrian to maritime.
    """

    def __init__(self, n_bins=12, n_factors=4, d_out=16, use_cog_for_bearing=False,
                 dist_scale=3.0, speed_scale=10.0):
        super().__init__()
        self.n_bins = n_bins
        self.use_cog_for_bearing = use_cog_for_bearing
        self.n_factors = n_factors
        self.d_out = d_out
        self.dist_scale = dist_scale
        self.speed_scale = speed_scale

        self.encode_mlp = nn.Sequential(
            nn.Linear(n_bins * n_factors, d_out * 2),
            nn.ReLU(),
            nn.Linear(d_out * 2, d_out),
        )

        bin_index = torch.arange(n_bins, dtype=torch.float) / n_bins
        self.register_buffer('bin_index', bin_index)
        self.register_buffer('pos_std', torch.ones(2))
        self.register_buffer('sog_std', torch.tensor(1.0))

    def set_norm_params(self, pos_std, sog_std):
        """Set normalization parameters for physical-space bearing/distance.

        Must be called before forward() for correct COLREGs sector alignment.

        Args:
            pos_std: [2] tensor (lat_std_nm, lon_std_nm)
            sog_std: scalar tensor (sog_std_knots)
        """
        self.pos_std.copy_(pos_std)
        self.sog_std.copy_(sog_std)

    def forward(self, obs, cri_matrix, mask):
        """
        obs: [B, N, T, 7] (lat_nm, lon_nm, sog, sin_cog, cos_cog, sin_hdg, cos_hdg)
        cri_matrix: [B, N, N] collision risk index
        mask: [B, N] valid ship mask
        return: [B, N, d_out]
        """
        B, N, T, _ = obs.shape
        device = obs.device

        last_pos = obs[:, :, -1, :2]
        last_sog = obs[:, :, -1, 2]
        if self.use_cog_for_bearing:
            sin_dir = obs[:, :, -1, 3]  # sin_cog
            cos_dir = obs[:, :, -1, 4]  # cos_cog
        else:
            sin_dir = obs[:, :, -1, 5]  # sin_hdg
            cos_dir = obs[:, :, -1, 6]  # cos_hdg
        heading = torch.atan2(sin_dir, cos_dir)

        # Compute displacements in physical nm space to preserve COLREGs bearing geometry
        # Normalized coords have different lat/lon std (~4.15 vs ~2.66 nm),
        # which distorts atan2 bearings by ~12° if not corrected.
        dy = (last_pos[:, :, 0].unsqueeze(1) - last_pos[:, :, 0].unsqueeze(2)) * self.pos_std[0]
        dx = (last_pos[:, :, 1].unsqueeze(1) - last_pos[:, :, 1].unsqueeze(2)) * self.pos_std[1]
        abs_bearing = torch.atan2(dx, dy)
        rel_bearing = (abs_bearing - heading.unsqueeze(2)) % (2 * math.pi)

        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        dist_norm = torch.clamp(dist / self.dist_scale, 0, 1)

        rel_speed = (last_sog.unsqueeze(2) - last_sog.unsqueeze(1)) * self.sog_std
        rel_speed_norm = torch.clamp(rel_speed / self.speed_scale, -1, 1)

        bin_idx = (rel_bearing / (2 * math.pi / self.n_bins)).long() % self.n_bins

        pair_mask = mask.unsqueeze(1).float() * mask.unsqueeze(2).float()
        diag_mask = 1.0 - torch.eye(N, device=device).unsqueeze(0)
        valid_mask = pair_mask * diag_mask

        # Vectorized: one_hot binning replaces Python for-loop
        one_hot = F.one_hot(bin_idx, self.n_bins).float()  # [B, N, N, n_bins]
        one_hot = one_hot * valid_mask.unsqueeze(-1)
        raw_count = one_hot.sum(dim=2)  # [B, N, n_bins]
        count = raw_count.clamp(min=1)
        has_neighbor = (raw_count > 0).float()

        # Min-pool distance (closest neighbor most critical for COLREGs)
        dist_for_min = dist_norm.unsqueeze(-1) + (1.0 - one_hot) * 2.0
        sc_dist = dist_for_min.min(dim=2)[0] * has_neighbor

        sc_speed = (rel_speed_norm.unsqueeze(-1) * one_hot).sum(dim=2) / count
        sc_bin = self.bin_index.view(1, 1, self.n_bins).expand(B, N, -1)

        # Max-pool CRI (highest risk neighbor most critical)
        sc_cri = (cri_matrix.unsqueeze(-1) * one_hot).max(dim=2)[0]

        sc_feat = torch.stack([sc_dist, sc_speed, sc_bin, sc_cri], dim=-1)
        sc_flat = sc_feat.reshape(B, N, -1)
        return self.encode_mlp(sc_flat)