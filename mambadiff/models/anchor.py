import torch
import torch.nn as nn
import numpy as np


class KinematicAnchorInitializer(nn.Module):
    def __init__(self, obs_steps=30, pred_steps=30, pred_dim=2, k_samples=20, d_model=256,
                 use_residual=True, initial_var_scale=0.1, n_smooth=3):
        super().__init__()
        self.pred_steps = pred_steps
        self.pred_dim = pred_dim
        self.k_samples = k_samples
        self.use_residual = use_residual
        self.initial_var_scale = initial_var_scale
        self.n_smooth = n_smooth

        self.var_encoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.var_head = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, k_samples * pred_steps * pred_dim),
        )
        self.scale_head = nn.Linear(128, 1)

        if use_residual:
            self.residual_mlp = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.ReLU(),
                nn.Linear(128, pred_steps * pred_dim),
            )
            nn.init.zeros_(self.residual_mlp[-1].weight)
            nn.init.zeros_(self.residual_mlp[-1].bias)

        self.register_buffer('norm_mean', torch.zeros(3))
        self.register_buffer('norm_std', torch.ones(3))

    def set_norm_params(self, mean, std):
        if abs(float(mean[0])) > 5.0 or abs(float(mean[1])) > 5.0:
            raise ValueError(
                f"norm_mean[:2]={mean[:2].tolist()} too large — expected ~0 for local (nm) coords."
            )
        if float(std[0]) < 1e-6 or float(std[1]) < 1e-6 or float(std[2]) < 1e-6:
            raise ValueError(f"norm_std has near-zero values: {std.tolist()}")
        self.norm_mean.copy_(mean)
        self.norm_std.copy_(std)

    def _kinematic_extrapolate(self, obs):
        """
        obs: [B, T_obs, 7] (lat, lon, sog, sin_cog, cos_cog, sin_hdg, cos_hdg)
        return: anchor_mean [B, T_pred, 2] — in normalized local coordinate space
        """
        last_lat_norm = obs[:, -1, 0]
        last_lon_norm = obs[:, -1, 1]

        last_lat_local = last_lat_norm * self.norm_std[0] + self.norm_mean[0]
        last_lon_local = last_lon_norm * self.norm_std[1] + self.norm_mean[1]

        # Exponentially weighted average of last n_smooth steps for SOG/COG stability
        n = min(self.n_smooth, obs.shape[1])
        weights = torch.exp(torch.linspace(-1.0, 0.0, n, device=obs.device))
        weights = weights / weights.sum()
        w = weights.unsqueeze(0)  # [1, n]

        recent_sog_norm = obs[:, -n:, 2]  # [B, n]
        recent_sog = recent_sog_norm * self.norm_std[2] + self.norm_mean[2]
        last_sog = (recent_sog * w).sum(dim=1)  # [B]

        recent_sin = obs[:, -n:, 3]  # [B, n]
        recent_cos = obs[:, -n:, 4]  # [B, n]
        avg_sin = (recent_sin * w).sum(dim=1)
        avg_cos = (recent_cos * w).sum(dim=1)
        cog_rad = torch.atan2(avg_sin, avg_cos)

        dt = 20.0
        speed_nm_per_step = last_sog * (dt / 3600.0)

        vx = speed_nm_per_step * torch.sin(cog_rad)
        vy = speed_nm_per_step * torch.cos(cog_rad)

        steps = torch.arange(1, self.pred_steps + 1, device=obs.device, dtype=obs.dtype)

        d_lat = vy.unsqueeze(1) * steps.unsqueeze(0)
        d_lon = vx.unsqueeze(1) * steps.unsqueeze(0)

        lat_pred = last_lat_local.unsqueeze(1) + d_lat
        lon_pred = last_lon_local.unsqueeze(1) + d_lon

        lat_pred_norm = (lat_pred - self.norm_mean[0]) / self.norm_std[0]
        lon_pred_norm = (lon_pred - self.norm_mean[1]) / self.norm_std[1]

        return torch.stack([lat_pred_norm, lon_pred_norm], dim=-1)

    def forward(self, obs, context):
        """
        obs: [B, T_obs, 7]
        context: [B, d_model]
        return:
          loc: [B, K, T_pred, 2]
          anchor_mean: [B, T_pred, 2]
        """
        anchor_mean = self._kinematic_extrapolate(obs)

        if self.use_residual:
            residual = self.residual_mlp(context).reshape(-1, self.pred_steps, self.pred_dim)
            anchor_mean = anchor_mean + residual

        h = self.var_encoder(context)
        scale = self.scale_head(h)
        var_raw = self.var_head(h).reshape(-1, self.k_samples, self.pred_steps, self.pred_dim)

        scale_clamped = scale.clamp(-4.0, 4.0)
        var_scaled = torch.exp(scale_clamped / 2).unsqueeze(-1).unsqueeze(-1) * var_raw
        std_norm = var_raw.std(dim=1, keepdim=True).mean(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        var_scaled = self.initial_var_scale * var_scaled / std_norm

        loc = var_scaled + anchor_mean.unsqueeze(1)
        return loc, anchor_mean