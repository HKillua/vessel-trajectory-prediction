import torch
import torch.nn as nn

from mambadiff.models.encoder import TemporalEncoder
from mambadiff.models.interaction import EncounterEdgeEncoder, EncounterAwareGAT
from mambadiff.models.anchor import KinematicAnchorInitializer
from mambadiff.models.denoiser import ConditionalDenoisingModel
from mambadiff.models.guidance import COLREGsEnergyVectorized, DynamicGuidanceSchedule, NomotoProjection
from mambadiff.models.social_circle import SocialCircleEncoder
from mambadiff.trainer.diffusion_utils import DiffusionSchedule, extract


class MambaDiffECR(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        m = cfg['model']
        inter = cfg['interaction']
        diff = cfg['diffusion']
        guid = cfg['guidance']
        sc_cfg = cfg.get('social_circle', {})

        self.d_model = m['d_model']
        self.n_feat = m['n_feat']
        self.pred_dim = m['pred_dim']
        self.obs_steps = m['obs_steps']
        self.pred_steps = m['pred_steps']
        self.k_samples = m['k_samples']
        self.cfg_dropout = diff['cfg_dropout']
        self.biased_t_sampling = cfg.get('training', {}).get('biased_t_sampling', True)
        self.use_gradient_checkpointing = cfg.get('training', {}).get('gradient_checkpointing', False)

        self.encoder = TemporalEncoder(
            encoder_type=m['encoder_type'],
            n_feat=m['n_feat'],
            d_model=m['d_model']
        )

        self.use_social_circle = sc_cfg.get('enabled', True)
        if self.use_social_circle:
            sc_d_out = sc_cfg.get('d_out', 64)
            self.social_circle = SocialCircleEncoder(
                n_bins=sc_cfg.get('n_bins', 12),
                n_factors=sc_cfg.get('n_factors', 4),
                d_out=sc_d_out,
                use_cog_for_bearing=sc_cfg.get('use_cog_for_bearing', False),
                dist_scale=sc_cfg.get('dist_scale', 3.0),
                speed_scale=sc_cfg.get('speed_scale', 10.0),
            )
            self.sc_proj = nn.Linear(sc_d_out, m['d_model'])
            self.sc_gate = nn.Sequential(
                nn.Linear(m['d_model'] + sc_d_out, m['d_model']),
                nn.Sigmoid(),
            )

        self.edge_encoder = EncounterEdgeEncoder(
            n_encounter_types=inter['n_encounter_types'],
            encounter_embed_dim=inter['encounter_embed_dim'],
            edge_dim=inter['edge_dim'],
            dcpa_scale=inter.get('dcpa_scale', 3.0),
            tcpa_scale=inter.get('tcpa_scale', 1800.0),
        )
        sda_cfg = inter.get('sda', {})
        self.use_sda = sda_cfg.get('enabled', False)

        self.gat = EncounterAwareGAT(
            d_model=m['d_model'],
            n_head=inter['gat_heads'],
            edge_dim=inter['edge_dim'],
            dropout=0.1,
            n_layers=inter.get('gat_layers', 2),
            use_sda=self.use_sda,
        )

        anchor_cfg = cfg.get('anchor', {})
        self.anchor = KinematicAnchorInitializer(
            obs_steps=m['obs_steps'],
            pred_steps=m['pred_steps'],
            pred_dim=m['pred_dim'],
            k_samples=m['k_samples'],
            d_model=m['d_model'],
            use_residual=anchor_cfg.get('use_residual', True),
            initial_var_scale=anchor_cfg.get('initial_var_scale', 0.1),
            n_smooth=anchor_cfg.get('n_smooth', 3),
        )

        den_cfg = cfg.get('denoiser', {})
        self.denoiser = ConditionalDenoisingModel(
            context_dim=m['d_model'],
            pred_dim=m['pred_dim'],
            pred_steps=m['pred_steps'],
            encounter_embed_dim=inter['encounter_embed_dim'],
            time_embed_dim=den_cfg.get('time_embed_dim', 64),
            enc_proj_dim=den_cfg.get('enc_proj_dim', 64),
            cfg_dropout=diff['cfg_dropout'],
            tf_layers=den_cfg.get('tf_layers', 4),
            n_head=den_cfg.get('n_head', 8),
            ff_mult=den_cfg.get('ff_mult', 4),
        )

        self.enc_attn = nn.Sequential(
            nn.Linear(m['d_model'] + inter['encounter_embed_dim'], 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        self.colregs_energy = COLREGsEnergyVectorized(
            safe_dcpa=guid.get('safe_dcpa', 0.5),
            maneuver_threshold_deg=guid.get('maneuver_threshold_deg', 5.0),
            w_proximity=guid.get('w_proximity', 1.0),
            w_direction=guid.get('w_direction', 0.1),
        )
        self.guidance_schedule = DynamicGuidanceSchedule(
            schedule=guid['dynamic_schedule'],
            w_max=guid['colregs_weight']
        )
        self.nomoto = NomotoProjection(
            max_turn_rate_deg=guid['nomoto_max_turn_rate'],
            speed_adaptive=guid.get('nomoto_speed_adaptive', True),
            low_speed_kn=guid.get('nomoto_low_speed_kn', 5.0),
            high_speed_kn=guid.get('nomoto_high_speed_kn', 15.0),
            low_speed_rate_mult=guid.get('nomoto_low_speed_rate_mult', 2.5),
        )

        self.diffusion = None

    def init_diffusion(self, cfg, device='cpu'):
        self.diffusion = DiffusionSchedule(cfg['diffusion']).to(device)

    def _augment_input(self, obs):
        """
        obs: [B, N, T, 7] → [B*N, T, 21]
        Features: [lat, lon, sog, sin_cog, cos_cog, sin_hdg, cos_hdg]

        Augmentation:
        - abs_feat: raw values
        - rel_feat: displacement from last timestep
          - lat/lon/sog: simple subtraction (physical displacement)
          - sin/cos angle pairs: proper angular difference via atan2
        - vel_feat: first-order differences (velocity)
          - lat/lon/sog: simple difference
          - sin/cos angle pairs: proper angular difference via atan2
        """
        B, N, T, C = obs.shape
        obs_flat = obs.reshape(B * N, T, C)

        abs_feat = obs_flat

        # --- rel_feat: relative to last timestep ---
        rel_feat = torch.zeros_like(obs_flat)
        # Position and speed: simple subtraction
        rel_feat[:, :, :3] = obs_flat[:, :, :3] - obs_flat[:, -1:, :3]
        # Angle pairs: proper angular difference
        for sin_idx, cos_idx in [(3, 4), (5, 6)]:
            sin_cur = obs_flat[:, :, sin_idx]
            cos_cur = obs_flat[:, :, cos_idx]
            sin_last = obs_flat[:, -1:, sin_idx]
            cos_last = obs_flat[:, -1:, cos_idx]
            d_angle = torch.atan2(
                sin_cur * cos_last - cos_cur * sin_last,
                cos_cur * cos_last + sin_cur * sin_last,
            )
            rel_feat[:, :, sin_idx] = torch.sin(d_angle)
            rel_feat[:, :, cos_idx] = torch.cos(d_angle)

        # --- vel_feat: first-order differences ---
        vel_feat = torch.zeros_like(obs_flat)
        vel_feat[:, 1:, :3] = obs_flat[:, 1:, :3] - obs_flat[:, :-1, :3]
        for sin_idx, cos_idx in [(3, 4), (5, 6)]:
            sin_cur = obs_flat[:, 1:, sin_idx]
            cos_cur = obs_flat[:, 1:, cos_idx]
            sin_prev = obs_flat[:, :-1, sin_idx]
            cos_prev = obs_flat[:, :-1, cos_idx]
            d_angle = torch.atan2(
                sin_cur * cos_prev - cos_cur * sin_prev,
                cos_cur * cos_prev + sin_cur * sin_prev,
            )
            # sin(Δθ) ≈ Δθ for small angles: signed turn direction
            vel_feat[:, 1:, sin_idx] = torch.sin(d_angle)
            # Replace cos(Δθ)≈1 (near-constant, uninformative) with |Δθ|
            # (turn magnitude), providing complementary information
            vel_feat[:, 1:, cos_idx] = d_angle.abs()

        return torch.cat([abs_feat, rel_feat, vel_feat], dim=-1)

    def _get_target_encounter_emb(self, encounter_type, cri_matrix, target_idx, mask, h=None):
        """Context-aware encounter embedding aggregation."""
        B, N, _ = encounter_type.shape
        batch_idx = torch.arange(B, device=encounter_type.device)
        target_encounters = encounter_type[batch_idx, target_idx]
        target_cri = cri_matrix[batch_idx, target_idx]

        all_emb = self.edge_encoder.enc_embed(target_encounters)

        valid = (target_encounters > 0).float() * mask.float()
        has_encounter = valid.sum(dim=1) > 0

        if h is not None:
            target_ctx = h[batch_idx, target_idx].unsqueeze(1).expand(-1, N, -1)
            attn_input = torch.cat([target_ctx, all_emb], dim=-1)
            attn_logits = self.enc_attn(attn_input).squeeze(-1)
            attn_logits = attn_logits + torch.log(target_cri + 0.1)
            attn_logits = attn_logits.masked_fill(valid == 0, float('-inf'))
            safe_logits = attn_logits.clone()
            safe_logits[~has_encounter, 0] = 0.0
            attn_weights = torch.softmax(safe_logits, dim=1)
        else:
            weights = target_cri * valid
            weights_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            attn_weights = weights / weights_sum

        agg_emb = (all_emb * attn_weights.unsqueeze(-1)).sum(dim=1)
        agg_emb = agg_emb * has_encounter.float().unsqueeze(-1)

        return agg_emb

    def encode(self, batch):
        """Encode observations → per-ship context vectors."""
        obs = batch['obs']
        B, N = obs.shape[:2]

        aug_input = self._augment_input(obs)
        h_flat = self.encoder(aug_input)
        h = h_flat.view(B, N, self.d_model)

        if self.use_social_circle:
            sc_feat = self.social_circle(obs, batch['cri_matrix'], batch['mask'])
            gate = self.sc_gate(torch.cat([h, sc_feat], dim=-1))
            h = h + gate * self.sc_proj(sc_feat)

        edge_feat = self.edge_encoder(
            batch['dcpa_matrix'], batch['tcpa_matrix'],
            batch['cri_matrix'], batch['encounter_type']
        )
        h = self.gat(h, edge_feat, batch['mask'])

        h = h * batch['mask'].unsqueeze(-1).float()

        return h

    def noise_estimation_loss(self, batch, detach_encoder=False, h=None, epoch_frac=1.0):
        """Phase 1: pretrain denoiser with noise estimation loss."""
        pred_gt = batch['pred'][:, :, :, :self.pred_dim]
        B, N, T, D = pred_gt.shape
        target_idx = batch.get('target_ship_idx', torch.zeros(B, dtype=torch.long, device=pred_gt.device))
        batch_idx = torch.arange(B, device=pred_gt.device)
        target_pred = pred_gt[batch_idx, target_idx]

        if h is None:
            if detach_encoder:
                with torch.no_grad():
                    h = self.encode(batch)
            else:
                h = self.encode(batch)
        context = h[batch_idx, target_idx].unsqueeze(1)
        encounter_emb = self._get_target_encounter_emb(
            batch['encounter_type'], batch['cri_matrix'], target_idx, batch['mask'],
            h=h,
        )

        # [C3] Progressive biased t-sampling: 30% tau-focused at epoch 0,
        # ramping to 70% by final epoch
        use_biased = self.training and self.biased_t_sampling
        if use_biased:
            tau_prob = 0.3 + 0.4 * min(epoch_frac, 1.0)
            low_mask = torch.rand(B, device=pred_gt.device) < tau_prob
            tau_indices = self.diffusion.tau_steps
            t_low = torch.tensor(tau_indices, device=pred_gt.device)[
                torch.randint(0, len(tau_indices), (B,))
            ]
            t_full = torch.randint(0, self.diffusion.n_steps, (B,), device=pred_gt.device)
            t = torch.where(low_mask, t_low, t_full)
        else:
            t = torch.randint(0, self.diffusion.n_steps, (B,), device=pred_gt.device)

        a = extract(self.diffusion.alphas_bar_sqrt, t, target_pred)
        am1 = extract(self.diffusion.one_minus_alphas_bar_sqrt, t, target_pred)
        noise = torch.randn_like(target_pred)

        x_noisy = a * target_pred + am1 * noise

        eps_pred = self.denoiser(x_noisy, t, context, encounter_emb)
        noise_loss = (noise - eps_pred).pow(2).mean()

        return noise_loss

    def p_sample(self, x, t_cur, context, encounter_emb,
                 t_prev=None, deterministic=False,
                 encounter_type=None, cri=None, neighbor_pred=None,
                 target_idx=None, mask=None, obs_last=None,
                 guidance_scale=0.0, cfg_scale=0.0):
        """Single reverse diffusion step. Supports both DDPM and DDIM."""
        B = x.size(0)
        t_tensor = torch.full((B,), t_cur, device=x.device, dtype=torch.long)

        if cfg_scale > 0:
            eps = self.denoiser.forward_cfg(x, t_tensor, context, encounter_emb, cfg_scale)
        else:
            eps = self.denoiser(x, t_tensor, context, encounter_emb)

        # Force FP32 for DDIM arithmetic to avoid AMP precision loss
        x = x.float()
        eps = eps.float()

        alpha_bar_cur = self.diffusion.alphas_prod[t_cur]

        if t_prev is not None and t_prev >= 0:
            alpha_bar_prev = self.diffusion.alphas_prod[t_prev]
        elif t_prev is not None and t_prev < 0:
            alpha_bar_prev = torch.tensor(1.0, device=x.device)
        else:
            t_p = max(t_cur - 1, 0)
            alpha_bar_prev = self.diffusion.alphas_prod[t_p]

        x0_pred = (x - (1 - alpha_bar_cur).sqrt() * eps) / alpha_bar_cur.sqrt().clamp(min=1e-8)

        if not deterministic and t_cur > 0:
            sigma_sq = ((1 - alpha_bar_prev) / (1 - alpha_bar_cur)) * (1 - alpha_bar_cur / alpha_bar_prev)
            sigma_sq = sigma_sq.clamp(min=0)
            dir_xt = (1 - alpha_bar_prev - sigma_sq).clamp(min=0).sqrt() * eps
        else:
            sigma_sq = None
            dir_xt = (1 - alpha_bar_prev).sqrt() * eps

        mean = alpha_bar_prev.sqrt() * x0_pred + dir_xt

        if guidance_scale > 0 and neighbor_pred is not None:
            w_t = self.guidance_schedule.weight(t_cur, self.diffusion.tau_steps[0])
            with torch.enable_grad():
                mean_req = mean.detach().requires_grad_(True)
                pred_multi = neighbor_pred.detach().clone()
                batch_idx = torch.arange(B, device=x.device)
                pred_multi[batch_idx, target_idx] = mean_req
                energy = self.colregs_energy(pred_multi, encounter_type, cri, mask)
                grad = torch.autograd.grad(energy.sum(), mean_req)[0]
                grad_norm = grad.reshape(B, -1).norm(dim=1, keepdim=True).clamp(min=1e-8)
                grad_norm = grad_norm.unsqueeze(-1)
                grad = grad * torch.clamp(10.0 / grad_norm, max=1.0)
                mean = mean - guidance_scale * w_t * grad

        if obs_last is not None:
            nomoto_window = max(1, len(self.diffusion.tau_steps) // 2)
            tau_list = self.diffusion.tau_steps
            cur_pos = tau_list.index(t_cur) if t_cur in tau_list else -1
            steps_from_end = len(tau_list) - 1 - cur_pos if cur_pos >= 0 else nomoto_window
            if steps_from_end < nomoto_window:
                mean = self.nomoto(mean, obs_last)

        if deterministic or t_cur == 0:
            return mean
        else:
            return mean + sigma_sq.sqrt() * torch.randn_like(x)

    def _get_neighbor_pred(self, batch, h):
        """Predict neighbor trajectories using kinematic anchor + learned residual."""
        B, N = batch['obs'].shape[:2]
        obs_flat = batch['obs'].reshape(B * N, batch['obs'].shape[2], batch['obs'].shape[3])
        neighbor_pred = self.anchor._kinematic_extrapolate(obs_flat)
        if self.anchor.use_residual:
            ctx_flat = h.reshape(B * N, self.d_model)
            residual = self.anchor.residual_mlp(ctx_flat)
            neighbor_pred = neighbor_pred + residual.reshape(neighbor_pred.shape)
        return neighbor_pred.reshape(B, N, self.pred_steps, self.pred_dim)

    def _run_reverse_loop(self, loc, context, encounter_emb, batch,
                          target_idx, obs_last_k, h_all=None,
                          deterministic=False, guidance_scale=0.0, cfg_scale=0.0,
                          use_gt_neighbors=False):
        """Shared reverse diffusion loop for both training and inference."""
        B, N = batch['obs'].shape[:2]
        batch_idx = torch.arange(B, device=loc.device)

        tau_steps = self.diffusion.tau_steps
        start_t = tau_steps[0]

        a_start = self.diffusion.alphas_bar_sqrt[start_t]
        am_start = self.diffusion.one_minus_alphas_bar_sqrt[start_t]
        noise = torch.randn_like(loc)
        cur_y = a_start * loc + am_start * noise

        if guidance_scale > 0:
            if use_gt_neighbors:
                neighbor_pred = batch['pred'][:, :, :, :self.pred_dim].detach()
            elif h_all is not None:
                neighbor_pred = self._get_neighbor_pred(batch, h_all).detach()
            else:
                obs_all = batch['obs']
                obs_flat = obs_all.reshape(B * N, obs_all.shape[2], obs_all.shape[3])
                neighbor_pred = self.anchor._kinematic_extrapolate(obs_flat)
                neighbor_pred = neighbor_pred.reshape(B, N, self.pred_steps, self.pred_dim)
        else:
            neighbor_pred = None

        for step_i, t_cur in enumerate(tau_steps):
            t_prev = tau_steps[step_i + 1] if step_i + 1 < len(tau_steps) else -1

            B_total = B * self.k_samples
            x_flat = cur_y.view(B_total, self.pred_steps, self.pred_dim)
            ctx_flat = context.repeat(1, self.k_samples, 1).view(B_total, 1, self.d_model)
            enc_flat = encounter_emb.unsqueeze(1).repeat(1, self.k_samples, 1).view(B_total, -1)
            obs_flat = obs_last_k.reshape(B_total, 2)

            if guidance_scale > 0:
                K = self.k_samples
                nb_k = neighbor_pred.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B_total, N, self.pred_steps, self.pred_dim)
                et_k = batch['encounter_type'].unsqueeze(1).expand(-1, K, -1, -1).reshape(B_total, N, N)
                cri_k = batch['cri_matrix'].unsqueeze(1).expand(-1, K, -1, -1).reshape(B_total, N, N)
                mask_k = batch['mask'].unsqueeze(1).expand(-1, K, -1).reshape(B_total, N)
                tidx_k = target_idx.unsqueeze(1).expand(-1, K).reshape(B_total)
            else:
                nb_k = et_k = cri_k = mask_k = tidx_k = None

            if self.use_gradient_checkpointing and self.training:
                x_denoised = torch.utils.checkpoint.checkpoint(
                    self.p_sample,
                    x_flat, t_cur, ctx_flat, enc_flat,
                    t_prev, deterministic,
                    et_k, cri_k, nb_k, tidx_k, mask_k, obs_flat,
                    guidance_scale, cfg_scale,
                    use_reentrant=False,
                )
            else:
                x_denoised = self.p_sample(
                    x_flat, t_cur, ctx_flat, enc_flat,
                    t_prev=t_prev, deterministic=deterministic,
                    encounter_type=et_k, cri=cri_k,
                    neighbor_pred=nb_k, target_idx=tidx_k,
                    mask=mask_k, obs_last=obs_flat,
                    guidance_scale=guidance_scale,
                    cfg_scale=cfg_scale,
                )
            cur_y = x_denoised.view(B, self.k_samples, self.pred_steps, self.pred_dim)

        return cur_y

    def predict(self, batch, cfg_scale=0.0, guidance_scale=0.0,
                deterministic=False, use_gt_neighbors=False):
        """Full inference pipeline.

        Args:
            use_gt_neighbors: if True, use ground-truth neighbor trajectories
                for COLREGs guidance (oracle mode for ablation studies).
        """
        obs = batch['obs']
        B, N = obs.shape[:2]
        target_idx = batch.get('target_ship_idx', torch.zeros(B, dtype=torch.long, device=obs.device))
        batch_idx = torch.arange(B, device=obs.device)

        h = self.encode(batch)
        context = h[batch_idx, target_idx].unsqueeze(1)

        encounter_emb = self._get_target_encounter_emb(
            batch['encounter_type'], batch['cri_matrix'], target_idx, batch['mask'],
            h=h,
        )

        target_obs = obs[batch_idx, target_idx]
        loc, anchor_mean = self.anchor(target_obs, context.squeeze(1))

        obs_last = target_obs[:, -1, :2]
        obs_last_k = obs_last.unsqueeze(1).expand(-1, self.k_samples, -1)

        cur_y = self._run_reverse_loop(
            loc, context, encounter_emb, batch,
            target_idx, obs_last_k,
            h_all=h,
            deterministic=deterministic,
            guidance_scale=guidance_scale,
            cfg_scale=cfg_scale,
            use_gt_neighbors=use_gt_neighbors,
        )

        return cur_y, anchor_mean, h

    def forward(self, batch):
        pred, anchor_mean, _ = self.predict(batch)
        return pred, anchor_mean