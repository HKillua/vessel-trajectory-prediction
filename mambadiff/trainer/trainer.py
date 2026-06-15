import csv
import json
import os
import sys
import time
import yaml
import torch
import random
import numpy as np
import torch.nn as nn

from mambadiff.models.mambadiff_ecr import MambaDiffECR
from mambadiff.losses.fredf import FreDFLoss
from mambadiff.models.guidance import NomotoLoss
from mambadiff.metrics.colregs_metrics import compute_ccr, compute_predicted_dcpa
from mambadiff.metrics.stratified import stratified_evaluate, format_stratified_results
from mambadiff.metrics.conformal import ConformalTrajectoryCalibrator


class ECRTrainer:
    def __init__(self, cfg_path, device=None, log_dir='results/mambadiff'):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log = open(os.path.join(log_dir, 'log.txt'), 'a+')

        train_cfg = self.cfg.get('training', {})
        self.use_amp = train_cfg.get('amp', True) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.colregs_aux_weight = train_cfg.get('colregs_aux_weight', 0.01)
        self.ccr_threshold_deg = train_cfg.get('ccr_threshold_deg', 5.0)
        self.warmup_epochs = train_cfg.get('warmup_epochs', 5)

        # [M4] Global seed for reproducibility
        seed = train_cfg.get('seed', 42)
        self._set_seed(seed)
        self._log(f'Global seed: {seed}')

        self._build_dataloaders()
        self._build_model()
        self._init_csv_loggers()
        self._log_environment(cfg_path)

    def _init_csv_loggers(self):
        p1_path = os.path.join(self.log_dir, 'phase1_metrics.csv')
        p2_path = os.path.join(self.log_dir, 'phase2_metrics.csv')
        self._p1_csv_path = p1_path
        self._p2_csv_path = p2_path
        self._p1_csv = open(p1_path, 'w', newline='')
        self._p2_csv = open(p2_path, 'w', newline='')
        self._p1_writer = csv.writer(self._p1_csv)
        self._p2_writer = csv.writer(self._p2_csv)
        self._p1_writer.writerow([
            'epoch', 'train_noise_loss', 'val_noise_loss', 'lr',
            'gpu_mem_mb', 'gpu_mem_peak_mb', 'epoch_time_s',
        ])
        self._p2_writer.writerow([
            'epoch', 'total_loss', 'dist_loss', 'fredf_loss', 'anchor_loss',
            'nomoto_loss', 'div_loss', 'pw_fde_nm',
            'val_ade', 'val_fde', 'val_ccr', 'val_dcpa_mae',
            'val_anchor_ade', 'val_anchor_fde',
            'lr', 'eff_div_weight', 'sda_gates',
            'gpu_mem_mb', 'gpu_mem_peak_mb', 'epoch_time_s',
            'best_ade', 'no_improve',
        ])
        self._p1_csv.flush()
        self._p2_csv.flush()

    def _log_environment(self, cfg_path):
        self._log('=' * 60)
        self._log('MambaDiff-ECR Training Session')
        self._log('=' * 60)
        self._log(f'Config: {os.path.abspath(cfg_path)}')
        self._log(f'Log dir: {os.path.abspath(self.log_dir)}')
        self._log(f'Files:')
        self._log(f'  log.txt:            {os.path.abspath(os.path.join(self.log_dir, "log.txt"))}')
        self._log(f'  phase1_metrics.csv: {self._p1_csv_path}')
        self._log(f'  phase2_metrics.csv: {self._p2_csv_path}')
        self._log(f'  checkpoints:        {self.log_dir}/checkpoint_*.pt')
        self._log(f'Device: {self.device}')
        if self.device.type == 'cuda':
            self._log(f'  GPU: {torch.cuda.get_device_name(self.device)}')
            total_mem = torch.cuda.get_device_properties(self.device).total_mem / 1024**3
            self._log(f'  GPU memory: {total_mem:.1f} GB')
            self._log(f'  CUDA version: {torch.version.cuda}')
        self._log(f'PyTorch: {torch.__version__}')
        self._log(f'AMP: {self.use_amp}')
        self._log(f'Train samples: {len(self.train_loader.dataset)}')
        self._log(f'Val samples: {len(self.val_loader.dataset) if self.val_loader else 0}')
        self._log(f'Test samples: {len(self.test_loader.dataset) if self.test_loader else 0}')
        self._log(f'Batch size: {self.cfg["training"]["batch_size"]}')

        cfg_dump_path = os.path.join(self.log_dir, 'config_snapshot.yaml')
        with open(cfg_dump_path, 'w') as f:
            yaml.dump(self.cfg, f, default_flow_style=False, allow_unicode=True)
        self._log(f'Config snapshot saved: {cfg_dump_path}')
        self._log('=' * 60)

    def _log_dataset_stats(self):
        """Log encounter type distribution and data statistics for diagnostics."""
        enc_names = {0: 'safe', 1: 'head-on', 2: 'cross-GW', 3: 'cross-SO', 4: 'overtake', 5: 'other'}
        for split_name, loader in [('Train', self.train_loader), ('Val', self.val_loader), ('Test', self.test_loader)]:
            if loader is None:
                continue
            enc_counts = {}
            n_ships_list = []
            n_samples = 0
            for batch in loader:
                if 'encounter_type' in batch:
                    et = batch['encounter_type']
                    mask = batch.get('mask', torch.ones(et.shape[0], et.shape[1], dtype=torch.bool))
                    for b in range(et.shape[0]):
                        n_valid = mask[b].sum().item()
                        n_ships_list.append(n_valid)
                        tidx = batch.get('target_ship_idx', torch.zeros(et.shape[0], dtype=torch.long))[b].item()
                        for j in range(et.shape[2]):
                            if j != tidx and mask[b, j]:
                                etype = et[b, tidx, j].item()
                                enc_counts[etype] = enc_counts.get(etype, 0) + 1
                n_samples += batch['obs'].shape[0]
            parts = [f'{enc_names.get(k, f"t{k}")}={v}' for k, v in sorted(enc_counts.items())]
            self._log(f'{split_name} encounter distribution ({n_samples} samples): {" ".join(parts)}')
            if n_ships_list:
                import statistics
                self._log(f'  ships/scene: mean={statistics.mean(n_ships_list):.1f} '
                           f'median={statistics.median(n_ships_list):.0f} '
                           f'max={max(n_ships_list)}')

    def _build_dataloaders(self):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data_provider.dataloader_multivessel import create_dataloaders

        data_cfg = self.cfg['data']
        train_cfg = self.cfg['training']
        loaders = create_dataloaders(
            data_root=data_cfg['data_root'],
            batch_size=train_cfg['batch_size'],
            num_workers=data_cfg.get('num_workers', 4),
        )
        self.train_loader = loaders['train']
        self.val_loader = loaders.get('val')
        self.test_loader = loaders.get('test')

    def _build_model(self):
        self.model = MambaDiffECR(self.cfg).to(self.device)
        self.model.init_diffusion(self.cfg, self.device)

        if hasattr(self.train_loader, 'dataset') and self.train_loader.dataset.norm_params is not None:
            np_mean = self.train_loader.dataset.norm_params['mean']
            np_std = self.train_loader.dataset.norm_params['std']
            self.model.anchor.set_norm_params(
                torch.from_numpy(np_mean).to(self.device),
                torch.from_numpy(np_std).to(self.device),
            )
            pos_std = torch.from_numpy(np_std[:2]).to(self.device)
            self.model.nomoto.set_pos_std(pos_std)
            self.model.colregs_energy.set_pos_std(pos_std)

            if self.model.use_social_circle:
                sog_std = torch.tensor(np_std[2], dtype=torch.float32).to(self.device)
                self.model.social_circle.set_norm_params(pos_std, sog_std)

        fredf_cfg = self.cfg.get('fredf', {})
        self.fredf_loss = FreDFLoss(
            k_freq=fredf_cfg.get('k_freq', 'auto'),
            log_magnitude=fredf_cfg.get('log_magnitude', False),
            low_weight=fredf_cfg.get('low_weight', 0.01),
            high_weight=fredf_cfg.get('high_weight', 1.0),
        )
        nomoto_cfg = self.cfg.get('nomoto_loss', {})
        self.nomoto_loss = NomotoLoss(
            max_turn_rate_deg=nomoto_cfg.get('max_turn_rate_deg', 3.0),
            speed_adaptive=nomoto_cfg.get('speed_adaptive', True),
            low_speed_kn=nomoto_cfg.get('low_speed_kn', 5.0),
            high_speed_kn=nomoto_cfg.get('high_speed_kn', 15.0),
            low_speed_rate_mult=nomoto_cfg.get('low_speed_rate_mult', 2.5),
        ) if nomoto_cfg.get('enabled', True) else None
        self.nomoto_weight = nomoto_cfg.get('weight', 1.0)

        if self.nomoto_loss is not None:
            self.nomoto_loss = self.nomoto_loss.to(self.device)
            if hasattr(self.train_loader, 'dataset') and self.train_loader.dataset.norm_params is not None:
                np_std = self.train_loader.dataset.norm_params['std']
                self.nomoto_loss.set_pos_std(torch.from_numpy(np_std[:2]).to(self.device))

        self.conformal = ConformalTrajectoryCalibrator(alpha=0.1)

        self._print_params(self.model, 'MambaDiffECR')
        self._log_dataset_stats()

    def _print_params(self, model, name='Model'):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._log(f'[{name}] Trainable/Total: {trainable:,}/{total:,}')
        module_params = {}
        for n, p in model.named_parameters():
            mod = n.split('.')[0]
            module_params.setdefault(mod, [0, 0])
            module_params[mod][0] += p.numel()
            module_params[mod][1] += p.numel() if p.requires_grad else 0
        for mod, (tot, trn) in sorted(module_params.items()):
            self._log(f'  {mod:20s}: {trn:>10,} / {tot:>10,}')
        if hasattr(model, 'use_sda') and model.use_sda:
            self._log(f'  SDA enabled: per-head learnable gate (4 params/layer × {len(model.gat.layers)} layers)')

    def _log(self, msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f'[{ts}] {msg}'
        print(line)
        self.log.write(line + '\n')
        self.log.flush()

    def _gpu_stats(self):
        if self.device.type != 'cuda':
            return 0.0, 0.0
        mem = torch.cuda.memory_allocated(self.device) / 1024**2
        peak = torch.cuda.max_memory_allocated(self.device) / 1024**2
        return mem, peak

    def _to_device(self, batch):
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def _make_scheduler(self, opt, epochs):
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, StepLR
        train_cfg = self.cfg['training']
        warmup = LinearLR(opt, start_factor=0.01, total_iters=self.warmup_epochs)
        decay = StepLR(opt, step_size=train_cfg.get('decay_step', 20),
                       gamma=train_cfg.get('decay_gamma', 0.5))
        return SequentialLR(opt, [warmup, decay], milestones=[self.warmup_epochs])

    def _backward_step(self, loss, opt, model_params, separate_clip=False,
                       encoder_params=None, denoiser_params=None):
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(opt)
            if separate_clip and encoder_params and denoiser_params:
                torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
                torch.nn.utils.clip_grad_norm_(denoiser_params, 0.1)
            else:
                torch.nn.utils.clip_grad_norm_(model_params, 1.0)
            self.scaler.step(opt)
            self.scaler.update()
        else:
            loss.backward()
            if separate_clip and encoder_params and denoiser_params:
                torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
                torch.nn.utils.clip_grad_norm_(denoiser_params, 0.1)
            else:
                torch.nn.utils.clip_grad_norm_(model_params, 1.0)
            opt.step()

    def pretrain_denoiser(self, epochs=50, lr=1e-3):
        """Phase 1: train denoiser with noise estimation loss."""
        self._log('=== Phase 1: Pretrain Denoiser ===')

        for param in self.model.parameters():
            param.requires_grad = True

        encoder_lr = lr * self.cfg.get('training', {}).get('encoder_lr_ratio_phase1', 0.2)
        param_groups = [
            {'params': [p for n, p in self.model.named_parameters() if 'denoiser' in n],
             'lr': lr},
            {'params': [p for n, p in self.model.named_parameters()
                        if 'denoiser' not in n and p.requires_grad],
             'lr': encoder_lr},
        ]

        opt = torch.optim.AdamW(param_groups)
        scheduler = self._make_scheduler(opt, epochs)
        self._optimizer = opt
        self._scheduler = scheduler

        best_val_loss = float('inf')

        for epoch in range(epochs):
            t_start = time.time()
            self.model.train()
            loss_total, count = 0, 0

            for batch in self.train_loader:
                batch = self._to_device(batch)
                opt.zero_grad()

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    h = self.model.encode(batch)
                    # [S3] Deferred biased sampling: only in last 20% of Phase 1
                    epoch_frac = epoch / max(epochs - 1, 1)
                    loss = self.model.noise_estimation_loss(
                        batch, detach_encoder=False, h=h, epoch_frac=epoch_frac,
                    )

                    # [MD1] COLREGs aux loss with warmup to avoid wasting compute on zero-residual early epochs
                    colregs_warmup_frac = self.cfg.get('training', {}).get('colregs_aux_warmup_frac', 0.3)
                    eff_colregs_w = self.colregs_aux_weight * min(1.0, epoch_frac / max(colregs_warmup_frac, 1e-8))
                    if eff_colregs_w > 0 and 'encounter_type' in batch:
                        B = batch['obs'].shape[0]
                        N = batch['obs'].shape[1]
                        tidx = batch.get('target_ship_idx',
                                         torch.zeros(B, dtype=torch.long, device=self.device))
                        bidx = torch.arange(B, device=self.device)

                        target_obs = batch['obs'][bidx, tidx]
                        ctx = h[bidx, tidx]
                        anchor_mean = self.model.anchor._kinematic_extrapolate(target_obs)
                        if self.model.anchor.use_residual:
                            residual = self.model.anchor.residual_mlp(ctx)
                            anchor_mean = anchor_mean + residual.reshape(anchor_mean.shape)

                        all_pred = batch['pred'][:, :, :, :self.model.pred_dim].detach()
                        one_hot = torch.zeros(B, N, 1, 1, device=self.device)
                        one_hot[bidx, tidx] = 1.0
                        anchor_exp = anchor_mean.unsqueeze(1).expand_as(all_pred)
                        all_pred = all_pred * (1 - one_hot) + anchor_exp * one_hot
                        colregs_e = self.model.colregs_energy(
                            all_pred, batch['encounter_type'],
                            batch['cri_matrix'], batch['mask']
                        )
                        loss = loss + eff_colregs_w * colregs_e.mean()

                self._backward_step(loss, opt, self.model.parameters())

                loss_total += loss.item()
                count += 1

                if self.cfg['training'].get('debug') and count == 2:
                    break

            scheduler.step()
            epoch_time = time.time() - t_start
            cur_lr = opt.param_groups[0]['lr']
            gpu_mem, gpu_peak = self._gpu_stats()
            train_noise = loss_total / count

            val_noise = float('nan')
            if (epoch + 1) % self.cfg['training'].get('test_interval', 5) == 0:
                if self.val_loader is not None:
                    val_noise = self._eval_noise_loss(self.val_loader)
                    if val_noise < best_val_loss:
                        best_val_loss = val_noise
                        self._save_checkpoint(epoch, 'phase1_best')
                self._save_checkpoint(epoch, 'phase1')

            self._log(
                f'Phase1 Epoch {epoch}: noise_loss={train_noise:.6f}'
                f'  val={val_noise:.6f}  lr={cur_lr:.2e}'
                f'  colregs_w={eff_colregs_w:.3f}  biased_t={epoch_frac >= 0.8}'
                f'  gpu={gpu_mem:.0f}/{gpu_peak:.0f}MB  t={epoch_time:.1f}s'
            )
            self._p1_writer.writerow([
                epoch, f'{train_noise:.6f}',
                f'{val_noise:.6f}' if not np.isnan(val_noise) else '',
                f'{cur_lr:.2e}', f'{gpu_mem:.0f}', f'{gpu_peak:.0f}',
                f'{epoch_time:.1f}',
            ])
            self._p1_csv.flush()

        self._save_checkpoint(epochs - 1, 'phase1_final')

    def train(self, epochs=None, pretrained_denoiser_path=None):
        """Phase 2: freeze denoiser, train encoder + GAT + anchor end-to-end."""
        if pretrained_denoiser_path:
            cp = torch.load(pretrained_denoiser_path, map_location='cpu')
            self.model.load_state_dict(cp['model_dict'], strict=False)
            if 'conformal' in cp:
                self.conformal.load_state_dict(cp['conformal'])
            self._log(f'Loaded pretrained denoiser from {pretrained_denoiser_path}')

        self._log('=== Phase 2: End-to-End Training ===')

        # [C1] Fine-tune denoiser with very low LR to follow encoder representation drift
        for param in self.model.parameters():
            param.requires_grad = True

        train_cfg = self.cfg['training']
        if epochs is None:
            epochs = train_cfg['num_epochs']

        # Disable CFG dropout during Phase 2 denoiser fine-tuning:
        # denoiser is in eval mode (no BN/Dropout randomness) but we still
        # need occasional CFG dropout to keep the unconditional path fresh.
        # Use reduced dropout rate since denoiser LR is very low.
        self.model.denoiser._apply_cfg_dropout = True

        denoiser_lr = train_cfg['lr'] * train_cfg.get('denoiser_lr_ratio_phase2', 0.02)
        param_groups = [
            {'params': [p for n, p in self.model.named_parameters() if 'denoiser' not in n],
             'lr': train_cfg['lr']},
            {'params': [p for n, p in self.model.named_parameters() if 'denoiser' in n],
             'lr': denoiser_lr},
        ]
        opt = torch.optim.AdamW(param_groups)
        scheduler = self._make_scheduler(opt, epochs)
        self._optimizer = opt
        self._scheduler = scheduler
        all_params = list(self.model.parameters())

        self._log(f'  Denoiser LR: {denoiser_lr:.2e} (ratio={train_cfg.get("denoiser_lr_ratio_phase2", 0.02)})')
        self._log(f'  Loss weights: dist={dist_weight} fredf={fredf_weight} anchor={anchor_weight} '
                  f'diversity={diversity_weight} nomoto={self.nomoto_weight} all_ship_anchor={all_ship_anchor_weight}')
        self._log(f'  Diversity warmup: {diversity_warmup_epochs} epochs | Patience: {patience} | CCR bonus: {ccr_bonus}')
        self._log(f'  Grad clip: encoder@1.0 denoiser@0.1 | AMP: {self.use_amp}')

        # Collect param groups for separate gradient clipping
        encoder_params = [p for n, p in self.model.named_parameters()
                          if 'denoiser' not in n and p.requires_grad]
        denoiser_params = [p for n, p in self.model.named_parameters()
                           if 'denoiser' in n and p.requires_grad]

        dist_weight = train_cfg.get('loss_dist_weight', 50)
        fredf_weight = train_cfg.get('loss_fredf_weight', 0.1)
        anchor_weight = train_cfg.get('loss_anchor_weight', 10)
        diversity_weight = train_cfg.get('loss_diversity_weight', 1.0)
        diversity_warmup_epochs = train_cfg.get('diversity_warmup_epochs', 5)
        # [S2] All-ship anchor loss weight
        all_ship_anchor_weight = train_cfg.get('loss_all_ship_anchor_weight', 2.0)

        raw_weights = torch.arange(
            1, self.cfg['model']['pred_steps'] + 1, dtype=torch.float,
        )
        temporal_reweight = torch.sqrt(raw_weights).clamp(min=2.0)
        temporal_reweight = (temporal_reweight / temporal_reweight.mean()).to(
            self.device,
        ).unsqueeze(0).unsqueeze(0)

        best_ade = float('inf')
        best_ccr = -1.0
        best_combined = float('inf')
        ccr_bonus = train_cfg.get('ccr_bonus', 0.05)
        patience = train_cfg.get('patience', 15)
        no_improve = 0

        # Diversity loss adaptive protection: halve weight if ADE worsens 3 evals in a row
        div_degrade_count = 0
        prev_val_ade = float('inf')
        effective_diversity_weight = diversity_weight

        for epoch in range(epochs):
            t_start = time.time()
            self.model.train()
            self.model.denoiser.eval()
            loss_total, loss_dt, loss_fd, loss_ac, loss_nm, loss_dv, pw_fde_sum, count = 0, 0, 0, 0, 0, 0, 0, 0

            for batch in self.train_loader:
                batch = self._to_device(batch)
                opt.zero_grad()

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    pred_gt = batch['pred'][:, :, :, :self.model.pred_dim]
                    B = pred_gt.shape[0]
                    target_idx = batch.get(
                        'target_ship_idx',
                        torch.zeros(B, dtype=torch.long, device=self.device)
                    )
                    batch_idx = torch.arange(B, device=self.device)
                    target_gt = pred_gt[batch_idx, target_idx]

                    pred_traj, anchor_mean, h_cached = self.model.predict(batch, deterministic=True)

                    distances = (pred_traj - target_gt.unsqueeze(1)).norm(p=2, dim=-1)
                    weighted_distances = distances * temporal_reweight

                    loss_dist = weighted_distances.mean(dim=-1).min(dim=1)[0].mean()

                    fredf_topk = min(self.cfg['training'].get('fredf_topk', 1), pred_traj.shape[1])
                    if fredf_topk <= 1:
                        best_k = weighted_distances.mean(dim=-1).argmin(dim=1)
                        best_pred = pred_traj[batch_idx, best_k]
                        loss_freq = self.fredf_loss(best_pred, target_gt)
                    else:
                        topk_indices = weighted_distances.mean(dim=-1).topk(
                            fredf_topk, dim=1, largest=False)[1]
                        loss_freq = sum(
                            self.fredf_loss(pred_traj[batch_idx, topk_indices[:, ki]], target_gt)
                            for ki in range(fredf_topk)
                        ) / fredf_topk

                    loss_anchor = (anchor_mean - target_gt).norm(p=2, dim=-1).mean()

                    loss = (loss_dist * dist_weight
                            + loss_freq * fredf_weight
                            + loss_anchor * anchor_weight)

                    # Diversity: penalize K samples that are too similar
                    # Conditional diversity: exclude the best sample (closest to GT)
                    # from being pushed away, preserving accuracy while promoting spread.
                    if effective_diversity_weight > 0 and pred_traj.shape[1] > 1:
                        div_scale = min(1.0, epoch / max(diversity_warmup_epochs, 1))
                        eff_div_weight = effective_diversity_weight * div_scale
                        K = pred_traj.shape[1]
                        T_div = pred_traj.shape[2]

                        best_k_div = weighted_distances.mean(dim=-1).argmin(dim=1)  # [B]
                        pw = pred_traj.unsqueeze(1) - pred_traj.unsqueeze(2)  # [B, K, K, T, 2]
                        pw_norm = pw.norm(dim=-1)  # [B, K, K, T]
                        div_tw = torch.linspace(0.5, 1.5, T_div, device=self.device)
                        div_tw = div_tw / div_tw.mean()
                        pw_dist = (pw_norm * div_tw.view(1, 1, 1, -1)).mean(dim=-1)  # [B, K, K]
                        diag_mask = 1.0 - torch.eye(K, device=self.device).unsqueeze(0)
                        # Mask out best sample rows/cols: its gradient comes only
                        # from minADE, avoiding conflict with diversity push.
                        best_mask = torch.ones(B, K, K, device=self.device)
                        best_mask[batch_idx, best_k_div, :] = 0
                        best_mask[batch_idx, :, best_k_div] = 0
                        pair_mask = diag_mask * best_mask
                        n_pairs = pair_mask.sum(dim=(1, 2)).clamp(min=1.0)
                        mean_pw_dist = (pw_dist * pair_mask).sum(dim=(1, 2)) / n_pairs
                        loss_div = -mean_pw_dist.mean()
                        loss = loss + loss_div * eff_div_weight

                        with torch.no_grad():
                            fde_pw = pw[:, :, :, -1, :].norm(dim=-1)
                            mean_pw_fde = (fde_pw * diag_mask).sum(dim=(1, 2)) / (K * (K - 1))
                            mean_pw_fde = mean_pw_fde.mean().item()

                    if self.nomoto_loss is not None:
                        obs_last = batch['obs'][batch_idx, target_idx, -1, :2]
                        l_nomoto = self.nomoto_loss(anchor_mean, obs_last)
                        loss = loss + l_nomoto * self.nomoto_weight
                        loss_nm += l_nomoto.item() * self.nomoto_weight

                    # [S2] All-ship anchor loss: train residual_mlp on ALL ships
                    if all_ship_anchor_weight > 0 and self.model.anchor.use_residual:
                        all_pred_gt = pred_gt  # [B, N, T, 2]
                        B_s, N_s = all_pred_gt.shape[:2]
                        obs_all = batch['obs']  # [B, N, T_obs, 7]
                        obs_flat = obs_all.reshape(B_s * N_s, obs_all.shape[2], obs_all.shape[3])
                        all_anchor = self.model.anchor._kinematic_extrapolate(obs_flat)
                        ctx_flat = h_cached.reshape(B_s * N_s, self.model.d_model)
                        all_residual = self.model.anchor.residual_mlp(ctx_flat)
                        all_anchor = all_anchor + all_residual.reshape(all_anchor.shape)
                        all_anchor = all_anchor.reshape(B_s, N_s, self.model.pred_steps, self.model.pred_dim)
                        # [L5] Per-sample mean then batch mean (consistent with other losses)
                        per_ship_err = (all_anchor - all_pred_gt).norm(p=2, dim=-1).mean(dim=-1)  # [B, N]
                        valid_ships = batch['mask'].float()
                        l_all_anchor = (per_ship_err * valid_ships).sum(dim=1) / valid_ships.sum(dim=1).clamp(min=1)
                        l_all_anchor = l_all_anchor.mean()
                        loss = loss + l_all_anchor * all_ship_anchor_weight

                self._backward_step(loss, opt, all_params,
                                    separate_clip=True,
                                    encoder_params=encoder_params,
                                    denoiser_params=denoiser_params)

                # Per-module gradient norm monitoring (periodic, not just first 3 epochs)
                grad_log_interval = train_cfg.get('test_interval', 5)
                if count % 50 == 0:
                    with torch.no_grad():
                        grad_norms = {}
                        nan_params = []
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    nan_params.append(name)
                                module = name.split('.')[0]
                                norm = param.grad.data.norm(2).item()
                                grad_norms[module] = grad_norms.get(module, 0.0) + norm ** 2
                        if nan_params:
                            self._log(f'  *** ANOMALY: NaN/Inf grads in: {", ".join(nan_params[:5])} ***')
                        grad_parts = [f'{k}={v**0.5:.4f}' for k, v in sorted(grad_norms.items())]
                        if grad_parts and (epoch < 3 or epoch % grad_log_interval == 0):
                            self._log(f'  grad_norms [{count}]: {" ".join(grad_parts)}')

                        if hasattr(self.model, 'use_sda') and self.model.use_sda:
                            gate_strs = []
                            for li, layer in enumerate(self.model.gat.layers):
                                if hasattr(layer, 'sda_gate'):
                                    alphas = torch.sigmoid(layer.sda_gate).cpu().tolist()
                                    gate_strs.append(f'L{li}=[{",".join(f"{a:.3f}" for a in alphas)}]')
                            if gate_strs and (epoch < 3 or epoch % grad_log_interval == 0):
                                self._log(f'  sda_gates [{count}]: {" ".join(gate_strs)}')

                loss_total += loss.item()
                loss_dt += loss_dist.item() * dist_weight
                loss_fd += loss_freq.item() * fredf_weight
                loss_ac += loss_anchor.item() * anchor_weight
                if effective_diversity_weight > 0 and pred_traj.shape[1] > 1:
                    loss_dv += loss_div.item() * eff_div_weight
                    pw_fde_sum += mean_pw_fde
                count += 1

                if train_cfg.get('debug') and count == 2:
                    break

                if count == 1 and epoch == 0 and self.device.type == 'cuda':
                    mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
                    self._log(f'Peak GPU memory after 1st batch: {mem_mb:.0f} MB')

            scheduler.step()
            epoch_time = time.time() - t_start
            cur_lr = opt.param_groups[0]['lr']
            gpu_mem, gpu_peak = self._gpu_stats()

            log_parts = [
                f'Epoch {epoch}: loss={loss_total / count:.6f}',
                f'dist={loss_dt / count:.6f}',
                f'fredf={loss_fd / count:.6f}',
                f'anchor={loss_ac / count:.6f}',
            ]
            if self.nomoto_loss:
                log_parts.append(f'nomoto={loss_nm / count:.6f}')
            if effective_diversity_weight > 0:
                log_parts.append(f'div={loss_dv / count:.6f}')
                log_parts.append(f'pw_fde={pw_fde_sum / count:.4f}nm')
            log_parts.append(f'lr={cur_lr:.2e}')
            log_parts.append(f'gpu={gpu_mem:.0f}/{gpu_peak:.0f}MB')
            log_parts.append(f't={epoch_time:.1f}s')
            self._log(' '.join(log_parts))

            # Loss component ratio breakdown (helps diagnose which loss dominates)
            avg_total = loss_total / max(count, 1)
            if avg_total > 0 and (epoch < 3 or (epoch + 1) % train_cfg.get('test_interval', 5) == 0):
                ratios = []
                for lname, lval in [('dist', loss_dt), ('fredf', loss_fd), ('anchor', loss_ac),
                                     ('nomoto', loss_nm), ('div', loss_dv)]:
                    avg_l = lval / max(count, 1)
                    if avg_l != 0:
                        ratios.append(f'{lname}={avg_l / avg_total:.1%}')
                self._log(f'  loss_ratios: {" ".join(ratios)}')

            # Anomaly detection
            if np.isnan(loss_total) or np.isinf(loss_total):
                self._log('  *** ANOMALY: total loss is NaN/Inf! Check gradients and data. ***')
            elif epoch > 0 and hasattr(self, '_prev_epoch_loss'):
                spike = (loss_total / count) / max(self._prev_epoch_loss, 1e-8)
                if spike > 5.0:
                    self._log(f'  *** ANOMALY: loss spiked {spike:.1f}x vs previous epoch ***')
            self._prev_epoch_loss = loss_total / max(count, 1)

            val_ade = val_fde = val_ccr = val_dcpa = float('nan')
            val_anchor_ade = val_anchor_fde = float('nan')
            if (epoch + 1) % train_cfg.get('test_interval', 5) == 0:
                if self.val_loader is not None:
                    metrics = self.evaluate()
                    self._log_metrics('  Val', metrics, verbose=True)
                    val_ade = metrics['ADE']
                    val_fde = metrics['FDE']
                    val_ccr = metrics.get('CCR', float('nan'))
                    val_dcpa = metrics.get('DCPA_MAE', float('nan'))
                    val_anchor_ade = metrics.get('Anchor_ADE', float('nan'))
                    val_anchor_fde = metrics.get('Anchor_FDE', float('nan'))

                    # CFG health check: detect unconditional path degradation
                    if epoch >= 10:
                        metrics_cfg = self.evaluate(cfg_scale=1.0)
                        cfg_delta = metrics_cfg['ADE'] - metrics['ADE']
                        self._log(
                            f'  CFG health: ADE_base={metrics["ADE"]:.4f} '
                            f'ADE_cfg={metrics_cfg["ADE"]:.4f} '
                            f'delta={cfg_delta:+.4f}'
                            f'{" [OK]" if cfg_delta < 0 else " [WARN: CFG hurts]"}'
                        )

                    # [C3] Multi-metric checkpoint strategy
                    if val_ade < best_ade:
                        best_ade = val_ade
                        self._save_checkpoint(epoch, 'best_ade')

                    # Diversity adaptive protection: if ADE worsens 3 evals in a row, halve diversity weight
                    if val_ade > prev_val_ade:
                        div_degrade_count += 1
                    else:
                        div_degrade_count = 0
                    prev_val_ade = val_ade
                    if div_degrade_count >= 3 and effective_diversity_weight > 0.1:
                        effective_diversity_weight *= 0.5
                        div_degrade_count = 0
                        self._log(f'  Diversity protection: ADE worsened 3x, reducing div_weight to {effective_diversity_weight:.3f}')

                    if not np.isnan(val_ccr) and val_ccr > best_ccr:
                        best_ccr = val_ccr
                        self._save_checkpoint(epoch, 'best_ccr')

                    combined = val_ade - ccr_bonus * (val_ccr if not np.isnan(val_ccr) else 0.0)
                    if combined < best_combined:
                        best_combined = combined
                        no_improve = 0
                        self._save_checkpoint(epoch, 'best')
                    else:
                        no_improve += 1
                        self._log(f'  No improvement {no_improve}/{patience} (combined={combined:.6f}, best={best_combined:.6f})')
                        if no_improve >= patience:
                            self._log(f'Early stopping: no improvement for {patience} evals')
                            self._log(f'  Best ADE={best_ade:.4f}  Best CCR={best_ccr:.2%}  Best combined={best_combined:.6f}')
                            self._write_p2_csv_row(
                                epoch, count, loss_total, loss_dt, loss_fd, loss_ac,
                                loss_nm, loss_dv, pw_fde_sum, val_ade, val_fde,
                                val_ccr, val_dcpa, val_anchor_ade, val_anchor_fde,
                                cur_lr, effective_diversity_weight, gpu_mem, gpu_peak,
                                epoch_time, best_ade, no_improve,
                            )
                            break
                self._save_checkpoint(epoch, 'latest')

            self._write_p2_csv_row(
                epoch, count, loss_total, loss_dt, loss_fd, loss_ac,
                loss_nm, loss_dv, pw_fde_sum, val_ade, val_fde,
                val_ccr, val_dcpa, val_anchor_ade, val_anchor_fde,
                cur_lr, effective_diversity_weight, gpu_mem, gpu_peak,
                epoch_time, best_ade, no_improve,
            )

    def _write_p2_csv_row(self, epoch, count, loss_total, loss_dt, loss_fd, loss_ac,
                          loss_nm, loss_dv, pw_fde_sum, val_ade, val_fde,
                          val_ccr, val_dcpa, val_anchor_ade, val_anchor_fde,
                          cur_lr, eff_div_weight, gpu_mem, gpu_peak,
                          epoch_time, best_ade, no_improve):
        def _f(v, fmt='.6f'):
            return '' if (isinstance(v, float) and np.isnan(v)) else f'{v:{fmt}}'

        sda_gates_str = ''
        if hasattr(self.model, 'use_sda') and self.model.use_sda:
            gate_vals = []
            for layer in self.model.gat.layers:
                if hasattr(layer, 'sda_gate'):
                    gate_vals.extend(torch.sigmoid(layer.sda_gate).detach().cpu().tolist())
            sda_gates_str = '|'.join(f'{g:.4f}' for g in gate_vals)

        self._p2_writer.writerow([
            epoch,
            _f(loss_total / max(count, 1)), _f(loss_dt / max(count, 1)),
            _f(loss_fd / max(count, 1)), _f(loss_ac / max(count, 1)),
            _f(loss_nm / max(count, 1)), _f(loss_dv / max(count, 1)),
            _f(pw_fde_sum / max(count, 1), '.4f'),
            _f(val_ade, '.4f'), _f(val_fde, '.4f'),
            _f(val_ccr, '.4f'), _f(val_dcpa, '.4f'),
            _f(val_anchor_ade, '.4f'), _f(val_anchor_fde, '.4f'),
            f'{cur_lr:.2e}', f'{eff_div_weight:.4f}', sda_gates_str,
            f'{gpu_mem:.0f}', f'{gpu_peak:.0f}',
            f'{epoch_time:.1f}', _f(best_ade, '.4f'), no_improve,
        ])
        self._p2_csv.flush()

    def _denormalize_positions(self, positions):
        mean = self.model.anchor.norm_mean[:2]
        std = self.model.anchor.norm_std[:2]
        return positions * std + mean

    def _log_metrics(self, prefix, metrics, verbose=False):
        parts = [f'{prefix}: ADE={metrics["ADE"]:.4f}  FDE={metrics["FDE"]:.4f}']
        if 'Anchor_ADE' in metrics and verbose:
            diff_pct = (1 - metrics['ADE'] / max(metrics['Anchor_ADE'], 1e-8)) * 100
            parts.append(
                f'Anchor_ADE={metrics["Anchor_ADE"]:.4f}  '
                f'Anchor_FDE={metrics["Anchor_FDE"]:.4f}  '
                f'Diffusion_improvement={diff_pct:.1f}%'
            )
        if 'CCR' in metrics:
            parts.append(f'CCR={metrics["CCR"]:.2%}({metrics["CCR_n"]})')
        if 'DCPA_MAE' in metrics:
            if verbose:
                parts.append(
                    f'DCPA_MAE={metrics["DCPA_MAE"]:.4f}nm  '
                    f'pred_DCPA={metrics["pred_DCPA"]:.4f}nm  '
                    f'gt_DCPA={metrics["gt_DCPA"]:.4f}nm'
                )
            else:
                parts.append(f'DCPA_MAE={metrics["DCPA_MAE"]:.4f}nm')
        self._log('  '.join(parts))

    def _save_rng_state(self):
        state = {
            'torch': torch.random.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
        if torch.cuda.is_available():
            state['cuda'] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state):
        torch.random.set_rng_state(state['torch'])
        np.random.set_state(state['numpy'])
        random.setstate(state['python'])
        if 'cuda' in state:
            torch.cuda.set_rng_state_all(state['cuda'])

    def evaluate(self, loader=None, guidance_scale=0.0, cfg_scale=0.0, use_gt_neighbors=False):
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        self.model.denoiser._apply_cfg_dropout = False
        ade_total, fde_total, count = 0, 0, 0
        anchor_ade_total, anchor_fde_total = 0, 0
        ccr_compliant, ccr_applicable = 0, 0
        dcpa_err_sum, pred_dcpa_sum, gt_dcpa_sum, dcpa_n = 0.0, 0.0, 0.0, 0

        rng_state = self._save_rng_state()
        self._set_seed(0)
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                pred_gt = batch['pred'][:, :, :, :self.model.pred_dim]
                B = pred_gt.shape[0]
                target_idx = batch.get(
                    'target_ship_idx',
                    torch.zeros(B, dtype=torch.long, device=self.device)
                )
                batch_idx = torch.arange(B, device=self.device)
                target_gt = pred_gt[batch_idx, target_idx]

                pred_traj, anchor_mean, _ = self.model.predict(batch, guidance_scale=guidance_scale,
                                                   cfg_scale=cfg_scale,
                                                   deterministic=True, use_gt_neighbors=use_gt_neighbors)

                target_gt_nm = self._denormalize_positions(target_gt)
                pred_traj_nm = self._denormalize_positions(pred_traj)

                # Anchor ADE/FDE (measures diffusion contribution)
                anchor_nm = self._denormalize_positions(anchor_mean)
                anchor_err = (anchor_nm - target_gt_nm).norm(p=2, dim=-1)
                anchor_ade_total += anchor_err.mean(dim=-1).sum().item()
                anchor_fde_total += anchor_err[:, -1].sum().item()

                gt_expand = target_gt_nm.unsqueeze(1).expand_as(pred_traj_nm)
                distances = (pred_traj_nm - gt_expand).norm(p=2, dim=-1)

                ade = distances.mean(dim=-1).min(dim=1)[0].sum()
                fde = distances[:, :, -1].min(dim=1)[0].sum()

                ade_total += ade.item()
                fde_total += fde.item()
                count += B

                if 'encounter_type' in batch:
                    best_k = distances.mean(dim=-1).argmin(dim=1)
                    pred_best = pred_traj_nm[batch_idx, best_k]

                    ccr_results = compute_ccr(
                        pred_best, batch['encounter_type'],
                        target_idx, batch['mask'],
                    )
                    primary_th = self.ccr_threshold_deg
                    if primary_th in ccr_results:
                        nc, na = ccr_results[primary_th]
                    else:
                        nc, na = ccr_results[min(ccr_results.keys())]
                    ccr_compliant += nc
                    ccr_applicable += na

                    pred_gt_all_nm = self._denormalize_positions(pred_gt)
                    de, pd, gd, dn = compute_predicted_dcpa(
                        pred_best, pred_gt_all_nm,
                        target_idx, batch['mask'],
                        batch['encounter_type'],
                    )
                    dcpa_err_sum += de
                    pred_dcpa_sum += pd
                    gt_dcpa_sum += gd
                    dcpa_n += dn

        self._restore_rng_state(rng_state)
        self.model.denoiser._apply_cfg_dropout = True

        metrics = {
            'ADE': ade_total / count, 'FDE': fde_total / count,
            'Anchor_ADE': anchor_ade_total / count,
            'Anchor_FDE': anchor_fde_total / count,
        }

        if ccr_applicable > 0:
            metrics['CCR'] = ccr_compliant / ccr_applicable
            metrics['CCR_n'] = ccr_applicable
        if dcpa_n > 0:
            metrics['DCPA_MAE'] = dcpa_err_sum / dcpa_n
            metrics['pred_DCPA'] = pred_dcpa_sum / dcpa_n
            metrics['gt_DCPA'] = gt_dcpa_sum / dcpa_n

        return metrics

    def evaluate_stratified(self, loader=None, guidance_scale=0.0, use_gt_neighbors=False):
        """Evaluate with stratified breakdown by encounter type and difficulty."""
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        self.model.denoiser._apply_cfg_dropout = False
        all_pred, all_gt, all_enc, all_tidx, all_mask, all_obs = [], [], [], [], [], []

        rng_state = self._save_rng_state()
        self._set_seed(0)
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                B = batch['obs'].shape[0]
                target_idx = batch.get(
                    'target_ship_idx',
                    torch.zeros(B, dtype=torch.long, device=self.device)
                )
                batch_idx = torch.arange(B, device=self.device)

                pred_traj, _, _ = self.model.predict(batch, guidance_scale=guidance_scale,
                                                   deterministic=True, use_gt_neighbors=use_gt_neighbors)
                pred_gt = batch['pred'][:, :, :, :self.model.pred_dim]
                target_gt = pred_gt[batch_idx, target_idx]

                pred_traj_nm = self._denormalize_positions(pred_traj)
                target_gt_nm = self._denormalize_positions(target_gt)
                obs_target = batch['obs'][batch_idx, target_idx]

                obs_target_nm = obs_target.clone()
                norm_mean = self.model.anchor.norm_mean
                norm_std = self.model.anchor.norm_std
                obs_target_nm[:, :, 0] = obs_target[:, :, 0] * norm_std[0] + norm_mean[0]
                obs_target_nm[:, :, 1] = obs_target[:, :, 1] * norm_std[1] + norm_mean[1]
                obs_target_nm[:, :, 2] = obs_target[:, :, 2] * norm_std[2] + norm_mean[2]

                all_pred.append(pred_traj_nm.cpu())
                all_gt.append(target_gt_nm.cpu())
                all_enc.append(batch['encounter_type'].cpu())
                all_tidx.append(target_idx.cpu())
                all_mask.append(batch['mask'].cpu())
                all_obs.append(obs_target_nm.cpu())

        pred_cat = torch.cat(all_pred, dim=0)
        gt_cat = torch.cat(all_gt, dim=0)
        enc_cat = torch.cat(all_enc, dim=0)
        tidx_cat = torch.cat(all_tidx, dim=0)
        mask_cat = torch.cat(all_mask, dim=0)
        obs_cat = torch.cat(all_obs, dim=0)

        self._restore_rng_state(rng_state)
        self.model.denoiser._apply_cfg_dropout = True

        results = stratified_evaluate(pred_cat, gt_cat, enc_cat, tidx_cat, mask_cat, obs_cat)
        self._log(format_stratified_results(results))
        return results

    def test(self, checkpoint_path=None, guidance_scale=0.0, cfg_scale=0.0, use_gt_neighbors=False):
        if checkpoint_path:
            cp = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(cp['model_dict'])
            if 'conformal' in cp:
                self.conformal.load_state_dict(cp['conformal'])
            self._log(f'Loaded checkpoint from {checkpoint_path}')

        if self.test_loader is None:
            self._log('No test split available, skipping test.')
            return {'ADE': float('inf'), 'FDE': float('inf')}

        # --- Baseline: no guidance, no CFG ---
        metrics = self.evaluate(self.test_loader, guidance_scale=0.0,
                                cfg_scale=0.0, use_gt_neighbors=use_gt_neighbors)
        self._log_metrics('Test (baseline)', metrics, verbose=True)

        self._log('--- Stratified Results (baseline) ---')
        self.evaluate_stratified(self.test_loader, guidance_scale=0.0,
                                 use_gt_neighbors=use_gt_neighbors)

        # --- Auto CFG scale sweep ---
        cfg_scales_to_test = [0.5, 1.0, 1.5]
        if cfg_scale > 0 and cfg_scale not in cfg_scales_to_test:
            cfg_scales_to_test.append(cfg_scale)
        self._log('--- CFG Scale Sweep ---')
        best_cfg_ade, best_cfg_scale = metrics['ADE'], 0.0
        for cs in sorted(cfg_scales_to_test):
            m = self.evaluate(self.test_loader, guidance_scale=0.0,
                              cfg_scale=cs, use_gt_neighbors=use_gt_neighbors)
            self._log_metrics(f'  cfg_scale={cs}', m)
            if m['ADE'] < best_cfg_ade:
                best_cfg_ade, best_cfg_scale = m['ADE'], cs
        self._log(f'  Best CFG: scale={best_cfg_scale} ADE={best_cfg_ade:.4f}')

        # --- COLREGs guidance sweep ---
        if guidance_scale > 0:
            guidance_scales_to_test = [0.1, 0.3, 0.5, 1.0]
            if guidance_scale not in guidance_scales_to_test:
                guidance_scales_to_test.append(guidance_scale)
            self._log('--- Guidance Scale Sweep ---')
            for gs in sorted(guidance_scales_to_test):
                m = self.evaluate(self.test_loader, guidance_scale=gs,
                                  cfg_scale=cfg_scale, use_gt_neighbors=use_gt_neighbors)
                self._log_metrics(f'  guidance={gs}', m)

            # Full stratified with best guidance
            metrics_guided = self.evaluate(self.test_loader, guidance_scale=guidance_scale,
                                           cfg_scale=cfg_scale, use_gt_neighbors=use_gt_neighbors)
            self._log_metrics(f'Test (guidance={guidance_scale}, cfg={cfg_scale})', metrics_guided, verbose=True)

            self._log(f'--- Stratified Results (guided={guidance_scale}) ---')
            self.evaluate_stratified(self.test_loader, guidance_scale=guidance_scale,
                                     use_gt_neighbors=use_gt_neighbors)

            if 'CCR' in metrics and 'CCR' in metrics_guided:
                self._log(
                    f'  Guidance effect:  '
                    f'CCR {metrics["CCR"]:.2%} → {metrics_guided["CCR"]:.2%}  '
                    f'DCPA_MAE {metrics.get("DCPA_MAE", float("nan")):.4f} → '
                    f'{metrics_guided.get("DCPA_MAE", float("nan")):.4f}nm'
                )

        if self.val_loader is not None and self.test_loader is not None:
            self._log('--- Conformal Prediction Calibration ---')
            val_preds, val_gt = self._collect_predictions(self.val_loader)
            self.conformal.calibrate(val_preds, val_gt)

            test_preds, test_gt = self._collect_predictions(self.test_loader)
            conf_metrics = self.conformal.evaluate(test_preds, test_gt)
            self._log(
                f'  Coverage: {conf_metrics["coverage"]:.2%} (target: 90%)'
                f'  Mean radius: {conf_metrics["mean_radius_nm"]:.4f}nm'
                f'  Max radius: {conf_metrics["max_radius_nm"]:.4f}nm'
            )

        return metrics

    def _collect_predictions(self, loader):
        """Collect all predictions and ground truths for conformal calibration."""
        self.model.eval()
        all_preds, all_gts = [], []

        rng_state = self._save_rng_state()
        self._set_seed(0)
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                B = batch['obs'].shape[0]
                target_idx = batch.get(
                    'target_ship_idx',
                    torch.zeros(B, dtype=torch.long, device=self.device)
                )
                batch_idx = torch.arange(B, device=self.device)

                pred_traj, _, _ = self.model.predict(batch, deterministic=True)
                pred_gt = batch['pred'][:, :, :, :self.model.pred_dim]
                target_gt = pred_gt[batch_idx, target_idx]

                all_preds.append(self._denormalize_positions(pred_traj).cpu())
                all_gts.append(self._denormalize_positions(target_gt).cpu())

        self._restore_rng_state(rng_state)
        return torch.cat(all_preds, dim=0), torch.cat(all_gts, dim=0)

    def _save_checkpoint(self, epoch, tag):
        path = os.path.join(self.log_dir, f'checkpoint_{tag}.pt')
        save_dict = {
            'epoch': epoch,
            'model_dict': self.model.state_dict(),
            'cfg': self.cfg,
            'conformal': self.conformal.state_dict(),
        }
        if hasattr(self, '_optimizer') and self._optimizer is not None:
            save_dict['optimizer_dict'] = self._optimizer.state_dict()
        if hasattr(self, '_scheduler') and self._scheduler is not None:
            save_dict['scheduler_dict'] = self._scheduler.state_dict()
        torch.save(save_dict, path)
        self._log(f'Saved checkpoint: {path}')

    def _set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def close(self):
        """Close all open file handles."""
        for f in [self.log, self._p1_csv, self._p2_csv]:
            if f and not f.closed:
                f.close()

    def __del__(self):
        self.close()

    def _eval_noise_loss(self, loader):
        self.model.eval()
        # Temporarily disable biased t-sampling so val loss uses uniform t
        # and is comparable across epochs (train uses progressive bias)
        orig_biased = self.model.biased_t_sampling
        self.model.biased_t_sampling = False
        loss_sum, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    loss = self.model.noise_estimation_loss(batch)
                loss_sum += loss.item() * batch['obs'].shape[0]
                n += batch['obs'].shape[0]
        self.model.biased_t_sampling = orig_biased
        return loss_sum / max(n, 1)