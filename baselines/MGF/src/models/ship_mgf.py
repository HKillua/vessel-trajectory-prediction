"""ShipMGF: top-level model for ship trajectory prediction using Mixed Gaussian Flow.

Combines ShipTransformerEncoder + CIF-wrapped RealNVP flow with GMM base distribution.
Self-contained within baselines/MGF — no cross-baseline imports.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from models.ship_encoder import ShipTransformerEncoder
from models.TP.fastpredNF import (
    fastpredNF_CIF_separate_cond_clusterGMM,
    create_RealNVP_step,
)


class ShipMGF(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.obs_len = cfg["obs_len"]
        self.pred_len = cfg["pred_len"]

        self.encoder = ShipTransformerEncoder(
            input_dim=cfg.get("input_dim", 7),
            d_model=cfg.get("d_model", 64),
            cond_dim=cfg.get("cond_dim", 16),
            n_heads=cfg.get("n_heads", 4),
            n_enc_layers=cfg.get("n_enc_layers", 3),
            n_dec_layers=cfg.get("n_dec_layers", 3),
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            dropout=cfg.get("dropout", 0.1),
        )

        self.flow = fastpredNF_CIF_separate_cond_clusterGMM(
            input_size=2,
            n_blocks=cfg.get("n_blocks", 3),
            hidden_size=cfg.get("flow_hidden", 64),
            n_hidden=cfg.get("n_hidden", 2),
            cond_label_size=cfg.get("cond_dim", 16),
            cluster_model_path=cfg["cluster_path"],
            var_init=cfg.get("var_init", 0.1),
            learnVAR=cfg.get("learn_var", True),
            normalize_direction=False,
            flow_architecture="realNVP",
            pred_len=self.pred_len,
        )

        self.dequantize = cfg.get("dequantize", False)
        self.grad_clip = cfg.get("grad_clip", 1.0)

        decay_params = [n for n, p in self.named_parameters() if "bias" not in n]
        self.optimizer = optim.Adam(
            [
                {
                    "params": [p for n, p in self.named_parameters() if n in decay_params],
                    "weight_decay": cfg.get("weight_decay", 1e-5),
                },
                {
                    "params": [p for n, p in self.named_parameters() if n not in decay_params],
                    "weight_decay": 0.0,
                },
            ],
            lr=cfg.get("lr", 1e-3),
        )

    def _compute_flow_loss(self, data_dict):
        dist_args = self.encoder(data_dict)
        gt = data_dict["gt_st"]
        if self.dequantize:
            gt = gt + torch.rand_like(gt) / 100
        base_pos = data_dict["base_pos"]
        flow_loss = -self.flow.log_prob(base_pos, gt, dist_args)
        flow_loss = flow_loss.mean()
        return flow_loss, dist_args, base_pos, gt

    def _optim_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()

    def update(self, data_dict, bp=True):
        """Training step: NLL loss on flow."""
        flow_loss, _, _, _ = self._compute_flow_loss(data_dict)
        if bp:
            self._optim_step(flow_loss)
        return {"loss": flow_loss.item()}

    def update_mse(self, data_dict, w_mse, n_sample=20):
        """Training step: NLL + MSE loss."""
        flow_loss, dist_args, base_pos, gt = self._compute_flow_loss(data_dict)

        sampled = self.flow.sample(
            base_pos,
            cond=dist_args[:, None].expand(-1, n_sample, -1, -1),
            n_sample=n_sample,
        )
        gt_expanded = gt.unsqueeze(1)
        ade_per_sample = torch.norm(sampled - gt_expanded, dim=-1).mean(dim=-1)
        min_ade = ade_per_sample.min(dim=1).values.mean()

        loss = flow_loss + w_mse * min_ade
        self._optim_step(loss)

        return {
            "loss": loss.item(),
            "flow_loss": flow_loss.item(),
            "mse_loss": (w_mse * min_ade).item(),
        }

    @torch.no_grad()
    def predict(self, data_dict, n_sample=20):
        """Generate n_sample trajectory predictions.

        Returns: (B, n_sample, pred_len, 2) in normalized space
        """
        dist_args = self.encoder(data_dict)
        base_pos = data_dict["base_pos"]

        dist_args_exp = dist_args[:, None].expand(-1, n_sample, -1, -1)
        sampled = self.flow.sample(base_pos, cond=dist_args_exp, n_sample=n_sample)

        return sampled

    def save(self, path, epoch=0):
        torch.save(
            {
                "epoch": epoch,
                "state": self.state_dict(),
                "optim_state": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        device = next(self.parameters()).device
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.load_state_dict(ckpt["state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        return ckpt["epoch"]
