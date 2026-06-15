import math
import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    """Standard DDPM-style timestep embedding: sinusoidal encoding → MLP."""

    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
        self.register_buffer('freqs', freqs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t):
        """t: [B] integer timesteps → [B, dim]"""
        t_float = t.float().unsqueeze(-1)
        emb = t_float * self.freqs
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        return self._layer(x) * gate + bias


class AdaLNTransformerBlock(nn.Module):
    """Transformer block with adaptive LayerNorm (DiT-style)."""

    def __init__(self, d_model, nhead, dim_feedforward, ctx_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=dropout
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ada1 = nn.Linear(ctx_dim, d_model * 2)
        self.ada2 = nn.Linear(ctx_dim, d_model * 2)

    def forward(self, x, ctx):
        s1, b1 = self.ada1(ctx).chunk(2, dim=-1)
        h = self.norm1(x) * (1 + s1) + b1
        x = x + self.self_attn(h, h, h, need_weights=False)[0]

        s2, b2 = self.ada2(ctx).chunk(2, dim=-1)
        h = self.norm2(x) * (1 + s2) + b2
        x = x + self.ff(h)
        return x


class ConditionalDenoisingModel(nn.Module):
    def __init__(self, context_dim=256, pred_dim=2, pred_steps=30,
                 encounter_embed_dim=8,
                 time_embed_dim=64, enc_proj_dim=64,
                 cfg_dropout=0.1, tf_layers=4, n_head=8, ff_mult=4):
        super().__init__()
        self.pred_dim = pred_dim
        self.pred_steps = pred_steps
        self.cfg_dropout = cfg_dropout
        self.encounter_embed_dim = encounter_embed_dim
        self.enc_proj_dim = enc_proj_dim
        self.context_dim = context_dim
        self._apply_cfg_dropout = True

        self.time_embed = SinusoidalTimestepEmbedding(time_embed_dim)

        self.enc_proj = nn.Sequential(
            nn.Linear(encounter_embed_dim, enc_proj_dim),
            nn.SiLU(),
            nn.Linear(enc_proj_dim, enc_proj_dim),
        )

        ctx_dim = context_dim + time_embed_dim + enc_proj_dim

        d_tf = 2 * context_dim
        self.pos_emb = PositionalEncoding(d_model=d_tf, dropout=0.1, max_len=pred_steps + 4)
        self.concat1 = ConcatSquashLinear(pred_dim, d_tf, ctx_dim)

        self.transformer_layers = nn.ModuleList([
            AdaLNTransformerBlock(d_tf, n_head, d_tf * ff_mult, ctx_dim, dropout=0.1)
            for _ in range(tf_layers)
        ])

        self.concat3 = ConcatSquashLinear(d_tf, context_dim, ctx_dim)
        self.concat4 = ConcatSquashLinear(context_dim, context_dim // 2, ctx_dim)
        self.linear = ConcatSquashLinear(context_dim // 2, pred_dim, ctx_dim)

    def _denoise(self, x, ctx_emb):
        x = self.concat1(ctx_emb, x)
        x = self.pos_emb(x)
        for layer in self.transformer_layers:
            x = layer(x, ctx_emb)
        x = self.concat3(ctx_emb, x)
        x = self.concat4(ctx_emb, x)
        return self.linear(ctx_emb, x)

    def forward(self, x, t, context, encounter_emb):
        """
        x: [B, T_pred, pred_dim]
        t: [B] integer timesteps
        context: [B, 1, context_dim]
        encounter_emb: [B, encounter_embed_dim]
        """
        time_emb = self.time_embed(t).unsqueeze(1)

        # Joint CFG dropout: drop BOTH context and encounter together
        # so the model learns a true "unconditional" mode.
        # Uses _apply_cfg_dropout flag instead of self.training so that
        # Phase 2 fine-tuning (denoiser in eval mode) can still apply dropout.
        if self._apply_cfg_dropout and self.cfg_dropout > 0:
            drop = torch.rand(x.shape[0], 1, 1, device=x.device) < self.cfg_dropout
            context = context.masked_fill(drop, 0)
            encounter_emb = encounter_emb.masked_fill(drop.squeeze(-1), 0)

        enc_ctx = self.enc_proj(encounter_emb).unsqueeze(1)

        ctx_emb = torch.cat([time_emb, context, enc_ctx], dim=-1)
        return self._denoise(x, ctx_emb)

    def forward_cfg(self, x, t, context, encounter_emb, cfg_scale=1.5):
        """CFG: amplify the full conditional signal (context + encounter)."""
        eps_cond = self.forward(x, t, context, encounter_emb)

        # Unconditional: zero out ALL conditioning except timestep.
        # Pass zeros through enc_proj to match training dropout behavior.
        B = x.size(0)
        time_emb = self.time_embed(t).unsqueeze(1)
        zero_ctx = torch.zeros(B, 1, self.context_dim, device=x.device)
        zero_enc_raw = torch.zeros(B, self.encounter_embed_dim, device=x.device)
        zero_enc = self.enc_proj(zero_enc_raw).unsqueeze(1)
        ctx_emb_uncond = torch.cat([time_emb, zero_ctx, zero_enc], dim=-1)
        eps_uncond = self._denoise(x, ctx_emb_uncond)

        return (1 + cfg_scale) * eps_cond - cfg_scale * eps_uncond