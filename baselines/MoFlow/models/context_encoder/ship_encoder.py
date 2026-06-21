"""Ship Context Encoder for MoFlow.

Replaces ETHEncoder for ship trajectory prediction:
- Temporal attention (not flatten) for 30-step × 7D features
- Variable agents (2–15) with padding mask
- Interaction bias injection into cross-agent attention
"""

import torch
import torch.nn as nn
from models.context_encoder.mtr_encoder import SinusoidalPosEmb
from models.utils.attn_utils import _forward_layer_no_fastpath, forward_encoder_with_mask


class InteractionBiasModule(nn.Module):
    """Project interaction matrices into per-head attention bias."""

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dcpa_proj = nn.Sequential(nn.Linear(1, num_heads), nn.Tanh())
        self.tcpa_proj = nn.Sequential(nn.Linear(1, num_heads), nn.Tanh())
        self.cri_proj = nn.Sequential(nn.Linear(1, num_heads), nn.Tanh())
        self.adj_weight = nn.Parameter(torch.zeros(num_heads))

    def forward(self, adj, dcpa, tcpa, cri, mask):
        """
        Args:
            adj, dcpa, tcpa, cri: [B, N, N]
            mask: [B, N] bool (True = real ship)
        Returns:
            bias: [B*H, N, N] additive attention bias
        """
        B, N, _ = adj.shape
        H = self.num_heads

        dcpa_safe = dcpa.clamp(min=0, max=20.0) / 20.0
        tcpa_safe = tcpa.clamp(min=-60.0, max=60.0) / 60.0

        dcpa_b = self.dcpa_proj(dcpa_safe.unsqueeze(-1))  # [B,N,N,H]
        tcpa_b = self.tcpa_proj(tcpa_safe.unsqueeze(-1))
        cri_b = self.cri_proj(cri.unsqueeze(-1))
        adj_b = adj.unsqueeze(-1) * self.adj_weight       # [B,N,N,H]

        total = adj_b + dcpa_b + tcpa_b + cri_b           # [B,N,N,H]
        total = total.permute(0, 3, 1, 2)                 # [B,H,N,N]

        if mask is not None:
            pad = ~mask                                    # True = padded
            pad_2d = pad.unsqueeze(1) | pad.unsqueeze(2)   # [B,N,N]
            total = total.masked_fill(pad_2d.unsqueeze(1), 0.0)

        return total.reshape(B * H, N, N)


class ShipTemporalEncoder(nn.Module):
    """Per-agent temporal encoding via TransformerEncoder.

    Input:  [B, A, P, D_in]  (P=30 obs steps, D_in=7)
    Output: [B, A, D_model]
    """

    def __init__(self, in_dim=7, d_model=128, n_layers=2, nhead=4):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)

        self.pos_emb = nn.Parameter(torch.randn(1, 300, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2, batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x, padding_mask=None):
        """x: [B, A, P, D_in] → [B, A, D_model]
           padding_mask: [B, A] bool (True=real)
        """
        B, A, P, _ = x.shape
        x = x.reshape(B * A, P, -1)
        x = self.proj(x) + self.pos_emb[:, :P, :]
        x = self.temporal_encoder(x)
        x = x[:, -1, :]                       # take last timestep
        x = self.out_mlp(x)
        out = x.reshape(B, A, -1)
        if padding_mask is not None:
            out = out * padding_mask.unsqueeze(-1).float()
        return out


class ShipEncoder(nn.Module):
    """Ship context encoder replacing ETHEncoder.

    Architecture:
    1. ShipTemporalEncoder: per-agent temporal encoding [B,A,P,7] → [B,A,D]
    2. Cross-agent TransformerEncoder with interaction bias + padding mask
    """

    def __init__(self, config, use_pre_norm):
        super().__init__()
        self.model_cfg = config
        dim = config.D_MODEL
        in_feat = config.get('INPUT_FEAT_DIM', 7)

        self.temporal_encoder = ShipTemporalEncoder(
            in_dim=in_feat, d_model=dim, n_layers=2,
            nhead=config.NUM_ATTN_HEAD,
        )

        max_agents = config.get('MAX_AGENTS', 15)
        self.agent_query_embedding = nn.Embedding(max_agents, dim)

        pos_dim = dim
        self.pos_encoding = nn.Sequential(
            SinusoidalPosEmb(pos_dim, theta=10000),
            nn.Linear(pos_dim, pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, pos_dim),
        )
        self.mlp_pe = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            dropout=config.get('DROPOUT_OF_ATTN', 0.1),
            nhead=config.NUM_ATTN_HEAD,
            dim_feedforward=dim * 4,
            norm_first=use_pre_norm,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            layer, num_layers=config.NUM_ATTN_LAYERS,
        )
        self.num_out_channels = dim

    def forward(self, past_traj, interaction_bias=None, padding_mask=None):
        """
        Args:
            past_traj:        [B, A, P, D_in]
            interaction_bias: [B*H, A, A] or None
            padding_mask:     [B, A] bool (True=real) or None
        Returns:
            [B, A, D]
        """
        B, A, P, _ = past_traj.shape

        agent_feat = self.temporal_encoder(past_traj, padding_mask=padding_mask)  # [B, A, D]

        pos_enc = self.pos_encoding(torch.arange(A, device=past_traj.device))
        a_max = self.agent_query_embedding.num_embeddings
        agent_q = self.agent_query_embedding(
            torch.arange(min(A, a_max), device=past_traj.device)
        )
        if A > a_max:
            agent_q = torch.cat([
                agent_q,
                agent_q[-1:].expand(A - a_max, -1),
            ], dim=0)

        pe = self.mlp_pe(torch.cat([agent_q, pos_enc], dim=-1))  # [A, D]
        agent_feat = agent_feat + pe.unsqueeze(0)

        src_key_padding_mask = None
        if padding_mask is not None:
            src_key_padding_mask = ~padding_mask  # True = padded (PyTorch convention)

        encoder_out = forward_encoder_with_mask(
            self.transformer_encoder, agent_feat,
            mask=interaction_bias,
            src_key_padding_mask=src_key_padding_mask,
        )

        return encoder_out
