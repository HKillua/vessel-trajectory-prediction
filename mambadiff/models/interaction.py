import torch
import torch.nn as nn


class EncounterEdgeEncoder(nn.Module):
    def __init__(self, n_encounter_types=6, encounter_embed_dim=8, edge_dim=16,
                 dcpa_scale=3.0, tcpa_scale=1800.0):
        super().__init__()
        self.dcpa_scale = dcpa_scale
        self.tcpa_scale = tcpa_scale
        self.enc_embed = nn.Embedding(n_encounter_types, encounter_embed_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 + encounter_embed_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
        )

    def forward(self, dcpa, tcpa, cri, encounter_type):
        """
        dcpa, tcpa, cri: [B, N, N]
        encounter_type: [B, N, N] (long)
        return: [B, N, N, edge_dim]
        """
        enc_emb = self.enc_embed(encounter_type)
        dcpa_norm = torch.clamp(dcpa / self.dcpa_scale, 0, 1)
        tcpa_norm = torch.clamp(tcpa / self.tcpa_scale, -1, 1)
        continuous = torch.stack([dcpa_norm, tcpa_norm, cri], dim=-1)
        edge_input = torch.cat([continuous, enc_emb], dim=-1)
        return self.edge_mlp(edge_input)


class GATLayer(nn.Module):
    """Single GAT layer with edge features and residual connection.

    When use_sda=True, uses Signed Dual Attention: both positive (standard)
    and negative (negated logits) softmax, blended by a per-head learnable gate.
    """

    def __init__(self, d_model, n_head, edge_dim, dropout=0.1, use_sda=False):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.use_sda = use_sda

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_edge = nn.Linear(edge_dim, n_head)
        self.w_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        if self.use_sda:
            # Initialize gate to ~0.88 (sigmoid(2.0)) so initial behavior
            # approximates standard softmax attention. Training can then
            # learn to activate negative attention heads where needed.
            self.sda_gate = nn.Parameter(torch.full((n_head,), 2.0))

    def forward(self, h, edge_feat, mask):
        B, N, D = h.shape

        Q = self.w_q(h).view(B, N, self.n_head, self.d_head).transpose(1, 2)
        K = self.w_k(h).view(B, N, self.n_head, self.d_head).transpose(1, 2)
        V = self.w_v(h).view(B, N, self.n_head, self.d_head).transpose(1, 2)

        edge_bias = self.w_edge(edge_feat).permute(0, 3, 1, 2)
        raw_logits = (Q @ K.transpose(-1, -2)) / (self.d_head ** 0.5) + edge_bias

        if mask is not None:
            pad_mask = ~mask.unsqueeze(1).unsqueeze(2).expand(-1, self.n_head, N, -1)
        else:
            pad_mask = None

        if self.use_sda:
            pos_logits = raw_logits.masked_fill(pad_mask, float('-inf')) if pad_mask is not None else raw_logits
            neg_logits = (-raw_logits).masked_fill(pad_mask, float('-inf')) if pad_mask is not None else -raw_logits
            A_pos = torch.softmax(pos_logits, dim=-1)
            A_neg = torch.softmax(neg_logits, dim=-1)
            alpha = torch.sigmoid(self.sda_gate).view(1, self.n_head, 1, 1)
            attn = alpha * A_pos - (1 - alpha) * A_neg
        else:
            logits = raw_logits.masked_fill(pad_mask, float('-inf')) if pad_mask is not None else raw_logits
            attn = torch.softmax(logits, dim=-1)

        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        out = self.w_out(out)
        h = self.layer_norm(h + out)
        h = self.norm2(h + self.ff(h))
        return h


class EncounterAwareGAT(nn.Module):
    """Multi-layer GAT with encounter-aware edge features.

    Multiple layers allow multi-hop interaction reasoning: with 2 layers,
    ship A can see ship B's representation that already incorporates B's
    interaction with C, enabling chain-reaction awareness.
    """

    def __init__(self, d_model=256, n_head=4, edge_dim=16, dropout=0.1, n_layers=2,
                 use_sda=False):
        super().__init__()
        self.layers = nn.ModuleList([
            GATLayer(d_model, n_head, edge_dim, dropout, use_sda=use_sda)
            for _ in range(n_layers)
        ])

    def forward(self, h, edge_feat, mask):
        """
        h: [B, N, d_model]
        edge_feat: [B, N, N, edge_dim]
        mask: [B, N] bool
        return: [B, N, d_model]
        """
        for layer in self.layers:
            h = layer(h, edge_feat, mask)
        return h
