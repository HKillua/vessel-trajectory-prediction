import math
import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(self, channel_in=15, channel_conv=32, d_model=256):
        super().__init__()
        self.spatial_conv = nn.Conv1d(channel_in, channel_conv, 3, stride=1, padding=1)
        self.temporal_encoder = nn.GRU(channel_conv, d_model, 1, batch_first=True)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.spatial_conv.weight)
        nn.init.zeros_(self.spatial_conv.bias)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
        nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
        nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

    def forward(self, x):
        """
        x: [B, T, C]
        return: [B, d_model]
        """
        x_t = x.transpose(1, 2)
        x_conv = self.relu(self.spatial_conv(x_t))
        x_embed = x_conv.transpose(1, 2)
        _, state = self.temporal_encoder(x_embed)
        return state.squeeze(0)


class MambaBlock(nn.Module):
    """Pure-PyTorch Mamba (S6) block — no CUDA kernel required.

    Implements selective state space with input-dependent B, C, Δ.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True
        )

        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.dt_rank, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(self.d_inner, -1))
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self._init_dt_proj()

    def _init_dt_proj(self, dt_min=0.001, dt_max=0.1):
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def _ssm_scan(self, x, dt, B_inp, C_inp):
        """Selective scan: sequential due to input-dependent transitions."""
        batch, seq_len, d_inner = x.shape
        d_state = B_inp.shape[-1]

        A = -torch.exp(self.A_log)
        dt_soft = nn.functional.softplus(dt)

        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(seq_len):
            dA = torch.exp(dt_soft[:, t, :].unsqueeze(-1) * A)
            dB = dt_soft[:, t, :].unsqueeze(-1) * B_inp[:, t, :].unsqueeze(1)
            h = h * dA + x[:, t, :].unsqueeze(-1) * dB
            y = (h * C_inp[:, t, :].unsqueeze(1)).sum(dim=-1)
            ys.append(y)

        return torch.stack(ys, dim=1)

    def forward(self, x):
        """x: [B, T, d_model] → [B, T, d_model]"""
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :x_inner.shape[1]]
        x_conv = x_conv.transpose(1, 2)
        x_conv = nn.functional.silu(x_conv)

        x_proj = self.x_proj(x_conv)
        dt = x_proj[..., :self.dt_rank]
        B_inp = x_proj[..., self.dt_rank:self.dt_rank + self.d_state]
        C_inp = x_proj[..., self.dt_rank + self.d_state:]

        dt = self.dt_proj(dt)

        y = self._ssm_scan(x_conv, dt, B_inp, C_inp)
        y = (y + x_conv * self.D) * nn.functional.silu(z)

        return self.out_proj(y) + residual


class MambaEncoder(nn.Module):
    """Mamba-based temporal encoder for ship trajectory sequences.

    Uses mamba-ssm library when available (CUDA), falls back to
    pure-PyTorch implementation for CPU/MPS.
    """

    def __init__(self, channel_in=21, d_model=256, n_layers=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.proj_in = nn.Linear(channel_in, d_model)

        try:
            from mamba_ssm import Mamba
            self.layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(n_layers)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            self._use_cuda_mamba = True
        except ImportError:
            self.layers = nn.ModuleList([
                MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(n_layers)
            ])
            self._use_cuda_mamba = False

    def forward(self, x):
        """
        x: [B, T, C]
        return: [B, d_model]
        """
        h = self.proj_in(x)
        if self._use_cuda_mamba:
            for norm, layer in zip(self.norms, self.layers):
                h = h + layer(norm(h))
        else:
            for layer in self.layers:
                h = layer(h)
        return h[:, -1, :]


class DeltaNetEncoder(nn.Module):
    def __init__(self, channel_in=15, d_model=256, n_layers=2, n_heads=4):
        super().__init__()
        self.proj_in = nn.Linear(channel_in, d_model)
        self.d_model = d_model

        try:
            from fla.models.delta_net import DeltaNetModel
            from fla.models.delta_net import DeltaNetConfig
            config = DeltaNetConfig(
                hidden_size=d_model,
                num_hidden_layers=n_layers,
                num_heads=n_heads,
                vocab_size=2,
                use_cache=False,
                fuse_cross_entropy=False,
            )
            self.backbone = DeltaNetModel(config)
            self._use_fla = True
        except ImportError:
            self.backbone = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                          dim_feedforward=d_model * 2, batch_first=True),
                num_layers=n_layers
            )
            self._use_fla = False
            pe = torch.zeros(5000, d_model)
            position = torch.arange(0, 5000, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('_pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        x: [B, T, C]
        return: [B, d_model]
        """
        h = self.proj_in(x)
        if self._use_fla:
            out = self.backbone(inputs_embeds=h).last_hidden_state
        else:
            h = h + self._pe[:, :h.size(1), :]
            out = self.backbone(h)
        return out[:, -1, :]


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for ship trajectory sequences.
    
    
    """

    def __init__(self, channel_in=21, d_model=256, n_layers=2, n_heads=8,
                dim_feedforward=336, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Linear(channel_in, d_model)

        pe = torch.zeros(5000, d_model)
        position = torch.arange(0, 5000, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('_pe', pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True, norm_first=True,
        )

        self.backbone = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )

    def forward(self, x):
        """
        x: [B, T, C]
        return: [B, d_model]
        """
        h = self.proj_in(x)
        h = h + self._pe[:, :h.size(1), :]
        h = self.backbone(h)
        return h[:, -1, :]

class TemporalEncoder(nn.Module):
    def __init__(self, encoder_type='mamba', n_feat=7, d_model=256):
        super().__init__()
        channel_in = n_feat * 3
        if encoder_type == 'mamba':
            self.encoder = MambaEncoder(channel_in, d_model)
        elif encoder_type == 'deltanet':
            self.encoder = DeltaNetEncoder(channel_in, d_model)
        elif encoder_type == 'transformer':
            self.encoder = TransformerEncoder(channel_in, d_model)
        else:
            self.encoder = GRUEncoder(channel_in, 32, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.encoder(x)