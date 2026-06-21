"""
统一 Baseline 模型定义
所有模型共享相同接口: forward(obs_seq) -> pred_seq
  obs_seq:  [B, T_obs, D]  目标船的观测序列 (lat,lon,sog,sin_cog,cos_cog,sin_hdg,cos_hdg)
  pred_seq: [B, T_pred, 2] 预测的 (lat, lon) 序列
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== LSTM / GRU / Bi-LSTM / Bi-GRU =====================
class RNNBaseline(nn.Module):
    """通用 RNN baseline: 支持 LSTM/GRU + 单向/双向
    
    Fixed: 使用 Linear(H, T_pred*2) 替代 repeat+Linear(H,2)
    每个预测步有独立的输出，而非共享同一个 hidden state
    """

    def __init__(self, rnn_type='lstm', input_size=7, hidden_size=128,
                 num_layers=2, dropout=0.1, pred_dim=2, bidirectional=False,
                 pred_steps=30):
        super().__init__()
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()
        self.pred_dim = pred_dim
        self._t_pred = pred_steps
        rnn_cls = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        enc_dim = hidden_size * (2 if bidirectional else 1)
        # Fixed: 输出 T_pred * pred_dim，每步独立预测
        self.decoder = nn.Sequential(
            nn.Linear(enc_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, pred_steps * pred_dim),
        )

    def forward(self, obs_seq):
        """obs_seq: [B, T_obs, D] -> [B, T_pred, 2]"""
        rnn_out, _ = self.rnn(obs_seq)          # [B, T_obs, H*(1+bi)]
        last_h = rnn_out[:, -1, :]              # [B, H*(1+bi)]
        pred = self.decoder(last_h)             # [B, T_pred * pred_dim]
        return pred.view(-1, self._t_pred, self.pred_dim)  # [B, T_pred, 2]

    def set_pred_steps(self, t_pred):
        self._t_pred = t_pred


class RNNSeq2Seq(nn.Module):
    """RNN Encoder-Decoder (autoregressive decoder)
    
    Encoder: bidirectional RNN
    Decoder: unidirectional RNN (autoregressive)
    """

    def __init__(self, rnn_type='lstm', input_size=7, hidden_size=128,
                 num_layers=2, dropout=0.1, pred_dim=2, bidirectional=False):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.enc_num_dir = 2 if bidirectional else 1
        self.pred_dim = pred_dim

        self.encoder = rnn_cls(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.decoder = rnn_cls(
            input_size=pred_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Bridge: convert encoder final state to decoder initial state
        bridge_in = hidden_size * self.enc_num_dir
        self.bridge_h = nn.Linear(bridge_in, hidden_size)
        if self.rnn_type == 'lstm':
            self.bridge_c = nn.Linear(bridge_in, hidden_size)
        self.out_proj = nn.Linear(hidden_size, pred_dim)

    def forward(self, obs_seq, pred_steps=30):
        """obs_seq: [B, T_obs, D] -> [B, T_pred, 2]"""
        B = obs_seq.shape[0]
        _, enc_state = self.encoder(obs_seq)

        # Convert encoder state to decoder initial state
        if self.rnn_type == 'lstm':
            h, c = enc_state
            # h: [num_layers * num_dir, B, H]
            # Reshape to [num_layers, num_dir, B, H], combine directions
            h_reshaped = h.view(self.num_layers, self.enc_num_dir, B, self.hidden_size)
            h_combined = h_reshaped.sum(dim=1)  # [num_layers, B, H*dir] before sum, [num_layers, B, H] after
            # Actually we need to concat or project
            h_combined = h_reshaped[:, 0, :, :]  # take forward direction
            if self.enc_num_dir == 2:
                h_combined = self.bridge_h(torch.cat([h_reshaped[:, 0], h_reshaped[:, 1]], dim=-1))
            c_reshaped = c.view(self.num_layers, self.enc_num_dir, B, self.hidden_size)
            if self.enc_num_dir == 2:
                c_combined = self.bridge_c(torch.cat([c_reshaped[:, 0], c_reshaped[:, 1]], dim=-1))
            else:
                c_combined = c_reshaped[:, 0, :, :]
            dec_state = (h_combined.contiguous(), c_combined.contiguous())
        else:
            h = enc_state
            h_reshaped = h.view(self.num_layers, self.enc_num_dir, B, self.hidden_size)
            if self.enc_num_dir == 2:
                h_combined = self.bridge_h(torch.cat([h_reshaped[:, 0], h_reshaped[:, 1]], dim=-1))
            else:
                h_combined = h_reshaped[:, 0, :, :]
            dec_state = h_combined.contiguous()

        # Autoregressive decoding
        dec_input = torch.zeros(B, 1, self.pred_dim, device=obs_seq.device)
        preds = []
        for _ in range(pred_steps):
            dec_out, dec_state = self.decoder(dec_input, dec_state)
            pred_step = self.out_proj(dec_out)   # [B, 1, 2]
            preds.append(pred_step)
            dec_input = pred_step
        return torch.cat(preds, dim=1)           # [B, T_pred, 2]


# ===================== Transformer =====================
class TransformerBaseline(nn.Module):
    """Transformer Encoder-Decoder for trajectory prediction.

    Decoder queries are seeded from the encoder's last output (not dead PE),
    with a causal mask so each prediction step only attends to previous steps.
    """

    def __init__(self, input_size=7, d_model=128, n_heads=4, n_layers=2,
                 dim_ff=256, dropout=0.1, pred_dim=2, max_len=100,
                 pred_steps=30):
        super().__init__()
        self.d_model = d_model
        self._pred_steps = pred_steps
        self.input_proj = nn.Linear(input_size, d_model)
        self.output_proj = nn.Linear(d_model, pred_dim)

        # Positional encoding (shared by encoder and decoder)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

        # Learned query embedding for decoder steps
        self.query_embed = nn.Parameter(torch.randn(1, pred_steps, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                              norm=nn.LayerNorm(d_model))
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers,
                                              norm=nn.LayerNorm(d_model))

    def forward(self, obs_seq, pred_steps=None):
        """obs_seq: [B, T_obs, D] -> [B, T_pred, 2]"""
        if pred_steps is None:
            pred_steps = self._pred_steps
        B = obs_seq.shape[0]
        h = self.input_proj(obs_seq) * math.sqrt(self.d_model)
        h = h + self.pe[:, :h.shape[1], :]
        memory = self.encoder(h)

        # Seed decoder with encoder's last output + learned per-step queries
        enc_last = memory[:, -1:, :].expand(-1, pred_steps, -1)
        tgt = enc_last + self.query_embed[:, :pred_steps, :] + self.pe[:, :pred_steps, :]

        # Causal mask: step t can only attend to steps 0..t
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            pred_steps, device=obs_seq.device)
        out = self.decoder(tgt, memory, tgt_mask=causal_mask)
        return self.output_proj(out)


# ===================== Mamba =====================
try:
    from mamba_ssm import Mamba as _OfficialMamba
    _HAS_MAMBA_SSM = True
except ImportError:
    _HAS_MAMBA_SSM = False


class _PurePytorchMambaBlock(nn.Module):
    """Fallback: pure-PyTorch selective SSM (no CUDA kernel)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.dt_rank, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).unsqueeze(0).expand(self.d_inner, -1).float())
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :x_inner.shape[1]]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_proj = self.x_proj(x_conv)
        dt = x_proj[..., :self.dt_rank]
        B_inp = x_proj[..., self.dt_rank:self.dt_rank + self.d_state]
        C_inp = x_proj[..., self.dt_rank + self.d_state:]
        dt = self.dt_proj(dt)
        dt_soft = F.softplus(dt)
        A = -torch.exp(self.A_log)

        batch, seq_len, d_inner = x_conv.shape
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            dA = torch.exp(dt_soft[:, t, :].unsqueeze(-1) * A)
            dB = dt_soft[:, t, :].unsqueeze(-1) * B_inp[:, t, :].unsqueeze(1)
            h = h * dA + x_conv[:, t, :].unsqueeze(-1) * dB
            y = (h * C_inp[:, t, :].unsqueeze(1)).sum(dim=-1)
            ys.append(y)
        y_seq = torch.stack(ys, dim=1)
        y_seq = (y_seq + x_conv * self.D) * F.silu(z)
        return self.out_proj(y_seq)


class MambaBaseline(nn.Module):
    """Mamba encoder + MLP decoder.

    Uses official mamba-ssm CUDA kernel when available (pip install mamba-ssm),
    falls back to pure-PyTorch SSM otherwise.
    """

    def __init__(self, input_size=7, d_model=256, n_layers=4,
                 d_state=16, d_conv=4, expand=2,
                 pred_dim=2, pred_steps=30):
        super().__init__()
        self.pred_dim = pred_dim
        self._t_pred = pred_steps
        self.proj_in = nn.Linear(input_size, d_model)

        if _HAS_MAMBA_SSM:
            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    'norm': nn.LayerNorm(d_model),
                    'mamba': _OfficialMamba(d_model=d_model, d_state=d_state,
                                           d_conv=d_conv, expand=expand),
                })
                for _ in range(n_layers)
            ])
            self._use_official = True
        else:
            import warnings
            warnings.warn(
                "mamba-ssm not found, using pure-PyTorch fallback. "
                "For best results: pip install mamba-ssm causal-conv1d>=1.2.0"
            )
            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    'norm': nn.LayerNorm(d_model),
                    'mamba': _PurePytorchMambaBlock(d_model, d_state, d_conv, expand),
                })
                for _ in range(n_layers)
            ])
            self._use_official = False

        self.norm_f = nn.LayerNorm(d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_steps * pred_dim),
        )

    def forward(self, obs_seq):
        h = self.proj_in(obs_seq)
        for block in self.blocks:
            h = h + block['mamba'](block['norm'](h))  # pre-norm residual
        h = self.norm_f(h)
        last_h = h[:, -1, :]
        pred = self.decoder(last_h)
        return pred.view(-1, self._t_pred, self.pred_dim)

    def set_pred_steps(self, t_pred):
        self._t_pred = t_pred


# ===================== iTransformer =====================
class iTransformerBaseline(nn.Module):
    """iTransformer: 把每个特征通道当作一个 token（变量维度做 attention）"""

    def __init__(self, input_size=7, d_model=128, n_heads=4, n_layers=2,
                 dim_ff=256, dropout=0.1, pred_dim=2, obs_steps=30, pred_steps=30):
        super().__init__()
        self.d_model = d_model
        self.n_variates = input_size
        self.pred_dim = pred_dim
        self._obs_steps = obs_steps
        self._t_pred = pred_steps

        # Each variate is embedded independently
        self.variate_embed = nn.ModuleList([
            nn.Sequential(nn.Linear(obs_steps, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            for _ in range(input_size)
        ])
        # Learnable variate tokens
        self.variate_tokens = nn.Parameter(torch.randn(1, input_size, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                              norm=nn.LayerNorm(d_model))
        # Decoder: from variate tokens to predictions
        self.decoder = nn.Linear(d_model * input_size, pred_dim * pred_steps)

    def forward(self, obs_seq, pred_steps=None):
        """obs_seq: [B, T_obs, D=7] -> [B, T_pred, 2]"""
        if pred_steps is None:
            pred_steps = self._t_pred
        B, T, D = obs_seq.shape
        # obs_seq: [B, T, D] -> transpose to [B, D, T]
        obs_t = obs_seq.transpose(1, 2)  # [B, D, T]

        # Embed each variate
        var_embs = []
        for d in range(D):
            var_embs.append(self.variate_embed[d](obs_t[:, d, :]))  # [B, d_model]
        var_embs = torch.stack(var_embs, dim=1)  # [B, D, d_model]

        # Add learnable variate tokens
        tokens = var_embs + self.variate_tokens.expand(B, -1, -1)

        # Transformer over variate dimension
        encoded = self.encoder(tokens)  # [B, D, d_model]

        # Flatten and predict
        flat = encoded.reshape(B, -1)   # [B, D*d_model]
        pred = self.decoder(flat)       # [B, pred_dim * T_pred]
        return pred.reshape(B, pred_steps, self.pred_dim)


# ===================== Social-LSTM =====================
class SocialLSTM(nn.Module):
    """Social-LSTM: LSTMCell + per-step hidden-state social pooling

    Faithful to Alahi et al. CVPR 2016:
    - Maintains hidden states for ALL ships (target + neighbors)
    - At each observation step: compute social tensor from neighbor hidden states
    - Social tensor = spatial grid pooling of neighbor hidden states
    - Input = embed(features) + embed(social_tensor) -> LSTMCell
    """

    def __init__(self, input_size=7, hidden_size=128, num_layers=1,
                 dropout=0.1, pred_dim=2, grid_size=4, grid_radius=2.0,
                 embedding_size=64, pred_steps=30):
        super().__init__()
        self.hidden_size = hidden_size
        self.grid_size = grid_size
        self.grid_radius = grid_radius
        self.pred_dim = pred_dim
        self._t_pred = pred_steps
        self.embedding_size = embedding_size

        self.input_embed = nn.Linear(input_size, embedding_size)
        social_pool_dim = grid_size * grid_size * hidden_size
        self.social_embed = nn.Linear(social_pool_dim, embedding_size)
        self.cell = nn.LSTMCell(2 * embedding_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, pred_steps * pred_dim),
        )

    def _compute_social_tensor(self, positions, hidden_states, full_mask):
        """Compute social tensor using neighbor hidden states in spatial grid.

        positions:    [B, N_total, 2]  all ships' positions at current step
        hidden_states:[B, N_total, H]  all ships' hidden states
        full_mask:    [B, N_total]     valid ship mask (1=valid)

        Returns: [B, G*G*H] social tensor for the target ship (index 0)
        """
        B, N, H = hidden_states.shape
        G = self.grid_size
        device = positions.device

        if N <= 1:
            return torch.zeros(B, G * G * H, device=device)

        target_pos = positions[:, 0:1, :]
        rel_pos = positions[:, 1:, :] - target_pos
        neighbor_h = hidden_states[:, 1:, :] * full_mask[:, 1:].unsqueeze(-1)

        cell_size = self.grid_radius / G
        gx = ((rel_pos[..., 0] / cell_size) + G / 2).long().clamp(0, G - 1)
        gy = ((rel_pos[..., 1] / cell_size) + G / 2).long().clamp(0, G - 1)
        cell_idx = gx * G + gy

        grid = torch.zeros(B, G * G, H, device=device)
        ci_exp = cell_idx.unsqueeze(-1).expand_as(neighbor_h)
        grid.scatter_add_(1, ci_exp, neighbor_h)

        return grid.reshape(B, G * G * H)

    def forward(self, obs_seq, neighbor_obs=None, mask=None):
        """
        obs_seq:      [B, T_obs, D=7]        target ship
        neighbor_obs: [B, N_max, T_obs, D=7]  neighbor ships (padded)
        mask:         [B, N_max]              valid neighbor mask
        Returns: [B, T_pred, 2]
        """
        B, T, D = obs_seq.shape
        device = obs_seq.device
        has_neighbors = (neighbor_obs is not None and neighbor_obs.shape[1] > 0)

        if has_neighbors:
            N_nb = neighbor_obs.shape[1]
            N_total = 1 + N_nb
            all_obs = torch.cat([obs_seq.unsqueeze(1), neighbor_obs], dim=1)
            if mask is None:
                full_mask = torch.ones(B, N_total, device=device)
            else:
                full_mask = torch.cat([torch.ones(B, 1, device=device), mask], dim=1)
        else:
            N_total = 1
            all_obs = obs_seq.unsqueeze(1)
            full_mask = torch.ones(B, 1, device=device)

        h = torch.zeros(B, N_total, self.hidden_size, device=device)
        c = torch.zeros(B, N_total, self.hidden_size, device=device)

        for t in range(T):
            current_pos = all_obs[:, :, t, :2]
            current_feat = all_obs[:, :, t, :]

            input_emb = self.dropout(self.relu(
                self.input_embed(current_feat.reshape(B * N_total, D))
            )).reshape(B, N_total, self.embedding_size)

            social_tensor = self._compute_social_tensor(current_pos, h, full_mask)
            social_emb = self.dropout(self.relu(self.social_embed(social_tensor)))

            target_input = torch.cat([input_emb[:, 0, :], social_emb], dim=-1)
            h_target, c_target = self.cell(target_input, (h[:, 0, :], c[:, 0, :]))
            h = h.clone()
            c = c.clone()
            h[:, 0, :] = h_target
            c[:, 0, :] = c_target

            if N_total > 1:
                N_nb = N_total - 1
                zero_social = torch.zeros(B * N_nb, self.embedding_size, device=device)
                nb_input = torch.cat([
                    input_emb[:, 1:, :].reshape(B * N_nb, self.embedding_size),
                    zero_social
                ], dim=-1)
                h_nb, c_nb = self.cell(
                    nb_input,
                    (h[:, 1:, :].reshape(B * N_nb, self.hidden_size),
                     c[:, 1:, :].reshape(B * N_nb, self.hidden_size))
                )
                mask_nb = full_mask[:, 1:].unsqueeze(-1)
                h = h.clone()
                c = c.clone()
                h[:, 1:, :] = h_nb.reshape(B, N_nb, self.hidden_size) * mask_nb
                c[:, 1:, :] = c_nb.reshape(B, N_nb, self.hidden_size) * mask_nb

        pred = self.decoder(h[:, 0, :])
        return pred.view(-1, self._t_pred, self.pred_dim)

    def set_pred_steps(self, t_pred):
        self._t_pred = t_pred


# ===================== Model Registry =====================
# Registry uses factory functions that accept pred_steps and obs_steps
MODEL_REGISTRY = {
    'lstm':         lambda pred_steps, obs_steps: RNNBaseline(rnn_type='lstm', bidirectional=False, pred_steps=pred_steps),
    'gru':          lambda pred_steps, obs_steps: RNNBaseline(rnn_type='gru', bidirectional=False, pred_steps=pred_steps),
    'bilstm':       lambda pred_steps, obs_steps: RNNBaseline(rnn_type='lstm', bidirectional=True, pred_steps=pred_steps),
    'bigru':        lambda pred_steps, obs_steps: RNNBaseline(rnn_type='gru', bidirectional=True, pred_steps=pred_steps),
    'seq2seq_lstm': lambda pred_steps, obs_steps: RNNSeq2Seq(rnn_type='lstm', bidirectional=False),
    'seq2seq_gru':  lambda pred_steps, obs_steps: RNNSeq2Seq(rnn_type='gru', bidirectional=False),
    'transformer':  lambda pred_steps, obs_steps: TransformerBaseline(
        d_model=256, n_heads=8, n_layers=3, dim_ff=512, pred_steps=pred_steps),
    'mamba':        lambda pred_steps, obs_steps: MambaBaseline(
        d_model=256, n_layers=4, pred_steps=pred_steps),
    'itransformer': lambda pred_steps, obs_steps: iTransformerBaseline(
        d_model=256, n_heads=8, n_layers=3, dim_ff=512,
        obs_steps=obs_steps, pred_steps=pred_steps),
    'social_lstm':  lambda pred_steps, obs_steps: SocialLSTM(pred_steps=pred_steps),
}


def build_model(model_name, pred_steps=30, obs_steps=30):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    model = MODEL_REGISTRY[model_name](pred_steps, obs_steps)
    return model