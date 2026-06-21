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
    """标准 Transformer Encoder-Decoder for trajectory prediction"""

    def __init__(self, input_size=7, d_model=128, n_heads=4, n_layers=2,
                 dim_ff=256, dropout=0.1, pred_dim=2, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_size, d_model)
        self.output_proj = nn.Linear(d_model, pred_dim)

        # Positional encoding
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

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

    def forward(self, obs_seq, pred_steps=30):
        """obs_seq: [B, T_obs, D] -> [B, T_pred, 2]"""
        B = obs_seq.shape[0]
        h = self.input_proj(obs_seq) * math.sqrt(self.d_model)
        h = h + self.pe[:, :h.shape[1], :]
        memory = self.encoder(h)

        # Decoder queries: learned start tokens repeated
        tgt = self.pe[:, :pred_steps, :].expand(B, -1, -1)
        out = self.decoder(tgt, memory)
        return self.output_proj(out)


# ===================== Mamba (Pure-PyTorch) =====================
class MambaBlock(nn.Module):
    """纯 PyTorch Mamba block (从 vessel-trajectory-prediction 复用)"""

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
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :x_inner.shape[1]]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_proj = self.x_proj(x_conv)
        dt = self.x_proj(x_conv)[..., :self.dt_rank]
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
        return self.out_proj(y_seq) + residual


class MambaBaseline(nn.Module):
    """Mamba encoder + linear decoder
    
    Fixed: 使用 Linear(H, T_pred*2) 替代 repeat+Linear(H,2)
    """

    def __init__(self, input_size=7, d_model=128, n_layers=2, pred_dim=2, pred_steps=30):
        super().__init__()
        self.pred_dim = pred_dim
        self._t_pred = pred_steps
        self.proj_in = nn.Linear(input_size, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layers)])
        # Fixed: 输出 T_pred * pred_dim
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, pred_steps * pred_dim),
        )

    def forward(self, obs_seq):
        h = self.proj_in(obs_seq)
        for block in self.blocks:
            h = block(h)
        last_h = h[:, -1, :]
        pred = self.decoder(last_h)  # [B, T_pred * pred_dim]
        return pred.view(-1, self._t_pred, self.pred_dim)  # [B, T_pred, 2]

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
    """Social-LSTM: LSTM + social pooling for multi-vessel interaction
    
    参考: Alahi et al. "Social LSTM" (CVPR 2016)
    
    Fixed: 使用 Linear(H, T_pred*2) 替代 repeat+Linear(H,2)
    """

    def __init__(self, input_size=7, hidden_size=128, num_layers=2,
                 dropout=0.1, pred_dim=2, grid_size=4, grid_radius=5.0, pred_steps=30):
        super().__init__()
        self.grid_size = grid_size
        self.grid_radius = grid_radius  # in nautical miles
        self.hidden_size = hidden_size
        self.pred_dim = pred_dim
        self._t_pred = pred_steps
        social_pool_dim = hidden_size * grid_size * grid_size

        # Embedding for social pooling features
        self.social_embed = nn.Linear(input_size, hidden_size)
        self.pool_proj = nn.Linear(social_pool_dim, hidden_size)

        # Main LSTM: input = obs_features + pooled_social
        self.rnn = nn.LSTM(input_size + hidden_size, hidden_size,
                           num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        # Fixed: 输出 T_pred * pred_dim
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, pred_steps * pred_dim),
        )

    def _build_social_pool(self, obs_seq, neighbor_obs):
        """构建 social pooling grid
        
        obs_seq:      [B, T, D]  目标船观测
        neighbor_obs: [B, N, T, D] 邻居船观测
        """
        B, T, D = obs_seq.shape
        N = neighbor_obs.shape[1]
        
        # 用最后时刻的位置计算相对关系
        target_pos = obs_seq[:, -1, :2]             # [B, 2] lat,lon
        neighbor_pos = neighbor_obs[:, :, -1, :2]    # [B, N, 2]
        rel_pos = neighbor_pos - target_pos.unsqueeze(1)  # [B, N, 2]

        # Embed neighbor features at last timestep
        neighbor_feats = self.social_embed(neighbor_obs[:, :, -1, :])  # [B, N, H]

        # Build grid
        grid = torch.zeros(B, self.grid_size, self.grid_size, self.hidden_size,
                          device=obs_seq.device)
        cell_size = self.grid_radius / self.grid_size
        for n in range(N):
            gx = ((rel_pos[:, n, 0] / cell_size) + self.grid_size / 2).long().clamp(0, self.grid_size - 1)
            gy = ((rel_pos[:, n, 1] / cell_size) + self.grid_size / 2).long().clamp(0, self.grid_size - 1)
            for b in range(B):
                grid[b, gx[b], gy[b]] += neighbor_feats[b, n]

        grid_flat = grid.reshape(B, -1)  # [B, grid*grid*H]
        return self.pool_proj(grid_flat)  # [B, H]

    def forward(self, obs_seq, neighbor_obs=None):
        """
        obs_seq:      [B, T_obs, D=7]
        neighbor_obs: [B, N, T_obs, D=7] or None
        """
        B, T, D = obs_seq.shape

        if neighbor_obs is not None and neighbor_obs.shape[1] > 0:
            social = self._build_social_pool(obs_seq, neighbor_obs)  # [B, H]
            social_expanded = social.unsqueeze(1).expand(-1, T, -1)  # [B, T, H]
            rnn_in = torch.cat([obs_seq, social_expanded], dim=-1)   # [B, T, D+H]
        else:
            # No neighbors: zero social features
            zeros = torch.zeros(B, T, self.hidden_size, device=obs_seq.device)
            rnn_in = torch.cat([obs_seq, zeros], dim=-1)

        rnn_out, _ = self.rnn(rnn_in)
        last_h = rnn_out[:, -1, :]
        pred = self.decoder(last_h)  # [B, T_pred * pred_dim]
        return pred.view(-1, self._t_pred, self.pred_dim)  # [B, T_pred, 2]

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
    'transformer':  lambda pred_steps, obs_steps: TransformerBaseline(),
    'mamba':        lambda pred_steps, obs_steps: MambaBaseline(pred_steps=pred_steps),
    'itransformer': lambda pred_steps, obs_steps: iTransformerBaseline(obs_steps=obs_steps, pred_steps=pred_steps),
    'social_lstm':  lambda pred_steps, obs_steps: SocialLSTM(pred_steps=pred_steps),
}


def build_model(model_name, pred_steps=30, obs_steps=30):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    model = MODEL_REGISTRY[model_name](pred_steps, obs_steps)
    return model
