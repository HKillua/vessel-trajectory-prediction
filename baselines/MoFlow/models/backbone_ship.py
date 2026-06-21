"""Ship Motion Transformer backbone for MoFlow.

Based on ETHMotionTransformer but adapted for:
- Variable number of agents (2–15) with padding mask
- Interaction-aware cross-agent attention bias
- Parameterized output dimension (MODEL_OUT_DIM)
- Target-ship-only loss extraction
"""

import torch
import torch.nn as nn
from einops import repeat, rearrange

from .context_encoder import build_context_encoder
from .context_encoder.ship_encoder import InteractionBiasModule
from .utils.attn_utils import _forward_layer_no_fastpath
from .motion_decoder import build_decoder
from .utils.common_layers import build_mlps
from .context_encoder.mtr_encoder import SinusoidalPosEmb


class ShipMotionTransformer(nn.Module):
    def __init__(self, model_config, logger, config):
        super().__init__()
        self.model_cfg = model_config
        self.dim = model_config.CONTEXT_ENCODER.D_MODEL
        self.config = config

        use_pre_norm = model_config.get('USE_PRE_NORM', False)

        self.context_encoder = build_context_encoder(
            model_config.CONTEXT_ENCODER, use_pre_norm,
        )

        K = model_config.NUM_PROPOSED_QUERY
        out_dim = model_config.MODEL_OUT_DIM
        max_agents = config.agents

        self.motion_query_embedding = nn.Embedding(K, self.dim)
        self.agent_order_embedding = nn.Embedding(max_agents, self.dim)
        self.post_pe_cat_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )

        sinu_pos_emb = SinusoidalPosEmb(self.dim, theta=10000)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )

        self.noisy_y_mlp = nn.Sequential(
            nn.Linear(out_dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )

        dropout_ = model_config.MOTION_DECODER.get('DROPOUT_OF_ATTN', 0.1)
        n_heads_attn = model_config.MOTION_DECODER.NUM_ATTN_HEAD
        self.noisy_y_attn_k = nn.TransformerEncoderLayer(
            d_model=self.dim, nhead=n_heads_attn,
            dim_feedforward=self.dim * 4, dropout=dropout_, batch_first=True,
        )
        self.noisy_y_attn_a = nn.TransformerEncoderLayer(
            d_model=self.dim, nhead=n_heads_attn,
            dim_feedforward=self.dim * 4, dropout=dropout_, batch_first=True,
        )
        self.n_heads_pre_decoder = n_heads_attn

        dim_decoder = model_config.MOTION_DECODER.D_MODEL
        assert self.dim == dim_decoder, (
            f"CONTEXT_ENCODER.D_MODEL ({self.dim}) must equal "
            f"MOTION_DECODER.D_MODEL ({dim_decoder}) in ShipMotionTransformer"
        )
        self.init_emb_fusion_mlp = nn.Sequential(
            nn.Linear(self.dim + self.dim + self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, dim_decoder),
        )

        self.motion_decoder = build_decoder(
            model_config.MOTION_DECODER, use_pre_norm, use_adaln=True,
        )

        self.reg_head = build_mlps(
            c_in=dim_decoder,
            mlp_channels=list(model_config.REGRESSION_MLPS),
            ret_before_act=True, without_norm=True,
        )
        self.cls_head = build_mlps(
            c_in=dim_decoder,
            mlp_channels=list(model_config.CLASSIFICATION_MLPS),
            ret_before_act=True, without_norm=True,
        )

        interaction_cfg = model_config.get('INTERACTION', {})
        if interaction_cfg.get('ENABLED', False):
            n_heads_inter = interaction_cfg.get('NUM_HEADS', 8)
            assert n_heads_inter == n_heads_attn, (
                f"INTERACTION.NUM_HEADS ({n_heads_inter}) must equal "
                f"MOTION_DECODER.NUM_ATTN_HEAD ({n_heads_attn})"
            )
            self.interaction_bias = InteractionBiasModule(n_heads_inter)
            self.encoder_interaction_bias = InteractionBiasModule(
                model_config.CONTEXT_ENCODER.NUM_ATTN_HEAD,
            )
        else:
            self.interaction_bias = None
            self.encoder_interaction_bias = None

        params_total = sum(p.numel() for p in self.parameters())
        params_enc = sum(p.numel() for p in self.context_encoder.parameters())
        params_dec = sum(p.numel() for p in self.motion_decoder.parameters())
        logger.info(
            "ShipMotionTransformer — Total: {:,}, Encoder: {:,}, "
            "Decoder: {:,}, Other: {:,}".format(
                params_total, params_enc, params_dec,
                params_total - params_enc - params_dec,
            )
        )

    def _build_interaction_bias(self, x_data, bias_module=None):
        """Build attention bias from interaction matrices."""
        if bias_module is None:
            bias_module = self.interaction_bias
        adj = x_data.get('adj_matrix')
        if adj is None or bias_module is None:
            return None
        dcpa = x_data.get('dcpa_matrix', torch.zeros_like(adj))
        tcpa = x_data.get('tcpa_matrix', torch.zeros_like(adj))
        cri = x_data.get('cri_matrix', torch.zeros_like(adj))
        mask = x_data.get('mask')
        return bias_module(adj, dcpa, tcpa, cri, mask)

    def _build_encoder_interaction_bias(self, x_data):
        """Build attention bias for the context encoder."""
        adj = x_data.get('adj_matrix')
        if adj is None or self.encoder_interaction_bias is None:
            return None
        dcpa = x_data.get('dcpa_matrix', torch.zeros_like(adj))
        tcpa = x_data.get('tcpa_matrix', torch.zeros_like(adj))
        cri = x_data.get('cri_matrix', torch.zeros_like(adj))
        mask = x_data.get('mask')
        return self.encoder_interaction_bias(adj, dcpa, tcpa, cri, mask)

    def forward(self, y, time, x_data):
        """
        Args:
            y:      [B, K, A, MODEL_OUT_DIM] noisy future trajectory
            time:   [B] denoising timestep
            x_data: data dict from ship_collate_fn
        """
        device = y.device
        B, K, A, _ = y.shape

        enc_inter_bias = self._build_encoder_interaction_bias(x_data)
        padding_mask = x_data.get('mask')

        encoder_out = self.context_encoder(
            x_data['past_traj_original_scale'],
            interaction_bias=enc_inter_bias,
            padding_mask=padding_mask,
        )  # [B, A, D]

        encoder_out_batch = repeat(encoder_out, 'b a d -> b k a d', k=K)

        y_emb = self.noisy_y_mlp(y)  # [B, K, A, D]

        time_ = time
        if self.config.denoising_method == 'fm':
            time = time * 1000.0

        t_emb = self.time_mlp(time)
        t_emb_batch = repeat(t_emb, 'b d -> b k a d', k=K, a=A)

        k_pe = self.motion_query_embedding(torch.arange(K, device=device))
        k_pe_batch = repeat(k_pe, 'k d -> b k a d', b=B, a=A)

        a_max = self.agent_order_embedding.num_embeddings
        a_indices = torch.arange(min(A, a_max), device=device)
        a_pe = self.agent_order_embedding(a_indices)
        if A > a_max:
            a_pe = torch.cat([a_pe, a_pe[-1:].expand(A - a_max, -1)], dim=0)
        a_pe_batch = repeat(a_pe, 'a d -> b k a d', b=B, k=K)

        y_emb = y_emb + k_pe_batch + a_pe_batch

        # Cross-mode attention (K-to-K)
        y_emb_k = rearrange(y_emb, 'b k a d -> (b a) k d')
        y_emb_k = self.noisy_y_attn_k(y_emb_k)
        y_emb = rearrange(y_emb_k, '(b a) k d -> b k a d', b=B, a=A)

        # Cross-agent attention (A-to-A) with interaction bias
        inter_bias = self._build_interaction_bias(x_data)
        # inter_bias is [B*H, A, A] → reshape to [B, H, A, A] → expand K → [B*K*H, A, A]
        if inter_bias is not None:
            H = self.n_heads_pre_decoder
            inter_bias_k = inter_bias.reshape(B, H, A, A)
            inter_bias_k = inter_bias_k.unsqueeze(1).expand(B, K, H, A, A)
            inter_bias_k = inter_bias_k.reshape(B * K * H, A, A)
        else:
            inter_bias_k = None

        pad_mask_a = None
        if padding_mask is not None:
            pad_mask_a = (~padding_mask).unsqueeze(1).expand(B, K, A)
            pad_mask_a = pad_mask_a.reshape(B * K, A)

        y_emb_a = rearrange(y_emb, 'b k a d -> (b k) a d')
        if inter_bias_k is not None:
            y_emb_a = _forward_layer_no_fastpath(
                self.noisy_y_attn_a, y_emb_a,
                src_mask=inter_bias_k, src_key_padding_mask=pad_mask_a,
            )
        else:
            y_emb_a = self.noisy_y_attn_a(
                y_emb_a, src_key_padding_mask=pad_mask_a,
            )
        y_emb = rearrange(y_emb_a, '(b k) a d -> b k a d', b=B, k=K)

        if self.training and self.config.get('drop_method', None) == 'emb':
            m, k = self.config.drop_logi_m, self.config.drop_logi_k
            p_m = 1 / (1 + torch.exp(-k * (time_ - m)))
            p_m = p_m[:, None, None, None]
            y_emb = y_emb.masked_fill(torch.rand_like(p_m) < p_m, 0.)

        emb_fusion = self.init_emb_fusion_mlp(
            torch.cat((encoder_out_batch, y_emb, t_emb_batch), dim=-1),
        )
        query_token = self.post_pe_cat_mlp(emb_fusion + k_pe_batch + a_pe_batch)

        # Decoder with interaction bias (reuse from cross-agent attention)
        readout_token = self.motion_decoder(
            query_token, t_emb,
            interaction_bias=inter_bias,
            padding_mask=padding_mask,
        )

        denoiser_x = self.reg_head(readout_token)
        denoiser_cls = self.cls_head(readout_token).squeeze(-1)

        return denoiser_x, denoiser_cls
