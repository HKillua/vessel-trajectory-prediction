"""ShipTransformerEncoder: Encoder-Decoder Transformer for ship trajectory conditioning.

Follows MGF's TF_encoder design but scaled for 7D ship features.
Decoder uses sinusoidal PE as queries, cross-attends to encoder output,
producing per-timestep conditioning for the flow model.
"""

import math

import torch
import torch.nn as nn


class ShipTransformerEncoder(nn.Module):
    """Transformer Encoder-Decoder that maps ship observations to per-timestep conditioning.

    Input:  data_dict['obs_st'] — (B, obs_len, 7)
    Output: (B, pred_len, cond_dim)
    """

    def __init__(
        self,
        input_dim=7,
        d_model=64,
        cond_dim=16,
        n_heads=4,
        n_enc_layers=3,
        n_dec_layers=3,
        obs_len=10,
        pred_len=30,
        dropout=0.1,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len

        self.input_proj = nn.Linear(input_dim, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.output_proj = nn.Linear(d_model, cond_dim)

        # Sinusoidal PE covering encoder + decoder positions
        pe = self._build_sinusoidal_pe(obs_len + pred_len, d_model)
        self.register_buffer("pe", pe)

    @staticmethod
    def _build_sinusoidal_pe(max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, data_dict):
        obs = data_dict["obs_st"]  # (B, obs_len, 7)

        enc_pe = self.pe[:, : self.obs_len]
        dec_pe = self.pe[:, self.obs_len : self.obs_len + self.pred_len]

        enc_out = self.transformer.encoder(self.input_proj(obs) + enc_pe)
        dec_out = self.transformer.decoder(
            dec_pe.expand(enc_out.shape[0], -1, -1), enc_out
        )

        return self.output_proj(dec_out)  # (B, pred_len, cond_dim)
