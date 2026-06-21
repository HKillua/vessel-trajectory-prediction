"""Attention helpers for bypassing PyTorch C++ fast-path NaN issues."""

import torch


def _forward_layer_no_fastpath(layer, src, src_mask=None, src_key_padding_mask=None):
    """Call TransformerEncoderLayer using the Python path, bypassing
    the C++ fast-path which produces NaN with 3D float additive masks
    on PyTorch >=2.12."""
    x = src
    if layer.norm_first:
        x = x + layer._sa_block(layer.norm1(x), src_mask, src_key_padding_mask, is_causal=False)
        x = x + layer._ff_block(layer.norm2(x))
    else:
        x = layer.norm1(x + layer._sa_block(x, src_mask, src_key_padding_mask, is_causal=False))
        x = layer.norm2(x + layer._ff_block(x))
    return x


def forward_encoder_with_mask(encoder, src, mask=None, src_key_padding_mask=None):
    """Forward through TransformerEncoder, bypassing C++ fast-path
    only when a 3D float mask is provided."""
    if mask is not None and mask.dim() == 3 and mask.is_floating_point():
        output = src
        for layer in encoder.layers:
            output = _forward_layer_no_fastpath(layer, output, mask, src_key_padding_mask)
        if encoder.norm is not None:
            output = encoder.norm(output)
        return output
    return encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask)
