import torch
import torch.nn as nn


class FreDFLoss(nn.Module):
    """Frequency Domain loss for trajectory prediction.

    Two modes:
    - Weighted MSE (default, backward-compatible): frequency-dependent weighting
      upweights high-frequency bins. Effective when high-freq energy is moderate.
    - Log-magnitude MSE (log_magnitude=True): computes MSE in log-spectral domain.
      Much more effective for preventing over-smoothing because a 50% relative error
      at any frequency produces the same loss magnitude, regardless of absolute energy.
      Inspired by log-STFT magnitude loss in speech synthesis (MelGAN, HiFi-GAN).
    """

    def __init__(self, k_freq=None, low_weight=0.01, high_weight=1.0,
                 log_magnitude=False, log_eps=1e-7):
        super().__init__()
        self.k_freq = k_freq
        self.low_weight = low_weight
        self.high_weight = high_weight
        self.log_magnitude = log_magnitude
        self.log_eps = log_eps

    def forward(self, pred, target):
        """
        pred: [B, T, 2] or [B, K, T, 2]
        target: [B, T, 2] or [B, K, T, 2]
        return: scalar loss
        """
        pred_fft = torch.fft.rfft(pred, dim=-2)
        target_fft = torch.fft.rfft(target, dim=-2)

        if self.k_freq == 'auto':
            k = pred.shape[-2] // 2
        elif self.k_freq is not None:
            k = self.k_freq
        else:
            k = None

        if k is not None:
            pred_fft = pred_fft[..., :k, :]
            target_fft = target_fft[..., :k, :]

        if self.log_magnitude:
            pred_mag = torch.sqrt(
                pred_fft.real ** 2 + pred_fft.imag ** 2 + self.log_eps
            )
            target_mag = torch.sqrt(
                target_fft.real ** 2 + target_fft.imag ** 2 + self.log_eps
            )
            loss = (torch.log(pred_mag) - torch.log(target_mag)) ** 2

            n_freq = loss.shape[-2]
            freq_weights = torch.linspace(
                self.low_weight, self.high_weight, n_freq,
                device=loss.device, dtype=loss.dtype,
            )
            shape = [1] * (loss.dim() - 2) + [n_freq, 1]
            loss = loss * freq_weights.view(*shape)
        else:
            loss = (pred_fft.real - target_fft.real) ** 2 + \
                   (pred_fft.imag - target_fft.imag) ** 2

            n_freq = loss.shape[-2]
            freq_weights = torch.linspace(
                self.low_weight, self.high_weight, n_freq,
                device=loss.device, dtype=loss.dtype,
            )
            shape = [1] * (loss.dim() - 2) + [n_freq, 1]
            loss = loss * freq_weights.view(*shape)

        return loss.mean()