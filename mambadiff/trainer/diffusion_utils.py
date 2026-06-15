import torch
import torch.nn as nn
import numpy as np


def make_beta_schedule(schedule='linear', n_timesteps=100, start=1e-4, end=5e-2):
    if schedule == 'linear':
        return torch.linspace(start, end, n_timesteps)
    elif schedule == 'quad':
        return torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == 'sigmoid':
        betas = torch.linspace(-6, 6, n_timesteps)
        return torch.sigmoid(betas) * (end - start) + start
    raise ValueError(f"Unknown schedule: {schedule}")


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


class DiffusionSchedule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_steps = cfg['steps']
        self.num_tau = cfg['num_tau']
        self.tau_start = cfg.get('tau_start', self.n_steps // 3)
        assert 0 < self.tau_start < self.n_steps, (
            f"tau_start={self.tau_start} must be in (0, {self.n_steps})"
        )

        betas = make_beta_schedule(
            schedule=cfg['beta_schedule'],
            n_timesteps=self.n_steps,
            start=cfg['beta_start'],
            end=cfg['beta_end']
        )
        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, 0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_prod', alphas_prod)
        self.register_buffer('alphas_bar_sqrt', torch.sqrt(alphas_prod))
        self.register_buffer('one_minus_alphas_bar_sqrt', torch.sqrt(1 - alphas_prod))

        self.tau_steps = self._make_tau_steps()

    def _make_tau_steps(self):
        """Sparse step schedule for true truncated diffusion.

        Spreads num_tau steps across [0, tau_start] so denoising starts
        from moderate noise (preserving anchor signal) rather than
        near-maximum noise.  With tau_start=33 and linear beta 1e-4→5e-2,
        alpha_bar[33] ≈ 0.75 → ~87% of anchor signal preserved (sqrt).
        """
        indices = np.linspace(0, self.tau_start, self.num_tau, dtype=int)
        return list(reversed(indices.tolist()))