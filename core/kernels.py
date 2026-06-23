from __future__ import annotations

import math

import torch

from .config import ModelConfig


def fractional_weights(alpha: float, modes: int, log_range: float, device=None, dtype=None):
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if modes < 1:
        raise ValueError("modes must be >= 1")
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    if alpha == 1.0:
        if modes != 1:
            raise ValueError("alpha=1.0 is the ordinary first-order model and requires modes=1")
        return torch.zeros(1, device=device, dtype=dtype), torch.ones(1, device=device, dtype=dtype)
    u = torch.linspace(-log_range, log_range, modes, device=device, dtype=dtype)
    numerator = math.sin(math.pi * alpha)
    denominator = torch.cosh(alpha * u) + math.cos(math.pi * alpha)
    raw = numerator / denominator
    raw = torch.clamp(raw, min=torch.finfo(dtype).eps)
    weights = raw / raw.sum()
    return u, weights


def brightness_dependent_tau(frame_mid: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
    with torch.no_grad():
        base = cfg.tau_ref * torch.pow(
            cfg.i_ref / torch.clamp(frame_mid + cfg.i_dark, min=cfg.eps),
            cfg.beta,
        )
        return torch.clamp(base, min=cfg.tau_min, max=cfg.tau_max)


def build_tau_modes(tau0: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return tau0.unsqueeze(0) * torch.exp(offsets).view(-1, 1, 1)
