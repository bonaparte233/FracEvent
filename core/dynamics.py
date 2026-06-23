from __future__ import annotations

import torch


def evolve_modes_at_delta(
    z_start: torch.Tensor,
    L0: torch.Tensor,
    L1: torch.Tensor,
    delta: torch.Tensor | float,
    dt: float,
    tau_modes: torch.Tensor,
) -> torch.Tensor:
    """Evaluate all voltage modes at ``delta`` from their interval-start state."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    with torch.no_grad():
        delta_t = torch.as_tensor(delta, device=z_start.device, dtype=z_start.dtype)
        if delta_t.ndim == 0:
            delta_t = delta_t.view(1, 1, 1)
        while delta_t.ndim < z_start.ndim:
            delta_t = delta_t.unsqueeze(0)
        slope = (L1 - L0) / dt
        decay = torch.exp(-delta_t / tau_modes)
        return decay * z_start + (1.0 - decay) * L0 + slope * (delta_t - tau_modes * (1.0 - decay))


def evolve_modes_closed_form(
    z_start: torch.Tensor,
    L0: torch.Tensor,
    L1: torch.Tensor,
    dt: float,
    tau_modes: torch.Tensor,
) -> torch.Tensor:
    return evolve_modes_at_delta(z_start, L0, L1, float(dt), dt, tau_modes)


def weighted_sum_modes(z_modes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        if weights.ndim == 1:
            weights = weights.view(-1, *([1] * (z_modes.ndim - 1)))
        return torch.sum(z_modes * weights, dim=0)
