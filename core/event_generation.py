from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import ModelConfig, SolverConfig
from .dynamics import evolve_modes_at_delta, evolve_modes_closed_form, weighted_sum_modes


@dataclass
class EventChunk:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray

    @property
    def size(self) -> int:
        return int(self.t.shape[0])


def compute_event_counts(v_end: torch.Tensor, ref: torch.Tensor, cfg: ModelConfig):
    with torch.no_grad():
        d = v_end - ref
        pos = torch.floor(torch.clamp(cfg.gain * d / cfg.theta_on, min=0)).to(torch.int32)
        neg = torch.floor(torch.clamp(cfg.gain * (-d) / cfg.theta_off, min=0)).to(torch.int32)
        return pos, neg


def _voltage_at_flat_delta(
    z_start_flat: torch.Tensor,
    L0_flat: torch.Tensor,
    L1_flat: torch.Tensor,
    tau_flat: torch.Tensor,
    weights: torch.Tensor,
    delta: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Return weighted voltage at an intra-interval time for flattened active pixels."""
    delta_m = delta.view(1, -1)
    decay = torch.exp(-delta_m / tau_flat)
    slope = (L1_flat - L0_flat) / dt
    z_delta = decay * z_start_flat + (1.0 - decay) * L0_flat.view(1, -1)
    z_delta = z_delta + slope.view(1, -1) * (delta_m - tau_flat * (1.0 - decay))
    if weights.ndim == 1:
        weights = weights.view(-1, 1)
    return torch.sum(z_delta * weights, dim=0)


def solve_crossing_times_bisection(
    z_start_flat: torch.Tensor,
    L0_flat: torch.Tensor,
    L1_flat: torch.Tensor,
    tau_flat: torch.Tensor,
    weights: torch.Tensor,
    targets: torch.Tensor,
    dt: float,
    polarity: int,
    steps: int = 12,
) -> torch.Tensor:
    if polarity not in (-1, 1):
        raise ValueError("polarity must be +1 or -1")
    with torch.no_grad():
        lo = torch.zeros_like(targets)
        hi = torch.full_like(targets, float(dt))
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            v_mid = _voltage_at_flat_delta(z_start_flat, L0_flat, L1_flat, tau_flat, weights, mid, dt)
            if polarity == 1:
                hit = v_mid >= targets
            else:
                hit = v_mid <= targets
            hi = torch.where(hit, mid, hi)
            lo = torch.where(hit, lo, mid)
        return hi


def generate_interval_events(
    z_start: torch.Tensor,
    ref: torch.Tensor,
    L0: torch.Tensor,
    L1: torch.Tensor,
    tau_modes: torch.Tensor,
    weights: torch.Tensor,
    t0: float,
    dt: float,
    model_cfg: ModelConfig,
    solver_cfg: SolverConfig,
    depth: int = 0,
):
    with torch.no_grad():
        z_end = evolve_modes_closed_form(z_start, L0, L1, dt, tau_modes)
        v_end = weighted_sum_modes(z_end, weights)
        ref0 = ref.clone()
        pos_count, neg_count = compute_event_counts(v_end, ref0, model_cfg)
        max_count = int(torch.maximum(pos_count.max(), neg_count.max()).item())

        if max_count > solver_cfg.max_events_per_interval and dt > solver_cfg.min_recursive_dt:
            half = 0.5 * dt
            L_mid = 0.5 * (L0 + L1)
            z_mid, chunks_a = generate_interval_events(
                z_start, ref, L0, L_mid, tau_modes, weights, t0, half, model_cfg, solver_cfg, depth + 1
            )
            z_final, chunks_b = generate_interval_events(
                z_mid, ref, L_mid, L1, tau_modes, weights, t0 + half, half, model_cfg, solver_cfg, depth + 1
            )
            return z_final, chunks_a + chunks_b

        if max_count > solver_cfg.max_events_per_interval:
            pos_count = torch.clamp(pos_count, max=solver_cfg.max_events_per_interval)
            neg_count = torch.clamp(neg_count, max=solver_cfg.max_events_per_interval)
            max_count = solver_cfg.max_events_per_interval

        chunks: list[EventChunk] = []
        height, width = ref.shape
        flat_index = torch.arange(height * width, device=ref.device).view(height, width)
        z_start_2d = z_start.reshape(z_start.shape[0], -1)
        L0_1d = L0.reshape(-1)
        L1_1d = L1.reshape(-1)
        tau_2d = tau_modes.reshape(tau_modes.shape[0], -1)
        weights_2d = weights.reshape(weights.shape[0], -1) if weights.ndim > 1 else None

        def append_for(mask: torch.Tensor, target: torch.Tensor, polarity: int) -> None:
            idx = flat_index[mask].reshape(-1)
            if idx.numel() == 0:
                return
            delta = solve_crossing_times_bisection(
                z_start_2d[:, idx],
                L0_1d[idx],
                L1_1d[idx],
                tau_2d[:, idx],
                weights if weights_2d is None else weights_2d[:, idx],
                target.reshape(-1),
                dt,
                polarity,
                solver_cfg.root_bisection_steps,
            )
            ys = torch.div(idx, width, rounding_mode="floor").to(torch.int64)
            xs = (idx - ys * width).to(torch.int64)
            chunks.append(
                EventChunk(
                    x=xs.detach().cpu().numpy().astype(np.uint16),
                    y=ys.detach().cpu().numpy().astype(np.uint16),
                    t=(np.float64(t0) + delta.detach().cpu().numpy().astype(np.float64)).astype(np.float64),
                    p=np.full(idx.numel(), polarity, dtype=np.int8),
                )
            )

        for j in range(1, int(pos_count.max().item()) + 1):
            mask = pos_count >= j
            target = ref0[mask] + j * model_cfg.theta_on / model_cfg.gain
            append_for(mask, target, 1)
        for j in range(1, int(neg_count.max().item()) + 1):
            mask = neg_count >= j
            target = ref0[mask] - j * model_cfg.theta_off / model_cfg.gain
            append_for(mask, target, -1)

        ref.copy_(torch.where(pos_count > 0, ref0 + pos_count.to(ref0.dtype) * model_cfg.theta_on / model_cfg.gain, ref))
        ref.copy_(torch.where(neg_count > 0, ref0 - neg_count.to(ref0.dtype) * model_cfg.theta_off / model_cfg.gain, ref))
        return z_end, chunks
