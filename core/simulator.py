from __future__ import annotations

import numpy as np
import torch

from .config import Config
from .dynamics import weighted_sum_modes
from .event_generation import EventChunk, generate_interval_events
from .io import EventBuffer, EventStream, iter_event_chunks
from .kernels import brightness_dependent_tau, build_tau_modes, fractional_weights


class FracEventSimulator:
    def __init__(self, config: Config):
        self.config = config
        requested = config.solver.device
        if requested == "cuda" and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.dtype = getattr(torch, config.solver.dtype)
        self.u_modes, self.weights = fractional_weights(
            config.model.alpha,
            config.model.modes,
            config.model.relaxation_log_range,
            self.device,
            self.dtype,
        )
        self.z: torch.Tensor | None = None
        self.ref: torch.Tensor | None = None
        self.height = 0
        self.width = 0
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(config.noise.seed)

    def _frame_tensor(self, frame: np.ndarray) -> torch.Tensor:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim == 3:
            arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        if arr.max(initial=0) > 1.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        return torch.as_tensor(arr, device=self.device, dtype=self.dtype)

    def log_intensity(self, frame: np.ndarray | torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(frame):
            frame_t = self._frame_tensor(frame)
        else:
            frame_t = frame.to(device=self.device, dtype=self.dtype)
        m = self.config.model
        return torch.log(frame_t + m.i_dark + m.eps)

    def initialize(self, frame0: np.ndarray, t0: float = 0.0) -> None:
        with torch.no_grad():
            L0 = self.log_intensity(frame0)
            self.height, self.width = int(L0.shape[-2]), int(L0.shape[-1])
            self.z = L0.unsqueeze(0).repeat(self.u_modes.numel(), 1, 1).clone()
            self.ref = weighted_sum_modes(self.z, self.weights).clone()

    def _tau_modes_for(self, frame0: np.ndarray, frame1: np.ndarray) -> torch.Tensor:
        f0 = self._frame_tensor(frame0)
        f1 = self._frame_tensor(frame1)
        tau0 = brightness_dependent_tau(0.5 * (f0 + f1), self.config.model)
        return build_tau_modes(tau0, self.u_modes)

    def process_frame_pair(self, frame0, frame1, t0: float, t1: float):
        if self.z is None or self.ref is None:
            self.initialize(frame0, t0)
        dt = float(t1 - t0)
        if dt <= 0:
            raise ValueError("timestamps must be strictly increasing")
        L0 = self.log_intensity(frame0)
        L1 = self.log_intensity(frame1)
        tau_modes = self._tau_modes_for(frame0, frame1)
        z_out, chunks = generate_interval_events(
            self.z,
            self.ref,
            L0,
            L1,
            tau_modes,
            self.weights,
            float(t0),
            dt,
            self.config.model,
            self.config.solver,
        )
        self.z = z_out
        if self.config.noise.enabled and self.config.noise.sigma > 0:
            decay = torch.exp(-2.0 * dt / tau_modes)
            scale = self.config.noise.sigma * torch.sqrt(
                self.weights.view(-1, 1, 1) * torch.clamp(1.0 - decay, min=0.0)
            )
            self.z = self.z + scale * torch.randn(
                self.z.shape,
                device=self.device,
                dtype=self.dtype,
                generator=self.rng,
            )
        if self.config.output.sort_by_time:
            chunks = self._sort_chunks_by_time(chunks)
        return chunks

    def _sort_chunks_by_time(self, chunks: list[EventChunk]) -> list[EventChunk]:
        nonempty = [c for c in chunks if c.size]
        if not nonempty:
            return []
        x = np.concatenate([c.x for c in nonempty]).astype(np.uint16, copy=False)
        y = np.concatenate([c.y for c in nonempty]).astype(np.uint16, copy=False)
        t = np.concatenate([c.t for c in nonempty]).astype(np.float64, copy=False)
        p = np.concatenate([c.p for c in nonempty]).astype(np.int8, copy=False)
        order = np.argsort(t, kind="stable")
        return [EventChunk(x=x[order], y=y[order], t=t[order], p=p[order])]

    def simulate(self, frames: np.ndarray, timestamps: np.ndarray, output_path: str | None = None) -> EventStream:
        if len(frames) != len(timestamps):
            raise ValueError("frames and timestamps must have the same length")
        if len(frames) < 2:
            raise ValueError("at least two frames are required")
        out_path = output_path if output_path is not None else self.config.output.path
        buffer = EventBuffer(out_path, self.config.output.format)
        self.initialize(frames[0], float(timestamps[0]))
        for i in range(len(frames) - 1):
            chunks = self.process_frame_pair(frames[i], frames[i + 1], float(timestamps[i]), float(timestamps[i + 1]))
            for x, y, t, p in iter_event_chunks(chunks):
                buffer.append(x, y, t, p)
        return buffer.close(
            self.height,
            self.width,
            config=self.config.to_metadata_dict(),
            sort=self.config.output.sort_by_time,
        )

    def estimate_peak_state_bytes(self, height: int, width: int, modes: int | None = None) -> int:
        modes = int(modes or self.u_modes.numel())
        return int((3 * modes + 6) * height * width * 4)
