from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class InputConfig:
    path: str | None = None
    timestamps: str | None = None
    timestamps_scale: float = 1.0
    fps: float | None = None
    color_mode: str = "grayscale"
    normalize: bool = True
    inverse_gamma: bool = False
    max_frames: int | None = None


@dataclass
class ModelConfig:
    alpha: float = 0.85
    modes: int = 6
    relaxation_log_range: float = 3.0
    tau_ref: float = 0.004
    tau_min: float = 0.00005
    tau_max: float = 0.030
    i_ref: float = 0.5
    beta: float = 1.0
    i_dark: float = 0.001
    eps: float = 1e-6
    gain: float = 1.0
    theta_on: float = 0.2
    theta_off: float = 0.2


@dataclass
class SolverConfig:
    root_bisection_steps: int = 12
    max_events_per_interval: int = 64
    min_recursive_dt: float = 1e-6
    device: str = "cuda"
    dtype: str = "float32"


@dataclass
class NoiseConfig:
    enabled: bool = False
    sigma: float = 0.0
    seed: int = 0


@dataclass
class OutputConfig:
    path: str = "events.npz"
    format: str = "npz"
    sort_by_time: bool = True


@dataclass
class Config:
    input: InputConfig = field(default_factory=InputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_metadata_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["input"].pop("path", None)
        data["input"].pop("timestamps", None)
        data["output"].pop("path", None)
        return data


def _merge_dataclass(obj: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown config key: {key}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    cfg = Config()
    if path:
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _merge_dataclass(cfg, data)
    if overrides:
        _merge_dataclass(cfg, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    m = cfg.model
    if not (0.0 < m.alpha <= 1.0):
        raise ValueError("model.alpha must be in (0, 1]")
    if m.modes < 1:
        raise ValueError("model.modes must be >= 1")
    if m.modes > 16:
        raise ValueError("model.modes must be <= 16")
    if m.alpha == 1.0 and m.modes != 1:
        raise ValueError("model.alpha=1.0 requires model.modes=1")
    if m.theta_on <= 0 or m.theta_off <= 0:
        raise ValueError("thresholds must be positive")
    if cfg.input.timestamps_scale <= 0:
        raise ValueError("input.timestamps_scale must be positive")
    if cfg.input.fps is not None and cfg.input.fps <= 0:
        raise ValueError("input.fps must be positive")
    if cfg.input.max_frames is not None and cfg.input.max_frames < 1:
        raise ValueError("input.max_frames must be positive")
    if m.gain <= 0:
        raise ValueError("gain must be positive")
    if m.tau_min <= 0 or m.tau_max <= 0 or m.tau_ref <= 0:
        raise ValueError("tau values must be positive")
    if m.tau_min > m.tau_max:
        raise ValueError("tau_min must be <= tau_max")
    if cfg.solver.root_bisection_steps < 1:
        raise ValueError("root_bisection_steps must be >= 1")
    if cfg.solver.root_bisection_steps > 64:
        raise ValueError("root_bisection_steps must be <= 64")
    if cfg.solver.max_events_per_interval < 1:
        raise ValueError("max_events_per_interval must be >= 1")
    if cfg.solver.max_events_per_interval > 1024:
        raise ValueError("max_events_per_interval must be <= 1024")
    if not (1e-9 <= cfg.solver.min_recursive_dt <= 1.0):
        raise ValueError("min_recursive_dt must be in [1e-9, 1.0]")
    if cfg.solver.dtype not in {"float32", "float64"}:
        raise ValueError("solver.dtype must be float32 or float64")
    device = str(cfg.solver.device)
    if device != "cpu" and device != "cuda" and not device.startswith("cuda:"):
        raise ValueError("solver.device must be cpu or cuda")
    if cfg.output.format not in {"npz", "h5", "hdf5"}:
        raise ValueError("output.format must be npz or h5")
    if cfg.noise.enabled and cfg.noise.sigma < 0:
        raise ValueError("noise.sigma must be non-negative")
