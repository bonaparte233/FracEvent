from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import h5py
import imageio.v3 as iio
import numpy as np
from PIL import Image


@dataclass
class EventStream:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray
    height: int
    width: int
    config: dict | None = None
    path: str | None = None
    event_count: int | None = None

    @property
    def size(self) -> int:
        return int(self.event_count if self.event_count is not None else self.t.size)

    def sort_by_time(self) -> "EventStream":
        if self.t.size:
            order = np.argsort(self.t, kind="stable")
            self.x = self.x[order]
            self.y = self.y[order]
            self.t = self.t[order]
            self.p = self.p[order]
        return self


class EventBuffer:
    def __init__(self, output_path: str | Path | None, fmt: str = "npz"):
        self.output_path = Path(output_path) if output_path else None
        self.fmt = "h5" if fmt == "hdf5" else fmt
        self._lists: dict[str, list[np.ndarray]] = {"x": [], "y": [], "t": [], "p": []}
        self._h5: h5py.File | None = None
        self._datasets: dict[str, h5py.Dataset] = {}
        self._is_time_sorted = True
        self._last_t: float | None = None
        if self.output_path and self.fmt == "h5":
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._h5 = h5py.File(self.output_path, "w")
            group = self._h5.create_group("events")
            self._datasets = {
                "x": group.create_dataset("x", shape=(0,), maxshape=(None,), chunks=True, dtype="uint16"),
                "y": group.create_dataset("y", shape=(0,), maxshape=(None,), chunks=True, dtype="uint16"),
                "t": group.create_dataset("t", shape=(0,), maxshape=(None,), chunks=True, dtype="float64"),
                "p": group.create_dataset("p", shape=(0,), maxshape=(None,), chunks=True, dtype="int8"),
            }

    def append(self, x, y, t, p) -> None:
        arrays = {
            "x": np.asarray(x, dtype=np.uint16),
            "y": np.asarray(y, dtype=np.uint16),
            "t": np.asarray(t, dtype=np.float64),
            "p": np.asarray(p, dtype=np.int8),
        }
        n = arrays["t"].shape[0]
        if n == 0:
            return
        t = arrays["t"]
        if (t.size > 1 and np.any(np.diff(t) < 0)) or (self._last_t is not None and t[0] < self._last_t):
            self._is_time_sorted = False
        self._last_t = float(t[-1])
        if self._h5:
            old = self._datasets["t"].shape[0]
            new = old + n
            for key, arr in arrays.items():
                ds = self._datasets[key]
                ds.resize((new,))
                ds[old:new] = arr
        else:
            for key, arr in arrays.items():
                self._lists[key].append(arr)

    def to_stream(self, height: int, width: int, config: dict | None = None, sort: bool = True) -> EventStream:
        arrays = {}
        for key, dtype in [("x", np.uint16), ("y", np.uint16), ("t", np.float64), ("p", np.int8)]:
            if self._lists[key]:
                arrays[key] = np.concatenate(self._lists[key]).astype(dtype, copy=False)
            else:
                arrays[key] = np.empty((0,), dtype=dtype)
        stream = EventStream(height=height, width=width, config=config, **arrays)
        return stream.sort_by_time() if sort else stream

    def close(self, height: int, width: int, config: dict | None = None, sort: bool = True) -> EventStream:
        if self._h5:
            event_count = int(self._datasets["t"].shape[0])
            meta = self._h5.create_group("metadata")
            meta.create_dataset("height", data=int(height))
            meta.create_dataset("width", data=int(width))
            meta.create_dataset("config", data=json.dumps(config or {}))
            meta.create_dataset("event_count", data=event_count)
            meta.create_dataset("sorted_by_time", data=bool(self._is_time_sorted))
            self._h5.close()
            return EventStream(
                x=np.empty((0,), dtype=np.uint16),
                y=np.empty((0,), dtype=np.uint16),
                t=np.empty((0,), dtype=np.float64),
                p=np.empty((0,), dtype=np.int8),
                height=height,
                width=width,
                config=config,
                path=str(self.output_path) if self.output_path else None,
                event_count=event_count,
            )
        stream = self.to_stream(height, width, config, sort)
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            save_npz(stream, self.output_path)
        return stream


def save_npz(stream: EventStream, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=stream.x.astype(np.uint16),
        y=stream.y.astype(np.uint16),
        t=stream.t.astype(np.float64),
        p=stream.p.astype(np.int8),
        height=np.array(stream.height, dtype=np.int32),
        width=np.array(stream.width, dtype=np.int32),
        config=json.dumps(stream.config or {}),
    )


def _as_gray_float(
    image: np.ndarray,
    normalize: bool = True,
    inverse_gamma: bool = False,
    color_mode: str = "grayscale",
) -> np.ndarray:
    arr = np.asarray(image)
    color_mode = color_mode.lower()
    if color_mode not in {"grayscale", "gray", "rgb"}:
        raise ValueError("color_mode must be grayscale or rgb")
    if arr.ndim == 3 and color_mode in {"grayscale", "gray"}:
        arr = arr[..., :3].astype(np.float32)
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    elif arr.ndim == 3:
        arr = arr[..., :3]
    arr = arr.astype(np.float32)
    if normalize and arr.max(initial=0) > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    if inverse_gamma:
        arr = np.power(arr, 2.2)
    return arr


def _load_timestamp_override(
    path: str | Path,
    max_frames: int | None = None,
    timestamps_scale: float = 1.0,
) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        timestamps = np.load(path)
    elif path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=False)
        key = "timestamps" if "timestamps" in data else data.files[0]
        timestamps = data[key]
    else:
        timestamps = np.loadtxt(path)
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    timestamps = timestamps * float(timestamps_scale)
    return timestamps[:max_frames] if max_frames else timestamps


def _apply_timestamp_override(
    timestamps: np.ndarray,
    timestamps_path: str | Path | None,
    max_frames: int | None,
    timestamps_scale: float = 1.0,
) -> np.ndarray:
    if timestamps_path is None:
        return timestamps.astype(np.float64)
    override = _load_timestamp_override(timestamps_path, max_frames, timestamps_scale)
    if override.shape[0] != timestamps.shape[0]:
        raise ValueError("timestamp override length must match loaded frame count")
    return override


def _video_fps(path: Path) -> float:
    try:
        meta = iio.immeta(path)
    except Exception:
        return 30.0
    for key in ("fps", "FPS", "framerate", "frame_rate"):
        value = meta.get(key)
        if value:
            try:
                fps = float(value)
            except (TypeError, ValueError):
                try:
                    fps = float(Fraction(str(value)))
                except (ValueError, ZeroDivisionError):
                    continue
            if fps > 0:
                return fps
    return 30.0


def load_image_folder(
    path: str | Path,
    max_frames: int | None = None,
    normalize: bool = True,
    inverse_gamma: bool = False,
    color_mode: str = "grayscale",
    timestamps_path: str | Path | None = None,
    timestamps_scale: float = 1.0,
    fps: float | None = None,
):
    root = Path(path)
    image_paths = sorted(
        p for p in root.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if max_frames:
        image_paths = image_paths[:max_frames]
    if not image_paths:
        raise ValueError(f"No image files found in {root}")
    frames = np.stack(
        [_as_gray_float(np.array(Image.open(p)), normalize, inverse_gamma, color_mode) for p in image_paths],
        axis=0,
    )
    timestamps = np.arange(len(frames), dtype=np.float64) / float(fps or 30.0)
    return frames, _apply_timestamp_override(timestamps, timestamps_path, max_frames, timestamps_scale)


def load_hdf5_frames(
    path: str | Path,
    max_frames: int | None = None,
    normalize: bool = True,
    inverse_gamma: bool = False,
    color_mode: str = "grayscale",
    timestamps_path: str | Path | None = None,
    timestamps_scale: float = 1.0,
):
    with h5py.File(path, "r") as f:
        if "frames" in f and "timestamps" in f:
            n = f["frames"].shape[0] if max_frames is None else min(max_frames, f["frames"].shape[0])
            frames = f["frames"][:n]
            timestamps = f["timestamps"][:n].astype(np.float64)
            frames = np.stack([_as_gray_float(fr, normalize, inverse_gamma, color_mode) for fr in frames], axis=0)
            return frames, _apply_timestamp_override(timestamps, timestamps_path, max_frames, timestamps_scale)
    raise ValueError(f"Unsupported HDF5 frame layout: {path}")


def load_frames(
    path: str | Path,
    max_frames: int | None = None,
    normalize: bool = True,
    inverse_gamma: bool = False,
    color_mode: str = "grayscale",
    timestamps_path: str | Path | None = None,
    timestamps_scale: float = 1.0,
    fps: float | None = None,
):
    path = Path(path)
    if path.is_dir():
        return load_image_folder(path, max_frames, normalize, inverse_gamma, color_mode, timestamps_path, timestamps_scale, fps)
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=False)
        frames = data["frames"]
        timestamps = data["timestamps"]
        if max_frames:
            frames = frames[:max_frames]
            timestamps = timestamps[:max_frames]
        frames = np.stack([_as_gray_float(fr, normalize, inverse_gamma, color_mode) for fr in frames], axis=0)
        return frames, _apply_timestamp_override(timestamps.astype(np.float64), timestamps_path, max_frames, timestamps_scale)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return load_hdf5_frames(path, max_frames, normalize, inverse_gamma, color_mode, timestamps_path, timestamps_scale)
    try:
        frames = []
        for i, frame in enumerate(iio.imiter(path)):
            if max_frames and i >= max_frames:
                break
            frames.append(_as_gray_float(frame, normalize, inverse_gamma, color_mode))
        if not frames:
            raise ValueError(f"No frames loaded from {path}")
        timestamps = np.arange(len(frames), dtype=np.float64) / float(fps or _video_fps(path))
        return np.stack(frames, axis=0), _apply_timestamp_override(timestamps, timestamps_path, max_frames, timestamps_scale)
    except Exception as exc:
        raise ValueError(f"Unsupported input path {path}: {exc}") from exc


def iter_event_chunks(chunks: Iterable) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    for chunk in chunks:
        yield chunk.x, chunk.y, chunk.t, chunk.p
