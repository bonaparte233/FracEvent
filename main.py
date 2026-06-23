from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.config import load_config
from core.io import load_frames
from core.simulator import FracEventSimulator


def _cli_overrides(args) -> dict:
    model = {}
    for name in ["alpha", "modes", "theta_on", "theta_off", "tau_ref"]:
        val = getattr(args, name, None)
        if val is not None:
            model[name] = val
    solver = {}
    if getattr(args, "device", None) is not None:
        solver["device"] = args.device
    output = {}
    if getattr(args, "output", None) is not None:
        output["path"] = args.output
        suffix = Path(args.output).suffix.lower()
        output["format"] = "h5" if suffix in {".h5", ".hdf5"} else "npz"
    input_cfg = {}
    if getattr(args, "input", None) is not None:
        input_cfg["path"] = args.input
    if getattr(args, "timestamps", None) is not None:
        input_cfg["timestamps"] = args.timestamps
    if getattr(args, "timestamps_scale", None) is not None:
        input_cfg["timestamps_scale"] = args.timestamps_scale
    if getattr(args, "fps", None) is not None:
        input_cfg["fps"] = args.fps
    if getattr(args, "max_frames", None) is not None:
        input_cfg["max_frames"] = args.max_frames
    merged = {}
    if model:
        merged["model"] = model
    if solver:
        merged["solver"] = solver
    if output:
        merged["output"] = output
    if input_cfg:
        merged["input"] = input_cfg
    return merged


def cmd_simulate(args) -> int:
    cfg = load_config(args.config, _cli_overrides(args))
    if not cfg.input.path:
        raise SystemExit("--input or input.path is required")
    frames, timestamps = load_frames(
        cfg.input.path,
        cfg.input.max_frames,
        normalize=cfg.input.normalize,
        inverse_gamma=cfg.input.inverse_gamma,
        color_mode=cfg.input.color_mode,
        timestamps_path=cfg.input.timestamps,
        timestamps_scale=cfg.input.timestamps_scale,
        fps=cfg.input.fps,
    )
    stream = FracEventSimulator(cfg).simulate(frames, timestamps, cfg.output.path)
    print(json.dumps({"events": stream.size, "height": stream.height, "width": stream.width, "output": cfg.output.path}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sim = sub.add_parser("simulate")
    sim.add_argument("--config", required=True, help="parameter setting YAML, e.g. configs/davis346.yaml")
    sim.add_argument("--input", help="input NPZ/HDF5/video file or image-sequence folder")
    sim.add_argument("--timestamps", help="optional frame timestamps (.txt, .npy, or .npz); overrides FPS-derived timestamps")
    sim.add_argument("--timestamps-scale", type=float, dest="timestamps_scale", help="scale applied to timestamp values")
    sim.add_argument("--fps", type=float, help="FPS for image folders or videos without usable FPS metadata")
    sim.add_argument("--output", help="output path; .npz, .h5, and .hdf5 are supported")
    sim.add_argument("--max-frames", type=int, help="process only the first N frames")
    sim.add_argument("--alpha", type=float, help="override model.alpha")
    sim.add_argument("--modes", type=int, help="override model.modes")
    sim.add_argument("--theta-on", type=float, dest="theta_on", help="override model.theta_on")
    sim.add_argument("--theta-off", type=float, dest="theta_off", help="override model.theta_off")
    sim.add_argument("--tau-ref", type=float, dest="tau_ref", help="override model.tau_ref")
    sim.add_argument("--device", help="cpu, cuda, or cuda:N")
    sim.set_defaults(func=cmd_simulate)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
