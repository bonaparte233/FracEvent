<p align="center">
  <img src="assets/overview.png" alt="FracEvent overview" width="100%">
</p>

<h1 align="center">FracEvent</h1>

<p align="center">
  <strong>Event-Camera Simulation via Fractional-Relaxation Pixel Dynamics</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Input" src="https://img.shields.io/badge/Input-NPZ%20%7C%20HDF5%20%7C%20Video%20%7C%20Frames-0f766e">
  <img alt="Output" src="https://img.shields.io/badge/Output-NPZ%20%7C%20HDF5-245c9f">
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#quick-check">Quick Check</a> ·
  <a href="#command-line">CLI</a> ·
  <a href="#python-api">Python API</a> ·
  <a href="#citation">Citation</a>
</p>

---

Official implementation of **FracEvent: Event-Camera Simulation via Fractional-Relaxation Pixel Dynamics**. FracEvent converts frame sequences into event streams `(x, y, t, p)` by modeling each pixel with retained fractional-relaxation voltage dynamics and sub-frame threshold timing.

## Features

- **Fractional-relaxation pixel dynamics** for event camera simulation.
- **Sub-frame event timing** from threshold localization inside frame intervals.
- **Flexible inputs**: NPZ, HDF5, video files, or image-sequence folders.
- **Simple outputs**: NPZ or HDF5 event streams with metadata.
- **Ready-to-use parameter settings** for DAVIS240-like and DAVIS346-like frame data.

## Installation

Clone this repository from GitHub and enter the project directory:

```bash
git clone https://github.com/bonaparte233/fracevent.git
cd fracevent
```

Use Python 3.10+ and install the package in editable mode:

```bash
pip install -e .
```

Or create the conda environment from the same directory:

```bash
conda env create -f environment.yml
conda activate fracevent
```

## Quick Check

```bash
python -c "import numpy as np; from core import FracEventSimulator, load_config; cfg=load_config('configs/davis346.yaml'); cfg.solver.device='cpu'; cfg.output.path=None; frames=np.full((3,4,4),0.5,dtype=np.float32); timestamps=np.array([0.0,0.1,0.2],dtype=np.float64); print(FracEventSimulator(cfg).simulate(frames,timestamps,None).t.size)"
```

The constant-input check should print `0`.

## Command Line

```bash
python main.py simulate --config configs/davis346.yaml --input frames.npz --output events.npz --device cpu
```

HDF5 input:

```bash
python main.py simulate --config configs/davis346.yaml --input frames.h5 --output events.npz --device cpu
```

Image-sequence folder:

```bash
python main.py simulate --config configs/davis346.yaml --input path/to/frame_sequence --fps 30 --output events.npz --device cpu
```

Explicit timestamps:

```bash
python main.py simulate --config configs/davis346.yaml --input path/to/frame_sequence --timestamps timestamps.txt --output events.npz --device cpu
```

CLI arguments such as `--input`, `--output`, `--device`, and model-parameter flags override values in the YAML file.

## Python API

```python
import numpy as np
from core import FracEventSimulator, load_config

cfg = load_config("configs/davis346.yaml")
cfg.solver.device = "cpu"
cfg.output.path = None

frames = np.zeros((4, 16, 16), dtype=np.float32)
frames[1:] = 0.8
timestamps = np.linspace(0.0, 0.1, 4)

events = FracEventSimulator(cfg).simulate(frames, timestamps, None)
print(events.x, events.y, events.t, events.p)
```

## Input Formats

`frames.npz` or `frames.h5` should contain:

```text
frames: [T,H,W] or [T,H,W,C]
timestamps: [T] in seconds
```

For HDF5 input, `frames` and `timestamps` are root datasets. `--input` can also be a video file or any image folder. Video timestamps use the file FPS metadata when available and otherwise fall back to 30 FPS. Image folders are sorted by filename and use `--fps` when provided; use zero-padded filenames or `--timestamps` for non-lexicographic frame names. `--timestamps` overrides FPS-derived timestamps.

## Output Format

Output format follows the output extension:

- `.npz`: flat arrays `x`, `y`, `t`, `p`, plus `height`, `width`, and `config`.
- `.h5` or `.hdf5`: `/events/{x,y,t,p}` and `/metadata`.

Event arrays use:

```text
x: uint16
y: uint16
t: float64 seconds
p: int8, +1 for ON and -1 for OFF
```

## Parameter Settings

- `configs/davis240.yaml`: DAVIS240-like parameter settings.
- `configs/davis346.yaml`: DAVIS346-like parameter settings.

The YAML files expose the main simulator parameters:

- `alpha`: fractional-memory exponent; smaller values retain longer temporal memory.
- `modes`: number of relaxation modes used in the finite approximation.
- `tau_ref`: reference relaxation time scale in seconds.
- `theta_on`: positive threshold for ON events.
- `theta_off`: positive threshold magnitude for OFF events.

## Repository Layout

```text
core/               simulator implementation
configs/            parameter settings
main.py             entry point
index.html          GitHub Pages project page
assets/             project-page assets
```

## Citation

If this code is useful for your research, please cite the paper:

```bibtex
@misc{chen2026fracevent,
      title={FracEvent: Event-Camera Simulation via Fractional-Relaxation Pixel Dynamics},
      author={Langyi Chen and Chuanzhi Xu and Haoxian Zhou and Pengfei Ye and Ziyu Luo and Haodong Chen and Qiang Qu and Xiaoming Chen and Weidong Cai},
      year={2026},
      eprint={2606.26636},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.26636},
}
```

## Limitations

- FracEvent cannot recover motion absent from the input frame trajectory.
- RGB inputs are converted to grayscale before simulation.
- The default configuration is deterministic.
