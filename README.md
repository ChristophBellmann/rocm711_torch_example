# rocm711_torch_example

Minimal example project that installs your custom ROCm 7.11 PyTorch wheel from:

- `/opt/rocm/wheels/pytorch_rocm711/`

This prevents accidentally downloading a different torch build from the internet.

## Setup

```bash
cd /path/to/ML-Lab/examples/rocm711_torch_example

./setup_rocm_venv.sh

. .venv/bin/activate_rocm_pytorch.sh
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Notes:
- `torch` is installed from the installed custom ROCm wheel under `/opt/rocm/wheels/pytorch_rocm711/`.
- `requirements.txt` contains only project-local Python dependencies.
- `activate_rocm_pytorch.sh` carries the required ROCm runtime environment for this custom wheel.

This project intentionally does **not** support an offline mode: internet downloads for
small Python dependencies are expected, but the ROCm 7.11 `torch` wheel is always taken from
the local path under `/opt/rocm/wheels/pytorch_rocm711/`.

## Run

```bash
. .venv/bin/activate_rocm_pytorch.sh
python run.py
```

Expected:

- `torch` imports
- HIP device is available (or a clear FAIL message)
- a small matmul runs on GPU and prints timing + TFLOPS estimate
