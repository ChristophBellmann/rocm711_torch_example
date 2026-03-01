# rocm711_torch_example

Minimal example project that installs your custom ROCm 7.11 PyTorch wheel from:

- `/opt/rocm/wheels/pytorch_rocm711/`

This prevents accidentally downloading a different torch build from the internet.

## Setup

```bash
cd /media/christoph/some_space/Compute/rocm711_torch_example
python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -r requirements.txt
```

Notes:
- `requirements.txt` forces **torch** to be installed from `/opt/rocm/wheels/pytorch_rocm711/...whl`.
- Small dependencies (e.g. `filelock`) are installed from PyPI.

This project intentionally does **not** support an offline mode: internet downloads for
dependencies are expected, but the ROCm 7.11 `torch` wheel is always taken from the local
path under `/opt/rocm/wheels/pytorch_rocm711/`.

## Run

```bash
source .venv/bin/activate
python run.py
```

Expected:

- `torch` imports
- HIP device is available (or a clear FAIL message)
- a small matmul runs on GPU and prints timing + TFLOPS estimate
