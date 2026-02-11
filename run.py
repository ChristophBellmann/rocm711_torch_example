#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time


def _fmt_s(x: float) -> str:
    if x < 1e-3:
        return f"{x * 1e6:.0f}us"
    if x < 1.0:
        return f"{x * 1e3:.1f}ms"
    return f"{x:.3f}s"


def _maybe_reexec_with_rocm_env() -> None:
    """
    Ensure ROCm runtime libs are discoverable for the process.

    This example wheel can depend on OpenMP runtime symbols (e.g. __kmpc_*)
    even when linked against libgomp; preloading libomp satisfies those.

    LD_LIBRARY_PATH/LD_PRELOAD are only honored at process startup, so we
    re-exec once if we need to add them.
    """
    if os.environ.get("ROCM711_EXAMPLE_REEXEC", "") == "1":
        return

    rocm = os.environ.get("ROCM_PATH", "").strip() or "/opt/rocm"
    env = dict(os.environ)
    env["ROCM_PATH"] = rocm

    # Keep it minimal: only add paths if missing.
    path = env.get("PATH", "")
    want_path = [f"{rocm}/bin", f"{rocm}/llvm/bin"]
    if not all(p in path.split(":") for p in want_path):
        env["PATH"] = ":".join(want_path + ([path] if path else []))

    ld = env.get("LD_LIBRARY_PATH", "")
    want_ld = [
        f"{rocm}/lib",
        f"{rocm}/lib64",
        f"{rocm}/lib/llvm/lib",
        f"{rocm}/lib/host-math/lib",
        f"{rocm}/lib/rocm_sysdeps/lib",
        f"{rocm}/llvm/lib",
    ]
    if not all(p in ld.split(":") for p in want_ld):
        env["LD_LIBRARY_PATH"] = ":".join(want_ld + ([ld] if ld else []))

    # Preload libomp to satisfy __kmpc_* symbols if needed.
    libomp = f"{rocm}/lib/llvm/lib/libomp.so"
    if os.path.exists(libomp):
        cur = env.get("LD_PRELOAD", "")
        if libomp not in cur.split(":"):
            env["LD_PRELOAD"] = ":".join([libomp] + ([cur] if cur else []))

    # Re-exec only if we actually changed something.
    if env.get("PATH") != os.environ.get("PATH") or env.get("LD_LIBRARY_PATH") != os.environ.get("LD_LIBRARY_PATH") or env.get("LD_PRELOAD") != os.environ.get("LD_PRELOAD"):
        env["ROCM711_EXAMPLE_REEXEC"] = "1"
        os.execvpe(sys.executable, [sys.executable, __file__] + sys.argv[1:], env)


def main() -> int:
    _maybe_reexec_with_rocm_env()

    try:
        import torch  # type: ignore
    except Exception as e:
        print("FAIL: import torch")
        print(f"  {e!r}")
        return 1

    print("== PyTorch ROCm smoke ==")
    print(f"torch.__version__      : {torch.__version__}")
    print(f"torch.version.hip      : {getattr(torch.version, 'hip', None)}")
    print(f"torch.version.rocm     : {getattr(torch.version, 'rocm', None)}")
    print(f"ROCM_PATH              : {os.environ.get('ROCM_PATH', '')}")
    print(f"LD_LIBRARY_PATH (set?) : {'yes' if os.environ.get('LD_LIBRARY_PATH') else 'no'}")

    # On ROCm builds, torch.cuda is the HIP backend.
    hip_ok = bool(torch.cuda.is_available())
    print(f"torch.cuda.is_available: {hip_ok}")
    if not hip_ok:
        print("FAIL: HIP device not available to PyTorch.")
        print("Hints:")
        print("- Check: /dev/kfd exists and your user is in 'video' and 'render' groups.")
        print("- Check: /opt/rocm is installed and working: /opt/rocm/bin/rocminfo")
        print("- Check: this venv installed torch from /opt/rocm/wheels/pytorch_rocm711 (see pip freeze).")
        return 2

    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(dev)
    print(f"device                : {props.name}")
    print(f"gcnArchName (if any)  : {getattr(props, 'gcnArchName', None)}")
    print(f"total_memory          : {props.total_memory/1024**3:.2f} GiB")

    # Sustained compute: matmul for ~5s.
    torch.set_float32_matmul_precision("high")

    # Use a size big enough to keep the GPU busy but not explode VRAM.
    # 4096^2 float16 ~= 128 MiB per matrix; we use two inputs + output.
    n = 4096
    dtype = torch.float16
    a = torch.randn((n, n), device=dev, dtype=dtype)
    b = torch.randn((n, n), device=dev, dtype=dtype)

    # Warmup.
    for _ in range(5):
        _ = a @ b
    torch.cuda.synchronize()

    start = time.perf_counter()
    iters = 0
    target_s = 5.0
    while True:
        # Launch a small burst, then synchronize so the wall-clock reflects
        # actual GPU execution (avoids queueing 100k kernels and waiting later).
        burst = 8
        for _ in range(burst):
            _ = a @ b
        iters += burst
        torch.cuda.synchronize()
        if (time.perf_counter() - start) >= target_s:
            break
    wall = time.perf_counter() - start

    # Rough FLOP estimate for GEMM: 2*n^3 per matmul (mul+add), ignore alpha/beta.
    flops = iters * (2.0 * (n**3))
    tflops = flops / wall / 1e12

    print("")
    print("== Matmul benchmark (sustained) ==")
    print(f"shape      : {n}x{n} ({dtype})")
    print(f"iters      : {iters}")
    print(f"wall       : {_fmt_s(wall)}")
    print(f"TFLOPS(est): {tflops:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
