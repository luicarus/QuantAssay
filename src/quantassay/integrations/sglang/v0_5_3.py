"""SGLang 0.5.3 RMSNorm adapter and server launcher."""

import argparse
import importlib
import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Callable, Literal

from kernscope import fused_add_rms_norm, rms_norm

Backend = Literal["torch", "triton"]


def install(backend: Backend) -> Callable[[], None]:
    """Patch SGLang's RMSNorm functions and return a restore callback."""
    if backend not in ("torch", "triton"):
        raise ValueError("backend must be 'torch' or 'triton'")
    try:
        version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("SGLang must be installed to use this adapter") from error
    if version != "0.5.3":
        raise RuntimeError(f"expected SGLang 0.5.3, found {version}")

    layernorm = importlib.import_module("sglang.srt.layers.layernorm")
    original_rmsnorm, original_fused = layernorm.rmsnorm, layernorm.fused_add_rmsnorm

    def rmsnorm_adapter(input, weight, eps=1e-6, out=None, enable_pdl=None):
        if enable_pdl:
            raise NotImplementedError("Kernscope does not support enable_pdl")
        output = rms_norm(input, weight, eps, backend=backend)
        if out is None:
            return output
        if out.shape != output.shape or out.dtype != output.dtype or out.device != output.device:
            raise ValueError("out must match the RMSNorm output shape, dtype, and device")
        return out.copy_(output)

    def fused_adapter(input, residual, weight, eps=1e-6, enable_pdl=None):
        if enable_pdl:
            raise NotImplementedError("Kernscope does not support enable_pdl")
        return fused_add_rms_norm(input, residual, weight, eps, backend=backend)

    layernorm.rmsnorm, layernorm.fused_add_rmsnorm = rmsnorm_adapter, fused_adapter

    def restore() -> None:
        layernorm.rmsnorm, layernorm.fused_add_rmsnorm = original_rmsnorm, original_fused

    return restore


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--kernscope-backend", choices=("torch", "triton"), required=True)
    args, server_args = parser.parse_known_args(argv)
    bootstrap = str(Path(__file__).with_name("v0_5_3_bootstrap"))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        part for part in (bootstrap, os.environ.get("PYTHONPATH", "")) if part
    )
    os.environ["KERNSCOPE_SGLANG_BACKEND"] = args.kernscope_backend
    os.execv(sys.executable, [sys.executable, "-m", "sglang.launch_server", *server_args])


if __name__ == "__main__":
    main()
