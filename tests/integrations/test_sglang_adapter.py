import importlib.metadata

import pytest
torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)
pytest.importorskip("sglang")

import sglang.srt.layers.layernorm as layernorm

from quantassay.integrations.sglang.v0_5_3 import install


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_adapter_patches_both_rmsnorm_paths(backend):
    if importlib.metadata.version("sglang") != "0.5.3":
        pytest.skip("adapter targets SGLang 0.5.3")
    norm = layernorm.RMSNorm(32, eps=1e-5).to(device="cuda", dtype=torch.float16)
    weight = norm.weight.data
    x = torch.linspace(-1.0, 1.0, 64, device="cuda", dtype=torch.float16).reshape(2, 32)
    residual = torch.linspace(0.5, -0.5, 64, device="cuda", dtype=torch.float16).reshape(2, 32)
    x_before, residual_before = x.clone(), residual.clone()
    summed = x_before.double() + residual_before.double()
    expected_residual = summed.to(x.dtype)
    expected_x = (
        summed * torch.rsqrt(summed.square().mean(dim=-1, keepdim=True) + 1e-5) * weight.double()
    ).to(x.dtype)
    original_rmsnorm, original_fused = layernorm.rmsnorm, layernorm.fused_add_rmsnorm
    restore = install(backend)

    try:
        output, residual_out = norm.forward_cuda(x, residual)
        assert output is x and residual_out is residual
        torch.testing.assert_close(x, expected_x, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(residual, expected_residual, rtol=1e-3, atol=1e-3)

        plain_input = torch.linspace(-1.0, 1.0, 64, device="cuda", dtype=torch.float16).reshape(2, 32)
        plain_expected = (
            plain_input.double()
            * torch.rsqrt(plain_input.double().square().mean(dim=-1, keepdim=True) + 1e-5)
            * weight.double()
        ).to(plain_input.dtype)
        torch.testing.assert_close(
            norm.forward_cuda(plain_input), plain_expected, rtol=1e-3, atol=1e-3
        )
    finally:
        restore()

    assert layernorm.rmsnorm is original_rmsnorm
    assert layernorm.fused_add_rmsnorm is original_fused
