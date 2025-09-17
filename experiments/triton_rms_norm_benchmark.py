"""
RMS Normalization Triton Benchmark
==================================

Benchmarking comparison between Triton-optimized RMS Norm kernel and PyTorch implementation.

RMS Normalization computes:
y = (x / sqrt(mean(x^2) + eps)) * weight

This is simpler than Layer Normalization as it doesn't subtract the mean,
only normalizes by the RMS (root mean square).
"""

import torch
import triton
import torch.nn.functional as F
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _rms_norm_fwd_fused(
    input_ptr,  # pointer to the input tensor
    output_ptr,  # pointer to the output tensor
    weight_ptr,  # pointer to the weight tensor
    rstd_ptr,  # pointer to the reciprocal standard deviation tensor
    row_stride,  # stride for moving to the next row
    feature_dim,  # number of features (columns) in input
    eps,  # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of input and output tensors to compute
    row_idx = tl.program_id(0)
    output_ptr += row_idx * row_stride
    input_ptr += row_idx * row_stride

    # Compute variance (mean of squared values for RMS)
    sum_of_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_offset in range(0, feature_dim, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
        input_values = tl.load(input_ptr + col_indices, mask=col_indices < feature_dim, other=0.0).to(tl.float32)
        sum_of_squares += input_values * input_values

    variance = tl.sum(sum_of_squares, axis=0) / feature_dim
    reciprocal_std = 1 / tl.sqrt(variance + eps)

    # Store reciprocal standard deviation for backward pass
    tl.store(rstd_ptr + row_idx, reciprocal_std)

    # Normalize input and apply weight transformation
    for block_offset in range(0, feature_dim, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
        valid_mask = col_indices < feature_dim

        weight_values = tl.load(weight_ptr + col_indices, mask=valid_mask)
        input_values = tl.load(input_ptr + col_indices, mask=valid_mask, other=0.0).to(tl.float32)

        normalized_values = input_values * reciprocal_std
        output_values = normalized_values * weight_values

        # Write final output
        tl.store(output_ptr + col_indices, output_values, mask=valid_mask)


@triton.jit
def _rms_norm_bwd_dx_fused(
    DX,  # pointer to the input gradient
    DY,  # pointer to the output gradient
    DW,  # pointer to the partial sum of weights gradient
    X,  # pointer to the input
    W,  # pointer to the weights
    Rstd,  # pointer to the 1/std
    Lock,  # pointer to the lock
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride

    # Offset locks and weights gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols

    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    rstd = tl.load(Rstd + row)

    # Compute dx for RMS norm
    # RMS norm: y = x * rstd * w
    # where rstd = 1 / sqrt(mean(x^2) + eps)
    #
    # Backward pass:
    # dx = (dy * w * rstd) - (x * rstd * mean(dy * w * x * rstd)) / mean(x^2 + eps)

    x_hat = x * rstd
    wdy = w * dy
    xhat = tl.where(mask, x_hat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)

    # For RMS norm: c1 = mean(dy * w * x_hat * x)
    c1 = tl.sum(wdy * xhat * x, axis=0) / N
    dx = (wdy - x * rstd * c1) * rstd

    # Write dx
    tl.store(DX + cols, dx, mask=mask)

    # Accumulate partial sums for dw
    partial_dw = (dy * x_hat).to(w.dtype)

    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
    tl.store(DW, partial_dw, mask=mask)

    # Release the lock
    tl.debug_barrier()
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _rms_norm_bwd_dw(
    DW,  # pointer to the partial sum of weights gradient
    FINAL_DW,  # pointer to the weights gradient
    M,  # GROUP_SIZE_M
    N,  # number of columns
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map the program id to the elements of DW it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate through the rows of DW to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)

    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)


class RMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This RMS norm doesn't support feature dim >= 64KB.")

        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        # enqueue kernel
        _rms_norm_fwd_fused[(M,)](
            x_arg,
            y,
            weight,
            rstd,
            x_arg.stride(0),
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_ctas=1,
        )

        ctx.save_for_backward(x, weight, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, rstd = ctx.saved_tensors
        # heuristics for amount of parallel reduction stream for DW
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192:
            GROUP_SIZE_M = 96
        if N <= 4096:
            GROUP_SIZE_M = 128
        if N <= 1024:
            GROUP_SIZE_M = 256

        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N,), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)

        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _rms_norm_bwd_dx_fused[(M,)](
            dx,
            dy,
            _dw,
            x,
            w,
            rstd,
            locks,
            x_arg.stride(0),
            N,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=ctx.num_warps,
        )

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
        # accumulate partial sums in separate kernel
        _rms_norm_bwd_dw[grid](
            _dw,
            dw,
            min(GROUP_SIZE_M, M),
            N,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=128,
            num_ctas=1,
        )

        return dx, None, dw, None


rms_norm = RMSNorm.apply


# def pytorch_rms_norm(x, weight, eps=1e-5):
#     """PyTorch reference implementation of RMS Norm."""
#     variance = x.pow(2).mean(dim=-1, keepdim=True)
#     x_normalized = x * torch.rsqrt(variance + eps)
#     return x_normalized * weight


def pytorch_rms_norm(x, weight, eps=1e-5):
    """PyTorch reference implementation of RMS Norm."""
    return F.rms_norm(x, (x.size(-1),), weight=weight, eps=eps)


# Compiled version for performance comparison
pytorch_rms_norm_compiled = torch.compile(pytorch_rms_norm)


def test_rms_norm(M, N, dtype, eps=1e-5, device=DEVICE):
    """Test correctness of Triton RMS Norm vs PyTorch implementation."""
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)

    # forward pass
    y_tri = rms_norm(x, w_shape, weight, eps)
    y_ref = pytorch_rms_norm(x, weight, eps)
    y_compiled = pytorch_rms_norm_compiled(x, weight, eps)

    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri = [_.grad.clone() for _ in [x, weight]]
    x.grad, weight.grad = None, None

    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref = [_.grad.clone() for _ in [x, weight]]
    x.grad, weight.grad = None, None

    # backward pass (torch compiled)
    y_compiled.backward(dy, retain_graph=True)
    dx_compiled, dw_compiled = [_.grad.clone() for _ in [x, weight]]

    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(y_tri, y_compiled, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_compiled, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_compiled, atol=1e-2, rtol=0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg="provider",
        line_vals=["triton", "torch", "torch_compile"],
        line_names=["Triton", "Torch", "Torch Compile"],
        styles=[("blue", "-"), ("green", "-"), ("red", "--")],
        ylabel="GB/s",
        plot_name="rms-norm-forward",
        args={"M": 4096, "dtype": torch.float16, "mode": "forward"},
    )
)
def bench_rms_norm_forward(
    M, N, dtype, provider, mode="forward", eps=1e-5, device=DEVICE
):
    """Benchmark RMS Norm forward pass."""
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == "triton":
            return rms_norm(x, w_shape, weight, eps)
        if provider == "torch":
            return pytorch_rms_norm(x, weight, eps)
        if provider == "torch_compile":
            return pytorch_rms_norm_compiled(x, weight, eps)

    # forward pass
    if mode == "forward":
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(
            y_fwd, quantiles=quantiles, rep=500
        )

    return gbps(ms), gbps(max_ms), gbps(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg="provider",
        line_vals=["triton", "torch", "torch_compile"],
        line_names=["Triton", "Torch", "Torch Compile"],
        styles=[("blue", "-"), ("green", "-"), ("red", "--")],
        ylabel="GB/s",
        plot_name="rms-norm-backward",
        args={"M": 4096, "dtype": torch.float16, "mode": "backward"},
    )
)
def bench_rms_norm_backward(
    M, N, dtype, provider, mode="backward", eps=1e-5, device=DEVICE
):
    """Benchmark RMS Norm backward pass."""
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == "triton":
            return rms_norm(x, w_shape, weight, eps)
        if provider == "torch":
            return pytorch_rms_norm(x, weight, eps)
        if provider == "torch_compile":
            return pytorch_rms_norm_compiled(x, weight, eps)

    # backward pass
    if mode == "backward":
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            quantiles=quantiles,
            grad_to_none=[x],
            rep=500,
        )

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    # Run correctness tests
    print("Testing RMS Norm correctness...")
    test_rms_norm(1151, 8192, torch.float16)
    print("✓ All tests passed!")

    # Run benchmarks
    print("\nRunning forward pass benchmark...")
    result = bench_rms_norm_forward.run(
        print_data=True, show_plots=True, save_path=".", return_df=True
    )
    print(result)

    # print("\nRunning backward pass benchmark...")
    # bench_rms_norm_backward.run(print_data=True, show_plots=True)
