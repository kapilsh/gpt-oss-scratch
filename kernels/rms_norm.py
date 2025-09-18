"""
RMS Normalization Triton Kernel
===============================

Optimized Triton implementation of RMS Normalization with forward and backward passes.

RMS Normalization computes:
y = (x / sqrt(mean(x^2) + eps)) * weight
"""

import torch
import triton
import torch.nn.functional as F
import triton.language as tl
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt

# Configure torch settings for compiled functions
try:
    torch._functorch.config.donated_buffer = False  # type: ignore
except AttributeError:
    # Handle cases where _functorch is not available
    pass

plt.style.use("seaborn-v0_8")

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
        input_values = tl.load(
            input_ptr + col_indices, mask=col_indices < feature_dim, other=0.0
        ).to(tl.float32)
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
        input_values = tl.load(input_ptr + col_indices, mask=valid_mask, other=0.0).to(
            tl.float32
        )

        normalized_values = input_values * reciprocal_std
        output_values = normalized_values * weight_values

        # Write final output
        tl.store(output_ptr + col_indices, output_values, mask=valid_mask)


@triton.jit
def _rms_norm_bwd_dx_fused(
    dx_ptr,  # pointer to the input gradient
    dy_ptr,  # pointer to the output gradient
    dw_ptr,  # pointer to the partial sum of weights gradient
    input_ptr,  # pointer to the input
    weight_ptr,  # pointer to the weights
    rstd_ptr,  # pointer to the reciprocal standard deviation
    lock_ptr,  # pointer to the lock
    row_stride,  # stride for moving to the next row
    feature_dim,  # number of features (columns) in input
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map the program id to the elements to compute
    row_idx = tl.program_id(0)
    col_indices = tl.arange(0, BLOCK_SIZE_N)
    valid_mask = col_indices < feature_dim

    input_ptr += row_idx * row_stride
    dy_ptr += row_idx * row_stride
    dx_ptr += row_idx * row_stride

    # Offset locks and weights gradient pointer for parallel reduction
    lock_id = row_idx % GROUP_SIZE_M
    lock_ptr += lock_id
    count_ptr = lock_ptr + GROUP_SIZE_M
    dw_ptr = dw_ptr + lock_id * feature_dim + col_indices

    # Load data to SRAM
    input_values = tl.load(input_ptr + col_indices, mask=valid_mask, other=0).to(
        tl.float32
    )
    dy_values = tl.load(dy_ptr + col_indices, mask=valid_mask, other=0).to(tl.float32)
    weight_values = tl.load(weight_ptr + col_indices, mask=valid_mask).to(tl.float32)
    reciprocal_std = tl.load(rstd_ptr + row_idx)

    # Compute gradients for RMS norm
    normalized_values = input_values * reciprocal_std
    weight_dy = weight_values * dy_values

    # Apply masking for valid elements
    normalized_masked = tl.where(valid_mask, normalized_values, 0.0)
    weight_dy_masked = tl.where(valid_mask, weight_dy, 0.0)

    # Compute correction term for RMS norm backward pass
    correction = (
        tl.sum(weight_dy_masked * normalized_masked * input_values, axis=0)
        / feature_dim
    )
    dx_values = (
        weight_dy_masked - input_values * reciprocal_std * correction
    ) * reciprocal_std

    # Write input gradients
    tl.store(dx_ptr + col_indices, dx_values, mask=valid_mask)

    # Accumulate partial sums for weight gradients
    partial_dw = (dy_values * normalized_values).to(weight_values.dtype)

    # Atomic operations for weight gradient accumulation
    while tl.atomic_cas(lock_ptr, 0, 1) == 1:
        pass
    count = tl.load(count_ptr)
    if count == 0:
        tl.atomic_xchg(count_ptr, 1)
    else:
        partial_dw += tl.load(dw_ptr, mask=valid_mask)
    tl.store(dw_ptr, partial_dw, mask=valid_mask)

    # Release the lock
    tl.debug_barrier()
    tl.atomic_xchg(lock_ptr, 0)


@triton.jit
def _rms_norm_bwd_dw(
    dw_partial_ptr,  # pointer to the partial sum of weights gradient
    dw_final_ptr,  # pointer to the final weights gradient
    num_groups,  # number of groups (GROUP_SIZE_M)
    feature_dim,  # number of features
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map the program id to the elements to compute
    program_id = tl.program_id(0)
    col_indices = program_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate through the rows of partial gradients to sum them
    for group_offset in range(0, num_groups, BLOCK_SIZE_M):
        row_indices = group_offset + tl.arange(0, BLOCK_SIZE_M)
        valid_mask = (row_indices[:, None] < num_groups) & (
            col_indices[None, :] < feature_dim
        )
        offsets = row_indices[:, None] * feature_dim + col_indices[None, :]
        dw_accumulator += tl.load(dw_partial_ptr + offsets, mask=valid_mask, other=0.0)

    # Sum across groups and write final result
    dw_final = tl.sum(dw_accumulator, axis=0)
    tl.store(dw_final_ptr + col_indices, dw_final, mask=col_indices < feature_dim)


class RMSNorm(torch.autograd.Function):
    """
    Triton-optimized RMS Normalization with automatic differentiation support.
    """

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps):
        # Allocate output tensor
        y = torch.empty_like(x)

        # Reshape input to 2D for processing
        x_reshaped = x.reshape(-1, x.shape[-1])
        batch_size, feature_dim = x_reshaped.shape
        rstd = torch.empty((batch_size,), dtype=torch.float32, device=x.device)

        # Determine optimal block size (limited by 64KB per feature)
        max_fused_size = 65536 // x.element_size()
        BLOCK_SIZE = min(max_fused_size, triton.next_power_of_2(feature_dim))

        if feature_dim > BLOCK_SIZE:
            raise RuntimeError("This RMS norm doesn't support feature dim >= 64KB.")

        # Heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        # Launch forward kernel
        _rms_norm_fwd_fused[(batch_size,)](  # type: ignore
            x_reshaped,
            y,
            weight,
            rstd,
            x_reshaped.stride(0),
            feature_dim,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,  # type: ignore
        )

        # Save tensors for backward pass
        ctx.save_for_backward(x, weight, rstd)
        ctx.block_size = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        feature_dim = weight.shape[0]

        # Heuristics for parallel reduction group size
        if feature_dim <= 1024:
            group_size = 256
        elif feature_dim <= 4096:
            group_size = 128
        elif feature_dim <= 8192:
            group_size = 96
        else:
            group_size = 64

        # Allocate output tensors
        locks = torch.zeros(2 * group_size, dtype=torch.int32, device=weight.device)
        dw_partial = torch.zeros(
            (group_size, feature_dim), dtype=x.dtype, device=weight.device
        )
        dw_final = torch.empty((feature_dim,), dtype=weight.dtype, device=weight.device)
        dx = torch.empty_like(dy)

        # Reshape inputs for processing
        x_reshaped = x.reshape(-1, x.shape[-1])
        batch_size, feature_dim = x_reshaped.shape

        # Launch backward kernel for input gradients and partial weight gradients
        _rms_norm_bwd_dx_fused[(batch_size,)](  # type: ignore
            dx,
            dy,
            dw_partial,
            x,
            weight,
            rstd,
            locks,
            x_reshaped.stride(0),
            feature_dim,
            BLOCK_SIZE_N=ctx.block_size,
            GROUP_SIZE_M=group_size,  # type: ignore
        )

        # Launch kernel to reduce partial weight gradients
        grid_size = lambda meta: (triton.cdiv(feature_dim, meta["BLOCK_SIZE_N"]),)
        _rms_norm_bwd_dw[grid_size](  # type: ignore
            dw_partial,
            dw_final,
            min(group_size, batch_size),
            feature_dim,
            BLOCK_SIZE_M=32,  # type: ignore
            BLOCK_SIZE_N=128,  # type: ignore
        )

        return dx, None, dw_final, None


# Convenience function for using the optimized RMS norm
rms_norm = RMSNorm.apply


def pytorch_rms_norm(x, weight, eps=1e-5):
    """PyTorch reference implementation of RMS Norm."""
    return F.rms_norm(x, (x.size(-1),), weight=weight, eps=eps)


# Compiled version for performance comparison
pytorch_rms_norm_compiled = torch.compile(pytorch_rms_norm)


def test_rms_norm_correctness(
    batch_size=1151, feature_dim=8192, dtype=torch.float16, eps=1e-5, device=None
):
    """Test correctness of Triton RMS Norm vs PyTorch implementation."""
    if device is None:
        device = DEVICE

    # Create test data
    weight = torch.rand(feature_dim, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(batch_size, feature_dim, dtype=dtype, device=device)
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)

    # Forward pass comparison
    y_triton: torch.Tensor = rms_norm(x, (feature_dim,), weight, eps)  # type: ignore
    y_pytorch = pytorch_rms_norm(x, weight, eps)
    y_compiled = pytorch_rms_norm_compiled(x, weight, eps)

    # Backward pass - Triton
    y_triton.backward(dy, retain_graph=True)  # type: ignore
    assert (
        x.grad is not None and weight.grad is not None
    ), "Gradients should be computed"
    dx_triton = x.grad.clone()
    dw_triton = weight.grad.clone()
    x.grad, weight.grad = None, None

    # Backward pass - PyTorch
    y_pytorch.backward(dy, retain_graph=True)
    assert (
        x.grad is not None and weight.grad is not None
    ), "Gradients should be computed"
    dx_pytorch = x.grad.clone()
    dw_pytorch = weight.grad.clone()
    x.grad, weight.grad = None, None

    # Backward pass - Compiled PyTorch
    y_compiled.backward(dy, retain_graph=True)
    assert (
        x.grad is not None and weight.grad is not None
    ), "Gradients should be computed"
    dx_compiled = x.grad.clone()
    dw_compiled = weight.grad.clone()

    # Assertions
    assert torch.allclose(
        y_triton, y_pytorch, atol=1e-2, rtol=0
    ), "Forward pass mismatch: Triton vs PyTorch"
    assert torch.allclose(
        y_triton, y_compiled, atol=1e-2, rtol=0
    ), "Forward pass mismatch: Triton vs Compiled"
    assert torch.allclose(
        dx_triton, dx_pytorch, atol=1e-2, rtol=0
    ), "Input grad mismatch: Triton vs PyTorch"
    assert torch.allclose(
        dx_triton, dx_compiled, atol=1e-2, rtol=0
    ), "Input grad mismatch: Triton vs Compiled"
    assert torch.allclose(
        dw_triton, dw_pytorch, atol=1e-2, rtol=0
    ), "Weight grad mismatch: Triton vs PyTorch"
    assert torch.allclose(
        dw_triton, dw_compiled, atol=1e-2, rtol=0
    ), "Weight grad mismatch: Triton vs Compiled"

    logger.success("All correctness tests passed!")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i if i < 9 else 512 * i for i in range(1, 54)],
        line_arg="provider",
        line_vals=["triton", "torch", "torch_compile"],
        line_names=["Triton", "Torch", "Torch Compile"],
        styles=[
            ("#1f77b4", "-", "o"),
            ("#2ca02c", "-.", "^"),
            ("#d62728", "--", "s"),
        ],
        ylabel="GB/s",
        plot_name="rms-norm-forward",
        args={"M": 4096, "dtype": torch.float16, "mode": "forward"},
    )
)
def bench_rms_norm_forward(
    M, N, dtype, provider, mode="forward", eps=1e-5, device=None
):
    """Benchmark RMS Norm forward pass."""
    if device is None:
        device = DEVICE

    # Create data
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

    # Forward pass benchmark
    if mode == "forward":
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(  # type: ignore
            y_fwd, quantiles=quantiles, rep=500, return_mode="all"
        )

    return gbps(ms), gbps(max_ms), gbps(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg="provider",
        line_vals=["triton", "torch", "torch_compile"],
        line_names=["Triton", "Torch", "Torch Compile"],
        styles=[
            ("#1f77b4", "-"),
            ("#2ca02c", "-."),
            ("#d62728", "--"),
        ],
        ylabel="GB/s",
        plot_name="rms-norm-backward",
        args={"M": 4096, "dtype": torch.float16, "mode": "backward"},
    )
)
def bench_rms_norm_backward(
    M, N, dtype, provider, mode="backward", eps=1e-5, device=None
):
    """Benchmark RMS Norm backward pass."""
    if device is None:
        device = DEVICE

    # Create data
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

    # Backward pass benchmark
    if mode == "backward":

        def backward_fn():
            # Clear gradients first
            if x.grad is not None:
                x.grad = None
            if weight.grad is not None:
                weight.grad = None

            # Get fresh forward pass
            y = y_fwd()

            # For compiled functions, we need create_graph=False, retain_graph=False
            if provider == "torch_compile":
                y.backward(dy, create_graph=False, retain_graph=False)  # type: ignore
            else:
                y.backward(dy, retain_graph=True)  # type: ignore

        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(  # type: ignore
            backward_fn,
            quantiles=quantiles,
            grad_to_none=[x, weight],
            rep=500,
        )

    return gbps(ms), gbps(max_ms), gbps(min_ms)


def run_performance_benchmarks(
    print_data=True,
    include_backward=False,
):
    """Run comprehensive performance benchmarks for RMS Norm."""
    logger.info("Starting RMS Norm performance benchmarks...")

    try:
        logger.info("Running forward pass benchmark...")
        forward_results = bench_rms_norm_forward.run(
            print_data=print_data, return_df=True, show_plots=True
        )
        logger.success("Forward pass benchmark completed successfully!")

        backward_results = None
        if include_backward:
            try:
                logger.info("Running backward pass benchmark...")
                backward_results = bench_rms_norm_backward.run(
                    print_data=print_data, return_df=True, show_plots=True
                )
                logger.success("Backward pass benchmark completed successfully!")
            except Exception as e:
                logger.warning(f"Backward benchmark failed: {e}")
                logger.info("Continuing with forward benchmark only...")

        logger.success("Performance benchmarks completed successfully!")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    # Run correctness test
    logger.info("Testing RMS Norm correctness...")
    test_rms_norm_correctness()

    # Run comprehensive benchmarks with professional plotting
    # Note: backward benchmark is disabled by default due to compilation issues
    logger.info("Running forward pass benchmark with visualization...")
    results = run_performance_benchmarks(
        print_data=True,
        include_backward=False,  # Set to True to enable backward benchmarking
    )
