#!/usr/bin/env python3

import sys
import torch
import click
from loguru import logger

# Import the existing softmax kernel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from softmax import softmax, softmax_tutorial

# Ensure CUDA is available
if not torch.cuda.is_available():
    logger.error("CUDA is not available. This script requires a CUDA-capable GPU.")
    sys.exit(1)

DEVICE = torch.device("cuda")


def benchmark_softmax(
    M, N, dtype=torch.float16, warmup_iters=10, bench_iters=100, backend="triton"
):
    """
    Benchmark softmax kernel with specific M and N values.

    Args:
        M: Number of rows (batch dimension)
        N: Number of columns (feature dimension)
        dtype: Data type for tensors
        warmup_iters: Number of warmup iterations
        bench_iters: Number of benchmark iterations
        backend: Either "triton" or "torch"
    """
    logger.info(f"Benchmarking {backend} softmax with M={M}, N={N}, dtype={dtype}")

    # Create input tensor
    x = torch.randn(M, N, dtype=dtype, device=DEVICE) * 2.0

    # Select softmax function
    if backend == "triton":
        softmax_fn = softmax
    elif backend == "triton_tutorial":
        softmax_fn = softmax_tutorial
    elif backend == "torch":
        softmax_fn = lambda x: torch.nn.functional.softmax(x, dim=-1)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Warmup
    logger.info(f"Warming up for {warmup_iters} iterations...")
    for _ in range(warmup_iters):
        _ = softmax_fn(x)
    torch.cuda.synchronize()

    # Benchmark
    logger.info(f"Running benchmark for {bench_iters} iterations...")

    # Collect individual timings for noise reduction
    times_ms = []

    for _ in range(bench_iters):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        output = softmax_fn(x)
        end_event.record()

        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))

    # Remove top and bottom 10% to reduce noise
    times_ms.sort()
    n_remove = int(0.1 * len(times_ms))
    if n_remove > 0:
        trimmed_times = times_ms[n_remove:-n_remove]
        logger.info(f"Trimmed {n_remove} outliers from each end ({2*n_remove} total)")
    else:
        trimmed_times = times_ms

    avg_time_ms = sum(trimmed_times) / len(trimmed_times)

    # Calculate metrics
    num_elements = M * N
    bytes_accessed = 2 * num_elements * x.element_size()  # input + output
    bandwidth_gbps = (bytes_accessed * 1e-9) / (avg_time_ms * 1e-3)

    logger.success(f"Benchmark completed:")
    logger.info(f"  Average time per iteration: {avg_time_ms:.4f} ms")
    logger.info(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
    logger.info(f"  Total elements: {num_elements:,}")
    logger.info(f"  Memory accessed: {bytes_accessed / 1e9:.2f} GB")

    return output, avg_time_ms, bandwidth_gbps


@click.command()
@click.option("--M", type=int, required=True, help="Number of rows (batch dimension)")
@click.option(
    "--N", type=int, required=True, help="Number of columns (feature dimension)"
)
@click.option(
    "--backend",
    type=click.Choice(["triton", "triton_tutorial", "torch"]),
    default="triton",
    help="Backend to use: triton (custom kernel), triton_tutorial (tutorial kernel), or torch (PyTorch)",
)
@click.option(
    "--dtype",
    type=click.Choice(["float16", "float32"]),
    default="float16",
    help="Data type for tensors",
)
@click.option("--warmup", type=int, default=10, help="Number of warmup iterations")
@click.option("--iters", type=int, default=100, help="Number of benchmark iterations")
@click.option(
    "--profile-only", is_flag=True, help="Run only a single iteration for profiling"
)
def main(m, n, backend, dtype, warmup, iters, profile_only):
    """Benchmark softmax kernel for ncu profiling."""

    # Convert dtype string to torch dtype
    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    dtype_tensor = dtype_map[dtype]

    logger.info(f"Starting {backend} softmax benchmark with parameters:")
    logger.info(f"  Backend: {backend}")
    logger.info(f"  M (rows): {m}")
    logger.info(f"  N (cols): {n}")
    logger.info(f"  dtype: {dtype}")
    logger.info(f"  warmup iterations: {warmup}")
    logger.info(f"  benchmark iterations: {iters}")

    if profile_only:
        logger.info("Profile-only mode: running single iteration for ncu profiling")
        x = torch.randn(m, n, dtype=dtype_tensor, device=DEVICE) * 2.0

        if backend == "triton":
            output = softmax(x)
        elif backend == "triton_tutorial":
            output = softmax_tutorial(x)
        else:  # torch
            output = torch.nn.functional.softmax(x, dim=-1)

        torch.cuda.synchronize()
        logger.success("Single iteration completed for profiling")
    else:
        # Run full benchmark
        output, avg_time_ms, bandwidth_gbps = benchmark_softmax(
            m, n, dtype_tensor, warmup, iters, backend
        )

    logger.success("Benchmark completed successfully!")


if __name__ == "__main__":
    main()
