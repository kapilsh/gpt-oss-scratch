#!/usr/bin/env python3

import sys
import torch
import click
from loguru import logger
import triton.profiler as proton

# Import the existing softmax kernel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from softmax import softmax, pytorch_softmax

# Ensure CUDA is available
if not torch.cuda.is_available():
    logger.error("CUDA is not available. This script requires a CUDA-capable GPU.")
    sys.exit(1)

DEVICE = torch.device("cuda")


def profile_softmax_with_proton(
    M, N, dtype=torch.float16, backend="triton", profile_name="softmax_profile"
):
    """
    Profile softmax kernel using Triton's Proton profiler.

    Args:
        M: Number of rows (batch dimension)
        N: Number of columns (feature dimension)
        dtype: Data type for tensors
        backend: Either "triton" or "torch"
        profile_name: Name for the profile output
    """
    logger.info(f"Profiling {backend} softmax with M={M}, N={N}, dtype={dtype}")
    logger.info(f"Profile will be saved as: {profile_name}")

    # Select softmax function
    if backend == "triton":
        softmax_fn = softmax
    elif backend == "torch":
        softmax_fn = pytorch_softmax
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Create input tensor
    x = torch.randn(M, N, dtype=dtype, device=DEVICE) * 2.0

    # Warmup run
    logger.info("Running warmup...")
    for _ in range(5):
        _ = softmax_fn(x)

    # Profile the kernel execution
    logger.info("Starting Proton profiling...")

    _ = proton.start(f"{profile_name}", backend="cupti")

    for _ in range(1000):
        output = softmax_fn(x)

    proton.finalize()

    # Move the generated file to the desired name with backend suffix
    import shutil
    import os
    final_filename = f"{profile_name}_{backend}.hatchet"
    if os.path.exists("proton.hatchet"):
        shutil.move("proton.hatchet", final_filename)
        logger.success(f"Profiling completed! Profile saved to: {final_filename}")
    else:
        logger.warning("proton.hatchet not found, profile may not have been generated")

    logger.info(f"To view the profile, use: proton-viewer {final_filename}")

    return output


@click.command()
@click.option("--M", type=int, required=True, help="Number of rows (batch dimension)")
@click.option(
    "--N", type=int, required=True, help="Number of columns (feature dimension)"
)
@click.option(
    "--backend",
    type=click.Choice(["triton", "torch"]),
    default="triton",
    help="Backend to use: triton (custom kernel) or torch (PyTorch)",
)
@click.option(
    "--dtype",
    type=click.Choice(["float16", "float32"]),
    default="float16",
    help="Data type for tensors",
)
@click.option(
    "--profile-name",
    type=str,
    default="softmax_profile",
    help="Name for the profile output file",
)
def main(m, n, backend, dtype, profile_name):
    """Profile softmax kernel using Triton's Proton profiler."""

    # Convert dtype string to torch dtype
    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    dtype_tensor = dtype_map[dtype]

    logger.info(f"Starting {backend} softmax profiling with parameters:")
    logger.info(f"  Backend: {backend}")
    logger.info(f"  M (rows): {m}")
    logger.info(f"  N (cols): {n}")
    logger.info(f"  dtype: {dtype}")
    logger.info(f"  Profile name: {profile_name}")

    # Run profiling
    output = profile_softmax_with_proton(m, n, dtype_tensor, backend, profile_name)

    logger.success("Profiling completed successfully!")
    final_filename = f"{profile_name}_{backend}.hatchet"
    logger.info(f"Profile file: {final_filename}")
    logger.info(f"View with: proton-viewer {final_filename}")


if __name__ == "__main__":
    main()
