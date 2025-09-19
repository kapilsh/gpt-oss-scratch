from matplotlib import pyplot as plt
import triton
import triton.language as tl
import torch
from loguru import logger

# Test device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.style.use("seaborn-v0_8")


@triton.jit
def softmax_forward(
    input_ptr,  # pointer to [n_rows, n_cols]
    output_ptr,  # pointer to [n_rows, n_cols]
    n_rows: tl.constexpr,  # number of rows
    n_cols: tl.constexpr,  # number of columns (feature dim)
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)

    # Base pointers for this row
    in_row_ptr = input_ptr + row_id * n_cols
    out_row_ptr = output_ptr + row_id * n_cols

    # ---- Fast path: entire row fits in BLOCK_SIZE ----
    if n_cols <= BLOCK_SIZE:
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        vals = tl.load(in_row_ptr + col_offsets, mask=mask, other=-float("inf")).to(
            tl.float32
        )

        row_max = tl.max(vals, axis=0)
        vals_stable = vals - row_max
        numer = tl.exp(vals_stable)
        denom = tl.sum(numer, axis=0)
        out = numer / denom

        tl.store(out_row_ptr + col_offsets, out, mask=mask)
        return

    # ---- Tiled path: handle rows larger than BLOCK_SIZE ----
    # ==== Reduction Pass ====
    # Pass 1: compute row max
    row_max = -float("inf")
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(in_row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(vals, axis=0))

    # ==== Reduction Pass ====
    # Pass 2: compute exp-sum
    row_sum = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(in_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        row_sum += tl.sum(tl.exp(vals - row_max), axis=0)

    # ==== Pointwise pass ====
    # Pass 3: normalize + write
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(in_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = tl.exp(vals - row_max) / row_sum
        tl.store(out_row_ptr + cols, out, mask=mask)


def softmax(x, dim=-1):
    """
    Fused softmax implementation using Triton.

    Args:
        x: Input tensor of shape (..., N) where N is the dimension to apply softmax
        dim: Dimension to apply softmax (only dim=-1 supported currently)

    Returns:
        Softmax output tensor of same shape as input
    """
    if dim != -1:
        raise NotImplementedError("Only dim=-1 is currently supported")

    # Flatten to 2D for kernel
    original_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape

    # Allocate output
    output = torch.empty_like(x_2d)

    # Choose block size - power of 2, at least 64
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = max(BLOCK_SIZE, 64)

    # Launch kernel
    grid = (n_rows,)
    softmax_forward[grid](
        x_2d, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE  # type: ignore
    )

    return output.view(original_shape)


def pytorch_softmax(x, dim=-1):
    """PyTorch reference implementation for comparison."""
    return torch.nn.functional.softmax(x, dim=dim)


@torch.compile
def pytorch_softmax_compiled(x, dim=-1):
    """PyTorch compiled reference implementation for comparison."""
    return torch.nn.functional.softmax(x, dim=dim)


class Softmax(torch.nn.Module):
    """
    Triton-accelerated Softmax module.

    Args:
        dim: Dimension to apply softmax (only dim=-1 supported currently)
    """

    def __init__(self, dim=-1):
        super().__init__()
        if dim != -1:
            raise NotImplementedError("Only dim=-1 is currently supported")
        self.dim = dim

    def forward(self, x):
        return softmax(x, dim=self.dim)


# ==================== BENCHMARKS ====================


def bench_softmax_forward(M, N, dtype, provider, device=None):
    """Benchmark Softmax forward pass."""
    if device is None:
        device = DEVICE

    # Create data
    x_shape = (M, N)
    x = (
        torch.randn(x_shape, dtype=dtype, device=device) * 2.0
    )  # Reasonable range for softmax

    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():
        if provider == "triton":
            return softmax(x, dim=-1)
        if provider == "torch":
            return pytorch_softmax(x, dim=-1)

    # Forward pass benchmark
    # Calculate bandwidth: input + output data movement
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    ms, min_ms, max_ms = triton.testing.do_bench(  # type: ignore
        y_fwd, quantiles=quantiles, rep=500, return_mode="all"
    )
    return gbps(ms), gbps(max_ms), gbps(min_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i if i < 12 else 512 * (i - 6) for i in range(6, 30)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[
            ("#1f77b4", "-", "o"),
            ("#2ca02c", "-.", "^"),
        ],
        ylabel="GB/s",
        plot_name="softmax-forward",
        args={"M": 4096, "dtype": torch.float16},
    )
)
def bench_softmax_small_batch(M, N, dtype, provider):
    return bench_softmax_forward(M, N, dtype, provider)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[2**i for i in range(6, 16)],  # 64 to 32K batch size
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[
            ("#1f77b4", "-", "o"),
            ("#2ca02c", "-.", "^"),
        ],
        ylabel="GB/s",
        plot_name="softmax-batch-scaling",
        args={
            "N": 32000,
            "dtype": torch.float16,
        },  # Large vocabulary like language models
    )
)
def bench_softmax_batch_scaling(M, N, dtype, provider):
    return bench_softmax_forward(M, N, dtype, provider)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32000,
            50257,
            100000,
        ],  # Common vocab sizes
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[
            ("#1f77b4", "-", "o"),
            ("#2ca02c", "-.", "^"),
        ],
        ylabel="GB/s",
        plot_name="softmax-vocab-scaling",
        args={"M": 512, "dtype": torch.float16},  # Typical batch*seq_len
    )
)
def bench_softmax_vocab_scaling(M, N, dtype, provider):
    return bench_softmax_forward(M, N, dtype, provider)


# ==================== BENCHMARK RUNNER ====================


def run_performance_benchmarks(print_data=True):
    """Run comprehensive performance benchmarks for Softmax."""
    logger.info("Starting Softmax performance benchmarks...")

    # logger.info("Running small batch benchmark...")
    # small_batch_results = bench_softmax_small_batch.run(
    #     print_data=print_data, return_df=True, show_plots=True
    # )
    # logger.success("Small batch benchmark completed successfully!")

    logger.info("Running batch scaling benchmark...")
    batch_scaling_results = bench_softmax_batch_scaling.run(
        print_data=print_data, return_df=True, show_plots=True
    )
    logger.success("Batch scaling benchmark completed successfully!")

    # logger.info("Running vocabulary scaling benchmark...")
    # vocab_scaling_results = bench_softmax_vocab_scaling.run(
    #     print_data=print_data, return_df=True, show_plots=True
    # )
    # logger.success("Vocabulary scaling benchmark completed successfully!")

    logger.success("All Softmax performance benchmarks completed successfully!")
    # return small_batch_results, batch_scaling_results, vocab_scaling_results


if __name__ == "__main__":
    run_performance_benchmarks()
