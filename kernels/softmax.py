import triton
import triton.language as tl
import torch


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
        x_2d, output, n_rows, n_cols, BLOCK_SIZE
    )

    return output.view(original_shape)


def pytorch_softmax(x, dim=-1):
    """PyTorch reference implementation for comparison."""
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
