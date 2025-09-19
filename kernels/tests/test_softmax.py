"""
Unit Tests for Softmax Triton Kernel
====================================

Comprehensive pytest-based correctness tests comparing Triton Softmax implementation
against PyTorch reference implementation across various configurations.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger
import sys
from pathlib import Path

# Add the parent directory to Python path to import softmax module
sys.path.insert(0, str(Path(__file__).parent.parent))

from softmax import softmax, pytorch_softmax, Softmax

# Test device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test tolerances
FORWARD_ATOL = 1e-4
FORWARD_RTOL = 1e-3

# Sum tolerances (different for float16 vs float32 due to precision)
SUM_ATOL_F32 = 1e-4    # Float32 precision
SUM_ATOL_F16 = 1e-3    # Float16 has lower precision


def get_sum_tolerance(dtype):
    """Get appropriate sum tolerance based on dtype."""
    if dtype == torch.float16:
        return SUM_ATOL_F16
    else:
        return SUM_ATOL_F32


class TestSoftmaxCorrectness:
    """Test suite for Softmax correctness across different configurations."""

    @pytest.mark.parametrize("M,N", [
        (1, 16),        # Single sample, very small feature dim
        (1, 64),        # Single sample, small feature dim
        (1, 128),       # Single sample, small-medium feature dim
        (1, 256),       # Single sample, medium feature dim
        (1, 512),       # Single sample, medium-large feature dim
        (1, 1024),      # Single sample, large feature dim
        (1, 2048),      # Single sample, very large feature dim
        (2, 128),       # Small batch, small feature dim
        (4, 256),       # Small batch, medium feature dim
        (8, 512),       # Small batch, large feature dim
        (16, 1024),     # Medium batch, large feature dim
        (32, 512),      # Medium batch, medium-large feature dim
        (64, 256),      # Large batch, medium feature dim
        (128, 128),     # Large batch, small-medium feature dim
        (256, 512),     # Large batch, large feature dim
        (512, 1024),    # Very large batch, large feature dim
        (1024, 2048),   # Stress test - very large dimensions
    ])
    @pytest.mark.parametrize("dtype", [
        torch.float16,
        torch.float32,
    ])
    def test_forward_correctness(self, M, N, dtype):
        """Test forward pass correctness against PyTorch implementation."""
        logger.info(f"Testing forward pass: M={M}, N={N}, dtype={dtype}")


        # Create test data
        torch.manual_seed(42)
        x_shape = (M, N)

        # Initialize input tensor with controlled range to avoid overflow
        x = torch.randn(x_shape, dtype=dtype, device=DEVICE)
        # Scale to reasonable range for softmax
        x = x * 2.0  # Keep in reasonable range to avoid overflow

        # Forward pass - Triton
        y_triton = softmax(x, dim=-1)

        # Forward pass - PyTorch reference
        y_pytorch = pytorch_softmax(x, dim=-1)

        # Check correctness
        max_diff = torch.max(torch.abs(y_triton - y_pytorch)).item()
        assert torch.allclose(y_triton, y_pytorch, atol=FORWARD_ATOL, rtol=FORWARD_RTOL), \
            f"Triton vs PyTorch forward mismatch: max_diff={max_diff:.6f}"

        # Check that outputs are valid probabilities
        # Sum should be approximately 1 along last dimension
        triton_sums = torch.sum(y_triton, dim=-1)
        pytorch_sums = torch.sum(y_pytorch, dim=-1)

        sum_tol = get_sum_tolerance(dtype)
        assert torch.allclose(triton_sums, torch.ones_like(triton_sums), atol=sum_tol), \
            f"Triton softmax output doesn't sum to 1 (max dev: {torch.max(torch.abs(triton_sums - 1.0)):.6f})"
        assert torch.allclose(pytorch_sums, torch.ones_like(pytorch_sums), atol=sum_tol), \
            f"PyTorch softmax output doesn't sum to 1 (max dev: {torch.max(torch.abs(pytorch_sums - 1.0)):.6f})"

        # Check that all values are non-negative
        assert torch.all(y_triton >= 0), "Triton softmax output contains negative values"
        assert torch.all(y_pytorch >= 0), "PyTorch softmax output contains negative values"

        logger.success(f"✓ Forward test passed: M={M}, N={N}, dtype={dtype}")

    @pytest.mark.parametrize("batch_size,seq_length,vocab_size", [
        # Small vocabulary tests - can test all batch sizes
        (1, 32, 128), (1, 128, 512), (1, 512, 1024), (1, 1024, 1024),
        (4, 32, 128), (4, 128, 512), (4, 512, 1024), (4, 1024, 1024),
        (16, 32, 128), (16, 128, 512), (16, 512, 1024), (16, 1024, 1024),
        (64, 32, 128), (64, 128, 512), (64, 256, 1024),

        # Medium vocabulary tests - limit batch sizes to avoid memory issues
        (1, 128, 4096), (1, 512, 4096), (1, 1024, 4096),
        (4, 128, 4096), (4, 256, 4096),
        (16, 128, 4096), (16, 256, 4096),
        (32, 128, 4096),

        # Large vocabulary tests - only small batches
        (1, 32, 32000), (1, 128, 32000), (1, 512, 32000),
        (2, 128, 32000), (2, 256, 32000),
        (4, 64, 32000), (4, 128, 32000),
    ])
    def test_transformer_dimensions(self, batch_size, seq_length, vocab_size):
        """Test typical transformer model dimensions (e.g., attention scores, language modeling)."""
        logger.info(f"Testing transformer dims: batch={batch_size}, seq={seq_length}, vocab={vocab_size}")

        # Create typical transformer input shapes
        # For attention: (batch_size, num_heads, seq_length, seq_length)
        # For language modeling: (batch_size, seq_length, vocab_size)

        # Test language modeling head dimensions
        M = batch_size * seq_length
        N = vocab_size

        # Use float16 for efficiency in large tests
        dtype = torch.float16 if vocab_size > 1024 else torch.float32

        torch.manual_seed(42)
        x_shape = (M, N)

        # Create logits with realistic range for language modeling
        x = torch.randn(x_shape, dtype=dtype, device=DEVICE) * 3.0

        # Forward pass
        y_triton = softmax(x, dim=-1)
        y_pytorch = pytorch_softmax(x, dim=-1)

        # Check correctness
        assert torch.allclose(y_triton, y_pytorch, atol=FORWARD_ATOL, rtol=FORWARD_RTOL), \
            f"Transformer dims mismatch: batch={batch_size}, seq={seq_length}, vocab={vocab_size}"

        # Verify probability properties
        sums = torch.sum(y_triton, dim=-1)
        sum_tol = get_sum_tolerance(dtype)
        assert torch.allclose(sums, torch.ones_like(sums), atol=sum_tol), \
            f"Softmax doesn't sum to 1 for transformer dimensions (max dev: {torch.max(torch.abs(sums - 1.0)):.6f})"

        logger.success(f"✓ Transformer test passed: {batch_size}x{seq_length}x{vocab_size}")

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        logger.info("Testing edge cases...")

        dtype = torch.float32

        # Test minimum supported dimensions
        M, N = 1, 1
        torch.manual_seed(42)
        x = torch.randn((M, N), dtype=dtype, device=DEVICE)

        y_triton = softmax(x, dim=-1)
        y_pytorch = pytorch_softmax(x, dim=-1)

        assert torch.allclose(y_triton, y_pytorch, atol=1e-5, rtol=1e-5)
        assert torch.allclose(y_triton, torch.ones_like(y_triton), atol=1e-6)  # Should be [1.0]

        # Test with identical values (should give uniform distribution)
        N = 10
        x_uniform = torch.ones((1, N), dtype=dtype, device=DEVICE) * 5.0
        y_triton_uniform = softmax(x_uniform, dim=-1)
        y_pytorch_uniform = pytorch_softmax(x_uniform, dim=-1)

        expected_uniform = torch.full((1, N), 1.0/N, dtype=dtype, device=DEVICE)
        assert torch.allclose(y_triton_uniform, expected_uniform, atol=1e-5)
        assert torch.allclose(y_pytorch_uniform, expected_uniform, atol=1e-5)

        # Test with zero input
        x_zero = torch.zeros((2, 5), dtype=dtype, device=DEVICE)
        y_triton_zero = softmax(x_zero, dim=-1)
        y_pytorch_zero = pytorch_softmax(x_zero, dim=-1)

        assert torch.allclose(y_triton_zero, y_pytorch_zero, atol=1e-6, rtol=1e-6)
        # Each row should be uniform [0.2, 0.2, 0.2, 0.2, 0.2]
        expected_zero = torch.full((2, 5), 0.2, dtype=dtype, device=DEVICE)
        assert torch.allclose(y_triton_zero, expected_zero, atol=1e-5)

        logger.success("✓ Edge case tests passed")

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        logger.info("Testing numerical stability...")

        M, N = 32, 256
        dtype = torch.float32

        torch.manual_seed(42)

        # Test with large values (potential overflow)
        x_large = torch.randn((M, N), dtype=dtype, device=DEVICE) * 10 + 50
        y_triton_large = softmax(x_large, dim=-1)
        y_pytorch_large = pytorch_softmax(x_large, dim=-1)

        assert torch.allclose(y_triton_large, y_pytorch_large, atol=1e-3, rtol=1e-2), \
            "Large values test failed"

        # Verify no NaN or Inf
        assert not torch.isnan(y_triton_large).any(), "Large values produced NaN in Triton"
        assert not torch.isinf(y_triton_large).any(), "Large values produced Inf in Triton"

        # Test with very negative values (potential underflow)
        x_negative = torch.randn((M, N), dtype=dtype, device=DEVICE) * 2 - 20
        y_triton_negative = softmax(x_negative, dim=-1)
        y_pytorch_negative = pytorch_softmax(x_negative, dim=-1)

        assert torch.allclose(y_triton_negative, y_pytorch_negative, atol=1e-5, rtol=1e-3), \
            "Very negative values test failed"

        # Test with mixed large positive and negative values
        x_mixed = torch.randn((M, N), dtype=dtype, device=DEVICE) * 20
        y_triton_mixed = softmax(x_mixed, dim=-1)
        y_pytorch_mixed = pytorch_softmax(x_mixed, dim=-1)

        assert torch.allclose(y_triton_mixed, y_pytorch_mixed, atol=1e-3, rtol=1e-2), \
            "Mixed extreme values test failed"

        logger.success("✓ Numerical stability tests passed")

    def test_different_block_sizes(self):
        """Test that kernel works correctly for various block size scenarios."""
        logger.info("Testing different block size scenarios...")

        dtype = torch.float32
        torch.manual_seed(42)

        # Test dimensions that exercise different block size paths
        test_cases = [
            (1, 32),    # Small - fits in single block
            (1, 63),    # Just under power of 2
            (1, 64),    # Exact power of 2
            (1, 65),    # Just over power of 2
            (1, 127),   # Just under next power of 2
            (1, 128),   # Next power of 2
            (1, 129),   # Just over, requires tiling
            (1, 1023),  # Large, just under power of 2
            (1, 1024),  # Large, exact power of 2
            (1, 1025),  # Large, just over, requires tiling
            (4, 2048),  # Multiple rows with large columns
        ]

        for M, N in test_cases:
            x = torch.randn((M, N), dtype=dtype, device=DEVICE) * 2.0

            y_triton = softmax(x, dim=-1)
            y_pytorch = pytorch_softmax(x, dim=-1)

            assert torch.allclose(y_triton, y_pytorch, atol=FORWARD_ATOL, rtol=FORWARD_RTOL), \
                f"Block size test failed for {M}x{N}"

            # Verify probability properties
            sums = torch.sum(y_triton, dim=-1)
            sum_tol = get_sum_tolerance(dtype)
            assert torch.allclose(sums, torch.ones_like(sums), atol=sum_tol), \
                f"Softmax doesn't sum to 1 for {M}x{N} (max dev: {torch.max(torch.abs(sums - 1.0)):.6f})"

        logger.success("✓ Block size tests passed")

    def test_multidimensional_input(self):
        """Test that softmax works correctly with multidimensional inputs."""
        logger.info("Testing multidimensional inputs...")

        dtype = torch.float32
        torch.manual_seed(42)

        # Test various multidimensional shapes
        test_shapes = [
            (2, 3, 128),           # 3D tensor
            (2, 4, 3, 64),         # 4D tensor (typical for attention)
            (8, 12, 512, 512),     # Large 4D (attention weights)
            (1, 1, 1, 1024),       # Edge case with many dimensions of size 1
        ]

        for shape in test_shapes:

            x = torch.randn(shape, dtype=dtype, device=DEVICE) * 2.0

            y_triton = softmax(x, dim=-1)
            y_pytorch = pytorch_softmax(x, dim=-1)

            assert torch.allclose(y_triton, y_pytorch, atol=FORWARD_ATOL, rtol=FORWARD_RTOL), \
                f"Multidimensional test failed for shape {shape}"

            # Verify shape preservation
            assert y_triton.shape == shape, f"Shape not preserved for {shape}"

            # Verify probability properties along last dimension
            sums = torch.sum(y_triton, dim=-1)
            expected_shape = shape[:-1]  # All dimensions except last
            assert sums.shape == expected_shape, f"Sum shape incorrect for {shape}"
            sum_tol = get_sum_tolerance(dtype)
            assert torch.allclose(sums, torch.ones_like(sums), atol=sum_tol), \
                f"Softmax doesn't sum to 1 for shape {shape} (max dev: {torch.max(torch.abs(sums - 1.0)):.6f})"

        logger.success("✓ Multidimensional input tests passed")


class TestSoftmaxModule:
    """Test the nn.Module wrapper."""

    def test_module_interface(self):
        """Test that the nn.Module interface works correctly."""
        logger.info("Testing nn.Module interface...")

        M, N = 16, 512
        dtype = torch.float32
        torch.manual_seed(42)

        x = torch.randn((M, N), dtype=dtype, device=DEVICE) * 2.0

        # Test Triton module
        softmax_module = Softmax(dim=-1)
        y_module = softmax_module(x)

        # Test against functional version
        y_functional = softmax(x, dim=-1)

        # Test against PyTorch
        y_pytorch = pytorch_softmax(x, dim=-1)

        assert torch.allclose(y_module, y_functional, atol=1e-6), \
            "Module and functional versions don't match"
        assert torch.allclose(y_module, y_pytorch, atol=FORWARD_ATOL, rtol=FORWARD_RTOL), \
            "Module doesn't match PyTorch"

        logger.success("✓ nn.Module interface tests passed")

    def test_module_parameters(self):
        """Test module parameter handling."""
        logger.info("Testing module parameters...")

        softmax_module = Softmax(dim=-1)

        # Should have no parameters
        assert len(list(softmax_module.parameters())) == 0, \
            "Softmax module should have no parameters"

        # Should be able to move to device
        if torch.cuda.is_available():
            softmax_module = softmax_module.cuda()
            assert next(softmax_module.parameters(), torch.tensor(0)).device.type == "cpu", \
                "Module device handling"

        logger.success("✓ Module parameter tests passed")


if __name__ == "__main__":
    # Run basic tests
    logger.info("Running Softmax correctness tests...")
    pytest.main([__file__, "-v", "--tb=short"])