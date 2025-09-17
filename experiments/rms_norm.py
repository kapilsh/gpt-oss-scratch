"""
RMSNorm Analysis Module

This module provides comprehensive analysis capabilities for RMSNorm and other neural network modules.
Includes FLOP calculations, memory bandwidth benchmarking, and performance analysis tools.
"""

import torch
import torch.nn as nn
import statistics
import itertools
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from tqdm.auto import tqdm


class RMSNorm(nn.Module):
    def __init__(self, embedding_dimension: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.embedding_dimension = embedding_dimension
        self.weight = nn.Parameter(torch.ones(embedding_dimension))

    def forward(self, x: torch.Tensor):
        means = x.pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(means + self.eps)) * self.weight


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking parameters."""

    warmup_runs: int = 10
    benchmark_runs: int = 100
    memory_factor: int = 6  # Memory access factor for bandwidth calculations
    percentile_filter: Tuple[float, float] = (0.1, 0.9)  # Filter outliers


@dataclass
class ParameterRanges:
    """Parameter ranges for comprehensive analysis."""

    batch_sizes: List[int]
    sequence_lengths: List[int]
    embedding_dims: List[int]


class ModuleAnalyzer:
    """
    Generic analyzer for neural network modules.
    Can be extended for different module types (RMSNorm, LayerNorm, etc.).
    """

    def __init__(
        self, module_class: type, flop_calculator: Callable, memory_factor: int = 6
    ):
        """
        Initialize analyzer for a specific module type.

        Args:
            module_class: The PyTorch module class to analyze
            flop_calculator: Function that calculates FLOPs for this module
            memory_factor: Memory access multiplier for bandwidth calculations
        """
        self.module_class = module_class
        self.flop_calculator = flop_calculator
        self.memory_factor = memory_factor

    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()

            return {
                "gpu_name": props.name,
                "total_memory_gb": total_memory / 1e9,
                "allocated_gb": allocated / 1e9,
                "cached_gb": cached / 1e9,
                "available_gb": (total_memory - cached) / 1e9,
                "l2_cache_mb": (
                    props.l2_cache_size / (1024 * 1024)
                    if hasattr(props, "l2_cache_size")
                    else "Unknown"
                ),
            }
        return {}

    def benchmark_module_cuda(
        self,
        module: torch.nn.Module,
        input_tensor: torch.Tensor,
        config: BenchmarkConfig = BenchmarkConfig(),
    ) -> Dict[str, float]:
        """
        Benchmark a module using CUDA events for precise GPU timing.

        Args:
            module: PyTorch module (eager or compiled)
            input_tensor: Input tensor on GPU
            config: Benchmark configuration

        Returns:
            Dictionary with timing statistics and memory bandwidth metrics
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for benchmarking")

        if not input_tensor.is_cuda:
            raise ValueError("Input tensor must be on GPU")

        # Ensure module is on GPU
        module = module.cuda()
        module.eval()

        # Create CUDA events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Warmup phase
        with torch.no_grad():
            for _ in range(config.warmup_runs):
                _ = module(input_tensor)

        torch.cuda.synchronize()

        # Benchmark phase
        times_ms = []
        with torch.no_grad():
            for _ in range(config.benchmark_runs):
                start_event.record()
                output = module(input_tensor)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
                times_ms.append(elapsed_ms)

        # Calculate statistics
        mean_ms = statistics.mean(times_ms)
        median_ms = statistics.median(times_ms)
        min_ms = min(times_ms)
        max_ms = max(times_ms)
        std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

        # Calculate memory bandwidth metrics
        tensor_shape = input_tensor.shape
        total_elements = input_tensor.numel()
        bytes_per_element = input_tensor.element_size()

        total_bytes = self.memory_factor * total_elements * bytes_per_element

        # Calculate GBPS for different timing metrics
        gbps_mean = (total_bytes * 1e-9) / (mean_ms * 1e-3)
        gbps_median = (total_bytes * 1e-9) / (median_ms * 1e-3)
        gbps_best = (total_bytes * 1e-9) / (min_ms * 1e-3)

        return {
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "std_ms": std_ms,
            "gbps_mean": gbps_mean,
            "gbps_median": gbps_median,
            "gbps_best": gbps_best,
            "total_bytes": total_bytes,
            "total_gb": total_bytes * 1e-9,
            "memory_factor": self.memory_factor,
            "tensor_shape": tensor_shape,
        }

    def safe_benchmark(
        self, B: int, S: int, D: int, config: BenchmarkConfig
    ) -> Tuple[float, float]:
        """Safely benchmark a configuration, return (bandwidth, tensor_size_gb) or (0.0, 0.0) if failed."""
        tensor_size_gb = B * S * D * 2 / 1e9  # Assuming FP16

        try:
            # Create tensors
            x = torch.randn(B, S, D, device="cuda", dtype=torch.float16)

            # Create module - this needs to be customized per module type
            if self.module_class.__name__ == "RMSNorm":
                module = self.module_class(embedding_dimension=D).cuda().half()
            else:
                # Generic fallback - may need customization for other modules
                module = self.module_class(D).cuda().half()

            # Ultra-fast benchmark for large-scale testing
            fast_config = BenchmarkConfig(
                warmup_runs=1, benchmark_runs=3, memory_factor=self.memory_factor
            )

            result = self.benchmark_module_cuda(module, x, fast_config)
            bandwidth = result["gbps_median"]

            # Clean up immediately
            del x, module
            torch.cuda.empty_cache()

            return bandwidth, tensor_size_gb

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return 0.0, tensor_size_gb
        except Exception as e:
            torch.cuda.empty_cache()
            return 0.0, tensor_size_gb

    def generate_parameter_ranges(self, expanded: bool = True) -> ParameterRanges:
        """Generate parameter ranges for comprehensive analysis."""
        if expanded:
            # Expanded ranges with powers of 2 + multiples
            batch_sizes = []
            for base_power in range(4, 13):  # 2^4 to 2^12
                base = 2**base_power
                batch_sizes.extend([base, int(base * 1.5), base * 2])
            batch_sizes = sorted(list(set(batch_sizes)))
            batch_sizes = [b for b in batch_sizes if b <= 8192]

            sequence_lengths = []
            for base_power in range(7, 16):  # 2^7 to 2^15
                base = 2**base_power
                sequence_lengths.extend(
                    [
                        base,
                        int(base * 1.25),
                        int(base * 1.5),
                        int(base * 1.75),
                        base * 2,
                    ]
                )
            sequence_lengths = sorted(list(set(sequence_lengths)))
            sequence_lengths = [s for s in sequence_lengths if s <= 65536]

            embedding_dims = []
            for base_power in range(8, 16):  # 2^8 to 2^15
                base = 2**base_power
                embedding_dims.extend(
                    [
                        base,
                        int(base * 1.25),
                        int(base * 1.5),
                        int(base * 1.75),
                        base * 2,
                    ]
                )
            embedding_dims = sorted(list(set(embedding_dims)))
            embedding_dims = [d for d in embedding_dims if d <= 65536]
        else:
            # Basic ranges - powers of 2 only
            batch_sizes = [2**i for i in range(4, 13) if 2**i <= 8192]
            sequence_lengths = [2**i for i in range(7, 16) if 2**i <= 65536]
            embedding_dims = [2**i for i in range(8, 16) if 2**i <= 65536]

        return ParameterRanges(batch_sizes, sequence_lengths, embedding_dims)

    def run_comprehensive_analysis(
        self,
        config: BenchmarkConfig = BenchmarkConfig(),
        expanded_ranges: bool = True,
        peak_bandwidth_gbps: float = 1008,  # RTX 4090 default
    ) -> pd.DataFrame:
        """
        Run comprehensive bandwidth analysis across parameter ranges.

        Args:
            config: Benchmark configuration
            expanded_ranges: Use expanded parameter ranges vs basic powers of 2
            peak_bandwidth_gbps: Theoretical peak GPU bandwidth for utilization calculations

        Returns:
            DataFrame with benchmark results
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for comprehensive analysis")

        print("Comprehensive Memory Bandwidth Analysis")
        print("=" * 60)

        # Get GPU info
        gpu_info = self.get_gpu_memory_info()
        available_memory_gb = gpu_info.get("available_gb", 20)

        print(f"GPU: {gpu_info.get('gpu_name', 'Unknown')}")
        print(f"Available Memory: {available_memory_gb:.1f} GB")
        print(f"Module: {self.module_class.__name__}")
        print(f"Memory Factor: {self.memory_factor}x")
        print()

        # Generate parameter ranges
        param_ranges = self.generate_parameter_ranges(expanded_ranges)

        print(
            f"Batch sizes ({len(param_ranges.batch_sizes)}): {param_ranges.batch_sizes[:5]}...{param_ranges.batch_sizes[-3:]}"
        )
        print(
            f"Sequence lengths ({len(param_ranges.sequence_lengths)}): {param_ranges.sequence_lengths[:5]}...{param_ranges.sequence_lengths[-3:]}"
        )
        print(
            f"Embedding dimensions ({len(param_ranges.embedding_dims)}): {param_ranges.embedding_dims[:5]}...{param_ranges.embedding_dims[-3:]}"
        )

        # Calculate total combinations and filter by memory
        total_combinations = (
            len(param_ranges.batch_sizes)
            * len(param_ranges.sequence_lengths)
            * len(param_ranges.embedding_dims)
        )
        print(f"\nTotal combinations: {total_combinations:,}")

        # Pre-filter viable configurations
        viable_configs = []
        memory_limit_gb = available_memory_gb * 0.9

        for B, S, D in itertools.product(
            param_ranges.batch_sizes,
            param_ranges.sequence_lengths,
            param_ranges.embedding_dims,
        ):
            tensor_size_gb = B * S * D * 2 / 1e9  # FP16
            if tensor_size_gb <= memory_limit_gb:
                viable_configs.append((B, S, D, tensor_size_gb))

        viable_configs.sort(key=lambda x: x[3])  # Sort by tensor size

        print(f"Viable configurations: {len(viable_configs):,}")
        if viable_configs:
            print(
                f"Memory range: {min(c[3] for c in viable_configs):.3f} - {max(c[3] for c in viable_configs):.2f} GB"
            )

        # Run benchmarks
        print(f"\nRunning benchmarks on {len(viable_configs):,} configurations...")

        results = []
        successful_tests = 0
        total_oom_count = 0

        with tqdm(
            total=len(viable_configs), desc="Benchmarking", unit="config"
        ) as pbar:
            for B, S, D, expected_size in viable_configs:
                bandwidth, actual_size = self.safe_benchmark(B, S, D, config)

                if bandwidth > 0:
                    # Calculate FLOPs if calculator provided
                    flops = self.flop_calculator(B, S, D) if self.flop_calculator else 0

                    results.append(
                        {
                            "batch_size": B,
                            "seq_length": S,
                            "embedding_dim": D,
                            "tensor_size_gb": actual_size,
                            "bandwidth_gbps": bandwidth,
                            "utilization_pct": (bandwidth / peak_bandwidth_gbps) * 100,
                            "flops": flops,
                        }
                    )
                    successful_tests += 1
                else:
                    total_oom_count += 1

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "Success": successful_tests,
                        "OOM": total_oom_count,
                    }
                )

        print(f"\nCompleted: {successful_tests:,} successful, {total_oom_count:,} OOM")

        if not results:
            raise RuntimeError("No successful benchmarks - all configurations hit OOM")

        # Convert to DataFrame and apply filtering
        df = pd.DataFrame(results)

        # Apply percentile filtering
        bandwidth_low = np.percentile(
            df["bandwidth_gbps"], config.percentile_filter[0] * 100
        )
        bandwidth_high = np.percentile(
            df["bandwidth_gbps"], config.percentile_filter[1] * 100
        )

        df_filtered = df[
            (df["bandwidth_gbps"] >= bandwidth_low)
            & (df["bandwidth_gbps"] <= bandwidth_high)
        ]

        print(f"Filtered results: {len(df_filtered):,} / {len(df):,} kept")

        return df_filtered


# RMSNorm-specific implementations
def calculate_rmsnorm_flops(
    batch_size: int, sequence_length: int, embedding_dim: int
) -> int:
    """
    Calculate the number of FLOPs for RMSNorm forward pass.

    RMSNorm operations:
    1. x.pow(2) - element-wise squaring: B × S × D FLOPs
    2. mean(dim=-1) - sum and divide: B × S × D FLOPs
    3. + eps - addition with scalar: B × S FLOPs
    4. rsqrt() - reciprocal square root: B × S FLOPs
    5. x * rsqrt_result - element-wise multiplication: B × S × D FLOPs
    6. result * weight - element-wise multiplication: B × S × D FLOPs

    Total ≈ 4 × B × S × D + 2 × B × S FLOPs
    """
    pow_flops = batch_size * sequence_length * embedding_dim
    mean_flops = batch_size * sequence_length * embedding_dim
    add_eps_flops = batch_size * sequence_length
    rsqrt_flops = batch_size * sequence_length
    mul_rsqrt_flops = batch_size * sequence_length * embedding_dim
    mul_weight_flops = batch_size * sequence_length * embedding_dim

    total_flops = (
        pow_flops
        + mean_flops
        + add_eps_flops
        + rsqrt_flops
        + mul_rsqrt_flops
        + mul_weight_flops
    )
    return total_flops


def format_flops(flops: int) -> str:
    """Format FLOP count in human-readable format."""
    if flops >= 1e12:
        return f"{flops / 1e12:.2f}T"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f}M"
    elif flops >= 1e3:
        return f"{flops / 1e3:.2f}K"
    else:
        return str(flops)


def create_rmsnorm_analyzer() -> ModuleAnalyzer:
    """Create a ModuleAnalyzer configured for RMSNorm."""

    return ModuleAnalyzer(
        module_class=RMSNorm,
        flop_calculator=calculate_rmsnorm_flops,
        memory_factor=6,  # RMSNorm has 6x memory access pattern
    )


def print_flop_analysis(configurations: List[Dict[str, Any]]) -> None:
    """Print FLOP analysis for given configurations."""
    print("RMSNorm FLOP Analysis")
    print("=" * 50)
    print(f"{'Config':<8} {'B':<4} {'S':<6} {'D':<6} {'Total FLOPs':<12}")
    print("-" * 50)

    for config in configurations:
        flops = calculate_rmsnorm_flops(config["B"], config["S"], config["D"])
        flops_str = format_flops(flops)
        print(
            f"{config['name']:<8} {config['B']:<4} {config['S']:<6} {config['D']:<6} {flops_str:<12}"
        )

    # Detailed breakdown for first example
    if configurations:
        config = configurations[0]
        B, S, D = config["B"], config["S"], config["D"]

        print(f"\nDetailed FLOP Breakdown ({config['name']} config)")
        print("=" * 50)
        print(f"Input shape: ({B}, {S}, {D})")
        print(f"x.pow(2):           {B * S * D:,} FLOPs")
        print(f"mean(dim=-1):       {B * S * D:,} FLOPs")
        print(f"+ eps:              {B * S:,} FLOPs")
        print(f"rsqrt():            {B * S:,} FLOPs")
        print(f"x * rsqrt_result:   {B * S * D:,} FLOPs")
        print(f"result * weight:    {B * S * D:,} FLOPs")
        print("-" * 30)
        total = calculate_rmsnorm_flops(B, S, D)
        print(f"Total:              {total:,} FLOPs ({format_flops(total)})")
        print(f"Approximation:      ≈ 4 × B × S × D = {4 * B * S * D:,} FLOPs")


if __name__ == "__main__":
    # Example usage
    configurations = [
        {"name": "Small", "B": 8, "S": 512, "D": 768},
        {"name": "Base", "B": 16, "S": 1024, "D": 1024},
        {"name": "Large", "B": 32, "S": 2048, "D": 4096},
        {"name": "XL", "B": 64, "S": 4096, "D": 8192},
    ]

    print_flop_analysis(configurations)
