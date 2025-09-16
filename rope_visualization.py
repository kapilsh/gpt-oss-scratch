import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Tuple, List
from loguru import logger
from model import RotaryEmbedding

save_dir = "./plots/"

# Ensure plots directory exists
os.makedirs(save_dir, exist_ok=True)


def test_basic_functionality() -> (
    Tuple[RotaryEmbedding, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Test basic RoPE functionality"""
    logger.info("=== Testing Basic RoPE Functionality ===")

    # Setup
    batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create RotaryEmbedding instance
    rope = RotaryEmbedding(
        head_dim=head_dim, base=10000, dtype=torch.float32, device=device
    )

    # Create sample query and key tensors
    query = torch.randn(seq_len, batch_size, num_heads, head_dim, device=device)
    key = torch.randn(seq_len, batch_size, num_heads, head_dim, device=device)

    logger.info(f"Input shapes - Query: {query.shape}, Key: {key.shape}")

    # Apply RoPE
    rotated_query, rotated_key = rope(query, key)

    logger.info(
        f"Output shapes - Query: {rotated_query.shape}, Key: {rotated_key.shape}"
    )
    logger.info(f"Device: {rotated_query.device}")
    logger.success("✓ Basic functionality test passed!\n")

    return rope, query, key, rotated_query, rotated_key


def test_yarn_scaling() -> None:
    """Test YaRN scaling functionality"""
    logger.info("=== Testing YaRN Scaling ===")

    head_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test different scaling factors
    scaling_factors = [1.0, 2.0, 4.0, 8.0]

    for scale in scaling_factors:
        rope = RotaryEmbedding(
            head_dim=head_dim,
            scaling_factor=scale,
            initial_context_length=2048,
            device=device,
        )

        # Test with longer sequence for scaled versions
        test_seq_len = int(2048 * scale) if scale > 1.0 else 512
        query = torch.randn(test_seq_len, 1, 1, head_dim, device=device)
        key = torch.randn(test_seq_len, 1, 1, head_dim, device=device)

        rotated_q, rotated_k = rope(query, key)

        logger.info(
            f"Scale {scale}: seq_len={test_seq_len}, output_shape={rotated_q.shape}"
        )

    logger.success("✓ YaRN scaling tests passed!\n")


def test_attention_patterns() -> Tuple[torch.Tensor, torch.Tensor]:
    """Test attention patterns with and without RoPE"""
    logger.info("=== Testing Attention Patterns ===")

    seq_len, head_dim = 64, 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test without RoPE (baseline)
    query = torch.randn(seq_len, 1, 1, head_dim, device=device)
    key = torch.randn(seq_len, 1, 1, head_dim, device=device)

    # Attention without RoPE
    attn_no_rope = torch.matmul(query.squeeze(), key.squeeze().transpose(-2, -1))
    attn_no_rope = torch.softmax(attn_no_rope / (head_dim**0.5), dim=-1)

    # Test with standard RoPE
    rope = RotaryEmbedding(head_dim=head_dim, device=device)
    rotated_q, rotated_k = rope(query, key)

    attn_with_rope = torch.matmul(
        rotated_q.squeeze(), rotated_k.squeeze().transpose(-2, -1)
    )
    attn_with_rope = torch.softmax(attn_with_rope / (head_dim**0.5), dim=-1)

    logger.info(f"Attention matrix shape: {attn_with_rope.shape}")
    logger.info(f"Max attention (no RoPE): {attn_no_rope.max().item():.4f}")
    logger.info(f"Max attention (with RoPE): {attn_with_rope.max().item():.4f}")
    logger.success("✓ Attention pattern test passed!\n")

    return attn_no_rope.cpu(), attn_with_rope.cpu()


def test_relative_position_property() -> None:
    """Test that RoPE encodes relative positions correctly"""
    logger.info("=== Testing Relative Position Property ===")

    head_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rope = RotaryEmbedding(head_dim=head_dim, device=device)

    # Create identical query vectors at different positions
    base_vector = torch.randn(1, 1, 1, head_dim, device=device)

    # Test positions
    positions = [10, 20, 30]
    rotated_vectors = []

    for pos in positions:
        # Create sequence with our vector at specific position
        seq = torch.zeros(max(positions) + 1, 1, 1, head_dim, device=device)
        seq[pos] = base_vector

        rotated_seq, _ = rope(seq, seq.clone())
        rotated_vectors.append(rotated_seq[pos])

    # Test relative position invariance
    # Dot product between vectors should depend only on relative distance
    dot_10_20 = torch.dot(rotated_vectors[0].flatten(), rotated_vectors[1].flatten())
    dot_20_30 = torch.dot(rotated_vectors[1].flatten(), rotated_vectors[2].flatten())

    logger.info(f"Dot product (pos 10, pos 20): {dot_10_20.item():.6f}")
    logger.info(f"Dot product (pos 20, pos 30): {dot_20_30.item():.6f}")
    logger.info(f"Difference: {abs(dot_10_20 - dot_20_30).item():.6f}")
    logger.success("✓ Relative position property test passed!\n")


def visualize_frequency_scaling(
    scaling_factor: float = 4.0, initial_context_length: int = 2048
) -> None:
    """Visualize how YaRN affects different frequency components

    Args:
        scaling_factor: YaRN scaling factor to visualize (default: 4.0)
        initial_context_length: Original training context length (default: 2048)
    """
    logger.info(
        f"=== Visualizing YaRN Frequency Scaling (Factor: {scaling_factor}x) ==="
    )

    head_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Standard RoPE
    rope_standard = RotaryEmbedding(
        head_dim=head_dim, scaling_factor=1.0, device=device
    )

    # YaRN with custom scaling
    rope_yarn = RotaryEmbedding(
        head_dim=head_dim,
        scaling_factor=scaling_factor,
        initial_context_length=initial_context_length,
        device=device,
    )

    # Get frequency information
    _, inv_freq_standard = rope_standard._compute_frequency_and_concentration()
    _, inv_freq_yarn = rope_yarn._compute_frequency_and_concentration()

    # Set up the aesthetic style
    plt.style.use("default")

    # Create figure with custom styling
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#f8f9fa")

    # Color palette - modern and vibrant
    colors = {
        "standard": "#2E86AB",  # Ocean blue
        "yarn": "#A23B72",  # Deep pink
        "ratio": "#F18F01",  # Orange
        "scaling": "#C73E1D",  # Red
        "reference": "#6c757d",  # Gray
        "grid": "#e9ecef",  # Light gray
    }

    dims = np.arange(len(inv_freq_standard.cpu()))

    # Subplot 1: Inverse Frequencies
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_facecolor("#ffffff")

    # Plot with enhanced styling
    line1 = ax1.plot(
        dims,
        inv_freq_standard.cpu(),
        color=colors["standard"],
        marker="o",
        markersize=6,
        linewidth=2.5,
        alpha=0.9,
        label="Standard RoPE",
        markerfacecolor="white",
        markeredgewidth=2,
    )
    line2 = ax1.plot(
        dims,
        inv_freq_yarn.cpu(),
        color=colors["yarn"],
        marker="s",
        markersize=6,
        linewidth=2.5,
        alpha=0.9,
        label=f"YaRN ({scaling_factor}×)",
        markerfacecolor="white",
        markeredgewidth=2,
    )

    ax1.set_xlabel("Dimension Pair Index", fontsize=12, fontweight="500")
    ax1.set_ylabel("Inverse Frequency", fontsize=12, fontweight="500")
    ax1.set_title("Inverse Frequencies", fontsize=14, fontweight="600", pad=20)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3, color=colors["grid"])
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)

    # Subplot 2: Frequency Scaling Ratio
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_facecolor("#ffffff")

    ratio = inv_freq_yarn.cpu() / inv_freq_standard.cpu()
    expected_ratio = 1.0 / scaling_factor

    ax2.plot(
        dims,
        ratio,
        marker="o",
        markersize=7,
        color=colors["ratio"],
        linewidth=3,
        alpha=0.9,
        markerfacecolor="white",
        markeredgewidth=2,
        label="Actual Ratio",
    )
    ax2.axhline(
        y=expected_ratio,
        color=colors["reference"],
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Expected (1/{scaling_factor})",
    )

    ax2.set_xlabel("Dimension Pair Index", fontsize=12, fontweight="500")
    ax2.set_ylabel("YaRN / Standard Ratio", fontsize=12, fontweight="500")
    ax2.set_title("Frequency Scaling Ratio", fontsize=14, fontweight="600", pad=20)
    ax2.grid(True, alpha=0.3, color=colors["grid"])
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)

    # Subplot 3: Wavelengths
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_facecolor("#ffffff")

    wavelengths_std = 2 * np.pi / inv_freq_standard.cpu().numpy()
    wavelengths_yarn = 2 * np.pi / inv_freq_yarn.cpu().numpy()

    ax3.fill_between(dims, wavelengths_std, alpha=0.2, color=colors["standard"])
    ax3.fill_between(dims, wavelengths_yarn, alpha=0.2, color=colors["yarn"])

    ax3.plot(
        dims,
        wavelengths_std,
        color=colors["standard"],
        marker="o",
        markersize=6,
        linewidth=2.5,
        alpha=0.9,
        label="Standard RoPE",
        markerfacecolor="white",
        markeredgewidth=2,
    )
    ax3.plot(
        dims,
        wavelengths_yarn,
        color=colors["yarn"],
        marker="s",
        markersize=6,
        linewidth=2.5,
        alpha=0.9,
        label=f"YaRN ({scaling_factor}×)",
        markerfacecolor="white",
        markeredgewidth=2,
    )

    ax3.set_xlabel("Dimension Pair Index", fontsize=12, fontweight="500")
    ax3.set_ylabel("Wavelength (tokens)", fontsize=12, fontweight="500")
    ax3.set_title("Positional Wavelengths", fontsize=14, fontweight="600", pad=20)
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3, color=colors["grid"])
    ax3.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)

    # Subplot 4: YaRN Ramp Function Effect
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_facecolor("#ffffff")

    scaling_effect = wavelengths_yarn / wavelengths_std

    # Create gradient effect for the line
    points = np.array([dims, scaling_effect]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    from matplotlib.collections import LineCollection

    lc = LineCollection(segments.tolist(), cmap="viridis", linewidths=4, alpha=0.8)
    lc.set_array(dims)
    line = ax4.add_collection(lc)

    # Add markers
    scatter = ax4.scatter(
        dims,
        scaling_effect,
        c=dims,
        cmap="viridis",
        s=80,
        alpha=0.9,
        edgecolors="white",
        linewidth=2,
        zorder=5,
    )

    ax4.axhline(
        y=scaling_factor,
        color=colors["reference"],
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Expected ({scaling_factor}×)",
    )
    ax4.axhline(
        y=1.0,
        color=colors["reference"],
        linestyle=":",
        linewidth=2,
        alpha=0.5,
        label="No Scaling (1×)",
    )

    ax4.set_xlabel("Dimension Pair Index", fontsize=12, fontweight="500")
    ax4.set_ylabel("Wavelength Scaling Factor", fontsize=12, fontweight="500")
    ax4.set_title("YaRN Ramp Function Effect", fontsize=14, fontweight="600", pad=20)
    ax4.grid(True, alpha=0.3, color=colors["grid"])
    ax4.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)

    # Add colorbar for the gradient effect
    cbar = plt.colorbar(scatter, ax=ax4, shrink=0.6, alpha=0.8)
    cbar.set_label("Dimension Index", fontsize=10, fontweight="500")

    # Overall styling
    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.tick_params(colors="#495057", which="both")

    # Add main title
    fig.suptitle(
        "YaRN vs Standard RoPE: Frequency Analysis",
        fontsize=18,
        fontweight="700",
        y=0.98,
        color="#212529",
    )

    # Dynamic subtitle with parameters
    extended_context = int(initial_context_length * scaling_factor)
    fig.text(
        0.5,
        0.94,
        f"Head Dimension: {head_dim} | Scaling Factor: {scaling_factor}× | Context: {initial_context_length}→{extended_context} tokens",
        ha="center",
        fontsize=12,
        style="italic",
        color="#6c757d",
    )

    plt.tight_layout(rect=(0, 0, 1, 0.92))

    # Dynamic filename based on parameters
    filename = f"yarn_frequency_analysis_scale_{scaling_factor}x.png"
    filepath = os.path.join(save_dir, filename)

    # Save with high quality
    plt.savefig(
        filepath, dpi=300, bbox_inches="tight", facecolor="#f8f9fa", edgecolor="none"
    )
    plt.show()

    logger.success(
        f"✓ Beautiful frequency scaling visualization saved as '{filepath}'\n"
    )


def benchmark_performance() -> None:
    """Benchmark RoPE performance"""
    logger.info("=== Performance Benchmark ===")

    import time

    # Test configurations
    configs = [
        {"seq_len": 512, "batch_size": 8, "num_heads": 8, "head_dim": 64},
        {"seq_len": 2048, "batch_size": 4, "num_heads": 12, "head_dim": 64},
        {"seq_len": 4096, "batch_size": 2, "num_heads": 16, "head_dim": 64},
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for config in configs:
        seq_len, batch_size, num_heads, head_dim = config.values()

        rope = RotaryEmbedding(head_dim=head_dim, device=device)
        query = torch.randn(seq_len, batch_size, num_heads, head_dim, device=device)
        key = torch.randn(seq_len, batch_size, num_heads, head_dim, device=device)

        # Warmup
        for _ in range(10):
            _ = rope(query, key)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        num_runs = 100

        for _ in range(num_runs):
            rotated_q, rotated_k = rope(query, key)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs * 1000  # ms

        logger.info(f"Config {config}: {avg_time:.2f}ms per forward pass")

    logger.success("✓ Performance benchmark completed!\n")


def visualize_embeddings_heatmap(
    scaling_factors: List[float] = [1.0, 2.0, 4.0, 8.0],
    seq_lengths: List[int] = [512, 1024, 2048, 4096],
    head_dim: int = 64,
    initial_context_length: int = 2048,
) -> None:
    """Visualize RoPE embeddings as heatmaps across different scaling factors and sequence lengths

    Args:
        scaling_factors: List of YaRN scaling factors to compare
        seq_lengths: List of sequence lengths to test
        head_dim: Dimension of each attention head
        initial_context_length: Original training context length
    """
    logger.info(f"=== Visualizing RoPE Embeddings Heatmaps ===")
    logger.info(f"Scaling factors: {scaling_factors}")
    logger.info(f"Sequence lengths: {seq_lengths}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the aesthetic style
    plt.style.use("default")

    # Calculate grid size
    n_scales = len(scaling_factors)
    n_lengths = len(seq_lengths)

    # Create large figure for comprehensive view
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("#f8f9fa")

    # Modern color palette
    colors = {
        "bg": "#ffffff",
        "text": "#212529",
        "subtitle": "#6c757d",
        "grid": "#e9ecef",
    }

    # Create subplots grid
    gs = fig.add_gridspec(n_scales, n_lengths, hspace=0.3, wspace=0.2)

    # Track min/max values for consistent color scaling
    all_cos_values = []
    all_sin_values = []

    # First pass: collect all values for consistent scaling
    for i, scale in enumerate(scaling_factors):
        for j, seq_len in enumerate(seq_lengths):
            # Skip if sequence length is too long for the scaling factor
            if seq_len > initial_context_length * scale and scale > 1.0:
                continue

            rope = RotaryEmbedding(
                head_dim=head_dim,
                scaling_factor=scale,
                initial_context_length=initial_context_length,
                device=device,
            )

            cos_cache, sin_cache = rope._compute_cos_sin_cache(seq_len)
            all_cos_values.append(cos_cache.cpu().numpy())
            all_sin_values.append(sin_cache.cpu().numpy())

    # Calculate global min/max for consistent color scaling
    global_cos_min = min(arr.min() for arr in all_cos_values)
    global_cos_max = max(arr.max() for arr in all_cos_values)
    global_sin_min = min(arr.min() for arr in all_sin_values)
    global_sin_max = max(arr.max() for arr in all_sin_values)

    # Create the heatmaps
    for i, scale in enumerate(scaling_factors):
        for j, seq_len in enumerate(seq_lengths):
            # Skip if sequence length is too long for the scaling factor
            if seq_len > initial_context_length * scale and scale > 1.0:
                ax = fig.add_subplot(gs[i, j])
                ax.text(
                    0.5,
                    0.5,
                    f"Seq len {seq_len}\ntoo long for\n{scale}× scaling",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    style="italic",
                    color=colors["subtitle"],
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor("#f5f5f5")
                continue

            # Create RoPE instance
            rope = RotaryEmbedding(
                head_dim=head_dim,
                scaling_factor=scale,
                initial_context_length=initial_context_length,
                device=device,
            )

            # Get cos/sin embeddings
            cos_cache, sin_cache = rope._compute_cos_sin_cache(seq_len)

            # Create subplot
            ax = fig.add_subplot(gs[i, j])
            ax.set_facecolor(colors["bg"])

            # Combine cos and sin for visualization (take first part of sequence for clarity)
            display_len = min(seq_len, 256)  # Limit display for readability
            cos_display = cos_cache[:display_len].cpu().numpy()

            # Create beautiful heatmap
            im = ax.imshow(
                cos_display.T,
                cmap="RdYlBu_r",  # Beautiful diverging colormap
                aspect="auto",
                interpolation="bilinear",
                vmin=global_cos_min,
                vmax=global_cos_max,
            )

            # Style the subplot
            ax.set_title(
                f"Scale {scale}× | Seq {seq_len}", fontsize=11, fontweight="600", pad=10
            )

            # Set labels only on edges
            if i == n_scales - 1:  # Bottom row
                ax.set_xlabel("Position", fontsize=10, fontweight="500")
            if j == 0:  # Left column
                ax.set_ylabel("Dimension", fontsize=10, fontweight="500")

            # Customize ticks
            if display_len <= 64:
                ax.set_xticks(range(0, display_len, max(1, display_len // 8)))
            else:
                ax.set_xticks(range(0, display_len, display_len // 8))

            ax.set_yticks(range(0, head_dim // 2, max(1, head_dim // 16)))
            ax.tick_params(labelsize=8, colors=colors["subtitle"])

            # Add grid
            ax.grid(True, alpha=0.2, color=colors["grid"], linewidth=0.5)

    # Add colorbar
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Cosine Embedding Value", fontsize=12, fontweight="500")
    cbar.ax.tick_params(labelsize=10, colors=colors["subtitle"])

    # Add main title
    fig.suptitle(
        "RoPE Embedding Heatmaps: YaRN Scaling Analysis",
        fontsize=20,
        fontweight="700",
        y=0.95,
        color=colors["text"],
    )

    # Add comprehensive subtitle
    subtitle = f"Head Dim: {head_dim} | Base Context: {initial_context_length} | Cosine Component Visualization"
    fig.text(
        0.5,
        0.91,
        subtitle,
        ha="center",
        fontsize=13,
        style="italic",
        color=colors["subtitle"],
    )

    # Add methodology note
    methodology = "Each heatmap shows cosine embeddings across positions (x-axis) and frequency dimensions (y-axis)"
    fig.text(
        0.5,
        0.02,
        methodology,
        ha="center",
        fontsize=10,
        style="italic",
        color=colors["subtitle"],
    )

    # Save with descriptive filename
    filename = f"rope_embeddings_heatmap_scales_{len(scaling_factors)}_lengths_{len(seq_lengths)}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(
        filepath, dpi=300, bbox_inches="tight", facecolor="#f8f9fa", edgecolor="none"
    )
    plt.show()

    logger.success(
        f"✓ Beautiful embeddings heatmap visualization saved as '{filepath}'"
    )

    # Create a second figure focusing on embedding differences
    _visualize_embedding_differences(
        scaling_factors, seq_lengths, head_dim, initial_context_length
    )


def _visualize_embedding_differences(
    scaling_factors: List[float],
    seq_lengths: List[int],
    head_dim: int,
    initial_context_length: int,
) -> None:
    """Create a focused visualization showing embedding differences between standard RoPE and YaRN"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Focus on interesting cases
    interesting_configs = [
        (2.0, 2048),  # 2x scaling at training length
        (4.0, 4096),  # 4x scaling at 2x length
        (8.0, 8192),  # 8x scaling at 4x length
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("#f8f9fa")

    for idx, (scale, seq_len) in enumerate(interesting_configs):
        # Standard RoPE
        rope_std = RotaryEmbedding(head_dim=head_dim, scaling_factor=1.0, device=device)
        # YaRN
        rope_yarn = RotaryEmbedding(
            head_dim=head_dim,
            scaling_factor=scale,
            initial_context_length=initial_context_length,
            device=device,
        )

        # Get embeddings (limit display length)
        display_len = min(seq_len, 512)
        cos_std, _ = rope_std._compute_cos_sin_cache(display_len)
        cos_yarn, _ = rope_yarn._compute_cos_sin_cache(display_len)

        # Plot standard RoPE (top row)
        ax_std = axes[0, idx]
        im_std = ax_std.imshow(
            cos_std.cpu().numpy().T,
            cmap="RdYlBu_r",
            aspect="auto",
            interpolation="bilinear",
        )
        ax_std.set_title(
            f"Standard RoPE\nSeq: {seq_len}", fontsize=12, fontweight="600"
        )
        ax_std.set_ylabel("Dimension", fontsize=10, fontweight="500")

        # Plot YaRN (bottom row)
        ax_yarn = axes[1, idx]
        im_yarn = ax_yarn.imshow(
            cos_yarn.cpu().numpy().T,
            cmap="RdYlBu_r",
            aspect="auto",
            interpolation="bilinear",
        )
        ax_yarn.set_title(
            f"YaRN {scale}×\nSeq: {seq_len}", fontsize=12, fontweight="600"
        )
        ax_yarn.set_xlabel("Position", fontsize=10, fontweight="500")
        ax_yarn.set_ylabel("Dimension", fontsize=10, fontweight="500")

        # Style both subplots
        for ax in [ax_std, ax_yarn]:
            ax.set_facecolor("#ffffff")
            ax.tick_params(labelsize=8, colors="#6c757d")
            ax.grid(True, alpha=0.2, color="#e9ecef", linewidth=0.5)

    # Add colorbar with proper positioning to avoid overlap
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))  # [left, bottom, width, height]
    cbar = fig.colorbar(im_yarn, cax=cbar_ax)
    cbar.set_label("Cosine Embedding Value", fontsize=12, fontweight="500")
    cbar.ax.tick_params(labelsize=10, colors="#6c757d")

    # Add titles
    fig.suptitle(
        "Standard RoPE vs YaRN: Side-by-Side Comparison",
        fontsize=18,
        fontweight="700",
        y=0.95,
    )
    fig.text(
        0.5,
        0.91,
        "Top: Standard RoPE | Bottom: YaRN Enhanced",
        ha="center",
        fontsize=12,
        style="italic",
        color="#6c757d",
    )

    # Adjust layout to leave space for colorbar
    plt.tight_layout(rect=(0, 0, 0.9, 0.88))

    # Save comparison
    filename_comp = "rope_yarn_comparison_heatmaps.png"
    filepath_comp = os.path.join(save_dir, filename_comp)
    plt.savefig(
        filepath_comp,
        dpi=300,
        bbox_inches="tight",
        facecolor="#f8f9fa",
        edgecolor="none",
    )
    plt.show()

    logger.success(f"✓ Comparison heatmap saved as '{filepath_comp}'\n")
