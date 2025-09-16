"""
Plotting utilities for neural network module analysis.

Provides comprehensive visualization functions for performance analysis results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, Any, Optional


def setup_plot_style():
    """Set up prettier plot style."""
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")


def create_comprehensive_analysis_plots(
    df_filtered: pd.DataFrame,
    module_name: str = "Module",
    peak_bandwidth_gbps: float = 1008,
    figsize: tuple = (20, 12)
) -> None:
    """
    Create comprehensive analysis plots for module performance.
    
    Args:
        df_filtered: Filtered DataFrame with benchmark results
        module_name: Name of the module being analyzed (e.g., "RMSNorm")
        peak_bandwidth_gbps: Theoretical peak GPU bandwidth
        figsize: Figure size tuple
    """
    setup_plot_style()
    
    # Calculate correlations
    batch_corr = df_filtered["batch_size"].corr(df_filtered["bandwidth_gbps"])
    seq_corr = df_filtered["seq_length"].corr(df_filtered["bandwidth_gbps"])
    emb_corr = df_filtered["embedding_dim"].corr(df_filtered["bandwidth_gbps"])
    
    # Calculate median utilization for reference lines
    median_utilization = df_filtered["utilization_pct"].median()
    median_bandwidth = (median_utilization / 100) * peak_bandwidth_gbps
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")
    
    # Custom color palette
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#8FBC8F", "#4B0082"]
    
    # Create subplot layout (2x3 grid)
    gs = fig.add_gridspec(
        2, 3, hspace=0.3, wspace=0.3, left=0.08, right=0.95, top=0.92, bottom=0.08
    )
    
    # Main title
    fig.suptitle(
        f"Expanded {module_name} Memory Bandwidth Analysis\n"
        + f"{len(df_filtered):,} Configurations Analyzed | RTX 4090 Performance Study",
        fontsize=18,
        fontweight="bold",
        y=0.96,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7),
    )
    
    # Plot 1: Main bandwidth vs tensor size
    ax1 = fig.add_subplot(gs[0, :2])
    scatter = ax1.scatter(
        df_filtered["tensor_size_gb"],
        df_filtered["bandwidth_gbps"],
        c=df_filtered["utilization_pct"],
        cmap="plasma",
        alpha=0.8,
        s=20,
        edgecolors="white",
        linewidth=0.3,
    )
    ax1.axhline(
        y=peak_bandwidth_gbps,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"RTX 4090 Peak ({peak_bandwidth_gbps} GB/s)",
    )
    ax1.axhline(
        y=median_bandwidth,
        color="green",
        linestyle="-.",
        linewidth=2,
        alpha=0.8,
        label=f"Median Utilization ({median_utilization:.1f}%: {median_bandwidth:.1f} GB/s)",
    )
    ax1.set_xlabel("Tensor Size (GB)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Memory Bandwidth (GB/s)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Memory Bandwidth vs Tensor Size (Expanded Ranges)\n(Color Scale = GPU Utilization %)",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    ax1.set_xscale("log")
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8, pad=0.02)
    cbar.set_label("GPU Utilization (%)", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)
    
    # Plot 2: Batch Size scaling
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(
        df_filtered["batch_size"],
        df_filtered["bandwidth_gbps"],
        alpha=0.6,
        s=15,
        c=colors[0],
        edgecolors="white",
        linewidth=0.2,
    )
    ax2.axhline(
        y=median_bandwidth,
        color="green",
        linestyle="-.",
        linewidth=2,
        alpha=0.8,
        label=f"Median ({median_utilization:.1f}%)",
    )
    ax2.set_xlabel("Batch Size", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Bandwidth (GB/s)", fontsize=11, fontweight="bold")
    ax2.set_title(
        f"Batch Size Impact\nCorr: {batch_corr:.3f}", fontsize=12, fontweight="bold"
    )
    ax2.set_xscale("log")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sequence Length scaling
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(
        df_filtered["seq_length"],
        df_filtered["bandwidth_gbps"],
        alpha=0.6,
        s=15,
        c=colors[1],
        edgecolors="white",
        linewidth=0.2,
    )
    ax3.axhline(
        y=median_bandwidth,
        color="green",
        linestyle="-.",
        linewidth=2,
        alpha=0.8,
        label=f"Median ({median_utilization:.1f}%)",
    )
    ax3.set_xlabel("Sequence Length", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Bandwidth (GB/s)", fontsize=11, fontweight="bold")
    ax3.set_title(
        f"Sequence Length Impact\nCorr: {seq_corr:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    ax3.set_xscale("log")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Embedding Dimension scaling
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(
        df_filtered["embedding_dim"],
        df_filtered["bandwidth_gbps"],
        alpha=0.6,
        s=15,
        c=colors[2],
        edgecolors="white",
        linewidth=0.2,
    )
    ax4.axhline(
        y=median_bandwidth,
        color="green",
        linestyle="-.",
        linewidth=2,
        alpha=0.8,
        label=f"Median ({median_utilization:.1f}%)",
    )
    ax4.set_xlabel("Embedding Dimension", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Bandwidth (GB/s)", fontsize=11, fontweight="bold")
    ax4.set_title(
        f"Embedding Dim Impact\nCorr: {emb_corr:.3f}", fontsize=12, fontweight="bold"
    )
    ax4.set_xscale("log")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Bandwidth Distribution
    ax5 = fig.add_subplot(gs[1, 2])
    n, bins, patches = ax5.hist(
        df_filtered["bandwidth_gbps"],
        bins=50,
        alpha=0.7,
        color=colors[3],
        edgecolor="white",
        linewidth=1,
    )
    # Color bars by height for gradient effect
    for i, p in enumerate(patches):
        p.set_facecolor(plt.cm.viridis(n[i] / max(n)))
    
    ax5.axvline(
        df_filtered["bandwidth_gbps"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {df_filtered["bandwidth_gbps"].mean():.1f} GB/s',
    )
    ax5.axvline(
        df_filtered["bandwidth_gbps"].median(),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f'Median: {df_filtered["bandwidth_gbps"].median():.1f} GB/s',
    )
    ax5.axvline(
        median_bandwidth,
        color="green",
        linestyle="-.",
        linewidth=2,
        label=f'Median Util: {median_utilization:.1f}%',
    )
    ax5.set_xlabel("Memory Bandwidth (GB/s)", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax5.set_title("Bandwidth Distribution", fontsize=12, fontweight="bold")
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    plt.show()


def print_analysis_summary(
    df_filtered: pd.DataFrame,
    module_name: str = "Module",
    peak_bandwidth_gbps: float = 1008
) -> None:
    """
    Print comprehensive analysis summary.
    
    Args:
        df_filtered: Filtered DataFrame with benchmark results
        module_name: Name of the module being analyzed
        peak_bandwidth_gbps: Theoretical peak GPU bandwidth
    """
    print(f"\n{module_name.upper()} ANALYSIS RESULTS")
    print("=" * 70)
    
    # Overall statistics
    print(f"Overall Performance:")
    print(f"   Best bandwidth: {df_filtered['bandwidth_gbps'].max():.1f} GB/s ({df_filtered['utilization_pct'].max():.1f}%)")
    print(f"   Average bandwidth: {df_filtered['bandwidth_gbps'].mean():.1f} GB/s ({df_filtered['utilization_pct'].mean():.1f}%)")
    print(f"   Median bandwidth: {df_filtered['bandwidth_gbps'].median():.1f} GB/s ({df_filtered['utilization_pct'].median():.1f}%)")
    print(f"   Bandwidth std dev: {df_filtered['bandwidth_gbps'].std():.1f} GB/s")
    
    # Find best configuration
    best_idx = df_filtered["bandwidth_gbps"].idxmax()
    best_config = df_filtered.loc[best_idx]
    
    print(f"\nBEST CONFIGURATION:")
    print(f"   Bandwidth: {best_config['bandwidth_gbps']:.1f} GB/s ({best_config['utilization_pct']:.1f}%)")
    print(f"   Shape: B={int(best_config['batch_size'])}, S={int(best_config['seq_length'])}, D={int(best_config['embedding_dim'])}")
    print(f"   Tensor size: {best_config['tensor_size_gb']:.2f} GB")
    
    # Scaling analysis
    print(f"\nScaling Analysis:")
    batch_corr = df_filtered["batch_size"].corr(df_filtered["bandwidth_gbps"])
    seq_corr = df_filtered["seq_length"].corr(df_filtered["bandwidth_gbps"])
    emb_corr = df_filtered["embedding_dim"].corr(df_filtered["bandwidth_gbps"])
    size_corr = df_filtered["tensor_size_gb"].corr(df_filtered["bandwidth_gbps"])
    
    print(f"   Batch size correlation with bandwidth: {batch_corr:.3f}")
    print(f"   Sequence length correlation with bandwidth: {seq_corr:.3f}")
    print(f"   Embedding dim correlation with bandwidth: {emb_corr:.3f}")
    print(f"   Tensor size correlation with bandwidth: {size_corr:.3f}")


def print_top_configurations(df_filtered: pd.DataFrame, top_n: int = 15) -> None:
    """
    Print top N configurations by bandwidth.
    
    Args:
        df_filtered: Filtered DataFrame with benchmark results
        top_n: Number of top configurations to show
    """
    print(f"\nTOP {top_n} CONFIGURATIONS:")
    print("=" * 85)
    print(f"{'Rank':<4} {'Batch':<8} {'SeqLen':<8} {'EmbDim':<8} {'Bandwidth':<12} {'Util%':<8} {'Size(GB)':<8}")
    print("-" * 85)
    
    top_configs = df_filtered.nlargest(top_n, "bandwidth_gbps")[
        ["batch_size", "seq_length", "embedding_dim", "tensor_size_gb", "bandwidth_gbps", "utilization_pct"]
    ]
    
    for i, (idx, row) in enumerate(top_configs.iterrows(), 1):
        B = int(row["batch_size"])
        S = int(row["seq_length"])
        D = int(row["embedding_dim"])
        bw = row["bandwidth_gbps"]
        util = row["utilization_pct"]
        size = row["tensor_size_gb"]
        
        # Performance indicators
        if util > 80:
            prefix = "***"
        elif util > 60:
            prefix = "** "
        elif util > 40:
            prefix = "*  "
        else:
            prefix = "   "
        
        print(f"{prefix} {i:2d}   {B:6d}    {S:6d}    {D:6d}    {bw:8.1f} GB/s  {util:6.1f}   {size:6.2f}")


def print_performance_tiers(df_filtered: pd.DataFrame) -> None:
    """
    Print performance tier analysis.
    
    Args:
        df_filtered: Filtered DataFrame with benchmark results
    """
    print(f"\nPERFORMANCE TIER ANALYSIS:")
    print("=" * 50)
    
    excellent = df_filtered[df_filtered["utilization_pct"] > 80]
    good = df_filtered[(df_filtered["utilization_pct"] > 60) & (df_filtered["utilization_pct"] <= 80)]
    moderate = df_filtered[(df_filtered["utilization_pct"] > 40) & (df_filtered["utilization_pct"] <= 60)]
    poor = df_filtered[df_filtered["utilization_pct"] <= 40]
    
    total = len(df_filtered)
    
    print(f"*** Excellent (>80%): {len(excellent):4d} configs ({len(excellent)/total*100:4.1f}%)")
    print(f"**  Good (60-80%):    {len(good):4d} configs ({len(good)/total*100:4.1f}%)")
    print(f"*   Moderate (40-60%): {len(moderate):4d} configs ({len(moderate)/total*100:4.1f}%)")
    print(f"    Poor (<40%):      {len(poor):4d} configs ({len(poor)/total*100:4.1f}%)")


def print_utilization_analysis(df_filtered: pd.DataFrame, peak_bandwidth_gbps: float = 1008) -> None:
    """
    Print GPU utilization analysis.
    
    Args:
        df_filtered: Filtered DataFrame with benchmark results
        peak_bandwidth_gbps: Theoretical peak GPU bandwidth
    """
    median_utilization = df_filtered["utilization_pct"].median()
    median_bandwidth = (median_utilization / 100) * peak_bandwidth_gbps
    
    print(f"\nGPU UTILIZATION ANALYSIS:")
    print("=" * 50)
    print(f"Median GPU Utilization: {median_utilization:.1f}%")
    print(f"Median Bandwidth: {median_bandwidth:.1f} GB/s")
    print(f"Configurations above median: {len(df_filtered[df_filtered['utilization_pct'] > median_utilization]):,}")
    print(f"Configurations below median: {len(df_filtered[df_filtered['utilization_pct'] <= median_utilization]):,}")


def create_complete_analysis_report(
    df_filtered: pd.DataFrame,
    module_name: str = "Module",
    peak_bandwidth_gbps: float = 1008,
    show_plots: bool = True,
    top_n: int = 15
) -> None:
    """
    Create a complete analysis report with plots and statistics.
    
    Args:
        df_filtered: Filtered DataFrame with benchmark results
        module_name: Name of the module being analyzed
        peak_bandwidth_gbps: Theoretical peak GPU bandwidth
        show_plots: Whether to show plots
        top_n: Number of top configurations to display
    """
    if show_plots:
        create_comprehensive_analysis_plots(df_filtered, module_name, peak_bandwidth_gbps)
    
    print_analysis_summary(df_filtered, module_name, peak_bandwidth_gbps)
    print_top_configurations(df_filtered, top_n)
    print_performance_tiers(df_filtered)
    print_utilization_analysis(df_filtered, peak_bandwidth_gbps)
    
    print(f"\nData Information:")
    print(f"   DataFrame contains {len(df_filtered):,} benchmark results")
    print(f"   Columns: {list(df_filtered.columns)}")