from dataclasses import dataclass
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import json
from dataclasses import dataclass


@dataclass
class ModelConfig:
    num_hidden_layers: int
    num_experts: int
    experts_per_token: int
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    swiglu_limit: float
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    sliding_window: int
    initial_context_length: int
    rope_theta: int
    rope_scaling_factor: float
    rope_ntk_alpha: int
    rope_ntk_beta: int

    @classmethod
    def from_file(cls, file_path: str) -> "ModelConfig":
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)


class RMSNorm(nn.Module):
    def __init__(self, embedding_dimension: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.embedding_dimension = embedding_dimension
        self.weight = nn.Parameter(torch.ones(embedding_dimension))

    def forward(self, x: torch.Tensor):
        means = x.pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(means + self.eps)) * self.weight


class RotaryEmbedding(torch.nn.Module):
    """Rotary Position Embedding with YaRN scaling support.

    Implements RoPE with optional YaRN-style scaling for context window extension.
    Supports both standard RoPE and YaRN enhancements including:
    - Non-uniform scaling across frequency dimensions
    - NTK-aware interpolation and extrapolation
    - Temperature-based concentration adjustment
    """

    def __init__(
        self,
        head_dim: int,
        base: int = 10000,
        dtype: torch.dtype = torch.float32,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize RotaryEmbedding.

        Args:
            head_dim: Dimension of each attention head
            base: Base for frequency calculation (default: 10000)
            dtype: Data type for computations
            initial_context_length: Original training context length
            scaling_factor: YaRN scaling factor (>1.0 enables YaRN)
            ntk_alpha: NTK alpha parameter for high-freq scaling
            ntk_beta: NTK beta parameter for low-freq scaling
            device: Device for tensor operations
        """
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    @staticmethod
    def _apply_rotary_embedding(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary position embedding to input tensor.

        Args:
            x: Input tensor of shape [..., seq_len, head_dim]
            cos: Cosine values of shape [seq_len, head_dim//2]
            sin: Sine values of shape [seq_len, head_dim//2]

        Returns:
            Rotated tensor with same shape as input
        """
        # Ensure cos/sin match input dtype and add dimension for broadcasting
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)

        # Split into even/odd dimensions and apply rotation
        x_even, x_odd = x.chunk(2, dim=-1)
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_odd * cos + x_even * sin

        return torch.cat([rotated_even, rotated_odd], dim=-1)

    def _compute_frequency_and_concentration(self) -> Tuple[float, torch.Tensor]:
        """Compute YaRN concentration factor and inverse frequencies.

        Based on YaRN paper: https://arxiv.org/abs/2309.00071

        Returns:
            Tuple of (concentration_factor, inverse_frequencies)
        """
        # Base frequencies for each dimension pair
        freq_indices = torch.arange(
            0, self.head_dim, 2, dtype=torch.float32, device=self.device
        )
        base_freqs = self.base ** (freq_indices / self.head_dim)

        if self.scaling_factor <= 1.0:
            # Standard RoPE - no scaling
            concentration = 1.0
            inv_freq = 1.0 / base_freqs
        else:
            # YaRN scaling enabled
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0

            # Calculate transition boundaries for YaRN ramp function
            d_half = self.head_dim // 2
            low_boundary = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high_boundary = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )

            assert (
                0 < low_boundary < high_boundary < d_half - 1
            ), "Invalid NTK boundaries - check alpha/beta parameters"

            # YaRN interpolation (PI) and extrapolation (NTK) frequencies
            interpolation_freq = 1.0 / (self.scaling_factor * base_freqs)
            extrapolation_freq = 1.0 / base_freqs

            # Create ramp function for smooth transition
            dim_indices = torch.arange(d_half, dtype=torch.float32, device=self.device)
            ramp = (dim_indices - low_boundary) / (high_boundary - low_boundary)
            blend_mask = 1.0 - ramp.clamp(0.0, 1.0)

            # Blend interpolation and extrapolation based on frequency
            inv_freq = (
                interpolation_freq * (1.0 - blend_mask)
                + extrapolation_freq * blend_mask
            )

        return concentration, inv_freq

    def _compute_cos_sin_cache(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute and cache cosine/sine values for given sequence length.

        Args:
            seq_len: Maximum sequence length to precompute

        Returns:
            Tuple of (cosine_cache, sine_cache) tensors
        """
        concentration, inv_freq = self._compute_frequency_and_concentration()

        # Create position indices
        position_ids = torch.arange(seq_len, dtype=torch.float32, device=self.device)

        # Compute frequency matrix: [seq_len, head_dim//2]
        freqs = torch.outer(position_ids, inv_freq)

        # Apply concentration (temperature) factor
        cos_cache = (freqs.cos() * concentration).to(self.dtype)
        sin_cache = (freqs.sin() * concentration).to(self.dtype)

        return cos_cache, sin_cache

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            query: Query tensor of shape [seq_len, ...]
            key: Key tensor of shape [seq_len, ...]

        Returns:
            Tuple of (rotated_query, rotated_key)
        """
        seq_len = query.shape[0]
        cos_cache, sin_cache = self._compute_cos_sin_cache(seq_len)

        # Reshape to apply rotation per head
        original_q_shape = query.shape
        query = query.view(seq_len, -1, self.head_dim)
        query = self._apply_rotary_embedding(query, cos_cache, sin_cache)
        query = query.reshape(original_q_shape)

        original_k_shape = key.shape
        key = key.view(seq_len, -1, self.head_dim)
        key = self._apply_rotary_embedding(key, cos_cache, sin_cache)
        key = key.reshape(original_k_shape)

        return query, key
