# gpt-oss-scratch

A deep-dive exploration of the [OpenAI GPT-OSS 20B](https://huggingface.co/openai/gpt-oss-20b) model. This repo covers everything from understanding the architecture and building components from scratch, to running inference, writing custom Triton CUDA kernels, and profiling the model on an RTX 4090.

## Overview

GPT-OSS 20B is a **Mixture-of-Experts (MoE)** transformer with:

- 24 transformer blocks (alternating sliding/full attention layers)
- 32 experts per MLP block, 4 selected per token via a learned gating network
- Grouped Query Attention (GQA): 64 query heads, 8 key/value heads, head_dim=64
- RoPE with YaRN scaling (32x, base context 4096 -> 131K)
- MXFP4 quantized expert weights
- SwiGLU activations with clamping (limit=7.0)
- RMSNorm (eps=1e-5)
- Vocab size: 201,088

## Repository Structure

```
gpt-oss-scratch/
|-- gpt-oss/                          # Git submodule: official OpenAI gpt-oss source
|-- data/gpt-oss-20b/original/        # Local model weights (OpenAI original format)
|
|-- Notebooks ──────────────────────────────────────────────────
|-- gpt_oss_from_scratch.ipynb         # Build every component from scratch (RMSNorm, RoPE, etc.)
|-- gpt_oss_reference_implementation.ipynb  # Load & inspect using official gpt-oss torch/triton code
|-- gpt_oss_from_unsloth.ipynb         # Load via Unsloth (LoRA, quantization, fine-tuning)
|-- gpt_oss_inference.ipynb            # HuggingFace pipeline inference + profiling
|
|-- Scripts ────────────────────────────────────────────────────
|-- gpt_oss_manual_inference.py        # Interactive CLI: manual inference with KV cache & streaming
|-- model.py                           # Standalone model components (ModelConfig, RotaryEmbedding, etc.)
|-- rope_visualization.py              # RoPE visualization, tests, and benchmarks
|
|-- Triton Kernels ─────────────────────────────────────────────
|-- kernels/
|   |-- softmax.py                     # Triton softmax kernel (fused online safe softmax)
|   |-- rms_norm.py                    # Triton RMSNorm kernel (forward + backward)
|   |-- benchmarks/
|   |   |-- softmax_benchmark.py       # Softmax kernel benchmarks vs PyTorch
|   |   |-- profile_softmax.py         # Softmax profiling script
|   |-- tests/
|       |-- test_softmax.py            # Softmax kernel correctness tests
|       |-- test_rms_norm.py           # RMSNorm kernel correctness tests
|
|-- Experiments ────────────────────────────────────────────────
|-- experiments/
|   |-- rms_norm.py                    # RMSNorm FLOP/bandwidth analysis module
|   |-- rms_norm_analysis.ipynb        # RMSNorm performance analysis notebook
|   |-- triton_rms_norm_benchmark.py   # Triton vs PyTorch RMSNorm benchmarks
|   |-- plotting_utils.py              # Benchmark visualization helpers
|   |-- rms-norm-forward.csv           # Benchmark results
|   |-- rms-norm-backward.csv
|
|-- Graphviz Diagrams ──────────────────────────────────────────
|-- graphviz/
|   |-- moe.viz                        # Mixture of Experts routing
|   |-- gqa.viz                        # Grouped Query Attention
|   |-- mha.viz                        # Multi-Head Attention
|   |-- mla.viz                        # Multi-head Latent Attention
|   |-- mqa.viz                        # Multi-Query Attention
|   |-- ffn.viz                        # Feed-Forward Network
|   |-- rope_embeddings.viz            # Rotary Position Embeddings
|   |-- rmsnorm.viz                    # RMS Normalization
|   |-- mxfp4.viz                      # MXFP4 quantization
|   |-- weights.viz                    # Weight layout
|   |-- gqa_inf_vs_train.viz           # GQA inference vs training
|
|-- Resources ──────────────────────────────────────────────────
|-- resources/
|   |-- gpt_oss_20b_architecture.jpg   # Architecture diagram
|   |-- gpt_oss_rope.png               # RoPE config from model card
|   |-- gpt_oss_rope_embeddings_viz.png
|   |-- rms_norm.png                   # RMSNorm formula
|   |-- rms_norm_viz.png               # RMSNorm computation visualization
|
|-- traces/                            # Torch profiler trace outputs
|-- requirements.txt                   # Pinned dependencies
|-- LICENSE                            # Apache 2.0
```

## Notebooks

### `gpt_oss_from_scratch.ipynb`

Build the core model components from first principles with inline tests and visualizations:

- **RMSNorm** -- implement from the formula, verify against `torch.nn.RMSNorm`
- **Rotary Position Embeddings (RoPE)** -- full implementation with YaRN scaling, NTK-aware interpolation, frequency ramp blending, attention pattern visualization, and benchmarks

### `gpt_oss_reference_implementation.ipynb`

Load the model using the official `gpt-oss` library (included as a git submodule):

- Load from the original OpenAI checkpoint format (`data/gpt-oss-20b/original/`)
- Inspect the pure PyTorch `Transformer` and Triton-optimized `Transformer` side by side
- Compare parameter counts and memory footprint (~18 GB on GPU)

### `gpt_oss_from_unsloth.ipynb`

Load and inspect via the [Unsloth](https://github.com/unslothai/unsloth) library:

- 4-bit quantization support (BnB and MXFP4)
- LoRA fine-tuning setup
- Model config inspection (layer types, sliding window, expert routing)

### `gpt_oss_inference.ipynb`

End-to-end inference using HuggingFace `pipeline`:

- Load model on RTX 4090 (~14 GB VRAM with HF)
- Run example prompts (quantum mechanics, PyTorch code, web servers)
- Torch profiler integration with TensorBoard trace export

## Interactive CLI (`gpt_oss_manual_inference.py`)

Bypasses the HuggingFace `pipeline` to make every inference step explicit:

```
prompt -> chat template -> tokenize -> autoregressive generation -> decode
```

The generation loop is fully manual:
1. **Forward pass** through the transformer (with KV cache)
2. **Temperature scaling** of logits
3. **Top-p (nucleus) filtering**
4. **Sampling** via `torch.multinomial`
5. **Stream** each token to the terminal as it's generated

### KV Cache

- **Prefill (step 0)**: full prompt processed, KV tensors cached for all positions
- **Decode (step 1+)**: only the new token is passed, reusing cached keys/values -- O(1) per layer instead of O(seq_len)

### Usage

```bash
python gpt_oss_manual_inference.py                    # interactive REPL
python gpt_oss_manual_inference.py -v                  # verbose: token tables, top-k probs
python gpt_oss_manual_inference.py -p                  # torch profiler enabled
python gpt_oss_manual_inference.py -p --profile-steps 10 --temperature 0.3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model-id` | `openai/gpt-oss-20b` | HuggingFace model ID |
| `--max-tokens` | `4096` | Max new tokens per prompt |
| `--temperature` | `0.7` | Sampling temperature |
| `--top-p` | `0.9` | Nucleus sampling cutoff |
| `-v` / `--verbose` | off | Show analysis: chat template, token tables, top-k logits |
| `-p` / `--profile` | off | Enable torch profiler per prompt |
| `--profile-steps` | `5` | Number of active profiler steps |

## Triton Kernels

Custom CUDA kernels written in [Triton](https://triton-lang.org/):

### Softmax (`kernels/softmax.py`)

Fused online safe softmax kernel. Computes the numerically stable softmax in a single pass using the online algorithm (running max + running sum). Benchmarked against PyTorch native softmax.

### RMSNorm (`kernels/rms_norm.py`)

Triton RMSNorm with forward and backward passes. Includes:
- Fused kernel: `y = (x / sqrt(mean(x^2) + eps)) * weight`
- Backward pass for training
- Tests against PyTorch reference (`kernels/tests/test_rms_norm.py`)
- Benchmarks (`experiments/triton_rms_norm_benchmark.py`)

## Profiling

The CLI and notebook both support `torch.profiler` with TensorBoard export:

```bash
# Generate traces via CLI
python gpt_oss_manual_inference.py -p

# View in TensorBoard
tensorboard --logdir traces/
```

The profiler uses a schedule per prompt: `wait=1 -> warmup=1 -> active=N` to skip CUDA lazy-init noise and produce clean traces. Each prompt gets a unique trace directory (`traces/trace_<8-char-hex>/`).

Notable kernels in traces:
- `_topk_forward` -- MoE expert routing (`torch.topk` in every MLP block, every layer)
- GEMM kernels -- attention projections and expert MLPs
- `elementwise_kernel` -- RMSNorm, SwiGLU activations

## Setup

Requires Python 3.10+, CUDA-capable GPU (tested on RTX 4090 24 GB).

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

Model weights are downloaded automatically from HuggingFace on first run and cached in `~/.cache/huggingface/hub/`.

## License

Apache 2.0
