# vLLM-like Inference Engine for GPT-OSS 20B — Implementation Plan

## Architecture Overview

The engine has 7 core components, loosely mirroring vLLM's design but tailored to GPT-OSS's MoE + GQA architecture:

```
┌──────────────────────────────────────────────────────────┐
│                      API Server (FastAPI)                  │
│           /generate, /v1/chat/completions (SSE)           │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                     LLMEngine (orchestrator)               │
│   add_request() → step() → outputs                        │
└───┬──────────────────┬──────────────────────┬────────────┘
    │                  │                      │
┌───▼──────┐  ┌────────▼────────┐  ┌─────────▼──────────┐
│Scheduler │  │  Block Manager  │  │  Output Processor  │
│(FCFS+    │  │ (PagedAttention)│  │  (sampling, detok) │
│preempt)  │  │  block tables   │  │                    │
└───┬──────┘  └────────┬────────┘  └────────────────────┘
    │                  │
┌───▼──────────────────▼────────────────────────────────┐
│              Model Executor (dispatch)                  │
└───┬──────────────────────────────────┬────────────────┘
    │                                  │
┌───▼──────────────────┐  ┌────────────▼──────────────┐
│  Worker 0 (GPU 0)    │  │  Worker 1 (GPU 1)         │
│  32 Q-heads, 4 KV    │  │  32 Q-heads, 4 KV         │
│  16 experts          │  │  16 experts               │
│  PagedAttn kernel    │  │  PagedAttn kernel         │
└──────────────────────┘  └───────────────────────────┘
```

---

## Phase 1: Block Manager (PagedAttention Core)

**File: `engine/block_manager.py`**

This is the centerpiece. Instead of one contiguous KV buffer per sequence, we manage a pool of
fixed-size **blocks** (pages) and map logical sequence positions to physical block slots.

### Block Layout for GPT-OSS 20B

```
Per block (on-GPU tensor shape):
  K cache: [num_layers, block_size, num_kv_heads, head_dim]
  V cache: [num_layers, block_size, num_kv_heads, head_dim]

  With block_size=16, num_layers=24, num_kv_heads=8, head_dim=64:
  → 24 × 16 × 8 × 64 × 2 (K+V) × 2 (bf16 bytes) = 786,432 bytes ≈ 768 KB per block
```

**With 2 GPUs (tensor parallel, TP=2):** each GPU holds 4 KV heads (not 8), so blocks are
halved in memory: ~384 KB each.

### Key Data Structures

```python
@dataclass
class PhysicalBlock:
    block_id: int          # index into the global KV cache tensor
    device: torch.device   # which GPU
    ref_count: int = 0     # for copy-on-write (prefix caching)

@dataclass
class LogicalBlock:
    physical_block: PhysicalBlock | None
    num_tokens: int        # tokens filled in this block

class BlockTable:
    """Maps a sequence's logical blocks → physical blocks."""
    logical_to_physical: list[PhysicalBlock]
```

### BlockManager Responsibilities

```python
class BlockManager:
    def __init__(self, num_gpu_blocks, block_size, num_layers, ...):
        # Preallocate the entire KV cache on GPU
        # k_cache: [num_blocks, num_layers, block_size, num_kv_heads, head_dim]
        # v_cache: [num_blocks, num_layers, block_size, num_kv_heads, head_dim]
        ...

    def can_allocate(self, seq: Sequence) -> AllocStatus
    def allocate(self, seq_group: SequenceGroup) -> None
    def append_slot(self, seq: Sequence) -> CopyOnWrite | None
    def fork(self, parent: Sequence, child: Sequence) -> None   # for beam search
    def free(self, seq: Sequence) -> None
    def swap_in(self, seq_group, cpu_blocks) -> None   # preemption recovery
    def swap_out(self, seq_group) -> dict              # preemption
```

### Block Sizing Strategy

At startup, profile available VRAM:
1. Load model weights, measure used memory
2. Reserve headroom for activations (default 1 GB per GPU)
3. Divide remaining memory by block size to get `num_gpu_blocks`
4. Target: 10–15% GPU memory for activations, 85–90% for KV cache

---

## Phase 2: Sequence & Request Lifecycle

**File: `engine/sequence.py`**

```python
class SequenceStatus(Enum):
    WAITING    = "waiting"    # in the request queue
    RUNNING    = "running"    # in current forward batch
    SWAPPED    = "swapped"    # preempted to CPU
    FINISHED   = "finished"

@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    block_table: BlockTable           # logical→physical mapping
    status: SequenceStatus
    sampling_params: SamplingParams
    # For sliding window: track which blocks are in-window vs evicted

@dataclass
class SequenceGroup:
    """Represents one user request (possibly multiple beam sequences)."""
    request_id: str
    sequences: list[Sequence]
    arrival_time: float
```

---

## Phase 3: Scheduler

**File: `engine/scheduler.py`**

Continuous batching: every `step()`, the scheduler decides which sequences to run.

```python
class SchedulerOutput:
    scheduled_seq_groups: list[SequenceGroup]   # prefill + decode
    blocks_to_swap_in: dict                      # CPU→GPU for preempted seqs
    blocks_to_swap_out: dict                     # GPU→CPU for preemption
    blocks_to_copy: dict                         # CoW for prefix cache

class Scheduler:
    def schedule(self) -> SchedulerOutput:
        # Priority order:
        # 1. Swap in preempted sequences (if GPU blocks available)
        # 2. Promote WAITING sequences into RUNNING (prefill)
        # 3. Continue RUNNING sequences (decode)
        # 4. Preempt lowest-priority if OOM (swap out or abort)
```

**Key policies:**
- **FCFS** within each priority tier
- **Preemption via swapping**: evict lowest-priority running sequences to CPU when a new prefill
  can't fit
- **Budget-based scheduling**: limit max tokens per step (prefill + decode combined) to keep
  latency predictable
- **Sliding window awareness**: GPT-OSS alternates full/sliding-window attention every other
  layer; sequences with long contexts can free evicted blocks outside the 128-token window

---

## Phase 4: PagedAttention Triton Kernel

**File: `kernels/paged_attention.py`**

This is the custom CUDA/Triton kernel that replaces the standard `attention()` call during
decode. During prefill, standard FlashAttention (already in
`gpt-oss/gpt_oss/triton/attention.py`) is used.

### Prefill vs Decode Distinction

| Phase | Sequence length | Use |
|-------|----------------|-----|
| Prefill | Full prompt (potentially 1000s of tokens) | Existing FlashAttention kernel |
| Decode | Single new token | **PagedAttention kernel** |

### PagedAttention Decode Kernel Design

```
Input:
  q:            [batch, 1, num_q_heads, head_dim]   (current token query)
  k_cache:      [num_blocks, num_layers, block_size, num_kv_heads, head_dim]
  v_cache:      [num_blocks, num_layers, block_size, num_kv_heads, head_dim]
  block_tables: [batch, max_blocks_per_seq]          (logical→physical mapping)
  context_lens: [batch]                              (actual token count per seq)
  layer_idx:    int

Algorithm (per query head, per sequence):
  1. For each logical block in block_tables[seq]:
     → look up physical block ID
     → load k_cache[block_id, layer_idx, :, kv_head, :]  (block_size × head_dim)
     → load v_cache[block_id, layer_idx, :, kv_head, :]
  2. Online softmax (flash-style) across all blocks:
     → accumulate (max, sum, output) incrementally
  3. GQA: each KV head serves 8 query heads (64Q / 8KV)
     → in kernel: query head → kv_head = q_head // 8
  4. Write output: [batch, 1, num_q_heads, head_dim]
```

**Sliding window in decode**: For sliding-window layers, only load blocks within the last 128
token positions. The block table and context_lens already encode which blocks are valid.

### Kernel Signature

```python
@triton.jit
def paged_attention_decode_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr,
    block_tables_ptr, context_lens_ptr,
    output_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    gqa_ratio: tl.constexpr,           # = num_q_heads // num_kv_heads = 8
    sliding_window: tl.constexpr,      # 128 or -1 for full attention
    BLOCK_KV: tl.constexpr,            # tokens loaded per program iteration
):
    ...
```

---

## Phase 5: Model Executor with Tensor Parallelism

**File: `engine/model_executor.py`, `engine/worker.py`**

### Tensor Parallelism (TP=2) Design

Split the model column-wise (Megatron-style):

| Layer Type | TP Split Strategy |
|------------|------------------|
| QKV projection | Q: split 64→32 heads/GPU; KV: split 8→4 heads/GPU |
| O projection | Split input columns, all-reduce output |
| MoE gate | Replicated (small) |
| MoE expert MLP1 | Split intermediate dim (column parallel) |
| MoE expert MLP2 | Split input dim (row parallel) + all-reduce |
| RMSNorm | Replicated |
| Embed / LM head | Vocab parallel (split vocab 201088 / 2) |

**Expert Parallelism (EP) for MoE:**
- 32 experts → 16 per GPU
- Tokens routed to experts not on local GPU → NCCL `all-to-all`
- With TP=2 on the same node, latency is very low (NVLink)

### Worker Per-GPU

```python
class Worker:
    def __init__(self, rank: int, world_size: int, model_config, ...):
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        # NCCL process group for all-reduce
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        self.model = self._load_sharded_model()
        self.kv_cache = self._allocate_kv_cache()   # local blocks for this GPU

    def execute_model(self, batch: BatchInput) -> BatchOutput:
        # 1. Embed tokens
        # 2. For each layer:
        #    a. RMSNorm (local)
        #    b. Attention (PagedAttention decode OR FlashAttn prefill)
        #       + all-reduce across TP group
        #    c. RMSNorm (local)
        #    d. MoE routing + expert computation
        #       + all-to-all for expert parallelism + all-reduce output
        # 3. Final RMSNorm + LM head
        # 4. Return logits to rank 0 (or gather)
```

### Mixed Batch (Prefill + Decode Together)

Each step can contain:
- Some sequences in **prefill** (variable-length prompts)
- Many sequences in **decode** (single-token forward)

These are processed in the same forward pass by chunking the input and using appropriate
attention kernels per sequence.

---

## Phase 6: Sampling

**File: `engine/sampling.py`**

```python
class SamplingParams:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1            # -1 = disabled
    repetition_penalty: float = 1.0
    max_tokens: int = 4096
    stop_sequences: list[str] = []

class Sampler:
    def forward(self, logits, sampling_params_list) -> list[int]:
        # Batched sampling supporting mixed strategies per request
        # 1. Apply repetition penalty
        # 2. Temperature scaling
        # 3. Top-k filter
        # 4. Top-p (nucleus) filter
        # 5. Multinomial sample (or argmax for greedy)
```

---

## Phase 7: LLMEngine (Orchestrator)

**File: `engine/llm_engine.py`**

```python
class LLMEngine:
    def __init__(self, model_config, parallel_config, cache_config):
        self.scheduler = Scheduler(...)
        self.block_manager = BlockManager(...)
        self.executor = ModelExecutor(...)    # spawns workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def add_request(self, request_id, prompt, sampling_params) -> None:
        token_ids = self.tokenizer.encode(prompt)
        seq = Sequence(seq_id=..., prompt_token_ids=token_ids, ...)
        seq_group = SequenceGroup(request_id=request_id, sequences=[seq], ...)
        self.scheduler.add_seq_group(seq_group)

    def step(self) -> list[RequestOutput]:
        # 1. Scheduler decides what runs this step
        sched_output = self.scheduler.schedule()

        # 2. Swap blocks between CPU/GPU as needed
        self.executor.execute_swaps(sched_output)

        # 3. Build batch input (token IDs, block tables, context lens, positions)
        batch = self._prepare_batch(sched_output)

        # 4. Forward pass
        logits = self.executor.execute_model(batch)

        # 5. Sample next tokens
        sampled = self.sampler.sample(logits, ...)

        # 6. Update sequences (append token, free finished seqs)
        outputs = self._process_outputs(sampled, sched_output)
        return outputs
```

---

## Phase 8: API Server

**File: `server/api_server.py`**

```python
# FastAPI server with OpenAI-compatible endpoints
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        return StreamingResponse(
            generate_stream(request), media_type="text/event-stream"
        )
    else:
        return await generate_full(request)

# Background loop driving the engine
async def engine_loop():
    while True:
        if engine.has_requests():
            outputs = engine.step()
            for output in outputs:
                if output.finished:
                    notify_waiting_client(output)
        await asyncio.sleep(0)  # yield to event loop
```

---

## File Structure

```
gpt-oss-scratch/
├── inference_engine/
│   ├── __init__.py
│   ├── config.py                    # EngineConfig, CacheConfig, ParallelConfig
│   ├── engine/
│   │   ├── llm_engine.py            # Main orchestrator
│   │   ├── scheduler.py             # Continuous batching scheduler
│   │   ├── block_manager.py         # PagedAttention block management
│   │   ├── sequence.py              # Sequence, SequenceGroup, BlockTable
│   │   ├── model_executor.py        # Dispatch to workers
│   │   ├── worker.py                # Per-GPU worker (model shard + KV cache)
│   │   └── sampling.py              # Batched sampler
│   ├── kernels/
│   │   ├── paged_attention.py       # PagedAttention Triton decode kernel
│   │   └── block_copy.py            # Copy-on-write block copy kernel
│   ├── model/
│   │   ├── gpt_oss_tp.py            # Tensor-parallel GPT-OSS model wrapper
│   │   └── weight_loader.py         # Shard and distribute checkpoint weights
│   └── server/
│       ├── api_server.py            # FastAPI server
│       └── protocol.py              # OpenAI-compatible request/response types
└── run_engine.py                    # Entry point
```

---

## Build Order (Recommended Sequence)

| Step | Component | Why First |
|------|-----------|-----------|
| 1 | `sequence.py` + `config.py` | Foundation data structures everything depends on |
| 2 | `block_manager.py` | Core PagedAttention logic, testable in isolation |
| 3 | `scheduler.py` | Drives the engine loop, testable with mock model |
| 4 | `paged_attention.py` (Triton kernel) | Most complex kernel; build/test independently |
| 5 | `worker.py` (single GPU first) | Get end-to-end working before adding TP |
| 6 | `llm_engine.py` | Wire everything together |
| 7 | `sampling.py` | Already partially exists in inference script |
| 8 | Tensor parallel (`model_executor.py`, `gpt_oss_tp.py`) | Additive; single GPU must work first |
| 9 | `api_server.py` | Final layer, easiest to add last |

---

## Key Design Decisions & Trade-offs

### PagedAttention Block Size
- **Larger blocks (32–64)**: Better GPU memory bandwidth utilization per kernel call, worse fragmentation
- **Smaller blocks (8–16)**: Less fragmentation, more kernel launch overhead
- **Recommendation**: Start with `block_size=16`; profile and tune

### KV Cache Dtype
- Current model uses `bfloat16`; KV cache should match to avoid cast overhead
- Future optimization: FP8 KV cache (2x memory saving, needs quantization-aware attention kernel)

### 2-GPU Memory Budget Example (2× RTX 4090, 24 GB each)

```
Total per GPU:    24 GB
Model weights:    ~7 GB  (MXFP4 expert weights give ~4× compression on expert params)
Activations:      ~1 GB  (reserved headroom)
KV cache:         ~16 GB → ~16 GB / (384 KB/block with TP2) ≈ 41,000 blocks
                  = 41,000 × 16 tokens = 656,000 max cached tokens simultaneously
```

### Sliding Window Interaction
- Layers 0, 2, 4, ... use sliding window = 128 tokens
- For sliding-window layers, blocks older than 128 positions can be freed during decode
- Block manager needs per-layer awareness (or conservatively keep all blocks)
- Optimization: track `min_active_block` per layer and free aggressively for SW layers

### MoE + Tensor Parallelism
- **Option A (simpler)**: TP-only — split experts column/row parallel within each GPU pair;
  no all-to-all, just all-reduce
- **Option B (optimal)**: EP (Expert Parallel) with TP — 16 experts/GPU + all-to-all for
  cross-GPU tokens
- **Recommendation**: Start with Option A (pure TP); revisit with EP when scaling beyond 2 GPUs

---

## Milestones

| # | Milestone | Goal |
|---|-----------|------|
| M1 | Single-GPU PagedAttention decode | Validate correctness vs. `gpt_oss_manual_inference.py` |
| M2 | Scheduler + continuous batching (1 GPU) | Measure throughput gain vs. sequential inference |
| M3 | 2-GPU tensor parallelism | Validate output matches single-GPU; measure latency reduction |
| M4 | Preemption + CPU swap | Handle more concurrent requests than GPU memory allows |
| M5 | HTTP server | Production-ready, OpenAI-compatible API |
| M6 | Prefix caching (CoW blocks) | Throughput improvement for shared system prompts |
