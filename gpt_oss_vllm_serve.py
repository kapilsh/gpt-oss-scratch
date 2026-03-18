"""
vLLM-backed server for GPT-OSS 20B.

Mirrors the same pipeline as gpt_oss_manual_inference.py but uses vLLM's
AsyncLLMEngine for continuous batching and PagedAttention instead of the
naive one-token-at-a-time HuggingFace loop.

Pipeline (explicit, just like the manual inference script):
  1. Parse chat messages from the HTTP request
  2. Apply the Harmony chat template via HuggingFace AutoTokenizer
  3. Tokenize to token IDs
  4. Hand token IDs to vLLM (no re-tokenization, full control)
  5. Stream generated tokens back via Server-Sent Events (SSE)

Usage:
  # Single GPU
  python gpt_oss_vllm_serve.py

  # Two GPUs (tensor parallel)
  python gpt_oss_vllm_serve.py --tp 2

  # Custom model / port
  python gpt_oss_vllm_serve.py --model-id openai/gpt-oss-20b --port 8080

  # Then query with any OpenAI-compatible client:
  curl http://localhost:8000/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
"""

import asyncio
import json
import logging
import os
import time
import uuid

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import click
import structlog
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.inputs import TokensPrompt

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

REQUESTS_TOTAL = Counter(
    "gptoss_requests_total",
    "Total number of chat completion requests",
    ["status"],          # "success" | "error"
)
PROMPT_TOKENS = Counter(
    "gptoss_prompt_tokens_total",
    "Total prompt tokens processed across all requests",
)
COMPLETION_TOKENS = Counter(
    "gptoss_completion_tokens_total",
    "Total completion tokens generated across all requests",
)
REQUEST_LATENCY = Histogram(
    "gptoss_request_latency_seconds",
    "End-to-end request latency (from token IDs ready to last token out)",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120],
)
TIME_TO_FIRST_TOKEN = Histogram(
    "gptoss_ttft_seconds",
    "Time to first token (prefill latency)",
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5],
)
REQUESTS_IN_FLIGHT = Gauge(
    "gptoss_requests_in_flight",
    "Number of requests currently being processed",
)

# ---------------------------------------------------------------------------
# Request / Response types  (OpenAI-compatible subset)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-oss-20b"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=4096, ge=1, le=131072)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=-1)                  # -1 = disabled
    stream: bool = False
    stop: list[str] | None = None
    profile: bool = False          # wrap generation with torch.profiler; trace saved to traces/


def _make_chunk(
    request_id: str, model: str, delta: dict, finish_reason: str | None = None
) -> str:
    payload = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _make_full_response(
    request_id: str, model: str, content: str, prompt_tokens: int, completion_tokens: int
) -> dict:
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Server class
# ---------------------------------------------------------------------------

class GPTOSSServer:
    """
    Wraps vLLM's AsyncLLMEngine with the GPT-OSS Harmony tokenizer/chat
    template, exposing the same explicit pipeline as gpt_oss_manual_inference.
    """

    def __init__(self, model_id: str, tp: int, gpu_memory_utilization: float):
        self.model_id = model_id
        self.app = FastAPI(title="gpt-oss vLLM server")
        self._register_routes()

        log.info("loading_tokenizer", model_id=model_id)
        # HuggingFace tokenizer carries the Harmony chat template and special
        # token definitions — same as gpt_oss_manual_inference.py.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.eos_token_id = self.tokenizer.eos_token_id

        log.info(
            "tokenizer_loaded",
            vocab_size=self.tokenizer.vocab_size,
            eos_token=self.tokenizer.eos_token,
            eos_id=self.eos_token_id,
        )

        log.info("building_vllm_engine", tp=tp, gpu_memory_utilization=gpu_memory_utilization)
        engine_args = AsyncEngineArgs(
            model=model_id,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,     # reuse KV blocks for shared prefixes
            dtype="bfloat16",               # match model weights dtype
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        log.info("engine_ready")

    # ------------------------------------------------------------------
    # Tokenization helpers (mirrors gpt_oss_manual_inference.py pipeline)
    # ------------------------------------------------------------------

    def apply_chat_template(self, messages: list[ChatMessage]) -> str:
        """Step 1: render messages into the Harmony prompt string."""
        raw = [{"role": m.role, "content": m.content} for m in messages]
        return self.tokenizer.apply_chat_template(
            raw, tokenize=False, add_generation_prompt=True
        )

    def tokenize(self, prompt_str: str) -> list[int]:
        """Step 2: convert the rendered prompt string to token IDs."""
        return self.tokenizer.encode(prompt_str)

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=False)

    # ------------------------------------------------------------------
    # Profiler helper
    # ------------------------------------------------------------------

    def _make_profiler(self, request_id: str) -> tuple:
        """
        Build a torch.profiler.profile context manager and return
        (profiler, trace_dir).  The caller is responsible for .start()
        and .stop().  Traces are written in TensorBoard format so you
        can inspect with:  tensorboard --logdir traces/
        """
        trace_dir = os.path.join("traces", f"trace_{request_id}")
        os.makedirs(trace_dir, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )
        return profiler, trace_dir

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    async def generate_stream(
        self, request_id: str, token_ids: list[int], sampling_params: SamplingParams
    ):
        """
        Async generator yielding (delta_text, finish_reason, is_finished).

        We pass pre-tokenized token IDs via TokensPrompt so vLLM never
        re-tokenizes the input — full transparency over the pipeline.

        Also records TTFT into Prometheus histograms.
        """
        prompt = TokensPrompt(prompt_token_ids=token_ids)
        gen = self.engine.generate(prompt, sampling_params, request_id=request_id)

        previous_text_len = 0
        previous_token_count = 0
        first_token = True
        t_start = time.perf_counter()

        async for output in gen:
            if not output.outputs:
                continue
            o = output.outputs[0]
            delta_text = o.text[previous_text_len:]
            previous_text_len = len(o.text)
            # Actual tokens generated this step (from vLLM's token ID list)
            new_tokens = len(o.token_ids) - previous_token_count
            previous_token_count = len(o.token_ids)

            if first_token and delta_text:
                TIME_TO_FIRST_TOKEN.observe(time.perf_counter() - t_start)
                first_token = False

            finished = o.finish_reason is not None
            yield delta_text, o.finish_reason, finished, new_tokens
            if finished:
                break

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------

    def _register_routes(self):
        app = self.app

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.get("/metrics")
        async def metrics():
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

        @app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "openai",
                    }
                ],
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest, raw: Request):
            request_id = uuid.uuid4().hex

            # ── Step 1: chat template ──────────────────────────────────
            prompt_str = self.apply_chat_template(request.messages)
            log.info("chat_template_applied", request_id=request_id, chars=len(prompt_str))

            # ── Step 2: tokenize ───────────────────────────────────────
            token_ids = self.tokenize(prompt_str)
            prompt_tokens = len(token_ids)
            log.info("tokenized", request_id=request_id, prompt_tokens=prompt_tokens)

            # ── Step 3: build sampling params ─────────────────────────
            stop_token_ids = [self.eos_token_id] if self.eos_token_id is not None else []
            sampling = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k if request.top_k > 0 else -1,
                max_tokens=request.max_tokens,
                stop=request.stop or [],
                stop_token_ids=stop_token_ids,
            )

            # ── Step 4: optionally set up profiler ────────────────────
            profiler, trace_dir = (
                self._make_profiler(request_id) if request.profile else (None, None)
            )
            if profiler:
                profiler.start()
                log.info("profiler_started", request_id=request_id, trace_dir=trace_dir)

            # ── Step 5: stream or collect ─────────────────────────────
            REQUESTS_IN_FLIGHT.inc()
            PROMPT_TOKENS.inc(prompt_tokens)
            t_request_start = time.perf_counter()

            if request.stream:
                async def event_stream():
                    completion_tokens = 0
                    try:
                        yield _make_chunk(
                            request_id, request.model, {"role": "assistant", "content": ""}
                        )
                        async for delta_text, finish_reason, finished, n_tokens in self.generate_stream(
                            request_id, token_ids, sampling
                        ):
                            completion_tokens += n_tokens
                            if delta_text:
                                yield _make_chunk(
                                    request_id, request.model, {"content": delta_text}
                                )
                            if finished:
                                yield _make_chunk(
                                    request_id, request.model, {}, finish_reason=finish_reason or "stop"
                                )
                        yield "data: [DONE]\n\n"
                        COMPLETION_TOKENS.inc(completion_tokens)
                        REQUESTS_TOTAL.labels(status="success").inc()
                    except Exception:
                        REQUESTS_TOTAL.labels(status="error").inc()
                        raise
                    finally:
                        if profiler:
                            profiler.stop()
                            log.info("profiler_saved", request_id=request_id, trace_dir=trace_dir)
                        REQUESTS_IN_FLIGHT.dec()
                        REQUEST_LATENCY.observe(time.perf_counter() - t_request_start)

                headers = {"X-Profile-Trace": trace_dir} if trace_dir else {}
                return StreamingResponse(
                    event_stream(), media_type="text/event-stream", headers=headers
                )

            else:
                try:
                    full_text = ""
                    completion_tokens = 0
                    async for delta_text, _, finished, n_tokens in self.generate_stream(
                        request_id, token_ids, sampling
                    ):
                        full_text += delta_text
                        completion_tokens += n_tokens

                    COMPLETION_TOKENS.inc(completion_tokens)
                    REQUESTS_TOTAL.labels(status="success").inc()
                    log.info(
                        "generation_complete",
                        request_id=request_id,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        trace_dir=trace_dir,
                    )
                    resp = _make_full_response(
                        request_id, request.model, full_text, prompt_tokens, completion_tokens
                    )
                    if trace_dir:
                        resp["profile_trace"] = trace_dir
                    headers = {"X-Profile-Trace": trace_dir} if trace_dir else {}
                    return JSONResponse(resp, headers=headers)
                except Exception:
                    REQUESTS_TOTAL.labels(status="error").inc()
                    raise
                finally:
                    if profiler:
                        profiler.stop()
                        log.info("profiler_saved", request_id=request_id, trace_dir=trace_dir)
                    REQUESTS_IN_FLIGHT.dec()
                    REQUEST_LATENCY.observe(time.perf_counter() - t_request_start)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option("--model-id", default="openai/gpt-oss-20b", show_default=True,
              help="HuggingFace model ID.")
@click.option("--tp", default=1, show_default=True,
              help="Tensor parallel size (number of GPUs).")
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host.")
@click.option("--port", default=8000, show_default=True, help="Bind port.")
@click.option("--gpu-memory-utilization", default=0.90, show_default=True,
              help="Fraction of GPU memory to use for the KV cache (0–1).")
@click.option("--log-level", default="info", show_default=True,
              type=click.Choice(["debug", "info", "warning", "error"]),
              help="Uvicorn log level.")
def serve(model_id, tp, host, port, gpu_memory_utilization, log_level):
    """
    Serve GPT-OSS 20B with vLLM (PagedAttention + continuous batching).

    Exposes an OpenAI-compatible /v1/chat/completions endpoint.
    Applies the Harmony chat template explicitly, just like the manual
    inference script, before handing token IDs to vLLM.
    """
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]
    )

    server = GPTOSSServer(
        model_id=model_id,
        tp=tp,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    log.info("starting_server", host=host, port=port, tp=tp, model_id=model_id)
    uvicorn.run(server.app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    serve()
