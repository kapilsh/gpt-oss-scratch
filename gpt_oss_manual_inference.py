"""
Interactive CLI for gpt-oss manual inference.

Breaks open the HuggingFace pipeline so you can see every step:
  chat template -> tokenize -> generate -> decode

Usage:
  python gpt_oss_manual_inference.py                        # defaults
  python gpt_oss_manual_inference.py --verbose              # show token tables
  python gpt_oss_manual_inference.py --max-tokens 512       # longer responses
  python gpt_oss_manual_inference.py --temperature 0.3      # less random
"""

import logging
import os
import uuid

# Suppress noisy progress bars and logs from HF ecosystem before imports
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import click
import structlog
import torch
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

transformers.logging.set_verbosity_error()

log = structlog.get_logger()
console = Console()

MODEL_ID = "openai/gpt-oss-20b"


def render_token_table(token_ids, tokenizer, title="Tokens"):
    """Build a rich Table of position / token-id / decoded-text."""
    table = Table(title=title, show_lines=False, pad_edge=False)
    table.add_column("pos", justify="right", style="dim")
    table.add_column("token_id", justify="right", style="cyan")
    table.add_column("text", style="green")
    for pos, tid in enumerate(token_ids):
        tid = tid if isinstance(tid, int) else tid.item()
        text = tokenizer.decode([tid])
        table.add_row(str(pos), str(tid), repr(text))
    return table


def render_top_k_table(top_k_indices, top_k_values, tokenizer, k=10):
    """Build a rich Table for top-k next-token probabilities."""
    table = Table(title=f"Top-{k} Next Tokens", show_lines=False, pad_edge=False)
    table.add_column("rank", justify="right", style="dim")
    table.add_column("token_id", justify="right", style="cyan")
    table.add_column("prob", justify="right", style="yellow")
    table.add_column("text", style="green")
    for i in range(k):
        tid = top_k_indices[i].item()
        p = top_k_values[i].item()
        text = tokenizer.decode([tid])
        table.add_row(str(i + 1), str(tid), f"{p:.4f}", repr(text))
    return table


def load_model(model_id):
    """Load tokenizer and model, return both."""
    console.print(Panel(f"Loading [bold]{model_id}[/bold] ...", style="blue"))

    with console.status("[bold blue]Loading tokenizer..."):
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    with console.status("[bold blue]Loading model weights..."):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )

    info_table = Table(title="Model Info", show_lines=False)
    info_table.add_column("property", style="dim")
    info_table.add_column("value", style="bold")
    info_table.add_row("tokenizer", type(tokenizer).__name__)
    info_table.add_row("vocab size", f"{tokenizer.vocab_size:,}")
    info_table.add_row("max length", f"{tokenizer.model_max_length:,}")
    info_table.add_row("eos token", f"{tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    info_table.add_row("device", str(model.device))
    console.print(info_table)
    console.print()

    return tokenizer, model


def top_p_filter(logits, top_p):
    """Zero out tokens outside the nucleus (top-p) probability mass."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Find where cumulative probability exceeds top_p
    cutoff_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[cutoff_mask] = float("-inf")

    # Scatter back to original ordering
    logits.scatter_(0, sorted_indices, sorted_logits)
    return logits


def sample_next_token(model, input_ids, temperature, top_p, past_key_values=None):
    """Single forward pass -> temperature scale -> top-p filter -> sample.

    With KV cache:
      - Prefill (step 0):  input_ids = full prompt, past_key_values = None
        The model processes every token and returns the KV cache.
      - Decode  (step 1+): input_ids = just the new token, past_key_values = cached KVs
        The model only computes attention for the new token, reusing cached
        keys/values from all prior positions.  This is O(1) per layer instead
        of O(seq_len).
    """
    # 1. Forward pass
    #    use_cache=True tells the model to return (and accept) past_key_values
    outputs = model(
        input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    # logits shape: (batch=1, seq_len_or_1, vocab_size)
    # We only care about the LAST position's prediction
    next_token_logits = outputs.logits[0, -1, :]

    # The model returns updated KV cache covering all positions seen so far
    new_past_key_values = outputs.past_key_values

    # 2. Temperature scaling: divide logits by temperature
    #    lower temp -> sharper distribution (more deterministic)
    #    higher temp -> flatter distribution (more random)
    scaled_logits = next_token_logits / temperature

    # 3. Top-p (nucleus) filtering: keep only tokens whose cumulative
    #    probability mass is within top_p, set the rest to -inf
    filtered_logits = top_p_filter(scaled_logits, top_p)

    # 4. Convert logits to probabilities via softmax
    probs = torch.softmax(filtered_logits, dim=-1)

    # 5. Sample: draw one token from the probability distribution
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token, probs, new_past_key_values


def run_inference(prompt, tokenizer, model, max_tokens, temperature, top_p, verbose, profile, profile_steps):
    """Run a single inference turn, continuing until EOS or context limit."""
    messages = [{"role": "user", "content": prompt}]
    eos_id = tokenizer.eos_token_id
    context_limit = getattr(tokenizer, "model_max_length", 8192)

    # ── Analysis ──────────────────────────────────────────────────────────

    # Step 1: Chat template
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Step 2: Tokenize
    input_ids = tokenizer.encode(prompt_str, return_tensors="pt").to(model.device)
    prompt_len = input_ids.shape[1]
    log.info("tokenized", seq_len=prompt_len)

    if verbose:
        console.print()
        console.rule("[bold dim]Analysis", style="dim")
        console.print(Panel(prompt_str, title="Chat Template Output", style="dim"))
        console.print(render_token_table(input_ids[0], tokenizer, title="Input Tokens"))
        console.print()

    # ── Generation (streamed) ─────────────────────────────────────────────

    # Step 3: Autoregressive generation — token by token
    #   This is what model.generate() hides:
    #     for each new token:
    #       a) forward pass (only new token with KV cache)
    #       b) take logits at the last position
    #       c) scale by temperature
    #       d) apply top-p nucleus filtering
    #       e) sample from the resulting distribution
    #       f) append to KV cache, repeat
    generated_tokens = []
    past_kv = None  # KV cache: None on first step (prefill), reused after

    trace_id = uuid.uuid4().hex[:8]
    trace_dir = os.path.join("traces", f"trace_{trace_id}")
    profiler = None

    if profile:
        os.makedirs(trace_dir, exist_ok=True)
        # Schedule: skip 1 step (wait) -> 1 warmup step (no recording) -> profile N active steps
        # This avoids capturing CUDA lazy-init overhead in the trace
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=profile_steps,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
            record_shapes=False,
            with_stack=True,
            profile_memory=False,
        )
        profiler.start()
        log.info(
            "profiler_started",
            trace_dir=trace_dir,
            schedule=f"wait=1, warmup=1, active={profile_steps}",
        )

    # Show top-k probabilities for the first token before streaming
    first_token_probs = None

    console.print()
    console.rule("[bold green]Response", style="green")

    with torch.no_grad():
        for step in range(max_tokens):
            total_seq_len = prompt_len + len(generated_tokens)
            if total_seq_len >= context_limit:
                log.warning("context_limit_reached", limit=context_limit)
                break

            if step == 0:
                # Prefill: feed the full prompt, build the KV cache
                forward_ids = input_ids
            else:
                # Decode: feed only the last generated token, reuse KV cache
                forward_ids = next_token.unsqueeze(0)  # shape (1, 1)

            next_token, probs, past_kv = sample_next_token(
                model, forward_ids, temperature, top_p, past_key_values=past_kv
            )
            token_id = next_token.item()
            generated_tokens.append(token_id)

            if step == 0:
                first_token_probs = probs

            # Stream: decode and print each token immediately
            if token_id != eos_id:
                token_text = tokenizer.decode([token_id])
                print(token_text, end="", flush=True)

            # Step the profiler every token so the schedule advances
            if profiler is not None:
                profiler.step()

            if token_id == eos_id:
                log.info("hit_eos", step=step + 1)
                break

    # End the streamed line
    print()

    # ── Summary ───────────────────────────────────────────────────────────

    total_generated = len(generated_tokens)
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    console.print()
    console.rule("[bold dim]Summary", style="dim")
    log.info("generation_complete", total_tokens=total_generated)

    # Render the full response as formatted markdown
    console.print(
        Panel(
            Markdown(response_text),
            title=f"Response  [dim]({total_generated} tokens)[/dim]",
            border_style="green",
        )
    )

    if profiler is not None:
        profiler.stop()
        console.print(
            Panel(
                f"Trace saved to [bold]{trace_dir}/[/bold]\n"
                f"[dim]Schedule: wait=1, warmup=1, active={profile_steps}[/dim]\n"
                "[dim]View with: tensorboard --logdir traces/[/dim]",
                title="Profiler",
                border_style="yellow",
            )
        )

    if verbose:
        if first_token_probs is not None:
            top10 = torch.topk(first_token_probs, k=10)
            console.print(render_top_k_table(
                top10.indices, top10.values, tokenizer,
            ))
            console.print()

        console.print(
            render_token_table(generated_tokens, tokenizer, title="Generated Tokens")
        )
        console.print()


@click.command()
@click.option("--model-id", default=MODEL_ID, show_default=True, help="HuggingFace model ID.")
@click.option("--max-tokens", default=4096, show_default=True, help="Max new tokens to generate.")
@click.option("--temperature", default=0.7, show_default=True, help="Sampling temperature.")
@click.option("--top-p", default=0.9, show_default=True, help="Nucleus sampling top-p.")
@click.option("--verbose", "-v", is_flag=True, help="Show token tables and logits.")
@click.option("--profile", "-p", is_flag=True, help="Profile each prompt with torch profiler (saved to traces/).")
@click.option("--profile-steps", default=5, show_default=True, help="Number of active profiler steps (token iterations).")
def cli(model_id, max_tokens, temperature, top_p, verbose, profile, profile_steps):
    """Interactive GPT-OSS inference — see every step the pipeline hides."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )

    tokenizer, model = load_model(model_id)

    console.print(
        Panel(
            "[bold]Type a prompt and press Enter. Ctrl-D or 'quit' to exit.[/bold]\n"
            f"[dim]max_tokens={max_tokens}  temperature={temperature}  top_p={top_p}  verbose={verbose}  profile={profile}  profile_steps={profile_steps}[/dim]",
            title="gpt-oss interactive",
            border_style="bright_blue",
        )
    )

    while True:
        try:
            prompt = console.input("[bold bright_blue]> [/bold bright_blue]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye![/dim]")
            break

        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            console.print("[dim]bye![/dim]")
            break

        run_inference(prompt, tokenizer, model, max_tokens, temperature, top_p, verbose, profile, profile_steps)
        console.print()


if __name__ == "__main__":
    cli()
