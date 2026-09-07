# Prefill Context-Parallel Attention POC for Trainium2

This repository is a from-scratch learning POC for the **history-only** part of
prefill context-parallel attention.  It is intentionally separate from
vLLM/Neuron integrations and prioritizes readable, testable behavior over
performance.

The target full-machine configuration is:

- `trn2.48xlarge`: 16 Trainium2 chips
- LNC=2, exposed as 64 logical PCP ranks
- Qwen3.5-397B-A17B dimensions: 32 Q heads, 2 KV heads, head dimension 256
- 4096 current tokens and 131072 historical tokens
- 512-token KV-cache blocks and 64-token interleave chunks
- 64 local Q tokens and 4 local historical KV blocks per rank

Only historical KV participates in attention in this first version.  Current
tokens do not attend to one another yet.

## Layout

For global token position `p`:

```text
chunk       = p // interleave_size
owner_rank  = chunk % pcp_size
local_chunk = chunk // pcp_size
local_pos   = local_chunk * interleave_size + p % interleave_size
```

Each rank owns:

```text
Q: [num_q_heads, local_q_len, head_dim]
K: [max_local_blocks, num_kv_heads, block_size, head_dim]
V: [max_local_blocks, num_kv_heads, block_size, head_dim]
O: [num_q_heads, local_q_len, head_dim]
```

The NKI kernel loads its Q shard into SBUF once.  The outer loop selects one
local KV block, and the inner loop circulates that block through a shared-HBM
ping/pong buffer.  Online softmax state remains in SBUF across both loops.

Neuron Compiler 2.27 currently rejects a collective inside a device-side
dynamic loop.  Consequently, this first hardware POC executes
`max_local_blocks` outer iterations and applies a runtime validity mask derived
from `actual_history_len`.  One compiled artifact therefore produces correct
results for shorter histories, but still pays the maximum bucket's
communication cost.  Length bucketing (or lifting this compiler restriction)
is the next step when performance work begins.

## Quick start

Activate the installed inference environment:

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/activate
```

Run host-only unit tests:

```bash
python -m pytest -q
```

Run the deterministic CPU ring model (small by default):

```bash
python run_poc.py --backend cpu --preset tiny
```

Run a small Neuron test on four logical ranks:

```bash
torchrun --standalone --nproc_per_node=4 run_poc.py \
  --backend neuron --preset tiny --pcp-size 4
```

## Benchmark

The benchmark compiles one NKI invocation with `torch.compile`, executes one
untimed warmup (including any cold compilation), and then measures synchronized
calls to that same artifact.  Every measured call copies only one output
element to the host as a completion marker.  For each iteration, rank 0 uses
the maximum latency across all PCP ranks and reports min/median/mean/P90/max.

Run the requested full-machine case with 20 measured calls:

```bash
torchrun --standalone --nproc_per_node=64 run_poc.py \
  --backend neuron --preset full --pcp-size 64 --check none \
  --benchmark-iterations 20 --benchmark-warmups 1
```

On the current `trn2.48xlarge`, one run produced:

```text
min=182.927 ms, median=183.638 ms, mean=184.046 ms,
P90=185.003 ms, max=188.024 ms
```

The mean corresponds to about 22,255 global current tokens/s for the 4096-token
request.  This is end-to-end compiled-call latency, including normal launch and
the one-element synchronization marker; compilation, input construction, and
the Gloo timing barrier are outside the measured interval.

Capturing all repetitions inside one outer `torch.compile` graph was also
tested, but this compiler version inlines the complete NKI body once per call,
making compile time and artifact size scale with the iteration count.  The
single-artifact synchronized loop avoids that benchmark-induced compile
expansion.  At least one warmup is required when benchmarking.

The `full` preset describes the requested 16-chip experiment.  It allocates a
large KV cache, so first compile and validate with `tiny`, then run:

```bash
torchrun --standalone --nproc_per_node=64 run_poc.py \
  --backend neuron --preset full --pcp-size 64 --check none
```

All ranks must use the same `actual_history_len`; otherwise a collective ring
can deadlock.  The CLI validates 4096-token alignment and the equal-shard
constraints before launching the kernel.
`--start-offset` is an alias for `--actual-history-len`, since this
history-only POC defines the new-token start offset as the historical length.

For a runtime history shorter than the compiled maximum, the final local
cache block may be partial.  The runner pads it to 512 tokens and supplies a
per-token mask, so global history lengths can vary in 4096-token increments
for the 64-rank configuration.  Use the small preset for dense correctness
checks; `--check none` avoids replicating the complete 128K cache and dense
oracle in every process during the full-machine smoke run.

The collective ring remains statically unrolled because this compiler version
does not allow a collective in a device-side dynamic loop.  The much larger
per-head attention body uses a dynamic device loop, keeping cold compilation
within the bridge's normal timeout instead of extending timeouts or cloning the
body 16 times for each ring step.

## Current POC constraints

- The first NKI kernel requires LNC=2 with two KV heads (one per physical
  core); Q-head count remains configurable.  The CPU reference supports
  general GQA head counts.
- `head_dim` must be a multiple of 128.
- `block_size` must be a multiple of 128 and of `interleave_size`.
- `current_len`, `actual_history_len`, and `max_history_len` must be divisible
  by `pcp_size * interleave_size` in the first hardware version.
- Q/K/V inputs are BF16; online softmax and accumulators are FP32.
- The CPU implementation is the executable specification.  The hardware path
  is isolated in `pcp_attention/kernel.py` so it can be tuned incrementally.
