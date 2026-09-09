# Prefill Context-Parallel Attention POC for Trainium2

This repository is a from-scratch learning POC for the **history-only** part of
prefill context-parallel attention. It is intentionally separate from
vLLM/Neuron integrations. The implementation remains readable and testable,
but now includes profiling-driven Trainium2 optimizations.

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

The NKI kernel loads its Q shard into SBUF once. The outer loop selects one
local KV block, and the inner loop circulates that block through a shared-HBM
ping/pong buffer. Q heads sharing a KV head are packed into 128 TensorE rows.
All four 128-token tiles in a 512-token cache block share one QK/softmax update,
and their PV products accumulate in PSUM. Online max/sum remain FP32 in SBUF;
the BF16 numerator remains in SBUF across both loops. FP8 K/V stay FP8 after
entering SBUF and are consumed directly by mixed-dtype TensorE matmuls.

Neuron Compiler 2.27 currently rejects a collective inside a device-side
dynamic loop.  Consequently, this first hardware POC executes
`max_local_blocks` outer iterations and applies a runtime additive score bias
derived from `actual_history_len`.  One compiled artifact therefore produces
correct results for shorter histories, but still pays the maximum bucket's
communication cost. Further length-bucket specialization can remove work on
masked cache blocks.

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

### Profiling-optimized 4-chip case

For 16 PCP ranks on four Trainium2 chips with LNC=2, global Q=1024, historical
KV=8192, FP8 K/V, owner-side K transpose, 32 Q heads, two KV heads, and D=256,
the original, previous, and freshly recompiled current 100-call measurements
are:

```text
                 median       mean       P90       device profile
original         13.450 ms   13.594 ms  13.752 ms  11.261 ms
round 8            2.931 ms    2.986 ms   3.322 ms   0.730 ms
current            2.839 ms    2.849 ms   3.126 ms   0.657 ms
```

The current code is 4.74x faster end to end and 17.14x faster in the device
profile than the original. At this small Q size, roughly 2.2 ms of host
launch/completion overhead hides the latest device improvement. The main
changes were static head
specialization, direct resident Q/state views, loading each K/V tile once per
KV head and reusing it across Q chunks, pairing two 64-token Q heads into one
128-row TensorE tile, fusing all 512 keys in a cache block into one
online-softmax update, additive masking, fused activation/reduction, BF16
numerator state, and FP8 resident K/V. Swapping the PV operands makes TensorE
produce `[Q,D]` directly and removes one result transpose. The runtime validity
mask is stored once per K tile and broadcast in SBUF. The final ring hop back
to the owner is also omitted because no consumer reads it.

The 2026-09-09 isolated Q=1K retest produced the following cumulative results.
Every row was compiled into a fresh cache and then captured with Neuron
Explorer. `device gain` is relative to the preceding row.

```text
stage                              host median   device time   model MFU   device gain
owner-transpose baseline             13.450 ms     11.261 ms      0.915%       -
static Q-head specialization           8.807 ms      6.437 ms      1.601%      42.84%
direct resident-state views            8.591 ms      6.219 ms      1.657%       3.38%
reuse KV across GQA heads              7.491 ms      5.306 ms      1.942%      14.69%
pair heads into 128 TensorE rows       4.066 ms      1.950 ms      5.283%      63.24%
omit final ring hop                    4.176 ms      1.903 ms      5.415%       2.45%
BF16 numerator state                   4.086 ms      1.821 ms      5.659%       4.30%
fused exp plus reduction               3.944 ms      1.747 ms      5.898%       4.05%
fuse four key tiles per block          3.108 ms      1.089 ms      9.458%      37.64%
initialize from first history tile     3.100 ms      0.912 ms     11.298%      16.29%
additive score mask                    2.957 ms      0.767 ms     13.430%      15.87%
direct score output and BF16 probs     2.767 ms      0.725 ms     14.205%       5.45%
resident FP8 KV                        2.868 ms      0.704 ms     14.641%       2.98%
current final, 100 calls               2.839 ms      0.657 ms     15.685%       -
```

The optimizations whose benefit requires multiple local Q chunks were isolated
at larger Q sizes:

```text
optimization/config         host median          device time          model MFU
KV hoist, Q=8K              7.972 -> 7.421 ms    5.562 -> 5.000 ms    14.818 -> 16.485%
direct PV, Q=8K             7.421 -> 6.211 ms    5.000 -> 3.934 ms    16.485 -> 20.953%
compact mask, Q=4K          4.012 -> 4.205 ms    1.861 -> 2.268/2.309 22.147 -> 18.173/17.851%
```

The compact mask reduces its input by 16x for Q=4K, but it is not a stable
latency optimization with neuronx-cc 2.27: older cached artifacts measured
1.859 -> 1.863 ms, while two fresh compiles both regressed. It is retained for
the lower HBM/input footprint, and its performance tradeoff is explicit in the
corresponding commit message.

Increasing Q amortizes fixed communication and runtime costs. With history
fixed at 8192, the current device profiles are:

```text
global Q   host median   device time   model MFU (dense BF16 peak)
1K          2.839 ms      0.657 ms      15.69%
4K          4.101 ms      1.863 ms      22.12%
8K          6.166 ms      3.844 ms      21.44%
16K         9.919 ms      7.781 ms      21.19%
```

MFU stabilizes around 21-22% once global Q reaches 4K. It counts only the QK
and PV model FLOPs and uses Trainium2's 667 BF16 TFLOPS/chip dense peak. The
mixed BF16-by-FP8 matmuls do not qualify for the double-FP8 mode. The device
profile excludes most host launch/completion overhead; this is why its absolute
time is lower than the synchronized PyTorch call time.

Run the requested full-machine case with 20 measured calls:

```bash
torchrun --standalone --nproc_per_node=64 run_poc.py \
  --backend neuron --preset full --pcp-size 64 --check none \
  --kv-dtype fp8_e4m3fn --pretranspose-k-on-owner \
  --benchmark-iterations 30 --benchmark-warmups 3
```

The optimized kernel produced the following 30-call result on the current
`trn2.48xlarge`:

```text
FP8 KV, owner transpose: min=14.561 ms, median=15.063 ms,
                         mean=15.366 ms, P90=15.580 ms,
                         max=18.361 ms, 266,562 global current tokens/s
Device profile: 11.638 ms, 14.16% model MFU
```

For comparison, before the profiling-guided kernel changes, 20-call runs
produced:

```text
BF16 KV: min=182.927 ms, median=183.638 ms, mean=184.046 ms,
          P90=185.003 ms, max=188.024 ms
FP8 KV:  min=192.200 ms, median=193.111 ms, mean=193.097 ms,
          P90=193.662 ms, max=193.864 ms
```

The optimized owner-transpose FP8 path is 12.82x faster by median than the old
FP8 path. These are end-to-end compiled-call latencies, including normal launch
and the one-element synchronization marker; compilation, input construction,
and the Gloo timing barrier are outside the measured interval. The owner still
uses a BF16 intermediate for FP8 K transpose because this compiler rejects a
direct FP8-to-FP8 transpose with the required layout. K/V are then retained in
FP8 through the ring and consumed directly by mixed-dtype TensorE matmuls.

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
per-token additive score bias, so global history lengths can vary in
4096-token increments
for the 64-rank configuration.  Use the small preset for dense correctness
checks; `--check none` avoids replicating the complete 128K cache and dense
oracle in every process during the full-machine smoke run.

The collective ring remains statically unrolled because this compiler version
does not allow a collective in a device-side dynamic loop. Head groups are also
statically specialized so the compiler can preserve resident views and schedule
the TensorE/vector pipeline effectively.

## Current POC constraints

- The first NKI kernel requires LNC=2 with two KV heads (one per physical
  core); Q-head count remains configurable.  The CPU reference supports
  general GQA head counts.
- `head_dim` must be a multiple of 128.
- `block_size` must be no greater than 512 and must be a multiple of 128 and of
  `interleave_size`.
- `current_len`, `actual_history_len`, and `max_history_len` must be divisible
  by `pcp_size * interleave_size` in the first hardware version.
- Q is BF16. K/V storage may be BF16 or FP8; the optimized owner-transpose path
  keeps FP8 K/V compressed through the ring and resident SBUF tiles.
- Online-softmax max/sum are FP32; probabilities and numerator state are BF16,
  with TensorE accumulation in FP32.
- The CPU implementation is the executable specification.  The hardware path
  is isolated in `pcp_attention/kernel.py` so it can be tuned incrementally.
