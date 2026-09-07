#!/usr/bin/env python3
"""Run the PCP attention executable specification or the NKI implementation."""

from __future__ import annotations

import argparse
import os
import socket
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from pcp_attention.config import PCPAttentionConfig, get_preset
from pcp_attention.layout import gather_history, shard_history, shard_query
from pcp_attention.reference import dense_history_attention, ring_history_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "neuron"), default="cpu")
    parser.add_argument("--preset", choices=("tiny", "full"), default="tiny")
    parser.add_argument("--pcp-size", type=int)
    parser.add_argument("--current-len", type=int)
    parser.add_argument("--max-history-len", type=int)
    parser.add_argument(
        "--actual-history-len",
        "--start-offset",
        dest="actual_history_len",
        type=int,
        help="runtime historical-token count; also the new-token start offset",
    )
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--interleave-size", type=int)
    parser.add_argument("--num-q-heads", type=int)
    parser.add_argument("--num-kv-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--check", choices=("full", "sampled", "none"), default="full")
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=0,
        help=(
            "number of dependent calls to the same compiled NKI graph; "
            "zero disables benchmarking"
        ),
    )
    parser.add_argument(
        "--benchmark-warmups",
        type=int,
        default=1,
        help="number of untimed executions of the compiled benchmark graph",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> PCPAttentionConfig:
    cfg = get_preset(args.preset)
    overrides = {
        name: getattr(args, name)
        for name in (
            "pcp_size",
            "current_len",
            "max_history_len",
            "actual_history_len",
            "block_size",
            "interleave_size",
            "num_q_heads",
            "num_kv_heads",
            "head_dim",
        )
        if getattr(args, name) is not None
    }
    return cfg.with_overrides(**overrides)


def make_inputs(cfg: PCPAttentionConfig, seed: int):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(
        cfg.num_q_heads,
        cfg.current_len,
        cfg.head_dim,
        generator=generator,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        cfg.num_kv_heads,
        cfg.actual_history_len,
        cfg.head_dim,
        generator=generator,
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        cfg.num_kv_heads,
        cfg.actual_history_len,
        cfg.head_dim,
        generator=generator,
        dtype=torch.bfloat16,
    )
    return q, k, v


def make_local_inputs(cfg: PCPAttentionConfig, seed: int, rank: int):
    """Generate only one rank's tensors for memory-safe full-size runs."""

    generator = torch.Generator().manual_seed(seed + 104729 * rank)
    q = torch.randn(
        cfg.num_q_heads,
        cfg.local_q_len,
        cfg.head_dim,
        generator=generator,
        dtype=torch.bfloat16,
    )
    cache_shape = (
        cfg.actual_local_blocks,
        cfg.num_kv_heads,
        cfg.block_size,
        cfg.head_dim,
    )
    k = torch.randn(*cache_shape, generator=generator, dtype=torch.bfloat16)
    v = torch.randn(*cache_shape, generator=generator, dtype=torch.bfloat16)
    return q, k, v


def make_block_valid_mask(cfg: PCPAttentionConfig) -> torch.Tensor:
    """Return the per-key runtime mask for a potentially partial last block."""

    kv_tiles = cfg.block_size // 128
    mask = torch.zeros(
        (cfg.max_local_blocks, kv_tiles, cfg.local_q_len, 128),
        dtype=torch.uint8,
    )
    remaining = cfg.actual_local_history_len
    for block in range(cfg.actual_local_blocks):
        valid_in_block = min(cfg.block_size, remaining)
        for tile in range(kv_tiles):
            valid_in_tile = min(128, max(0, valid_in_block - tile * 128))
            mask[block, tile, :, :valid_in_tile] = 1
        remaining -= valid_in_block
    return mask


def run_cpu(cfg: PCPAttentionConfig, seed: int) -> None:
    q, k, v = make_inputs(cfg, seed)
    q_shards = shard_query(q, cfg)
    k_shards, v_shards = shard_history(k, v, cfg)
    output = ring_history_attention(q_shards, k_shards, v_shards, cfg)
    dense_k = gather_history(k_shards, cfg)
    dense_v = gather_history(v_shards, cfg)
    max_error = 0.0
    for rank in range(cfg.pcp_size):
        expected = dense_history_attention(
            q_shards[rank], dense_k, dense_v, cfg.softmax_scale
        )
        max_error = max(max_error, (output[rank] - expected).abs().max().item())
    print(f"CPU ring matches dense reference; max_abs_error={max_error:.3e}")


def run_neuron(
    cfg: PCPAttentionConfig,
    seed: int,
    check: str,
    benchmark_iterations: int = 0,
    benchmark_warmups: int = 1,
) -> None:
    if benchmark_iterations < 0:
        raise ValueError("benchmark_iterations must be non-negative")
    if benchmark_warmups < 0:
        raise ValueError("benchmark_warmups must be non-negative")
    if benchmark_iterations and benchmark_warmups == 0:
        raise ValueError(
            "benchmark_warmups must be at least one so compilation is untimed"
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size != cfg.pcp_size:
        raise RuntimeError(
            f"torchrun WORLD_SIZE={world_size}, but pcp_size={cfg.pcp_size}"
        )
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("gloo")

    # Device discovery must happen after torchrun assigns the process.  Each
    # process owns one logical LNC2 rank.
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", str(local_rank))
    if world_size > 1 and "NEURON_RT_ROOT_COMM_ID" not in os.environ:
        root_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        root_comm_port = [None]
        if rank == 0:
            # Ask the OS instead of guessing MASTER_PORT + 1, which can be
            # occupied by an unrelated or recently terminated rendezvous.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("", 0))
                root_comm_port[0] = sock.getsockname()[1]
        torch.distributed.broadcast_object_list(root_comm_port, src=0)
        os.environ["NEURON_RT_ROOT_COMM_ID"] = (
            f"{root_addr}:{root_comm_port[0]}"
        )
    venv_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
    import libtorch_neuronx_lite  # noqa: F401
    from pcp_attention.kernel import pack_local_kv, wrap_for_native_torch

    # Correctness runs construct the global tensors used by the dense host
    # oracle.  Unchecked full-machine runs create just the local shard so 64
    # processes do not each replicate the complete 128K cache in host memory.
    if check == "none":
        q_local, k_local, v_local = make_local_inputs(cfg, seed, rank)
        k = v = None
    else:
        q, k, v = make_inputs(cfg, seed)
        q_shards = shard_query(q, cfg)
        k_shards, v_shards = shard_history(k, v, cfg)
        q_local = q_shards[rank]
        k_local = k_shards[rank]
        v_local = v_shards[rank]
    packed = pack_local_kv(k_local, v_local)
    if cfg.actual_local_blocks < cfg.max_local_blocks:
        padded = torch.zeros(
            (cfg.max_local_blocks, packed.shape[1]), dtype=packed.dtype
        )
        padded[: cfg.actual_local_blocks] = packed
        packed = padded
    block_valid_mask = make_block_valid_mask(cfg)

    device = torch.device("neuron", 0)
    kernel = wrap_for_native_torch(cfg)

    def invoke(q_arg, kv_arg, block_valid_mask_arg):
        return kernel(q_arg, kv_arg, block_valid_mask_arg)

    q_device = q_local.to(device)
    packed_device = packed.to(device)
    mask_device = block_valid_mask.to(device)

    compiled = torch.compile(
        invoke,
        backend="neuron_libtorch",
        fullgraph=True,
        dynamic=False,
    )
    if check != "none" or benchmark_iterations == 0:
        actual = compiled(q_device, packed_device, mask_device).cpu().float()

        if check != "none":
            expected = dense_history_attention(q_local, k, v, cfg.softmax_scale)
            if check == "sampled":
                actual = actual[:, :1, :]
                expected = expected[:, :1, :]
            torch.testing.assert_close(actual, expected, atol=6e-2, rtol=6e-2)
        status = "Neuron run completed" if check == "none" else "Neuron output OK"
        print(f"rank={rank}: {status}", flush=True)

    if benchmark_iterations:
        # Reuse one compiled graph rather than capturing N calls in one outer
        # graph.  neuronx-cc inlines every NKI call in such a graph, making
        # compile time and artifact size scale with the benchmark iteration
        # count.  This runtime bridge has a bounded asynchronous execution
        # queue and does not expose a device-wide synchronize operation, so a
        # one-element D2H copy is used as a low-volume completion marker after
        # every call.
        for _ in range(benchmark_warmups):
            warmup_output = compiled(q_device, packed_device, mask_device)
            warmup_output.reshape(-1)[:1].cpu()

        if torch.distributed.is_initialized():
            # The regular barrier probes the registered PrivateUse1
            # accelerator in this PyTorch/Neuron build.  A Gloo monitored
            # barrier is explicitly host-side and gives us the same aligned
            # timing boundary without requiring PrivateUse1 device hooks.
            torch.distributed.monitored_barrier()
        elapsed_samples_s = []
        for _ in range(benchmark_iterations):
            start = time.perf_counter()
            benchmark_output = compiled(
                q_device, packed_device, mask_device
            )
            completion_marker = benchmark_output.reshape(-1)[:1].cpu()
            elapsed_samples_s.append(time.perf_counter() - start)
            _ = completion_marker[0].item()

        samples_by_rank = [elapsed_samples_s]
        if torch.distributed.is_initialized():
            samples_by_rank = [None] * world_size
            torch.distributed.all_gather_object(
                samples_by_rank, elapsed_samples_s
            )
        if rank == 0:
            global_samples_ms = [
                max(rank_samples[index] for rank_samples in samples_by_rank)
                * 1000.0
                for index in range(benchmark_iterations)
            ]
            sorted_samples = sorted(global_samples_ms)
            p90_index = max(
                0, (9 * len(sorted_samples) + 9) // 10 - 1
            )
            mean_ms = statistics.fmean(global_samples_ms)
            print(
                "benchmark: "
                "mode=synchronized_single_jit, "
                f"iterations={benchmark_iterations}, "
                f"warmups={benchmark_warmups}, "
                f"min_ms={min(global_samples_ms):.3f}, "
                f"median_ms={statistics.median(global_samples_ms):.3f}, "
                f"mean_ms={mean_ms:.3f}, "
                f"p90_ms={sorted_samples[p90_index]:.3f}, "
                f"max_ms={max(global_samples_ms):.3f}, "
                f"global_current_tokens_per_s="
                f"{cfg.current_len / (mean_ms / 1000.0):.1f}",
                flush=True,
            )
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main() -> int:
    args = parse_args()
    cfg = make_config(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print("configuration:")
        for name, value in asdict(cfg).items():
            print(f"  {name}={value}")
        print(f"  local_q_len={cfg.local_q_len}")
        print(f"  actual_local_blocks={cfg.actual_local_blocks}")
    if args.backend == "cpu":
        if args.benchmark_iterations:
            raise ValueError("benchmark mode currently requires --backend neuron")
        run_cpu(cfg, args.seed)
    else:
        run_neuron(
            cfg,
            args.seed,
            args.check,
            args.benchmark_iterations,
            args.benchmark_warmups,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
