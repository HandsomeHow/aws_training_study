#!/usr/bin/env python3
"""Benchmark the installed nkilib ring-attention kernel without modifying /opt.

This is intentionally separate from the PCP POC runner.  The installed
``ring_attention_spmd_fwd`` requires equal local Q/K/V sequence lengths,
MHA, and head_dim <= 128, so ``--local-seqlen 2048`` preserves the 128K
global KV length at CP=64 but is not shape-equivalent to the history-only
PCP workload.
"""

from __future__ import annotations

import argparse
import os
import socket
import statistics
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-seqlen", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--lnc", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("gloo")

    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", str(local_rank))
    if world_size > 1 and "NEURON_RT_ROOT_COMM_ID" not in os.environ:
        root_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        root_comm_port = [None]
        if rank == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("", 0))
                root_comm_port[0] = sock.getsockname()[1]
        torch.distributed.broadcast_object_list(root_comm_port, src=0)
        os.environ["NEURON_RT_ROOT_COMM_ID"] = (
            f"{root_addr}:{root_comm_port[0]}"
        )

    # libtorch-neuron's subprocesses need the active virtualenv binaries.
    venv_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")
    return world_size, rank, local_rank


def percentile_nearest_rank(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, int((percentile * len(ordered) + 0.999999)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def main() -> int:
    args = parse_args()
    if args.head_dim > 128:
        raise ValueError("official ring_attention_spmd_fwd requires head_dim <= 128")
    if args.warmups < 1 or args.iterations < 1:
        raise ValueError("warmups and iterations must both be positive")

    world_size, rank, _ = setup_distributed()

    import libtorch_neuronx_lite  # noqa: F401
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from nkilib.experimental.attention.ring_attention_fwd import (
        ring_attention_spmd_fwd,
    )

    replica_groups = (tuple(range(world_size)),)
    ring = wrap_nki(ring_attention_spmd_fwd)[args.lnc]

    def invoke(q_arg, k_arg, v_arg):
        return ring(
            q=q_arg,
            k=k_arg,
            v=v_arg,
            replica_groups=replica_groups,
            num_workers=world_size,
            softmax_scale=1.0 / (args.head_dim**0.5),
            use_causal_mask=False,
            striped_input=False,
            training=False,
            tp_q=True,
            tp_k=True,
        )

    generator = torch.Generator().manual_seed(args.seed + 104729 * rank)
    shape = (1, args.heads, args.local_seqlen, args.head_dim)
    q = torch.randn(shape, dtype=torch.bfloat16, generator=generator)
    k = torch.randn(shape, dtype=torch.bfloat16, generator=generator)
    v = torch.randn(shape, dtype=torch.bfloat16, generator=generator)
    device = torch.device("neuron", 0)
    q_device = q.to(device)
    k_device = k.to(device)
    v_device = v.to(device)

    compiled = torch.compile(
        invoke,
        backend="neuron_libtorch",
        fullgraph=True,
        dynamic=False,
    )

    for _ in range(args.warmups):
        output = compiled(q_device, k_device, v_device)
        output.reshape(-1)[:1].cpu()

    if torch.distributed.is_initialized():
        torch.distributed.monitored_barrier()

    local_samples_s: list[float] = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        output = compiled(q_device, k_device, v_device)
        marker = output.reshape(-1)[:1].cpu()
        local_samples_s.append(time.perf_counter() - start)
        _ = marker[0].item()

    samples_by_rank: list[list[float] | None] = [local_samples_s]
    if torch.distributed.is_initialized():
        samples_by_rank = [None] * world_size
        torch.distributed.all_gather_object(samples_by_rank, local_samples_s)

    if rank == 0:
        global_samples_ms = [
            max(rank_samples[i] for rank_samples in samples_by_rank if rank_samples)
            * 1000.0
            for i in range(args.iterations)
        ]
        mean_ms = statistics.fmean(global_samples_ms)
        global_seq = args.local_seqlen * world_size
        # Non-causal full self-attention: QK plus PV, counting FMA as 2 FLOPs.
        useful_flops = (
            4
            * args.heads
            * args.head_dim
            * global_seq
            * global_seq
        )
        print(
            "official_ring_benchmark: "
            f"world_size={world_size}, lnc={args.lnc}, "
            f"local_shape={shape}, global_seqlen={global_seq}, "
            f"warmups={args.warmups}, iterations={args.iterations}, "
            f"min_ms={min(global_samples_ms):.3f}, "
            f"median_ms={statistics.median(global_samples_ms):.3f}, "
            f"mean_ms={mean_ms:.3f}, "
            f"p90_ms={percentile_nearest_rank(global_samples_ms, 0.9):.3f}, "
            f"max_ms={max(global_samples_ms):.3f}, "
            f"effective_tflops={useful_flops / (mean_ms / 1000.0) / 1e12:.3f}",
            flush=True,
        )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
