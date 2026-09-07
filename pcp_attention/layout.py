"""Striped token ownership and host-side shard/gather helpers."""

from __future__ import annotations

import torch

from .config import PCPAttentionConfig


def owner_and_local_position(
    global_position: int, pcp_size: int, interleave_size: int
) -> tuple[int, int]:
    chunk, within_chunk = divmod(global_position, interleave_size)
    owner = chunk % pcp_size
    local_chunk = chunk // pcp_size
    return owner, local_chunk * interleave_size + within_chunk


def global_positions_for_rank(
    total_len: int, rank: int, pcp_size: int, interleave_size: int
) -> torch.Tensor:
    if not 0 <= rank < pcp_size:
        raise ValueError(f"rank {rank} is outside [0, {pcp_size})")
    if total_len % (pcp_size * interleave_size):
        raise ValueError("total_len must divide evenly across striped ranks")
    local_chunks = total_len // (pcp_size * interleave_size)
    base = (
        torch.arange(local_chunks, dtype=torch.int64) * pcp_size + rank
    ) * interleave_size
    within = torch.arange(interleave_size, dtype=torch.int64)
    return (base[:, None] + within[None, :]).reshape(-1)


def shard_query(q: torch.Tensor, config: PCPAttentionConfig) -> list[torch.Tensor]:
    """Shard global ``[Hq, current_len, D]`` Q using striped ownership.

    Q positions are global positions starting at ``actual_history_len``.  With
    the current alignment constraints that offset is a full rank-stride, so the
    same rank ordering can be applied to the current segment directly.
    """

    config.validate()
    expected = (config.num_q_heads, config.current_len, config.head_dim)
    if tuple(q.shape) != expected:
        raise ValueError(f"expected Q shape {expected}, got {tuple(q.shape)}")
    return [
        q[:, global_positions_for_rank(
            config.current_len, rank, config.pcp_size, config.interleave_size
        ), :].contiguous()
        for rank in range(config.pcp_size)
    ]


def shard_history(
    k: torch.Tensor, v: torch.Tensor, config: PCPAttentionConfig
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Shard global history and reshape each shard to the block-cache layout."""

    config.validate()
    expected = (config.num_kv_heads, config.actual_history_len, config.head_dim)
    if tuple(k.shape) != expected or tuple(v.shape) != expected:
        raise ValueError(
            f"expected K/V shape {expected}, got {tuple(k.shape)} and {tuple(v.shape)}"
        )
    k_shards: list[torch.Tensor] = []
    v_shards: list[torch.Tensor] = []
    for rank in range(config.pcp_size):
        positions = global_positions_for_rank(
            config.actual_history_len,
            rank,
            config.pcp_size,
            config.interleave_size,
        )
        def to_cache_layout(tensor: torch.Tensor) -> torch.Tensor:
            local = tensor[:, positions, :]
            # A 4K-aligned global history can leave only 64 valid tokens in
            # the last per-rank 512-token cache block.  Pad that physical
            # block here; the runtime mask keeps the padding out of softmax.
            padded = torch.zeros(
                (
                    config.num_kv_heads,
                    config.actual_local_blocks * config.block_size,
                    config.head_dim,
                ),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            padded[:, : config.actual_local_history_len, :] = local
            return padded.reshape(
                config.num_kv_heads,
                config.actual_local_blocks,
                config.block_size,
                config.head_dim,
            ).permute(1, 0, 2, 3).contiguous()

        k_shards.append(to_cache_layout(k))
        v_shards.append(to_cache_layout(v))
    return k_shards, v_shards


def gather_history(shards: list[torch.Tensor], config: PCPAttentionConfig) -> torch.Tensor:
    """Invert ``shard_history`` for one of K or V."""

    config.validate()
    if len(shards) != config.pcp_size:
        raise ValueError(f"expected {config.pcp_size} shards, got {len(shards)}")
    sample = shards[0]
    result = torch.empty(
        (config.num_kv_heads, config.actual_history_len, config.head_dim),
        dtype=sample.dtype,
        device=sample.device,
    )
    for rank, shard in enumerate(shards):
        expected = (
            config.actual_local_blocks,
            config.num_kv_heads,
            config.block_size,
            config.head_dim,
        )
        if tuple(shard.shape) != expected:
            raise ValueError(f"rank {rank}: expected {expected}, got {tuple(shard.shape)}")
        positions = global_positions_for_rank(
            config.actual_history_len,
            rank,
            config.pcp_size,
            config.interleave_size,
        ).to(shard.device)
        local = shard.permute(1, 0, 2, 3).reshape(
            config.num_kv_heads, -1, config.head_dim
        )[:, : config.actual_local_history_len, :]
        result[:, positions, :] = local
    return result
