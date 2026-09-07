"""Dense and ring-ordered executable specifications."""

from __future__ import annotations

import torch

from .config import PCPAttentionConfig


def _kv_head_for_query_head(
    query_head: int, num_q_heads: int, num_kv_heads: int
) -> int:
    return query_head // (num_q_heads // num_kv_heads)


def dense_history_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """Exact history-only GQA attention in FP32.

    Shapes are Q ``[Hq, Q, D]`` and K/V ``[Hkv, K, D]``.
    """

    hq, _, dim = q.shape
    hkv, _, kdim = k.shape
    if k.shape != v.shape or dim != kdim or hq % hkv:
        raise ValueError("incompatible Q/K/V shapes for grouped-query attention")
    group = hq // hkv
    k_expanded = k.repeat_interleave(group, dim=0).float()
    v_expanded = v.repeat_interleave(group, dim=0).float()
    scores = torch.matmul(q.float(), k_expanded.transpose(-1, -2)) * scale
    return torch.matmul(torch.softmax(scores, dim=-1), v_expanded)


def ring_history_attention(
    q_shards: list[torch.Tensor],
    k_shards: list[torch.Tensor],
    v_shards: list[torch.Tensor],
    config: PCPAttentionConfig,
) -> list[torch.Tensor]:
    """Simulate the requested two-loop ring and online-softmax update.

    At outer iteration ``b`` every rank begins with local cache block ``b``.
    Ring step ``s`` processes the block from rank ``(rank - s) % pcp_size``;
    this matches receive-from-predecessor collective permute semantics.
    """

    config.validate()
    if not (len(q_shards) == len(k_shards) == len(v_shards) == config.pcp_size):
        raise ValueError("one Q/K/V shard is required for every PCP rank")

    outputs: list[torch.Tensor] = []
    for rank, q in enumerate(q_shards):
        qf = q.float()
        running_max = torch.full(
            (config.num_q_heads, config.local_q_len, 1),
            -torch.inf,
            dtype=torch.float32,
        )
        running_sum = torch.zeros_like(running_max)
        running_out = torch.zeros_like(qf)

        for block in range(config.actual_local_blocks):
            valid_tokens = min(
                config.block_size,
                config.actual_local_history_len - block * config.block_size,
            )
            for step in range(config.pcp_size):
                source_rank = (rank - step) % config.pcp_size
                k_block = k_shards[source_rank][block, :, :valid_tokens, :].float()
                v_block = v_shards[source_rank][block, :, :valid_tokens, :].float()

                for q_head in range(config.num_q_heads):
                    kv_head = _kv_head_for_query_head(
                        q_head, config.num_q_heads, config.num_kv_heads
                    )
                    scores = torch.matmul(
                        qf[q_head], k_block[kv_head].transpose(0, 1)
                    ) * config.softmax_scale
                    block_max = scores.max(dim=-1, keepdim=True).values
                    new_max = torch.maximum(running_max[q_head], block_max)
                    old_scale = torch.exp(running_max[q_head] - new_max)
                    probs = torch.exp(scores - new_max)
                    running_out[q_head] = (
                        running_out[q_head] * old_scale
                        + torch.matmul(probs, v_block[kv_head])
                    )
                    running_sum[q_head] = (
                        running_sum[q_head] * old_scale
                        + probs.sum(dim=-1, keepdim=True)
                    )
                    running_max[q_head] = new_max

        outputs.append(running_out / running_sum)
    return outputs
