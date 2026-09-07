import torch

from pcp_attention.config import get_preset
from pcp_attention.layout import (
    gather_history,
    global_positions_for_rank,
    owner_and_local_position,
    shard_history,
)


def test_owner_round_trip() -> None:
    pcp_size, interleave = 4, 64
    for rank in range(pcp_size):
        positions = global_positions_for_rank(2048, rank, pcp_size, interleave)
        for local_position, global_position in enumerate(positions.tolist()):
            assert owner_and_local_position(global_position, pcp_size, interleave) == (
                rank,
                local_position,
            )


def test_history_shard_and_gather_are_inverse() -> None:
    cfg = get_preset("tiny")
    shape = (cfg.num_kv_heads, cfg.actual_history_len, cfg.head_dim)
    k = torch.arange(torch.tensor(shape).prod(), dtype=torch.float32).reshape(shape)
    v = -k
    k_shards, v_shards = shard_history(k, v, cfg)
    torch.testing.assert_close(gather_history(k_shards, cfg), k)
    torch.testing.assert_close(gather_history(v_shards, cfg), v)


def test_partial_cache_block_shard_and_gather_are_inverse() -> None:
    cfg = get_preset("tiny").with_overrides(actual_history_len=1024)
    shape = (cfg.num_kv_heads, cfg.actual_history_len, cfg.head_dim)
    k = torch.arange(torch.tensor(shape).prod(), dtype=torch.float32).reshape(shape)
    k_shards, _ = shard_history(k, -k, cfg)
    assert k_shards[0].shape[0] == 1
    assert torch.count_nonzero(k_shards[0][:, :, 256:, :]) == 0
    torch.testing.assert_close(gather_history(k_shards, cfg), k)
