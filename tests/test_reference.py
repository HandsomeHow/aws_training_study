import torch

from pcp_attention.config import get_preset
from pcp_attention.layout import gather_history, shard_history, shard_query
from pcp_attention.reference import dense_history_attention, ring_history_attention


def test_ring_online_softmax_matches_dense_attention() -> None:
    cfg = get_preset("tiny").with_overrides(
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=128,
    )
    generator = torch.Generator().manual_seed(1234)
    q = torch.randn(
        cfg.num_q_heads, cfg.current_len, cfg.head_dim, generator=generator
    )
    k = torch.randn(
        cfg.num_kv_heads, cfg.actual_history_len, cfg.head_dim, generator=generator
    )
    v = torch.randn(
        cfg.num_kv_heads, cfg.actual_history_len, cfg.head_dim, generator=generator
    )
    q_shards = shard_query(q, cfg)
    k_shards, v_shards = shard_history(k, v, cfg)

    actual = ring_history_attention(q_shards, k_shards, v_shards, cfg)
    dense_k = gather_history(k_shards, cfg)
    dense_v = gather_history(v_shards, cfg)
    for rank, q_shard in enumerate(q_shards):
        expected = dense_history_attention(q_shard, dense_k, dense_v, cfg.softmax_scale)
        torch.testing.assert_close(actual[rank], expected, atol=2e-5, rtol=2e-5)


def test_ring_partial_last_block_matches_dense_attention() -> None:
    cfg = get_preset("tiny").with_overrides(
        actual_history_len=1024,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=128,
    )
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(cfg.num_q_heads, cfg.current_len, cfg.head_dim, generator=generator)
    k = torch.randn(
        cfg.num_kv_heads, cfg.actual_history_len, cfg.head_dim, generator=generator
    )
    v = torch.randn(
        cfg.num_kv_heads, cfg.actual_history_len, cfg.head_dim, generator=generator
    )
    q_shards = shard_query(q, cfg)
    k_shards, v_shards = shard_history(k, v, cfg)

    actual = ring_history_attention(q_shards, k_shards, v_shards, cfg)
    for rank, q_shard in enumerate(q_shards):
        expected = dense_history_attention(q_shard, k, v, cfg.softmax_scale)
        torch.testing.assert_close(actual[rank], expected, atol=2e-5, rtol=2e-5)
