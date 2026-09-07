import pytest

from pcp_attention.config import get_preset


def test_full_shape_math() -> None:
    cfg = get_preset("full")
    assert cfg.local_q_len == 64
    assert cfg.actual_local_history_len == 2048
    assert cfg.actual_local_blocks == 4
    assert cfg.softmax_scale == 1 / 16


def test_rejects_unequal_striped_shards() -> None:
    cfg = get_preset("tiny")
    with pytest.raises(ValueError, match=r"pcp_size \* interleave_size"):
        cfg.with_overrides(current_len=320)


def test_partial_last_local_cache_block() -> None:
    cfg = get_preset("tiny").with_overrides(
        max_history_len=2048,
        actual_history_len=1024,
    )
    assert cfg.actual_local_history_len == 256
    assert cfg.actual_local_blocks == 1
