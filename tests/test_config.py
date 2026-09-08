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


def test_accepts_multiple_local_q_tiles() -> None:
    cfg = get_preset("full").with_overrides(
        pcp_size=16,
        current_len=4096,
        max_history_len=8192,
        actual_history_len=8192,
    )
    assert cfg.local_q_len == 256


def test_owner_pretranspose_is_configurable() -> None:
    cfg = get_preset("tiny").with_overrides(pretranspose_k_on_owner=True)
    assert cfg.pretranspose_k_on_owner


def test_rejects_partial_local_q_tile_above_128() -> None:
    with pytest.raises(ValueError, match="128-token NKI Q tile"):
        get_preset("full").with_overrides(
            pcp_size=16,
            current_len=3072,
        )
