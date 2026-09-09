import torch

from pcp_attention.config import get_preset
from run_poc import kv_storage_dtype, make_block_valid_mask


def test_partial_block_valid_mask() -> None:
    cfg = get_preset("tiny").with_overrides(actual_history_len=1024)
    mask = make_block_valid_mask(cfg)

    assert mask.shape == (1, 4, 1, 128)
    assert mask.dtype == torch.bfloat16
    assert torch.all(mask[0, :2] == 0)
    assert torch.all(mask[0, 2:] == -9984)


def test_kv_storage_dtype() -> None:
    assert kv_storage_dtype("bf16") == torch.bfloat16
    assert kv_storage_dtype("fp8_e4m3fn") == torch.float8_e4m3fn
