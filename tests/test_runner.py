import torch

from pcp_attention.config import get_preset
from run_poc import make_block_valid_mask, make_local_inputs


def test_partial_block_valid_mask() -> None:
    cfg = get_preset("tiny").with_overrides(actual_history_len=1024)
    mask = make_block_valid_mask(cfg)

    assert mask.shape == (1, 4, 1, 128)
    assert mask.dtype == torch.bfloat16
    assert torch.all(mask[0, :2] == 0)
    assert torch.all(mask[0, 2:] == -9984)


def test_local_kv_inputs_use_final_fp8_storage() -> None:
    cfg = get_preset("tiny")
    _, k, v = make_local_inputs(cfg, seed=1234, rank=0)

    assert k.dtype == torch.float8_e4m3fn
    assert v.dtype == torch.float8_e4m3fn
