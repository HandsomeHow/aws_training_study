import torch

from pcp_attention.config import get_preset
from run_poc import make_block_valid_mask


def test_partial_block_valid_mask() -> None:
    cfg = get_preset("tiny").with_overrides(actual_history_len=1024)
    mask = make_block_valid_mask(cfg)

    assert mask.shape == (1, 4, cfg.local_q_len, 128)
    assert torch.all(mask[0, :2] == 1)
    assert torch.all(mask[0, 2:] == 0)
