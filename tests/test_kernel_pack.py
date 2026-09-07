import torch

from pcp_attention.kernel import pack_local_kv


def test_pack_local_kv_order() -> None:
    k = torch.arange(2 * 2 * 4 * 8).reshape(2, 2, 4, 8)
    v = -k
    packed = pack_local_kv(k, v)
    restored = packed.reshape(2, 2, 2, 4, 8)
    torch.testing.assert_close(restored[:, 0], k)
    torch.testing.assert_close(restored[:, 1], v)
