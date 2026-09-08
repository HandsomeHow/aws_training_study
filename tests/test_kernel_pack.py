import torch

from pcp_attention.kernel import pack_local_kv


def test_pack_local_kv_order() -> None:
    k = torch.arange(2 * 2 * 4 * 8).reshape(2, 2, 4, 8)
    v = -k
    packed = pack_local_kv(k, v)
    restored = packed.reshape(2, 2, 2, 4, 8)
    torch.testing.assert_close(restored[:, 0], k)
    torch.testing.assert_close(restored[:, 1], v)


def test_pack_local_kv_preserves_fp8_storage() -> None:
    k = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    v = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    packed = pack_local_kv(k, v)

    assert packed.dtype == torch.float8_e4m3fn
    assert packed.element_size() == 1
    restored = packed.reshape(1, 2, 2, 4, 8)
    torch.testing.assert_close(restored[:, 0].float(), k.float())
    torch.testing.assert_close(restored[:, 1].float(), v.float())
