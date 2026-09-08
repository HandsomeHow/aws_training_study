"""Static and runtime configuration for the PCP attention POC."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, sqrt


@dataclass(frozen=True)
class PCPAttentionConfig:
    """Configuration shared by layout, reference, runner, and NKI kernel.

    Static fields participate in NKI compilation. ``actual_history_len`` is a
    runtime value and may be no greater than ``max_history_len``.
    """

    pcp_size: int
    current_len: int
    max_history_len: int
    actual_history_len: int
    block_size: int = 512
    interleave_size: int = 64
    num_q_heads: int = 32
    num_kv_heads: int = 2
    head_dim: int = 256
    lnc: int = 2
    alignment: int = 4096
    pretranspose_k_on_owner: bool = False

    @property
    def softmax_scale(self) -> float:
        return 1.0 / sqrt(self.head_dim)

    @property
    def q_heads_per_kv_head(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    @property
    def local_q_len(self) -> int:
        return self.current_len // self.pcp_size

    @property
    def max_local_history_len(self) -> int:
        return self.max_history_len // self.pcp_size

    @property
    def actual_local_history_len(self) -> int:
        return self.actual_history_len // self.pcp_size

    @property
    def max_local_blocks(self) -> int:
        return ceil(self.max_local_history_len / self.block_size)

    @property
    def actual_local_blocks(self) -> int:
        return ceil(self.actual_local_history_len / self.block_size)

    @property
    def rank_stride(self) -> int:
        return self.pcp_size * self.interleave_size

    def with_overrides(self, **kwargs: int) -> "PCPAttentionConfig":
        return replace(self, **kwargs).validate()

    def validate(self) -> "PCPAttentionConfig":
        positive = {
            "pcp_size": self.pcp_size,
            "current_len": self.current_len,
            "max_history_len": self.max_history_len,
            "actual_history_len": self.actual_history_len,
            "block_size": self.block_size,
            "interleave_size": self.interleave_size,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "lnc": self.lnc,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.actual_history_len > self.max_history_len:
            raise ValueError("actual_history_len must not exceed max_history_len")
        if self.num_q_heads % self.num_kv_heads:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        if self.num_q_heads % self.lnc or self.num_kv_heads % self.lnc:
            raise ValueError("Q and KV heads must divide evenly across LNC cores")
        if self.head_dim % 128:
            raise ValueError("head_dim must be a multiple of the TensorE K tile (128)")
        if self.block_size % 128:
            raise ValueError("block_size must be a multiple of the KV tile (128)")
        if self.block_size % self.interleave_size:
            raise ValueError("block_size must be divisible by interleave_size")
        for name, value in (
            ("current_len", self.current_len),
            ("max_history_len", self.max_history_len),
            ("actual_history_len", self.actual_history_len),
        ):
            if value % self.rank_stride:
                raise ValueError(
                    f"{name} must be divisible by pcp_size * interleave_size "
                    f"({self.rank_stride}), got {value}"
                )
        if self.local_q_len > 128 and self.local_q_len % 128:
            raise ValueError(
                "local_q_len values above 128 must be divisible by the "
                "128-token NKI Q tile"
            )
        return self


_PRESETS = {
    # One local KV block per rank; retains the Qwen model dimensions.
    "tiny": PCPAttentionConfig(
        pcp_size=4,
        current_len=256,
        max_history_len=2048,
        actual_history_len=2048,
        block_size=512,
        interleave_size=64,
        alignment=256,
    ),
    # Requested full-machine experiment.
    "full": PCPAttentionConfig(
        pcp_size=64,
        current_len=4096,
        max_history_len=131072,
        actual_history_len=131072,
        block_size=512,
        interleave_size=64,
        alignment=4096,
    ),
}


def get_preset(name: str) -> PCPAttentionConfig:
    try:
        return _PRESETS[name].validate()
    except KeyError as exc:
        choices = ", ".join(sorted(_PRESETS))
        raise ValueError(f"unknown preset {name!r}; choose one of: {choices}") from exc
