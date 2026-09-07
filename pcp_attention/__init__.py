"""From-scratch prefill context-parallel attention POC."""

from .config import PCPAttentionConfig, get_preset
from .layout import gather_history, shard_history, shard_query
from .reference import dense_history_attention, ring_history_attention

__all__ = [
    "PCPAttentionConfig",
    "dense_history_attention",
    "gather_history",
    "get_preset",
    "ring_history_attention",
    "shard_history",
    "shard_query",
]
