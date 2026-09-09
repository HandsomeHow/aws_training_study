"""NKI implementation of history-only PCP attention.

The code is deliberately direct.  It keeps Q and online-softmax state in SBUF,
uses shared-HBM ping/pong communication buffers, and processes one 128-token
KV tile at a time.  Optimized layouts and scheduling are future work.
"""

from __future__ import annotations

from typing import Any

from .config import PCPAttentionConfig


def pack_local_kv(k: Any, v: Any) -> Any:
    """Pack cache tensors for the kernel's contiguous ring payload.

    Input shapes are ``[blocks, Hkv, block_size, D]``.  The returned tensor is
    ``[blocks, 2 * Hkv * block_size * D]`` with all K followed by all V in each
    block.  ``Any`` keeps importing this module from requiring torch or NKI.
    """

    import torch

    if k.shape != v.shape or k.ndim != 4:
        raise ValueError("K and V must have equal [blocks, Hkv, block, D] shapes")
    return torch.stack((k, v), dim=1).reshape(k.shape[0], -1).contiguous()


def make_history_attention_kernel(config: PCPAttentionConfig):
    """Build a JIT kernel specialized for every field except actual length.

    ``block_valid_mask_ref`` is generated at runtime from actual history
    length.  The current compiler rejects collectives inside device-side
    dynamic loops, so the first hardware version executes the maximum static
    block count and masks blocks beyond the runtime length.
    """

    config.validate()
    if config.num_kv_heads != config.lnc:
        raise ValueError(
            "first NKI kernel requires one KV head per LNC core "
            f"(num_kv_heads={config.num_kv_heads}, lnc={config.lnc})"
        )

    import nki
    global ncc, nisa, nl
    import nki.collectives as ncc
    import nki.isa as nisa
    import nki.language as nl

    pcp_size = config.pcp_size
    q_len = config.local_q_len
    q_tile_size = min(q_len, 128)
    q_tiles = q_len // q_tile_size
    block_size = config.block_size
    num_q_heads = config.num_q_heads
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim
    lnc = config.lnc
    max_local_blocks = config.max_local_blocks
    heads_per_core = num_q_heads // lnc
    heads_per_compute_tile = min(heads_per_core, 128 // q_tile_size)
    if heads_per_core % heads_per_compute_tile:
        raise ValueError(
            "heads_per_core must divide evenly into 128-row compute tiles"
        )
    head_groups = heads_per_core // heads_per_compute_tile
    compute_rows = q_tile_size * heads_per_compute_tile
    kv_heads_per_core = num_kv_heads // lnc
    q_per_kv = num_q_heads // num_kv_heads
    d_tiles = head_dim // 128
    kv_tiles = block_size // 128
    payload_elements = 2 * num_kv_heads * block_size * head_dim
    replica_group_spec = (tuple(range(pcp_size)),)
    softmax_scale = config.softmax_scale
    pretranspose_k_on_owner = config.pretranspose_k_on_owner

    @nki.jit
    def history_attention_kernel(
        q_ref,
        local_kv_ref,
        block_valid_mask_ref,
        pcp_size=pcp_size,
        q_len=q_len,
        q_tile_size=q_tile_size,
        q_tiles=q_tiles,
        block_size=block_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        lnc=lnc,
        max_local_blocks=max_local_blocks,
        heads_per_core=heads_per_core,
        heads_per_compute_tile=heads_per_compute_tile,
        head_groups=head_groups,
        compute_rows=compute_rows,
        kv_heads_per_core=kv_heads_per_core,
        q_per_kv=q_per_kv,
        d_tiles=d_tiles,
        kv_tiles=kv_tiles,
        payload_elements=payload_elements,
        replica_group_spec=replica_group_spec,
        softmax_scale=softmax_scale,
        pretranspose_k_on_owner=pretranspose_k_on_owner,
    ):
        assert tuple(q_ref.shape) == (num_q_heads, q_len, head_dim)
        assert tuple(local_kv_ref.shape) == (max_local_blocks, payload_elements)
        assert tuple(block_valid_mask_ref.shape) == (
            max_local_blocks,
            kv_tiles,
            q_len,
            128,
        )
        replica_group = ncc.ReplicaGroup(replica_group_spec)
        resident_kv_dtype = (
            local_kv_ref.dtype
            if pretranspose_k_on_owner
            else nl.bfloat16
        )

        # Collective endpoints use the historical KV storage dtype. This keeps
        # FP8 caches compressed while they circulate through the HBM ring;
        # individual tiles are converted to BF16 only when loaded into SBUF.
        out_ref = nl.ndarray(q_ref.shape, dtype=q_ref.dtype, buffer=nl.shared_hbm)
        comm0 = nl.ndarray(
            (2, num_kv_heads, block_size, head_dim),
            dtype=local_kv_ref.dtype,
            buffer=nl.shared_hbm,
            name="kv_ping",
        )
        comm1 = nl.ndarray(
            (2, num_kv_heads, block_size, head_dim),
            dtype=local_kv_ref.dtype,
            buffer=nl.shared_hbm,
            name="kv_pong",
        )

        core = nl.program_id(0)
        # Q is loaded once, scaled once, and remains live in SBUF for the full
        # outer loop.  Each LNC core owns a disjoint group of heads.
        q_local = nl.ndarray(
            (128, q_tiles, head_groups, d_tiles, compute_rows),
            dtype=nl.bfloat16,
            buffer=nl.sbuf,
            name="resident_q",
        )
        for head_group in nl.static_range(head_groups):
            for group_head in nl.static_range(heads_per_compute_tile):
                local_head = head_group * heads_per_compute_tile + group_head
                global_head = core * heads_per_core + local_head
                row_start = group_head * q_tile_size
                row_end = row_start + q_tile_size
                for q_chunk in nl.static_range(q_tiles):
                    q_start = q_chunk * q_tile_size
                    q_end = q_start + q_tile_size
                    for d_tile in nl.static_range(d_tiles):
                        q_input = q_ref.select(0, global_head).slice(
                            0, q_start, q_end
                        ).slice(
                            1, d_tile * 128, (d_tile + 1) * 128
                        )
                        q_tile = q_local.select(1, q_chunk).select(
                            1, head_group
                        ).select(1, d_tile).slice(
                            1, row_start, row_end
                        )
                        q_tile[:, :] = nl.multiply(
                            nl.load_transpose2d(
                                q_input,
                                dtype=nl.bfloat16,
                            ),
                            softmax_scale,
                        )

        # Every state tile is initialized explicitly by the first history
        # block, so avoid zero/-inf memsets over the full resident state.
        running_max = nl.ndarray(
            (compute_rows, q_tiles, head_groups, 1),
            dtype=nl.float32,
            buffer=nl.sbuf,
            name="online_max",
        )
        running_sum = nl.ndarray(
            (compute_rows, q_tiles, head_groups, 1),
            dtype=nl.float32,
            buffer=nl.sbuf,
            name="online_sum",
        )
        running_out = nl.ndarray(
            (compute_rows, q_tiles, head_groups, d_tiles, 128),
            # The output numerator is consumed by BF16 TensorE matmuls and
            # eventually returned as BF16. Keeping only max/sum in FP32 cuts
            # the dominant recurrent vector-state traffic in half.
            dtype=nl.bfloat16,
            buffer=nl.sbuf,
            name="online_numerator",
        )
        # neuronx-cc 2.27 rejects collective instructions inside a hardware
        # dynamic loop.  Iterate over the compiled maximum and use a runtime
        # score mask so one artifact still supports shorter actual histories.
        for block_index in nl.static_range(max_local_blocks):
            local_block = local_kv_ref.select(0, block_index).reshape(
                (2, num_kv_heads, block_size, head_dim)
            )
            if pretranspose_k_on_owner:
                # Each LNC core owns one KV head. Transpose its local K tiles
                # once before the first ring hop and preserve that physical
                # FP8 layout through all subsequent collective permutations.
                owner_k = local_block.select(0, 0).select(0, core)
                ring_k = comm0.select(0, 0).select(0, core)
                for owner_kv_tile in nl.static_range(kv_tiles):
                    for owner_d_tile in nl.static_range(d_tiles):
                        owner_k_tile = owner_k.slice(
                            0,
                            owner_kv_tile * 128,
                            (owner_kv_tile + 1) * 128,
                        ).slice(
                            1,
                            owner_d_tile * 128,
                            (owner_d_tile + 1) * 128,
                        )
                        owner_k_sbuf = nl.ndarray(
                            (128, 128),
                            dtype=nl.bfloat16,
                            buffer=nl.sbuf,
                        )
                        owner_k_t_psum = nl.ndarray(
                            (128, 128),
                            dtype=nl.bfloat16,
                            buffer=nl.psum,
                        )
                        owner_k_t_sbuf = nl.ndarray(
                            (128, 128),
                            dtype=nl.bfloat16,
                            buffer=nl.sbuf,
                        )
                        owner_k_sbuf[:, :] = nl.load(
                            owner_k_tile,
                            dtype=nl.bfloat16,
                        )
                        nisa.nc_transpose(
                            dst=owner_k_t_psum,
                            data=owner_k_sbuf,
                            engine=nisa.engine.tensor,
                        )
                        nisa.tensor_copy(
                            dst=owner_k_t_sbuf,
                            src=owner_k_t_psum,
                        )
                        ring_k_tile = ring_k.slice(
                            0,
                            owner_kv_tile * 128,
                            (owner_kv_tile + 1) * 128,
                        ).slice(
                            1,
                            owner_d_tile * 128,
                            (owner_d_tile + 1) * 128,
                        )
                        nl.store(
                            ring_k_tile,
                            value=nl.copy(
                                owner_k_t_sbuf,
                                dtype=local_kv_ref.dtype,
                            ),
                        )

                # V is already in the orientation required by the PV matmul.
                nisa.dma_copy(
                    dst=comm0.select(0, 1).select(0, core),
                    src=local_block.select(0, 1).select(0, core),
                )
            elif core == 0:
                nisa.dma_copy(
                    dst=comm0.reshape((1, payload_elements)),
                    src=local_block.reshape((1, payload_elements)),
                )
            nisa.core_barrier(data=comm0, cores=(0, 1))

            # Static PCP size gives a fixed collective topology.  Collective
            # starts before arithmetic; the barrier on its destination is the
            # point at which both cores wait before consuming the next buffer.
            for ring_step in nl.static_range(pcp_size):
                if ring_step % 2 == 0:
                    current, incoming = comm0, comm1
                else:
                    current, incoming = comm1, comm0

                # The final payload has already visited every rank. Sending it
                # once more would only return it to its owner, and the result
                # is never consumed, so omit that collective and its tail
                # barrier.
                has_next_ring_step = ring_step + 1 < pcp_size
                if pcp_size > 1 and has_next_ring_step:
                    ncc.collective_permute_implicit(
                        srcs_by_channel=[[current]],
                        dsts_by_channel=[[incoming]],
                        replica_group=replica_group,
                        channel_ids=[0],
                    )

                # One KV head is assigned to each LNC core. Load every K/V
                # tile once and reuse it across all Q heads mapped to that KV
                # head, instead of issuing the same HBM reads 16 times.
                kv_head = core
                for q_chunk in nl.static_range(q_tiles):
                    q_start = q_chunk * q_tile_size
                    q_end = q_start + q_tile_size
                    block_valid_mask = nl.ndarray(
                        (compute_rows, block_size),
                        dtype=nl.bfloat16,
                        buffer=nl.sbuf,
                    )
                    resident_k = nl.ndarray(
                        (128, d_tiles, block_size),
                        dtype=resident_kv_dtype,
                        buffer=nl.sbuf,
                    )
                    resident_v = nl.ndarray(
                        (128, kv_tiles, d_tiles, 128),
                        dtype=resident_kv_dtype,
                        buffer=nl.sbuf,
                    )
                    for kv_tile in nl.static_range(kv_tiles):
                        kv_start = kv_tile * 128
                        kv_end = kv_start + 128
                        base_block_valid_mask = nl.load(
                            block_valid_mask_ref.select(
                                0, block_index
                            ).select(0, kv_tile).slice(0, q_start, q_end),
                            dtype=nl.bfloat16,
                        )
                        for group_head in nl.static_range(
                            heads_per_compute_tile
                        ):
                            row_start = group_head * q_tile_size
                            row_end = row_start + q_tile_size
                            block_valid_mask.slice(
                                0, row_start, row_end
                            ).slice(1, kv_start, kv_end)[:, :] = nl.copy(
                                base_block_valid_mask,
                                dtype=nl.bfloat16,
                            )
                        for d_tile in nl.static_range(d_tiles):
                            current_k = current.select(0, 0).select(
                                0, kv_head
                            ).slice(
                                0, kv_tile * 128, (kv_tile + 1) * 128
                            ).slice(
                                1, d_tile * 128, (d_tile + 1) * 128
                            )
                            k_tile = resident_k.select(1, d_tile).slice(
                                1, kv_start, kv_end
                            )
                            if pretranspose_k_on_owner:
                                k_tile[:, :] = nl.load(
                                    current_k,
                                    dtype=resident_kv_dtype,
                                )
                            elif local_kv_ref.dtype == nl.bfloat16:
                                k_tile[:, :] = nl.load_transpose2d(
                                    current_k,
                                    dtype=nl.bfloat16,
                                )
                            else:
                                k_tile_untransposed = nl.ndarray(
                                    (128, 128),
                                    dtype=nl.bfloat16,
                                    buffer=nl.sbuf,
                                )
                                k_tile_transposed_psum = nl.ndarray(
                                    (128, 128),
                                    dtype=nl.bfloat16,
                                    buffer=nl.psum,
                                )
                                k_tile_untransposed[:, :] = nl.load(
                                    current_k,
                                    dtype=nl.bfloat16,
                                )
                                nisa.nc_transpose(
                                    dst=k_tile_transposed_psum,
                                    data=k_tile_untransposed,
                                    engine=nisa.engine.tensor,
                                )
                                nisa.tensor_copy(
                                    dst=k_tile,
                                    src=k_tile_transposed_psum,
                                )

                            current_v = current.select(0, 1).select(
                                0, kv_head
                            ).slice(
                                0, kv_tile * 128, (kv_tile + 1) * 128
                            ).slice(
                                1, d_tile * 128, (d_tile + 1) * 128
                            )
                            resident_v.select(1, kv_tile).select(
                                1, d_tile
                            )[:, :] = nl.load(
                                current_v, dtype=resident_kv_dtype
                            )

                    # Fuse all four 128-token KV tiles in the cache block into
                    # one 512-column QK/softmax. This updates online state once
                    # per block and keeps Q stationary while all K columns
                    # stream through TensorE.
                    for head_group in nl.static_range(head_groups):
                        head_max = running_max.select(
                            1, q_chunk
                        ).select(1, head_group)
                        head_sum = running_sum.select(
                            1, q_chunk
                        ).select(1, head_group)
                        head_out = running_out.select(
                            1, q_chunk
                        ).select(1, head_group)
                        head_q = q_local.select(1, q_chunk).select(
                            1, head_group
                        )

                        score_psum = nl.ndarray(
                            (compute_rows, block_size),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )
                        for d_tile in nl.static_range(d_tiles):
                            nisa.nc_matmul(
                                dst=score_psum,
                                stationary=head_q.select(1, d_tile),
                                moving=resident_k.select(1, d_tile),
                                accumulate=d_tile != 0,
                            )

                        scores = nl.ndarray(
                            (compute_rows, block_size),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        nisa.tensor_tensor(
                            dst=scores,
                            data1=score_psum,
                            data2=block_valid_mask,
                            op=nl.add,
                            engine=nisa.engine.vector,
                        )
                        tile_max = nl.max(scores, axis=1, keepdims=True)
                        is_first_history_tile = (
                            block_index == 0 and ring_step == 0
                        )
                        if is_first_history_tile:
                            new_max = tile_max
                        else:
                            new_max = nl.maximum(head_max, tile_max)
                            alpha = nl.exp(
                                nl.subtract(head_max, new_max)
                            )
                        probabilities = nl.ndarray(
                            (compute_rows, block_size),
                            dtype=nl.bfloat16,
                            buffer=nl.sbuf,
                        )
                        tile_sum = nl.ndarray(
                            (compute_rows, 1),
                            dtype=nl.float32,
                            buffer=nl.sbuf,
                        )
                        nisa.activation(
                            dst=probabilities,
                            op=nl.exp,
                            data=scores,
                            bias=nl.negative(new_max),
                            reduce_op=nl.add,
                            reduce_res=tile_sum,
                            reduce_cmd=nisa.reduce_cmd.reset_reduce,
                        )

                        probabilities_t = nl.ndarray(
                            (128, kv_tiles, compute_rows),
                            dtype=nl.bfloat16,
                            buffer=nl.sbuf,
                        )
                        for kv_tile in nl.static_range(kv_tiles):
                            kv_start = kv_tile * 128
                            kv_end = kv_start + 128
                            probabilities_t_psum = nl.ndarray(
                                (128, compute_rows),
                                dtype=nl.bfloat16,
                                buffer=nl.psum,
                            )
                            nisa.nc_transpose(
                                dst=probabilities_t_psum,
                                data=probabilities.slice(
                                    1, kv_start, kv_end
                                ),
                                engine=nisa.engine.tensor,
                            )
                            nisa.tensor_copy(
                                dst=probabilities_t.select(1, kv_tile),
                                src=probabilities_t_psum,
                            )

                        for d_tile in nl.static_range(d_tiles):
                            pv_psum = nl.ndarray(
                                (128, compute_rows),
                                dtype=nl.float32,
                                buffer=nl.psum,
                            )
                            for kv_tile in nl.static_range(kv_tiles):
                                nisa.nc_matmul(
                                    dst=pv_psum,
                                    stationary=resident_v.select(
                                        1, kv_tile
                                    ).select(1, d_tile),
                                    moving=probabilities_t.select(
                                        1, kv_tile
                                    ),
                                    accumulate=kv_tile != 0,
                                )
                            pv_dq = nl.ndarray(
                                (128, compute_rows),
                                dtype=nl.bfloat16,
                                buffer=nl.sbuf,
                            )
                            nisa.tensor_copy(dst=pv_dq, src=pv_psum)
                            pv_qd = nl.ndarray(
                                (compute_rows, 128),
                                dtype=nl.bfloat16,
                                buffer=nl.sbuf,
                            )
                            pv_qd_psum = nl.ndarray(
                                (compute_rows, 128),
                                dtype=nl.bfloat16,
                                buffer=nl.psum,
                            )
                            nisa.nc_transpose(
                                dst=pv_qd_psum,
                                data=pv_dq,
                                engine=nisa.engine.tensor,
                            )
                            nisa.tensor_copy(dst=pv_qd, src=pv_qd_psum)
                            out_state = head_out.select(1, d_tile)
                            if is_first_history_tile:
                                out_state[:, :] = nl.copy(
                                    pv_qd, dtype=nl.bfloat16
                                )
                            else:
                                out_state[:, :] = nl.add(
                                    nl.multiply(out_state, alpha),
                                    pv_qd,
                                )

                        if is_first_history_tile:
                            head_sum[:, :] = nl.copy(
                                tile_sum, dtype=nl.float32
                            )
                        else:
                            head_sum[:, :] = nl.add(
                                nl.multiply(head_sum, alpha),
                                tile_sum,
                            )
                        head_max[:, :] = nl.copy(
                            new_max, dtype=nl.float32
                        )

                if pcp_size > 1 and has_next_ring_step:
                    nisa.core_barrier(data=incoming, cores=(0, 1))

        for head_group in nl.static_range(head_groups):
            for q_chunk in nl.static_range(q_tiles):
                q_start = q_chunk * q_tile_size
                q_end = q_start + q_tile_size
                sum_state = running_sum.select(1, q_chunk).select(
                    1, head_group
                )
                inverse_sum = nl.ndarray(
                    (compute_rows, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.reciprocal(dst=inverse_sum, data=sum_state)
                for d_tile in nl.static_range(d_tiles):
                    out_state = running_out.select(1, q_chunk).select(
                        1, head_group
                    ).select(1, d_tile)
                    normalized = nl.multiply(out_state, inverse_sum)
                    for group_head in nl.static_range(
                        heads_per_compute_tile
                    ):
                        local_head = (
                            head_group * heads_per_compute_tile + group_head
                        )
                        global_head = core * heads_per_core + local_head
                        row_start = group_head * q_tile_size
                        row_end = row_start + q_tile_size
                        output_tile = out_ref.select(
                            0, global_head
                        ).slice(0, q_start, q_end).slice(
                            1, d_tile * 128, (d_tile + 1) * 128
                        )
                        nl.store(
                            output_tile,
                            value=nl.copy(
                                normalized.slice(0, row_start, row_end),
                                dtype=q_ref.dtype,
                            ),
                        )
        nisa.core_barrier(data=out_ref, cores=(0, 1))
        return out_ref

    return history_attention_kernel


def wrap_for_native_torch(config: PCPAttentionConfig):
    """Return a native-PyTorch callable using the installed libtorch bridge."""

    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(make_history_attention_kernel(config))[config.lnc]
