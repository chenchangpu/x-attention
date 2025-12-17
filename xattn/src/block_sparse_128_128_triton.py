import math

import torch
import torch.nn.functional as F
import os

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from block_sparse_attn import block_sparse_attn_func

BLOCK_SIZE = 128

NUM_STAGES_OPTIONS = [2, 3, 4]
NUM_WARPS_OPTIONS = [4, 8]


DEVICE = triton.runtime.driver.active.get_active_torch_device()

def get_sparse_attn_mask_from_topk(x, topk, use_dense_for_last_block=False):
    bsz, num_head, downsample_len, _ = x.shape
    # N_CTX = downsample_len * BLOCK
    sparse_index = torch.topk(x, topk, dim=-1).indices
    dense_mask = torch.full([bsz, num_head, downsample_len, downsample_len], False, dtype=torch.bool, device=x.device)
    dense_mask.scatter_(-1, sparse_index, True)
    if use_dense_for_last_block:
        dense_mask[:, :, -2:, :] = True
    dense_mask.tril_()
    return dense_mask

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    desc_k, desc_v,  #
                    block_mask_ptr, stride_bmask_n, # dahu: block spare extra parameters
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:  # dahu: causal=True情况下，第一次调用_attn_fwd_inner
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:    # dahu: causal=True情况下，第二次调用_attn_fwd_inner
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:       # dahu: STAGE==3, causal=False情况下，有且仅调用一次_attn_fwd_inner
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_block_col_idx = start_n // BLOCK_N
        mask_value = tl.load(block_mask_ptr + k_block_col_idx * stride_bmask_n)
        if mask_value == True:
            # -- compute qk ----
            k = desc_k.load([offsetk_y, 0]).T
            qk = tl.dot(q, k)
            if STAGE == 2:
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                qk -= m_ij[:, None]
            else:
                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                qk = qk * qk_scale - m_ij[:, None]
            p = tl.math.exp2(qk)
            # -- compute correction factor
            alpha = tl.math.exp2(m_i - m_ij)
            l_ij = tl.sum(p, 1)
            # -- update output accumulator --
            if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
                BM: tl.constexpr = acc.shape[0]
                BN: tl.constexpr = acc.shape[1]
                acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
                acc0 = acc0 * alpha[:, None]
                acc1 = acc1 * alpha[:, None]
                acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
            else:
                acc = acc * alpha[:, None]
            # prepare p and v for the dot
            if dtype == tl.float8e5:
                v = desc_v.load([0, offsetv_y]).T
            else:
                v = desc_v.load([offsetv_y, 0])
            p = p.to(dtype)
            # note that this non transposed v for FP8 is only supported on Blackwell
            acc = tl.dot(p, v, acc)
            # update m_i and l_i
            # place this at the end of the loop to reduce register pressure
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            
        offsetk_y += BLOCK_N    # fix: add BLOCK_SIZE anyway
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


if supports_host_descriptor():
    NUM_STAGES_OPTIONS = [2, 3, 4]
else:
    NUM_STAGES_OPTIONS = [2, 3, 4]

configs = [
    triton.Config({}, num_stages=s, num_warps=w)
    for s in NUM_STAGES_OPTIONS
    for w in NUM_WARPS_OPTIONS
]
if "PYTEST_VERSION" in os.environ:
    # Use a single config in testing for reproducibility
    configs = [
        triton.Config(dict(BLOCK_M=128, BLOCK_N=64), num_stages=2, num_warps=4),
    ]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(configs=configs, key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"])
@triton.jit
def _attn_fwd(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, #
              block_mask_ptr, stride_bmz, stride_bmh, stride_bmm, stride_bmn, # dahu: block sparse mask 额外参数
              N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              IS_HOPPER: tl.constexpr,  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.bfloat16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0) # dahu: m_block
    off_hz = tl.program_id(1)  # dahu: bh
    off_z = off_hz // H        # dahu: b
    off_h = off_hz % H         # dahu: h

    y_dim = Z * H * N_CTX # B * H * L
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[HEAD_DIM, y_dim], strides=[N_CTX, 1],
                                         block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    
    block_mask_ptr += off_z * stride_bmz + off_h * stride_bmh + start_m * stride_bmm # dahu: block mask ptr

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX # dahu: 当前batch, head所在offset
    qo_offset_y = offset_y + start_m * BLOCK_M     # dahu: 当前batch, head，seq维度并行，所在offset
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # dahu: M维度的offs, 0 <= off < seq_len
    offs_n = tl.arange(0, BLOCK_N)                     # dahu: N维度的offs, [0,1,... BLOCK_N-1]
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # dahu: registers / shared ?
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        block_mask_ptr, stride_bmn, # dahu: block sparse额外参数
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        block_mask_ptr, stride_bmn, # dahu: block sparse额外参数
                                        offset_y, dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


# dahu: only optimize for hopper
class block_sparse_attention_128_128(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, block_mask, causal, sm_scale, warp_specialize=True):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        stride_bmz, stride_bmh, stride_bmm, stride_bmn = block_mask.stride()

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32) # dahu: tensor: [B, H, L]
        # Use device_descriptor for Hopper + warpspec.
        if supports_host_descriptor() and not (is_hopper() and warp_specialize):
            # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
            y_dim = q.shape[0] * q.shape[1] * q.shape[2] # dahu: B, H, L

            dummy_block = [1, 1]
            desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            if q.dtype == torch.float8_e5m2:
                desc_v = TensorDescriptor(v, shape=[HEAD_DIM_K, y_dim], strides=[q.shape[2], 1],
                                          block_shape=dummy_block)
            else:
                desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1],
                                          block_shape=dummy_block)
            desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(META):
            return (triton.cdiv(q.shape[2], BLOCK_SIZE), q.shape[0] * q.shape[1], 1) # dahu: m_num_blocks, B * H, 1

        _attn_fwd[grid]( # dahu:
            sm_scale, M,  # scale, tensor:[B,H,L]
            q.shape[0], q.shape[1],  # B, H
            desc_q, desc_k, desc_v, desc_o,  # global_q, global_k, global_v, global_o
            block_mask, stride_bmz, stride_bmh, stride_bmm, stride_bmn, # dahu: 传递额外参数
            N_CTX=q.shape[2],  # N_CTX = L
            HEAD_DIM=HEAD_DIM_K,  # head_dim (64 or 128)
            BLOCK_M=BLOCK_SIZE,
            BLOCK_N=BLOCK_SIZE,
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  # bool
            STAGE=stage,  # causal=True -> stage=3
            warp_specialize=warp_specialize,  #
            IS_HOPPER=is_hopper(),  #
            **extra_kern_args)

        return o

    @staticmethod
    def backward(ctx, do):
        # No gradient propagation.
        raise NotImplementedError("It does not support gradient propagation yet")
        return None, None, None, None, None, None, None


block_sparse_triton_128_128 = block_sparse_attention_128_128.apply


def test_correctness():
    BATCH, N_HEADS, SEQ_LEN, D_HEAD = 1, 1, 1024, 128
    TOPK = 8
    BLOCK = BLOCK_SIZE
    torch.manual_seed(0)

    q = torch.randn(
        BATCH,
        N_HEADS,
        SEQ_LEN,
        D_HEAD,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    sm_scale = 1.0 / (D_HEAD**0.5)

    downsample_len = math.ceil(SEQ_LEN / BLOCK)
    x_ds = torch.randn(
        [BATCH, N_HEADS, downsample_len, downsample_len],
        device="cuda",
        dtype=torch.bfloat16,
    )
    x_ds[:, :, :, 0] = 100
    block_mask = get_sparse_attn_mask_from_topk(x_ds, topk=TOPK)

    triton_output = block_sparse_triton_128_128(
        q,
        k,
        v,
        block_mask,
        True,
        sm_scale,
        True,
    )

    query_states = q.transpose(1, 2).view(SEQ_LEN, N_HEADS, D_HEAD)
    key_states = k.transpose(1, 2).view(SEQ_LEN, N_HEADS, D_HEAD)
    value_states = v.transpose(1, 2).view(SEQ_LEN, N_HEADS, D_HEAD)
    q_cu_seq_lens = torch.tensor(
        [0, SEQ_LEN], dtype=torch.int32, device=query_states.device
    )
    k_cu_seq_lens = torch.tensor(
        [0, SEQ_LEN], dtype=torch.int32, device=query_states.device
    )
    head_mask_type = torch.tensor(
        [1 for _ in range(N_HEADS)], device=query_states.device, dtype=torch.int32
    )
    assert head_mask_type.device == query_states.device
    assert q_cu_seq_lens.device == query_states.device
    assert k_cu_seq_lens.device == query_states.device
    assert key_states.device == query_states.device
    assert value_states.device == query_states.device
    assert block_mask.device == query_states.device

    attn_output = block_sparse_attn_func(   # dahu: here to optimize
        query_states,
        key_states,
        value_states,
        q_cu_seq_lens,
        k_cu_seq_lens,
        head_mask_type,
        None,
        block_mask[:, :, :SEQ_LEN, :SEQ_LEN].contiguous(),
        SEQ_LEN,
        SEQ_LEN,
        p_dropout=0.0,
        deterministic=True,
        is_causal=True,
    )
    ref_output = attn_output.view(BATCH, SEQ_LEN, N_HEADS, D_HEAD).transpose(
        1, 2
    )

    torch.testing.assert_close(
        triton_output, ref_output, atol=1e-2, rtol=1e-2
    )
    print("Correctness check passed (BLOCK_SIZE=128).")


def benchmark():
    BATCH, N_HEADS, SEQ_LEN, D_HEAD = 1, 32, 32768, 128
    BLOCK = BLOCK_SIZE
    TOPK = 8
    torch.manual_seed(0)

    q = torch.randn(
        BATCH,
        N_HEADS,
        SEQ_LEN,
        D_HEAD,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    sm_scale = 1.0 / (D_HEAD**0.5)

    downsample_len = math.ceil(SEQ_LEN / BLOCK)
    x_ds = torch.randn(
        [BATCH, N_HEADS, downsample_len, downsample_len],
        device="cuda",
        dtype=torch.bfloat16,
    )
    x_ds[:, :, :, 0] = 100
    block_mask = get_sparse_attn_mask_from_topk(x_ds, topk=TOPK)

    # 计算有效块数量与稀疏率
    used_mask = block_mask[:, :, :downsample_len, :downsample_len]
    true_blocks = int(used_mask.sum().item())
    sparsity = used_mask.float().mean().item()

    # FLOPs 估算：每个被选中块有两次 GEMM，各 2*M*N*D
    flops = 4 * BLOCK * BLOCK * D_HEAD * true_blocks

    print("=============== Triton ===============")
    for ws in [True]:
        # 预热
        torch.cuda.synchronize()
        _ = block_sparse_triton_128_128(
            q,
            k,
            v,
            block_mask,
            True,
            sm_scale,
            ws,
        )
        torch.cuda.synchronize()

        iters = 20
        times_ms = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = block_sparse_triton_128_128(
                q,
                k,
                v,
                block_mask,
                True,
                sm_scale,
                ws,
            )
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))

        avg_ms = sum(times_ms) / len(times_ms)
        tflops = (flops / 1e12) / (avg_ms / 1e3)

        print(
            f"BLOCK_SIZE={BLOCK}, warp_specialize={ws}: "
            f"avg {avg_ms:.3f} ms, sparsity={sparsity*100:.2f}%, TFLOPS={tflops:.3f}"
        )
        
    print("=============== Block Sparse ===============")
    # 预热
    torch.cuda.synchronize()
    query_states = q.transpose(1, 2).view(SEQ_LEN, N_HEADS, D_HEAD)
    key_states = k.transpose(1, 2).view(SEQ_LEN, N_HEADS, D_HEAD)
    value_states = v.transpose(1, 2).view(SEQ_LEN, N_HEADS, D_HEAD)
    q_cu_seq_lens = torch.tensor(
        [0, SEQ_LEN], dtype=torch.int32, device=query_states.device
    )
    k_cu_seq_lens = torch.tensor(
        [0, SEQ_LEN], dtype=torch.int32, device=query_states.device
    )
    head_mask_type = torch.tensor(
        [1 for _ in range(N_HEADS)], device=query_states.device, dtype=torch.int32
    )
    assert head_mask_type.device == query_states.device
    assert q_cu_seq_lens.device == query_states.device
    assert k_cu_seq_lens.device == query_states.device
    assert key_states.device == query_states.device
    assert value_states.device == query_states.device
    assert block_mask.device == query_states.device

    attn_output = block_sparse_attn_func(   # dahu: here to optimize
        query_states,
        key_states,
        value_states,
        q_cu_seq_lens,
        k_cu_seq_lens,
        head_mask_type,
        None,
        block_mask[:, :, :SEQ_LEN, :SEQ_LEN].contiguous(),
        SEQ_LEN,
        SEQ_LEN,
        p_dropout=0.0,
        deterministic=True,
        is_causal=True,
    )
    ref_output = attn_output.view(BATCH, SEQ_LEN, N_HEADS, D_HEAD).transpose(
        1, 2
    )
    torch.cuda.synchronize()

    iters = 20
    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        attn_output = block_sparse_attn_func(   # dahu: here to optimize
            query_states,
            key_states,
            value_states,
            q_cu_seq_lens,
            k_cu_seq_lens,
            head_mask_type,
            None,
            block_mask[:, :, :SEQ_LEN, :SEQ_LEN].contiguous(),
            SEQ_LEN,
            SEQ_LEN,
            p_dropout=0.0,
            deterministic=True,
            is_causal=True,
        )
        ref_output = attn_output.view(BATCH, SEQ_LEN, N_HEADS, D_HEAD).transpose(
            1, 2
        )
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    avg_ms = sum(times_ms) / len(times_ms)
    tflops = (flops / 1e12) / (avg_ms / 1e3)

    print(
        f"BLOCK_SIZE={BLOCK}: "
        f"avg {avg_ms:.3f} ms, sparsity={sparsity*100:.2f}%, TFLOPS={tflops:.3f}"
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    test_correctness()
    benchmark()


if __name__ == "__main__":
    main()


