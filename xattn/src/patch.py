# This file is derived from:
# https://github.com/ByteDance-Seed/FlexPrefill/blob/main/flex_prefill/modules/patch.py
# - patch xattention for llama model in vllm(0.6.1.post1)

# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import types
import torch

from xattn.src.Xattention import Xattention_prefill

import vllm
from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.backends.flash_attn import FlashAttentionMetadata


def patch_vllm_model(model, pattern: str, cfg: dict = None):
    assert isinstance(model, vllm.LLM)
    assert pattern in [
        "default",
        "flash",
        "xattn",
    ], "only support default, flash and xattn"

    if pattern != "xattn":
        return

    model.llm_engine.scheduler_config.chunked_prefill_enabled = False
    model.llm_engine.scheduler_config.max_num_seqs = 1

    def xattn_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashAttentionImpl"
            )

        # NOTE(woosuk): FlashAttention does not support FP8 KV cache.
        assert (
            k_scale == 1.0 and v_scale == 1.0
        ), "key/v_scale is not supported in FlashAttention."

        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping.flatten(),
                self.kv_cache_dtype,
                k_scale,
                v_scale,
            )

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert key.shape[0] == num_prefill_tokens + num_decode_tokens
        assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_tokens]
        key = key[:num_prefill_tokens]
        value = value[:num_prefill_tokens]
        if self.num_heads != self.num_kv_heads: # repeat kv head
            group = self.num_heads // self.num_kv_heads
            assert(group * self.num_kv_heads == self.num_heads)
            key = key.repeat_interleave(group, dim=1).contiguous()
            value = value.repeat_interleave(group, dim=1).contiguous()

        assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if (
                kv_cache is None
                or prefill_meta.block_tables is None
                or prefill_meta.block_tables.numel() == 0
            ):
                # normal attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                
                # out = torch.ops.vllm.flash_attn_varlen_func(
                #     q=query,
                #     k=key,
                #     v=value,
                #     cu_seqlens_q=prefill_meta.seq_start_loc,
                #     cu_seqlens_k=prefill_meta.seq_start_loc,
                #     max_seqlen_q=prefill_meta.max_prefill_seq_len,
                #     max_seqlen_k=prefill_meta.max_prefill_seq_len,
                #     softmax_scale=self.scale,
                #     causal=True,
                #     window_size=self.sliding_window,
                #     alibi_slopes=self.alibi_slopes,
                #     softcap=self.logits_soft_cap,
                # )
                cfg_use_triton = cfg.get("use_triton", 1)
                if cfg_use_triton == 1:
                    use_triton = True
                else:
                    use_triton = False
                cfg_use_pooling = cfg.get("use_pooling", 1)
                if cfg_use_pooling == 1:
                    use_pooling = True
                else:
                    use_pooling = False
                out = Xattention_prefill(
                    query.transpose(0,1).contiguous().unsqueeze(0),
                    key.transpose(0,1).contiguous().unsqueeze(0),
                    value.transpose(0,1).contiguous().unsqueeze(0),
                    stride=cfg.get("stride", 16),
                    block_size=cfg.get("block_size", 128),
                    use_triton=use_triton,
                    chunk_size=cfg.get("chunk_size", 2048),
                    threshold=cfg.get("threshold",0.8),
                    use_pooling=use_pooling
                ).squeeze(0).transpose(0,1).contiguous()
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out
            else:
                # prefix-enabled attention
                assert prefill_meta.seq_lens is not None
                max_seq_len = max(prefill_meta.seq_lens)
                output[:num_prefill_tokens] = (
                    torch.ops.vllm.flash_attn_varlen_func(  # noqa
                        q=query,
                        k=key_cache,
                        v=value_cache,
                        cu_seqlens_q=prefill_meta.query_start_loc,
                        max_seqlen_q=prefill_meta.max_query_len,
                        cu_seqlens_k=prefill_meta.seq_start_loc,
                        max_seqlen_k=max_seq_len,
                        softmax_scale=self.scale,
                        causal=True,
                        alibi_slopes=self.alibi_slopes,
                        block_table=prefill_meta.block_tables,
                        softcap=self.logits_soft_cap,
                    )
                )

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            output[num_prefill_tokens:] = torch.ops.vllm.flash_attn_with_kvcache(
                decode_query.unsqueeze(1),
                key_cache,
                value_cache,
                block_table=decode_meta.block_tables,
                cache_seqlens=decode_meta.seq_lens_tensor,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                softcap=self.logits_soft_cap,
            ).squeeze(1)

        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)

    for (
        idx,
        m,
    ) in (
        model.llm_engine.model_executor.driver_worker.model_runner.model.named_modules()
    ):
        if "attention" in type(m).__name__.lower() and hasattr(m, "impl"):
            m.impl.forward = types.MethodType(xattn_forward, m.impl)


def patch_model(model, pattern: str, cfg: dict):
    """Patch the attention mechanism of a transformers model or vllm model to use a specified pattern.

    Args:
        model: The model to be patched. It must be either a `PreTrainedModel` from the Hugging Face transformers
            library or a vllm model.
        pattern (str): The attention pattern to apply to the model. Supported patterns include:
            - 'default': Default attention mechanism with no additional configuration.
            - 'flash': Flash attention mechanism with no additional configuration.
            - 'xattn': Xattention.
        cfg (dict): Configuration settings for the specified pattern. The required keys and values depend on
        {
            "stride": int, choices=[8,16], default=16
            "block_size", int, default=128
            "use_triton", int(bool), 1(True) or 0(False), default=1
            "chunk_size", int, default=2048
            "threshold", float, default=0.8
            "use_pooling", int(bool), 1(True) or 0(False), default=1 
            # note: when use_pooling=1, cfg params stride/use_triton/chunk_size is useless
        }
    """
    patch_vllm_model(model, pattern, cfg)
    return model
