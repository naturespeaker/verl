# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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

import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from megatron.core import ModelParallelConfig, tensor_parallel
from megatron.core import parallel_state as mpu
from torch import nn
from transformers import LlamaConfig

# Potentially reuse or adapt components from parallel_attention
from .parallel_linear import QKVParallelLinear
from .parallel_attention import LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv # Check if RoPE needs modification for ring
from verl.utils.megatron import tensor_parallel as tp_utils

# Potentially use flash_attn components if integrating Flash Attention
# from flash_attn import flash_attn_varlen_func
# from flash_attn.bert_padding import unpad_input, pad_input # For handling unpadded inputs if needed


class ParallelLlamaRingAttention(nn.Module):
    """
    Parallel Llama Attention layer with Ring Attention implementation.

    This layer implements Ring Attention, where the sequence is split across
    devices in the Tensor Parallel group, and Key/Value blocks are communicated
    in a ring fashion to compute attention scores distributively.
    """

    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        super().__init__()
        self.config = config
        self.megatron_config = megatron_config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # --- Tensor Parallelism Setup ---
        self.tp_group = mpu.get_tensor_model_parallel_group()
        self.tp_size = mpu.get_tensor_model_parallel_world_size()
        self.tp_rank = mpu.get_tensor_model_parallel_rank()

        assert self.num_heads % self.tp_size == 0, "num_heads must be divisible by tp_size"
        assert self.num_key_value_heads % self.tp_size == 0, "num_key_value_heads must be divisible by tp_size"

        self.num_heads_per_tp = self.num_heads // self.tp_size
        self.num_key_value_heads_per_tp = self.num_key_value_heads // self.tp_size
        self.hidden_size_per_tp = self.hidden_size // self.tp_size

        # --- Get Parallelism Kwargs ---
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        row_kwargs = tp_utils.get_default_kwargs_for_row_parallel_linear()
        if megatron_config is not None:
            # Ensure config objects are present in default kwargs (they should be)
            if "config" not in column_kwargs: column_kwargs["config"] = megatron_config
            if "config" not in row_kwargs: row_kwargs["config"] = megatron_config
            # Update kwargs based on the provided megatron_config
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)
            tp_utils.update_kwargs_with_config(row_kwargs, megatron_config)
        else:
            # Handle case where megatron_config might be None if necessary
            # For now, assume it's always provided
            pass

        # --- Layers ---
        # QKV Projection
        self.qkv_proj = QKVParallelLinear(
            input_size=self.hidden_size,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            bias=config.attention_bias,
            gather_output=False, # Keep QKV split across TP ranks
            skip_bias_add=False, # Let the layer handle bias addition
            **column_kwargs,
        )
        self.q_size = self.num_heads_per_tp * self.head_dim
        self.k_size = self.num_key_value_heads_per_tp * self.head_dim
        self.v_size = self.num_key_value_heads_per_tp * self.head_dim


        self.o_proj = tensor_parallel.RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            # bias=config.attention_bias,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            **row_kwargs,
        )

        # --- RoPE ---
        # Initialize RoPE based on LlamaConfig
        # We assume position_ids passed to forward will be global and handle slicing there.
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # --- Ring Communication Setup ---
        self.send_rank = (self.tp_rank + 1) % self.tp_size
        self.recv_rank = (self.tp_rank - 1 + self.tp_size) % self.tp_size


    def forward(
        self,
        hidden_states: torch.Tensor, # Shape: (local_seq_len_chunk, batch_size, hidden_size) or (batch_size, local_seq_len_chunk, hidden_size) - TBD
        # attention_mask: Optional[torch.Tensor] = None, # Causal mask is often handled internally by Flash Attn or Ring logic
        position_ids: Optional[torch.LongTensor] = None, # Shape should correspond to the *global* sequence positions for the local chunk
        # cu_seqlens: Optional[torch.Tensor] = None, # Needed if handling unpadded data with Flash Attn
        # max_seqlen_in_batch: Optional[int] = None, # Needed if handling unpadded data with Flash Attn
    ) -> torch.Tensor: # Output shape: same as input hidden_states

        # --- 0. Input Shape and Preparation ---
        # TODO: Determine the expected input shape. Megatron SP often uses (seq_len/tp, batch, hidden).
        # TODO: Handle potential unpadded input (if using flash_attn_varlen_func later).
        #       This might involve receiving unpadded hidden_states, cu_seqlens, etc.
        # bsz, local_seq_len, _ = hidden_states.size() # Example if input is (bs, local_seq, hidden)
        local_seq_len, bsz, _ = hidden_states.size() # Example if input is (local_seq, bs, hidden)
        # global_seq_len = local_seq_len * self.tp_size # Assuming sequence is perfectly divisible for now

        # --- 1. QKV Projection ---
        # Project hidden_states to Q, K, V using the parallel linear layer.
        # qkv_proj likely returns (output, bias), we take the output.
        qkv_output, _ = self.qkv_proj(hidden_states)
        # Split the combined QKV tensor into individual Q, K, V tensors.
        # The dimension for splitting depends on the output shape of QKVParallelLinear.
        # Assuming output shape is (local_seq_len, bsz, hidden_size_per_tp * (num_q_groups + 2)) where num_q_groups = num_heads / num_kv_heads
        # Or more likely (local_seq_len, bsz, q_size + k_size + v_size) where sizes are per TP rank.
        query_states, key_states, value_states = qkv_output.split([self.q_size, self.k_size, self.v_size], dim=2) # dim=2 if shape is (seq, batch, features)

        # (No direct replacement needed here as the placeholder is removed by the change above)

        # Reshape Q, K, V for attention calculation
        # Example: (local_seq_len, bsz, num_heads, head_dim) -> (bsz, num_heads, local_seq_len, head_dim)
        # query_states = query_states.view(local_seq_len, bsz, self.num_heads_per_tp, self.head_dim).permute(1, 2, 0, 3)
        # key_states = key_states.view(local_seq_len, bsz, self.num_key_value_heads_per_tp, self.head_dim).permute(1, 2, 0, 3)
        value_states = value_states.view(local_seq_len, bsz, self.num_key_value_heads_per_tp, self.head_dim).permute(1, 2, 0, 3)

        # Reshape Q, K for attention calculation: (seq_len, bsz, num_heads, head_dim) -> (bsz, num_heads, seq_len, head_dim)
        query_states = query_states.view(local_seq_len, bsz, self.num_heads_per_tp, self.head_dim).permute(1, 2, 0, 3)
        key_states = key_states.view(local_seq_len, bsz, self.num_key_value_heads_per_tp, self.head_dim).permute(1, 2, 0, 3)
        # value_states is already reshaped above


        # --- 2. Apply RoPE ---
        # Calculate cos and sin frequencies based on the maximum sequence length.
        # We assume position_ids are provided correctly for the local chunk's global positions.
        cos, sin = self.rotary_emb(value_states, seq_len=self.max_position_embeddings) # Use max_position_embeddings for cache size

        # Select the cosine and sine values corresponding to the specific position_ids for this chunk.
        # position_ids shape needs to be compatible for indexing. Expected: (local_seq_len) or (1, local_seq_len)
        # cos/sin shape: (max_pos, 1, head_dim/2) or similar. Indexing needs care.
        # Let's assume position_ids is (local_seq_len) and needs unsqueezing.
        # cos = cos[position_ids].unsqueeze(1) # Shape: (local_seq_len, 1, dim)
        # sin = sin[position_ids].unsqueeze(1) # Shape: (local_seq_len, 1, dim)
        # The apply_rotary_pos_emb function likely expects cos/sin indexed by position_ids directly.
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # --- 3. Prepare for Ring Communication ---
        # Make K/V contiguous if needed for communication
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        # Initialize output accumulator
        attn_output_accumulator = torch.zeros_like(query_states)

        # Initialize buffers for receiving K/V
        key_recv_buffer = torch.empty_like(key_states)
        value_recv_buffer = torch.empty_like(value_states)

        # --- 4. Ring Attention Loop ---
        current_key_states = key_states
        current_value_states = value_states

        # (Removing Online Softmax placeholders)

        for ring_iter in range(self.tp_size):
            # --- a. Communication (Send current K/V, Receive next K/V) ---
            # Start non-blocking recv first
            recv_op_k = dist.P2POp(dist.irecv, key_recv_buffer, self.recv_rank, group=self.tp_group)
            recv_op_v = dist.P2POp(dist.irecv, value_recv_buffer, self.recv_rank, group=self.tp_group)
            # Send current K/V
            send_op_k = dist.P2POp(dist.isend, current_key_states, self.send_rank, group=self.tp_group)
            send_op_v = dist.P2POp(dist.isend, current_value_states, self.send_rank, group=self.tp_group)

            ops = [recv_op_k, recv_op_v, send_op_k, send_op_v]
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

            # --- b. Local Attention Calculation ---
            # Calculate attention between local Q and current K/V block
            # TODO: Implement actual attention calculation.
            #       Consider using flash_attn_varlen_func here if inputs are unpadded.
            #       Need to handle causal masking appropriately within the ring context.
            #       The mask should prevent attending to future tokens *within the current block*
            #       and potentially across blocks depending on the ring iteration.

            # --- b. Local Attention Calculation ---
            # Calculate attention between local Q and the K/V block for this iteration.

            # Repeat K/V heads if using Grouped Query Attention (GQA)
            if self.num_key_value_groups > 1:
                current_key_states_rep = repeat_kv(current_key_states, self.num_key_value_groups)
                current_value_states_rep = repeat_kv(current_value_states, self.num_key_value_groups)
            else:
                current_key_states_rep = current_key_states
                current_value_states_rep = current_value_states

            # Calculate attention scores: (bsz, num_heads, q_len, k_len)
            # q_len is local_seq_len, k_len is also local_seq_len (length of the received block)
            attn_weights = torch.matmul(query_states, current_key_states_rep.transpose(2, 3)) / math.sqrt(self.head_dim)

            # Apply causal mask based on ring iteration
            if ring_iter == 0:
                # Local block: Standard causal mask
                # attn_weights shape: (bsz, num_heads_per_tp, local_seq_len, local_seq_len)
                causal_mask = torch.triu(
                    torch.ones((local_seq_len, local_seq_len), device=attn_weights.device, dtype=torch.bool),
                    diagonal=1
                )
                # Expand mask to broadcast: (1, 1, local_seq_len, local_seq_len)
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                attn_weights.masked_fill_(causal_mask, torch.finfo(attn_weights.dtype).min) # Use min value of dtype
            else:
                # Remote block: Check if it's from the future
                source_rank = (self.tp_rank - ring_iter + self.tp_size) % self.tp_size
                # Assuming ranks process contiguous blocks (0, 1, 2, ...),
                # if source_rank > current rank, it's a future block.
                if source_rank > self.tp_rank:
                    # Mask all attention to future blocks
                    attn_weights.fill_(torch.finfo(attn_weights.dtype).min) # Use min value of dtype
                # Else (source_rank < self.tp_rank): It's a past block, no mask needed.

            # Apply softmax
            # Upcast attention to fp32 for stability
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # Calculate attention output for this block: (bsz, num_heads, q_len, head_dim)
            attn_output_block = torch.matmul(attn_weights, current_value_states_rep)

            # (Placeholder removed by the change above)

            # Accumulate results
            attn_output_accumulator += attn_output_block

            # --- c. Update K/V for next iteration ---
            current_key_states = key_recv_buffer
            current_value_states = value_recv_buffer
            # Swap buffers for next recv
            key_recv_buffer, current_key_states = current_key_states, key_recv_buffer
            value_recv_buffer, current_value_states = current_value_states, value_recv_buffer


        # --- 5. Finalize Attention Output ---
        # The accumulator now holds the sum of attention outputs from all blocks.
        # If not using Online Softmax, this is the final attention output before projection.
        attn_output = attn_output_accumulator

        # Reshape output for RowParallelLinear input
        # Input shape: (bsz, num_heads_per_tp, local_seq_len, head_dim)
        # Target shape: (bsz * local_seq_len, hidden_size_per_tp)
        attn_output = attn_output.transpose(1, 2).contiguous() # -> (bsz, local_seq_len, num_heads_per_tp, head_dim)
        attn_output = attn_output.view(bsz * local_seq_len, self.hidden_size_per_tp) # -> (bsz * local_seq_len, hidden_size_per_tp)


        # --- 6. Output Projection ---
        # The RowParallelLinear with input_is_parallel=True should handle gathering across TP ranks internally.
        output = self.o_proj(attn_output)[0]

        # --- 7. Reshape output to original format ---
        # Example: If input was (local_seq_len, bsz, hidden_size), output should be the same.
        # output = output.view(local_seq_len, bsz, self.hidden_size) # Example reshape

        # Reshape final output to match the input hidden_states shape: (local_seq_len, bsz, hidden_size)
        # o_proj output shape: (bsz * local_seq_len, hidden_size)
        output = output.view(bsz, local_seq_len, self.hidden_size).transpose(0, 1).contiguous() # -> (local_seq_len, bsz, hidden_size)

        return output