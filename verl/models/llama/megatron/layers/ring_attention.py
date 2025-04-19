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

        # --- Layers ---
        # QKV Projection (Consider reusing or adapting QKVParallelLinear)
        # TODO: Verify if QKVParallelLinear needs changes for Ring Attention context
        #       or if a standard ColumnParallelLinear followed by manual splitting is better.
        self.qkv_proj = QKVParallelLinear(
            input_size=self.hidden_size,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            bias=config.attention_bias,
            # gather_output=False, # Ring attention handles distribution differently
            # skip_bias_add=False,
            # **column_kwargs, # Need to get appropriate kwargs
        )
        self.q_size = self.num_heads_per_tp * self.head_dim
        self.k_size = self.num_key_value_heads_per_tp * self.head_dim
        self.v_size = self.num_key_value_heads_per_tp * self.head_dim


        # Output Projection (RowParallelLinear)
        self.o_proj = tensor_parallel.RowParallelLinear(
            input_size=self.num_heads * self.head_dim, # Input is gathered across TP group before projection
            output_size=self.hidden_size,
            bias=config.attention_bias,
            input_is_parallel=True, # Output of Ring Attention should be gathered
            # skip_bias_add=False,
            # **row_kwargs, # Need to get appropriate kwargs
        )

        # --- RoPE ---
        # TODO: Initialize RoPE. Need to carefully consider how position_ids are handled
        #       across the distributed sequence length. The standard RoPE might need adaptation
        #       or careful application based on global position IDs for each chunk.
        # self.rotary_emb = LlamaRotaryEmbedding(...) # Or scaled versions

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
        # TODO: Project hidden_states to Q, K, V.
        # qkv = self.qkv_proj(hidden_states)[0] # Assuming qkv_proj handles TP splitting internally
        # query_states, key_states, value_states = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1) # Or dim=2 depending on input shape

        # --- Placeholder for QKV calculation ---
        # Assuming hidden_states is (local_seq_len, bsz, hidden_size)
        # This part needs careful implementation based on QKVParallelLinear or alternatives
        query_states = torch.randn(local_seq_len, bsz, self.num_heads_per_tp, self.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        key_states = torch.randn(local_seq_len, bsz, self.num_key_value_heads_per_tp, self.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        value_states = torch.randn(local_seq_len, bsz, self.num_key_value_heads_per_tp, self.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        # --- End Placeholder ---

        # Reshape Q, K, V for attention calculation
        # Example: (local_seq_len, bsz, num_heads, head_dim) -> (bsz, num_heads, local_seq_len, head_dim)
        # query_states = query_states.view(local_seq_len, bsz, self.num_heads_per_tp, self.head_dim).permute(1, 2, 0, 3)
        # key_states = key_states.view(local_seq_len, bsz, self.num_key_value_heads_per_tp, self.head_dim).permute(1, 2, 0, 3)
        # value_states = value_states.view(local_seq_len, bsz, self.num_key_value_heads_per_tp, self.head_dim).permute(1, 2, 0, 3)
        # --- Placeholder for reshape ---
        query_states = query_states.permute(1, 2, 0, 3) # (bsz, num_heads_per_tp, local_seq_len, head_dim)
        key_states = key_states.permute(1, 2, 0, 3)   # (bsz, num_key_value_heads_per_tp, local_seq_len, head_dim)
        value_states = value_states.permute(1, 2, 0, 3) # (bsz, num_key_value_heads_per_tp, local_seq_len, head_dim)
        # --- End Placeholder ---


        # --- 2. Apply RoPE ---
        # TODO: Apply Rotary Positional Embeddings.
        #       Requires `cos` and `sin` calculated based on *global* `position_ids`.
        #       The `position_ids` passed to forward should correspond to the global positions
        #       for the `local_seq_len` chunk this rank is processing.
        # cos, sin = self.rotary_emb(value_states, seq_len=global_seq_len) # Calculate for full sequence? Or just needed part?
        # cos_chunk = cos[position_ids] # Select the relevant part
        # sin_chunk = sin[position_ids]
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos_chunk, sin_chunk, position_ids=None) # position_ids already used for slicing cos/sin

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

        # Placeholder for Online Softmax stats if needed
        # local_max_score = torch.full((bsz, self.num_heads_per_tp, local_seq_len, 1), -float('inf'), ...)
        # local_sum_exp = torch.zeros((bsz, self.num_heads_per_tp, local_seq_len, 1), ...)

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

            # Placeholder calculation:
            # Repeat K/V if using Grouped Query Attention
            # current_key_states_rep = repeat_kv(current_key_states, self.num_key_value_groups)
            # current_value_states_rep = repeat_kv(current_value_states, self.num_key_value_groups)
            # attn_weights = torch.matmul(query_states, current_key_states_rep.transpose(2, 3)) / math.sqrt(self.head_dim)

            # TODO: Apply causal mask for this block/iteration
            # TODO: Implement Online Softmax update here if needed
            # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            # attn_output_block = torch.matmul(attn_weights, current_value_states_rep)

            # --- Placeholder for block output ---
            attn_output_block = torch.randn_like(query_states)
            # --- End Placeholder ---

            # Accumulate results
            attn_output_accumulator += attn_output_block

            # --- c. Update K/V for next iteration ---
            current_key_states = key_recv_buffer
            current_value_states = value_recv_buffer
            # Swap buffers for next recv
            key_recv_buffer, current_key_states = current_key_states, key_recv_buffer
            value_recv_buffer, current_value_states = current_value_states, value_recv_buffer


        # --- 5. Finalize Attention Output ---
        # TODO: If using Online Softmax, perform final normalization.
        attn_output = attn_output_accumulator

        # Reshape output to match expected format for RowParallelLinear
        # Example: (bsz, num_heads_per_tp, local_seq_len, head_dim) -> (bsz, local_seq_len, hidden_size_per_tp) -> (local_seq_len * bsz, hidden_size_per_tp) ?
        # The exact reshape depends on RowParallelLinear's expectation when input_is_parallel=True
        # It might expect (seq_len * bsz / tp, hidden_size) or similar. Needs verification.

        # --- Placeholder reshape ---
        # attn_output = attn_output.transpose(1, 2).contiguous() # (bsz, local_seq_len, num_heads_per_tp, head_dim)
        # attn_output = attn_output.reshape(bsz, local_seq_len, self.hidden_size_per_tp) # (bsz, local_seq_len, hidden_size_per_tp)
        # Need to match o_proj input format when input_is_parallel=True
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(local_seq_len * bsz, self.hidden_size_per_tp) # Example: (local_seq_len * bsz, hidden_size_per_tp)
        # --- End Placeholder ---


        # --- 6. Output Projection ---
        # The RowParallelLinear with input_is_parallel=True should handle gathering across TP ranks internally.
        output = self.o_proj(attn_output)[0]

        # --- 7. Reshape output to original format ---
        # Example: If input was (local_seq_len, bsz, hidden_size), output should be the same.
        # output = output.view(local_seq_len, bsz, self.hidden_size) # Example reshape

        # --- Placeholder reshape ---
        output = output.view(local_seq_len, bsz, self.hidden_size)
        # --- End Placeholder ---

        return output