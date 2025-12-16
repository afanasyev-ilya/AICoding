import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from dataclasses import dataclass
from typing import Optional, Tuple
from dataclasses import dataclass, field
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from contextlib import nullcontext
from torch.utils.checkpoint import checkpoint  # <-- NEW

########################################################################################################

CONTEXT_SIZE = 2048

@dataclass
class BaseGPTConfig:
    vocab_size: int
    
    block_size: int = CONTEXT_SIZE
    dropout: float = 0.1

    aux_loss_weight: float = 0.01

    # rope settings
    pos_encoding: str = "rope"
    rope_base: float = 10000.0
    rope_scale: float = 1.0

@dataclass
class MoEGPTConfig(BaseGPTConfig):
    # MHA settings
    n_layer: int = 6
    n_head: int = 16
    n_embd: int = 1024

    # MoE settings
    num_experts: int = 8
    expert_dim: int = 3072

# ---- RoPE helper ----
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, scale: float = 1.0, max_seq_len: int = 4096):
        super().__init__()
        assert dim % 2 == 0, "Rotary dim must be even"
        self.dim = dim
        self.base = base
        self.scale = scale
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _build_cache(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype) * self.scale
        freqs = torch.einsum("t,d->td", t, self.inv_freq)  # [T, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)            # [T, dim]
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        # Store as [T, dim] - we'll reshape when applying
        self.cos_cached = cos
        self.sin_cached = sin
        self.max_seq_len_cached = seq_len

    def get_cos_sin(self, seq_len: int, device, dtype):
        if seq_len > self.max_seq_len_cached or self.cos_cached.device != device:
            self._build_cache(max(seq_len, self.max_seq_len_cached + 1), device, dtype)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


import torch.backends.cuda as cuda
cuda.enable_flash_sdp(True)  # Enable Flash Attention if available

class MHA(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        # Use fused QKV projection for better memory efficiency
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self.use_rope = (getattr(config, "pos_encoding", "rope") == "rope")
        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim, base=config.rope_base, 
                                      scale=config.rope_scale, max_seq_len=config.block_size)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, q, k):
        # Same RoPE implementation as before
        B, H, T, D = q.shape
        cos, sin = self.rope.get_cos_sin(T, q.device, q.dtype)
        cos = cos.view(1, 1, T, D)
        sin = sin.view(1, 1, T, D)
        
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        return q_rotated, k_rotated

    def forward(
        self,
        x,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache: bool = False,
        **kwargs,
    ):
        # NOTE: we accept these args for HF/SGLang compatibility.
        # We don't implement KV-cache yet (past_key_value/use_cache are ignored).
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, nh, T, hd)

        if self.use_rope:
            q, k = self._apply_rope(q, k)

        # If an attention_mask is provided, Flash SDPA may not support it depending on torch version.
        # So: use Flash only when mask is None; otherwise allow math/mem_efficient fallback.
        if attention_mask is None:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout.p if self.training else 0,
                    is_causal=True,
                )
        else:
            # We accept common HF-style masks: [B, T] with 1 for tokens, 0 for padding.
            # We'll convert to an additive mask [B,1,1,T] where masked keys get -inf.
            if attention_mask.dim() == 2:
                am = attention_mask.to(dtype=torch.bool, device=q.device)  # True = keep
                # additive mask: 0 for keep, -inf for mask
                neg_inf = torch.finfo(q.dtype).min
                attn_mask = (~am).to(dtype=q.dtype) * neg_inf
                attn_mask = attn_mask[:, None, None, :]  # [B,1,1,T]
            else:
                # If it's already a broadcastable mask, just use it as-is.
                attn_mask = attention_mask

            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout.p if self.training else 0,
                    is_causal=True,
                )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.dropout(y)
        return y

######################################

class DenseFFN(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MHA(config)
        self.ff = DenseFFN(config.n_embd, config.dropout)

    def forward(self, x, **kwargs):
        x = x + self.attn(self.ln1(x), **kwargs)
        x = x + self.ff(self.ln2(x), **kwargs)
        return x, x.new_zeros(())

class MoEExpert(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.expert_fc = nn.Sequential(
            nn.Linear(config.n_embd, config.expert_dim),
            nn.GELU(),
            nn.Linear(config.expert_dim, config.n_embd),
        )

    def forward(self, x, **kwargs):
        return self.expert_fc(x)

class MoELayer(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.n_embd, config.num_experts, bias=False)
        self.experts = nn.ModuleList([MoEExpert(config) for _ in range(config.num_experts)])
        self.top_k = 2
        self.noise_epsilon = 1e-2

    def forward(self, x, **kwargs):
        B, T, C = x.shape
        x_flat = x.reshape(B * T, C)
        
        # Router with noise for load balancing
        router_logits = self.router(x_flat)
        if self.training:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_epsilon
        
        # Get top-k experts - VECTORIZED
        router_probs = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # Create expert masks - VECTORIZED
        expert_mask = torch.zeros(B * T, self.config.num_experts, device=x.device, dtype=router_probs.dtype)
        expert_mask.scatter_(1, topk_indices, topk_weights)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each expert in batch - VECTORIZED
        for expert_idx, expert in enumerate(self.experts):
            # Get tokens that use this expert
            mask = expert_mask[:, expert_idx] > 0
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = expert(expert_input, **kwargs)
                
                # Apply weights - VECTORIZED
                weights = expert_mask[mask, expert_idx].unsqueeze(-1)
                output[mask] += weights * expert_output
        
        output = output.reshape(B, T, C)
        
        # Aux loss
        aux_loss = self._compute_aux_loss(router_probs, topk_indices) if self.training else x.new_zeros(())
        
        return output, aux_loss

    def _compute_aux_loss(self, router_probs, topk_indices):
        # Compute in fp32 for stability, then cast back
        router_probs_f32 = router_probs.float()
        expert_usage = torch.zeros(self.config.num_experts, device=router_probs.device, dtype=torch.float32)
        for expert_idx in range(self.config.num_experts):
            expert_usage[expert_idx] = (topk_indices == expert_idx).float().mean()
        
        target_usage = torch.ones_like(expert_usage) / self.config.num_experts
        aux_loss = F.mse_loss(expert_usage, target_usage)
        return aux_loss.to(router_probs.dtype)

# DeepSeek-style alternating MHA -> MoE blocks
class MHAThenMoEBlock(nn.Module):
    """DeepSeek-style block: MHA followed by MoE (replaces dense FFN)"""
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MHA(config)
        self.moe = MoELayer(config)  # Replace dense FFN with MoE

    def forward(self, x, **kwargs):
        x = x + self.attn(self.ln1(x), **kwargs)
        moe_out, aux_loss = self.moe(self.ln2(x), **kwargs)
        x = x + moe_out
        return x, aux_loss


class MoEGPT(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = None
        if config.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # DeepSeek-style: alternating MHA -> MoE blocks
        self.blocks = nn.ModuleList([
            MHAThenMoEBlock(config) for _ in range(config.n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        idx=None,
        targets=None,
        input_ids=None,
        labels=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ):
        # Alias HF names -> your internal names
        if idx is None and input_ids is not None:
            idx = input_ids
        if targets is None and labels is not None:
            targets = labels

        B, T = idx.shape

        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
            x = x + self.pos_emb(pos)

        x = self.drop(x)
        aux_total = x.new_zeros(())

        if self.training:
            for block in self.blocks:
                def custom_forward(x_in, block=block):
                    # NOTE: checkpoint can't accept kwargs directly; but in training
                    # you typically don't have attention_mask etc anyway.
                    return block(x_in)

                x, aux_loss = checkpoint(custom_forward, x)
                aux_total = aux_total + aux_loss
        else:
            for block in self.blocks:
                x, aux_loss = block(
                    x,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,      # ignored
                    use_cache=False,          # ignored
                    **kwargs,
                )
                aux_total = aux_total + aux_loss

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.view(-1)
            )

        return logits, loss, aux_total

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature=0.6, top_p=0.85, top_k=0, repetition_penalty=1.15):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _, _ = self.forward(idx_cond)

            logits = logits[:, -1, :] / max(temperature, 1e-6)

            # --- repetition penalty (simple, effective) ---
            if repetition_penalty is not None and repetition_penalty > 1.0:
                for b in range(idx.size(0)):
                    prev = idx[b].tolist()
                    # penalize tokens already generated
                    logits[b, prev] /= repetition_penalty

            # --- top-p / nucleus ---
            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)

                mask = cum > top_p
                mask[..., 0] = False  # keep at least 1 token
                sorted_logits[mask] = -float("inf")
                logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx


