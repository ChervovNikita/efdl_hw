"""
Attention with RoPE
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TransformerConfig
from flash_attn import flash_attn_qkvpacked_func
from flash_attn.ops.triton.rotary import apply_rotary

from functools import partial
from typing import Optional, Tuple, Union

import torch
from torch import Tensor


class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, Tensor] = 0,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        out = apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cu_seqlens)  # Can't save int with save_for_backward
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        dx = apply_rotary(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=ctx.inplace,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x,
    cos,
    sin,
    interleaved=False,
    inplace=False,
    seqlen_offsets: Union[int, Tensor] = 0,
    cu_seqlens: Optional[Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmb.apply(
        x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
    )


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    """
    # TODO: Use fused RoPE from flash_attn library instead

    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        self._build_cache(max_seq_len)
    
    # def _build_cache(self, seq_len: int):
    #     """Build sin/cos cache up to seq_len."""
    #     positions = torch.arange(seq_len, device=self.inv_freq.device)
    #     freqs = torch.outer(positions, self.inv_freq)
    #     # emb = torch.cat([freqs, freqs], dim=-1)

    #     self.register_buffer('cos', freqs.cos(), persistent=False)
    #     self.register_buffer('sin', freqs.sin(), persistent=False)

    #     # self.register_buffer('cos', emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
    #     # self.register_buffer('sin', emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)
    
    # def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Apply rotary positional embedding to q and k.
        
    #     Args:
    #         q: (B, num_heads, S, head_dim)
    #         k: (B, num_heads, S, head_dim)
    #         seq_len: sequence length (must be <= max_seq_len)
            
    #     Returns:
    #         q_rotated, k_rotated with same shapes
    #     """
        
    #     q = apply_rotary_emb(q, self.cos, self.sin)
    #     k = apply_rotary_emb(k, self.cos, self.sin)
    #     return q, k

    def _build_cache(self, seq_len: int):
        """Build sin/cos cache up to seq_len."""
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer('cos', emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer('sin', emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embedding to q and k.
        
        Args:
            q: (B, num_heads, S, head_dim)
            k: (B, num_heads, S, head_dim)
            seq_len: sequence length (must be <= max_seq_len)
            
        Returns:
            q_rotated, k_rotated with same shapes
        """
        assert seq_len <= self.max_seq_len, \
            f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})"
        
        cos = self.cos[:, :, :seq_len, :]
        sin = self.sin[:, :, :seq_len, :]

        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to tensor x."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with vanilla implementation and RoPE.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads

        # TODO: Replace with fused QKV projection
        self.qkv_proj = nn.Linear(config.hidden_dim, config.hidden_dim * 3, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.rope = RotaryPositionalEmbedding(head_dim=self.head_dim, max_seq_len=config.max_seq_len, theta=config.rope_theta)

        # self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, H = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)

        q, k = self.rope(qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), S)
        qkv[:, :, 0] = q.transpose(1, 2)
        qkv[:, :, 1] = k.transpose(1, 2)

        # TODO: Replace vanilla attention with Flash Attention

        scale = 1.0 / math.sqrt(self.head_dim)
        
        out = flash_attn_qkvpacked_func(qkv, causal=True, dropout_p=self.config.dropout if self.training else 0)
        out = out.contiguous().view(B, S, H)
        out = self.out_proj(out)

        return out
