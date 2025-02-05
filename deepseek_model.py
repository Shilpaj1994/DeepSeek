#! /usr/bin/env python
"""
SmollmV2 model implementation
Author: Shilpaj Bhalerao
Date: 2025-01-19
"""
# Third-Party Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Local Imports
from deepseek_config import DeepSeekConfig, LatentAttentionConfig


class LatentAttention(nn.Module):
    """Modified RoPE implementation for latent attention"""
    def __init__(self, head_dim, kv_dim, config: LatentAttentionConfig):
        super().__init__()
        self.head_dim = head_dim
        self.kv_dim = kv_dim
        self.base = config.base
        self.scaling_factor = config.scaling_factor
        
        # Rotary embeddings with scaling
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, kv_dim, 2).float() / kv_dim))
        self.register_buffer('inv_freq', self.inv_freq)
        
    def _apply_rope(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    def forward(self, q, k):
        seq_len = q.shape[2]
        cos, sin = self._apply_rope(k, seq_len)
        
        # Apply scaling to rotary embeddings
        cos = cos * self.scaling_factor
        sin = sin * self.scaling_factor

        # Rotary transformations
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.compression_ratio = config.compression_ratio
        self.latent_dim = self.hidden_size // self.compression_ratio

        # Projection layers
        self.kv_proj_d = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        self.q_proj_d = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        self.k_proj_u = nn.Linear(self.latent_dim, self.head_dim, bias=False)
        self.q_proj_u = nn.Linear(self.latent_dim, self.head_dim, bias=False)
        self.v_proj_u = nn.Linear(self.latent_dim, self.hidden_size, bias=False)
        
        # Rotary components
        self.rope_k = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.rope_q = nn.Linear(self.latent_dim, self.head_dim, bias=False)
        self.rotary_emb = LatentAttention(self.head_dim, self.head_dim, config.latent_attention_config)

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, x):
        B, T, _ = x.size()
        
        # Project to latent space
        kv_latent = self.kv_proj_d(x)
        q_latent = self.q_proj_d(x)
        
        # Decompress projections
        k = self.k_proj_u(kv_latent).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_proj_u(q_latent).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj_u(kv_latent).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Rotary embeddings
        q_rot = self.rope_q(q_latent).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_rot = self.rope_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary_emb(q_rot, k_rot)

        # Attention with compressed KV
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        return self.o_proj(y)


class DeepSeekExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekMoE(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_shared = config.num_shared_experts
        self.top_k = config.top_k
        self.intermediate_size = int(config.hidden_size * config.mlp_ratio)

        # Experts
        self.shared_experts = nn.ModuleList([
            DeepSeekExpert(self.hidden_size, self.intermediate_size) 
            for _ in range(self.num_shared)
        ])
        self.routed_experts = nn.ModuleList([
            DeepSeekExpert(self.hidden_size, self.intermediate_size)
            for _ in range(self.num_experts - self.num_shared)
        ])
        
        # Routing
        self.router = nn.Linear(self.hidden_size, self.num_experts - self.num_shared, bias=False)
        self.register_parameter('routing_bias', nn.Parameter(torch.zeros(self.num_experts - self.num_shared)))

        self.expert_load = None  # Track expert utilization

    def forward(self, x):
        # Shared experts
        shared_out = sum(expert(x) for expert in self.shared_experts) / self.num_shared
        
        # Routing
        routing_logits = self.router(x) + self.routing_bias
        routing_probs = torch.sigmoid(routing_logits)
        scores, indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Calculate expert load (EXACT reference implementation)
        self.expert_load = torch.zeros(
            len(self.routed_experts), 
            device=x.device
        )
        batch_size, seq_len = x.size(0), x.size(1)
        
        for k in range(self.top_k):
            current_indices = indices[..., k]
            for i in range(len(self.routed_experts)):
                self.expert_load[i] += (current_indices == i).sum()

        # Normalize by total possible assignments
        self.expert_load /= (batch_size * seq_len * self.top_k)

        # Expert processing
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_mask = indices[..., k]
            for expert_idx in range(len(self.routed_experts)):
                mask = (expert_mask == expert_idx)
                if mask.any():
                    expert_out = self.routed_experts[expert_idx](x[mask])
                    out[mask] += expert_out * scores[mask][..., None]
        
        return shared_out + out

    def update_bias_terms(self, expert_load):
        """EXACT reference implementation of bias update"""
        target_load = 1.0 / len(self.routed_experts)
        load_diff = expert_load - target_load
        self.routing_bias.data -= 0.1 * load_diff


class MOEFeedForward(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.moe = DeepSeekMoE(config)
        
    def forward(self, x):
        return self.moe(x)


class DeepSeekBlock(nn.Module):
    """
    Transformer block
    """
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, bias=False)
        self.attn = MultiHeadLatentAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, bias=False)
        self.mlp = MOEFeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class DeepSeekLM(nn.Module):
    """
    DeepSeekLM model
    """
    def __init__(self, config=DeepSeekConfig()):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([DeepSeekBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=False),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        # weight initialization
        self.apply(self._init_weights)
        
        # Enable memory efficient attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            self.use_flash_attention = True
        else:
            self.use_flash_attention = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.04)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
