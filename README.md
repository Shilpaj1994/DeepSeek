# DeepSeek-V3 Architecture

In this project, I have recreated the Deepseek v3 architecture. In one of my previous project, I have created Smollm2 architecture. In this project, I will modify that architecture and add MHLA (MultiHead Latent Attention), Mixture of Experts (MoE) with Loss-less load balancing to recreate DeepSeek v3 architecture.



## MultiHead Latent Attention

![MHLA](https://planetbanatt.net/images/mla/mla.mp4)

Source - https://planetbanatt.net/articles/mla.html

Starting with MultiHeadLatentAttention. Looking at the class, it initializes with a hidden size, number of heads, and compression ratio. The latent_dim is hidden_size divided by compression_ratio, which suggests they're compressing the key and value projections. There are two sets of projections: kv_proj_d and q_proj_d for compression, then k_proj_u, q_proj_u, v_proj_u for decompression. The RoPE (Rotary Positional Embedding) components are applied to parts of the queries and keys.

In the forward method, they project the input x into compressed KV and Q vectors. Then they decompress them into smaller dimensions. The RoPE is applied to specific parts of the queries and keys. They concatenate the projected parts with the RoPE-processed parts, which might be a way to maintain positional information while reducing computation. Then they use scaled dot-product attention as usual. So this attention mechanism is different because it compresses the dimensions before computing attention, which could reduce memory usage and computation cost compared to traditional attention that uses the full hidden size everywhere.



## Mixture of Experts (MoE)

![MoE](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/01_moe_layer.png)

he DeepSeekMoE class has shared experts and routed experts. The router decides which experts to use for each token. The forward function calculates routing scores, selects top-k experts, and combines their outputs. The update_bias_terms method adjusts the routing biases based on expert load, which probably helps balance expert usage.

In traditional MoE, all experts are routed, but here they have a split between shared and routed experts. Shared experts are always used, which might capture common patterns, while routed experts handle specialized cases. The top-k selection is done per token, allowing dynamic routing. The bias update mechanism is interesting—it adjusts based on how much each expert is used compared to a target, which could prevent some experts from being underutilized.



## Traditional vs DeepSeek

Comparing to traditional transformers: The attention here uses compression and decomposed projections, which makes it more efficient. The MoE adds a mixture of shared and specialized experts with adaptive routing, whereas standard transformers use dense feed-forward networks. Together, these changes likely improve efficiency (less computation in attention) and effectiveness (better handling of diverse tokens via MoE).

1. **MultiHeadLatentAttention (Different from Standard Attention):**

   - Key Innovation: Uses compressed "latent" representations to reduce computation
   - How it Works:
     - Compresses Keys/Values using kv_proj_d (down-projection)
     - Compresses Queries using q_proj_d
     - Then decompresses them using _proj_u layers
     - Combines compressed representations with rotary positional embeddings (RoPE)
     - Uses standard scaled dot-product attention on the final representations
   - Why Better? Reduces memory usage and computation by working with compressed representations while maintaining positional awareness through RoPE

2. **Mixture of Experts (DeepSeekMoE):**

   - Key Innovation: Hybrid expert system with shared + routed experts
   - Structure:
     - Shared Experts (2 in this code): Always active, process all inputs
     - Routed Experts (2 in this code): Specialized experts selected per-token
     - Smart Routing: Uses learned biases and sigmoid activation to select experts

   - Key Differences from Traditional MoE:
     - Maintains shared experts as "base knowledge" while routed experts handle specialized patterns
     - Uses bias terms that adapt based on expert utilization (update_bias_terms method)
     - Combines outputs from shared and routed experts instead of using MoE exclusively

3. **Overall Architecture Differences from Traditional Transformers:**

   - Memory Efficiency: Latent attention reduces KV cache size through compression
   - Compute Efficiency: Processes most operations in compressed latent space
   - Expert Specialization: Hybrid expert system allows both general and specialized processing
   - Dynamic Adaptation: Router biases automatically adjust to balance expert usage



---



### 1. Core Architecture Changes

| Component           | Original Implementation | DeepSeek Implementation             |
| ------------------- | ----------------------- | ----------------------------------- |
| Attention Mechanism | Standard Self-Attention | MultiHeadLatentAttention            |
| MLP                 | Dense FFN               | DeepSeekMoE Hybrid Experts          |
| Positional Encoding | Learned Embeddings      | Rotary Positional Embeddings (RoPE) |
| Normalization       | LayerNorm               | LlamaRMSNorm                        |



### 2. Key Implemented Features

- Latent Attention (Memory Efficient):

```python
kv_proj_d = nn.Linear(hidden_size, latent_dim) *# 4:1 compression*
q_proj_d = nn.Linear(hidden_size, latent_dim)
```



- Hybrid MoE System:

```python
num_shared_experts = 2 *# Always active*
num_routed_experts = 6 *# Top-2 selected*
```



- Adaptive Router Biases:

```python
update_rate = 0.1 * (expert_load - 1/num_experts)
routing_bias -= update_rate
```

 

### 3. Critical Code Modifications

File: deepseek_model.py

```python
# New Attention

class MultiHeadLatentAttention(nn.Module):

  def __init__(self, config):
      self.kv_proj_d = nn.Linear(config.hidden_size, config.latent_dim)
      self.rotary_emb = LlamaRotaryEmbedding(...)

    
# MoE System
class DeepSeekMoE(nn.Module):

  def forward(self, x):
    # Shared experts process all tokens*
    shared_out = sum(expert(x) *for* expert *in* self.shared_experts)
    
    # Routed experts process selected tokens*
    scores, indices = torch.topk(router_output, self.top_k)
```

File: deepseek_lightning.py

```python
# Training Loop Integration
def training_step(self, batch, batch_idx):
    # Every 100 steps
    if self.global_step % 100 == 0:
        self._update_moe_biases()

def _update_moe_biases(self):
    for module in self.model.modules():
        if isinstance(module, DeepSeekMoE):
            module.update_bias_terms()
```

File: deepseek_config.py

```python
@dataclass
class DeepSeekConfig:
    # New Parameters
    num_experts: int = 8
    num_shared_experts: int = 2  
    top_k: int = 2
    compression_ratio: int = 4
    latent_dim: int = field(init=False)
    
    def __post_init__(self):
        self.latent_dim = self.hidden_size // self.compression_ratio
```



### 4. Monitoring Additions

```python
# In DeepSeekMoE forward():
self.register_buffer('expert_utilization', expert_counts / total_tokens)

# In LightningModule:
def on_train_batch_end(self, outputs, batch, batch_idx):
    log_data = {
        'moe/avg_utilization': torch.mean(self.expert_utilization),
        'moe/router_bias': self.routing_bias.mean()
    }
    self.log_dict(log_data)
```



### 5. Performance Critical Paths

- KV Compression: Reduces attention memory usage by 75%
- Expert Parallelism: Routed experts process only selected tokens
- Bias Warmup: Router biases initialized at 0, adaptively updated



### 6. Architecture Summary

```bash
Input
│
├─ Rotary Position Embeddings
│
├─ Latent Attention (4:1 KV compression)
│  └─ RoPE-enhanced Q/K projections
│
├─ Hybrid MoE FFN
│  ├─ 2 Shared Experts (process all tokens)
│  └─ 6 Routed Experts (top-2 selection)
│     └─ Adaptive Load Balancing
│
└─ RMSNorm Residual Connections
```

This implementation captures DeepSeek's key innovations:

1. Memory-efficient attention through latent projections

2. Stable MoE training via shared experts + adaptive routing

3. Hardware-aware optimizations (compression, expert parallelism)

4. Dynamic load balancing of specialized experts



---



## Training Details

