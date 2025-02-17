#!/usr/bin/env python3
"""
Configuration class for GPT model
Author: Shilpaj Bhalerao
Date: 2025-01-19
"""
# Standard Library Imports
from dataclasses import dataclass, field


@dataclass
class LatentAttentionConfig:
    """
    Configuration for Latent Attention
    """
    base: int = 10000                # Base for the angle calculations
    scaling_factor: float = 1.0      # Scaling factor for rotary embeddings
    head_dim_fraction: float = 0.3125  # Set to get exactly kv_dim=24 (216 total)
    round_multiple: int = 8          # Round kv_dim to nearest multiple of this number


@dataclass
class DeepSeekConfig:
    """
    Configuration for DeepSeekLM training setup
    """
    # Model configuration
    num_experts: int = 8
    expert_capacity: int = 64
    block_size: int = 2048         # max sequence length 
    vocab_size: int = 49152        # vocabulary size
    n_layer: int = 8               # number of transformer layers
    num_attention_heads: int = 8   # number of attention heads
    hidden_size: int = 768         # embedding dimension
    mlp_ratio: float = 2.67        # Based on MLP implementation (1536/576)
    dropout: float = 0.0           # No dropout used in implementation
    
    # Training configuration
    batch_size: int = 1                # Minimum batch size (from smollv2_lightning.py)
    num_workers: int = 0               # No additional workers to save memory
    shuffle_buffer_size: int = 1000    # Shuffle buffer size for dataset
    max_length: int = 2048             # Sequence length for training
    learning_rate: float = 3e-5        # From LitGPT initialization
    weight_decay: float = 1e-4         # From LitGPT initialization
    
    # Generation configuration
    max_new_tokens: int = 100          # From generation code in training_step
    
    # Training control
    seed: int = 1337
    max_steps: int = 5000
    clear_cache_every: int = 1000  # Clear GPU cache every N steps, 0 to disable
    
    # Generation parameters
    context_length: int = 10      # Number of tokens to use as context
    temperature: float = 1.0      # Sampling temperature
    top_k: int = 2              # Top-k sampling parameter
    
    # New MoE parameters
    num_shared_experts: int = 1
    compression_ratio: int = 8
    
    # Latent attention config
    latent_attention_config: LatentAttentionConfig = field(default_factory=LatentAttentionConfig)


@dataclass
class CheckpointConfig:
    """
    Configuration for checkpointing
    """
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 500  # Save checkpoint every 500 steps
    save_last: bool = True
    save_top_k: int = 1  # Changed from checkpoint_save_top_k
    save_weights_only: bool = True  # Changed from checkpoint_save_weights_only
    monitor: str = "train_loss"  # Monitor training loss for checkpointing
    mode: str = "min"  # Mode for the monitor metric
    save_on_train_epoch_end: bool = False  # Whether to save on training epoch end


@dataclass
class LoggingConfig:
    """
    Configuration for logging
    """
    log_every: int = 50      # Log metrics every 50 steps
    generate_every: int = 500  # Generate sample text every 500 steps
    log_metrics: bool = True
    log_progress_bar: bool = True
    log_model_summary: bool = True


@dataclass
class OptimizerConfig:
    """
    Configuration for optimizer
    """
    optimizer: str = "AdamW"  # Using AdamW optimizer
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    max_lr: float = 3e-4      # max_lr = learning_rate * 10
    div_factor: float = 25.0  # From OneCycleLR config
    final_div_factor: float = 100.0  # From OneCycleLR config
    pct_start: float = 0.2    # From OneCycleLR config
    
    # Additional optimizer settings
    optimizer_kwargs: dict = field(default_factory=lambda: {
        'betas': (0.9, 0.95),  # Default betas for AdamW
        'eps': 1e-8,           # Default epsilon value
    })
    three_phase: bool = False     # Use three-phase learning rate schedule
    anneal_strategy: str = 'linear'  # Learning rate annealing strategy


@dataclass
class DataConfig:
    """
    Configuration for dataset and tokenizer
    """
    # Dataset configuration
    dataset_path: str = "HuggingFaceTB/smollm-corpus"
    dataset_name: str = "cosmopedia-v2"
    
    # Tokenizer configuration
    tokenizer_path: str = "HuggingFaceTB/cosmo2-tokenizer"
    
    # DataLoader configuration
    batch_size: int = 32
    num_workers: int = 4
    shuffle_buffer_size: int = 1000
    max_length: int = 512
    
    # Dataset splits
    validation_split: float = 0.1  # 10% for validation
    pin_memory: bool = True
    streaming: bool = True         # Use streaming mode for dataset


@dataclass
class TrainerConfig:
    """
    Configuration for PyTorch Lightning Trainer
    """
    accelerator: str = 'auto'
    devices: int = 1
    precision: str = '16-mixed'
    log_every_n_steps: int = 10
    strategy: str = 'auto'
    deterministic: bool = False
    benchmark: bool = True
    enable_progress_bar: bool = True
    enable_model_summary: bool = True
    profiler: str = 'simple'
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 2
    val_check_interval: int = 1000  # Run validation every N training steps
    check_val_every_n_epoch: None = None  # Disable epoch-based validation
