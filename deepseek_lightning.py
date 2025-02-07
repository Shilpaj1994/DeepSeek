#!/usr/bin/env python
"""
Lightning module for SmollmV2 model training
"""

# Standard Library Imports
import os
from typing import Tuple

# Third-Party Imports
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import time
import numpy as np
from contextlib import nullcontext
import torch.nn.functional as F

# Local Imports
from deepseek_config import (DeepSeekConfig, OptimizerConfig, CheckpointConfig, 
                   LoggingConfig, TrainerConfig)
from deepseek_model import DeepSeekLM, DeepSeekMoE
from cosmopedia_datamodule import CosmopediaDataModule

class LitDeepSeekLM(pl.LightningModule):
    """
    Lightning module for DeepSeekLM model training
    """
    def __init__(
        self, 
        learning_rate=OptimizerConfig.learning_rate, 
        weight_decay=OptimizerConfig.weight_decay, 
        total_epochs=None, 
        total_steps=None,
        interupt_steps=DeepSeekConfig.max_steps,
        compile_model=True
    ):
        """
        Constructor
        :param learning_rate: Learning rate for the optimizer
        :param weight_decay: Weight decay for the optimizer
        :param total_epochs: Total number of epochs (optional)
        :param total_steps: Total number of steps (optional)
        :param compile_model: Whether to compile the model for faster training
        Note: Provide either total_epochs or total_steps, not both
        """
        super().__init__()
        self.save_hyperparameters()
        
        if total_epochs is None and total_steps is None:
            raise ValueError("Must provide either total_epochs or total_steps")
        if total_epochs is not None and total_steps is not None:
            raise ValueError("Provide either total_epochs or total_steps, not both")
        
        # Set seeds from config
        torch.manual_seed(DeepSeekConfig.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(DeepSeekConfig.seed)
        
        # Initialize the model
        self.model = DeepSeekLM(DeepSeekConfig())
        
        # Compile the model if requested and supported
        if compile_model and hasattr(torch, 'compile'):
            print("Compiling model for faster training...")
            self.model = torch.compile(self.model)
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Print model size info
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)  # Size in MB
        print(f"Model size in memory: {model_size:.2f} MB")
        
        # OneCycleLR parameters from OptimizerConfig
        self.max_lr = OptimizerConfig.max_lr
        self.div_factor = OptimizerConfig.div_factor
        self.final_div_factor = OptimizerConfig.final_div_factor
        self.pct_start = OptimizerConfig.pct_start
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        
        # Add performance monitoring attributes
        self.iter_num = 0
        self.iter_time = 0.0
        self.tokens_processed = 0
        self.interupt_steps = interupt_steps
        
    def on_load_checkpoint(self, checkpoint):
        """Restore iter_num when loading from checkpoint"""
        if 'iter_num' in checkpoint:
            self.iter_num = checkpoint['iter_num']
    
    def on_save_checkpoint(self, checkpoint):
        """Save iter_num in checkpoint"""
        checkpoint['iter_num'] = self.iter_num
        
    def forward(self, x, targets=None):
        """
        Method to forward the input through the model
        """
        return self.model(x, targets)
    
    def training_step(self, batch, batch_idx):
        """
        Method to perform a training step with performance monitoring
        """
        try:
            # Stop training at max steps from config
            if self.iter_num >= self.interupt_steps:
                self.trainer.should_stop = True
                return None

            # Start timing
            t0 = time.time()
            
            # Process batch
            input_ids = batch['input_ids']
            labels = batch['labels']
            attention_mask = batch['attention_mask']
            
            # Clear cache before forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Forward pass
            logits, loss = self(input_ids, targets=labels)
            
            # Update MoE routing biases
            self._update_moe_biases()
            
            # Calculate tokens processed
            tokens_per_iter = np.prod(input_ids.shape)
            self.tokens_processed += tokens_per_iter
            
            # Ensure CUDA synchronization after forward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Calculate iteration time
            dt = time.time() - t0
            self.iter_time += dt
            
            # Log metrics
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
            
            # Generate sample prediction
            if self.iter_num % LoggingConfig.generate_every == 0:
                # Get a sample input from the batch
                context_length = DeepSeekConfig.context_length  # Number of tokens to use as context
                sample_input = input_ids[0:1, :context_length]
                
                # Generate prediction
                self.model.eval()
                with torch.no_grad():
                    max_new_tokens = DeepSeekConfig.max_new_tokens
                    temperature = DeepSeekConfig.temperature
                    top_k = DeepSeekConfig.top_k
                    
                    for _ in range(max_new_tokens):
                        # Get model predictions
                        logits, _ = self(sample_input)
                        logits = logits[:, -1, :] / temperature
                        
                        # Apply top-k sampling
                        if top_k is not None:
                            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                            logits[logits < v[:, [-1]]] = -float('Inf')
                        
                        probs = F.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        sample_input = torch.cat([sample_input, next_token], dim=1)
                    
                    # Convert tokens to text using the tokenizer from datamodule
                    try:
                        input_text = self.trainer.datamodule.tokenizer.decode(sample_input[0, :10].tolist())
                        generated_text = self.trainer.datamodule.tokenizer.decode(sample_input[0, 10:].tolist())
                        print(f"\nStep {self.iter_num} - Sample Generation:")
                        print(f"Input: {input_text}")
                        print(f"Generated: {generated_text}\n")
                    except Exception as e:
                        print(f"Error decoding text: {str(e)}")
                
                self.model.train()  # Set back to training mode
            
            # Log performance metrics
            if self.iter_num % LoggingConfig.log_every == 0:
                tokens_per_sec = self.tokens_processed / self.iter_time if self.iter_time > 0 else 0
                
                self.log('tokens_per_sec', tokens_per_sec, on_step=True)
                self.log('iter_time_ms', dt * 1000, on_step=True)
                
                print(f"\nstep {self.iter_num} | loss: {loss.item():.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
                
                if torch.cuda.is_available():
                    self.log('gpu_memory', torch.cuda.memory_allocated() / 1e9, on_step=True)
                    self.log('gpu_memory_reserved', torch.cuda.memory_reserved() / 1e9, on_step=True)
                    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.memory_reserved() / 1e9:.2f}GB")
                
                # Clear GPU cache periodically if enabled
                if DeepSeekConfig.clear_cache_every > 0 and self.iter_num % DeepSeekConfig.clear_cache_every == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                self.tokens_processed = 0
                self.iter_time = 0.0
            
            self.iter_num += 1
            return loss
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"WARNING: out of memory - {str(e)}")
                return None
            raise e
    
    def validation_step(self, batch, batch_idx):
        """
        Method to perform a validation step
        """
        # Start timing for validation
        t0 = time.time()
        
        # Ensure CUDA synchronization for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # Process batch - updated for Cosmopedia format
        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        
        # Forward pass
        logits, loss = self(input_ids, targets=labels)
        
        # Ensure CUDA synchronization after forward pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # Calculate validation time
        dt = time.time() - t0
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        if batch_idx == 0:  # Only print for first batch
            print(f"\nValidation - loss: {loss.item():.4f} | dt: {dt*1000:.2f}ms")
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.memory_reserved() / 1e9:.2f}GB")
        
        return loss
    
    def configure_optimizers(self):
        """
        Method to configure the optimizer and scheduler
        """
        # Use Adam 8-bit if available
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        except ImportError:
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        
        # Calculate total steps
        if self.total_steps is None:
            total_steps = len(self.trainer.datamodule.train_dataloader()) * self.total_epochs
        else:
            total_steps = self.total_steps
        
        scheduler = {
            'scheduler': optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.max_lr,
                total_steps=total_steps,
                pct_start=self.pct_start,
                div_factor=self.div_factor,
                final_div_factor=self.final_div_factor,
                three_phase=OptimizerConfig.three_phase,
                anneal_strategy=OptimizerConfig.anneal_strategy
            ),
            'interval': 'step'
        }
        
        return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        """
        Called at the end of training epoch
        """
        # Reset performance counters at epoch end
        self.tokens_processed = 0
        self.iter_time = 0.0

    def _update_moe_biases(self):
        """Update MoE routing biases based on expert utilization"""
        for module in self.model.modules():
            if isinstance(module, DeepSeekMoE) and module.expert_load is not None:
                module.update_bias_terms(module.expert_load)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Matches reference update pattern"""
        # Update every 100 steps to prevent over-adjustment
        if self.global_step % 100 == 0:
            with torch.no_grad():
                for module in self.model.modules():
                    if isinstance(module, DeepSeekMoE):
                        if module.expert_load is not None:
                            # Use synchronized expert load
                            expert_load = module.expert_load.detach()
                            # Apply reference update rule
                            module.update_bias_terms(expert_load)
                            # Reset for next batch
                            module.expert_load = None

def plot_learning_rate(log_dir):
    """
    Plot learning rate from TensorBoard logs
    """
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "events.out.tfevents" in file:
                event_files.append(os.path.join(root, file))
    
    lr_data = []
    steps = []
    
    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(
            event_file,
            size_guidance={'scalars': 0}
        )
        ea.Reload()
        
        if 'lr' in ea.Tags()['scalars']:
            events = ea.Scalars('lr')
            for event in events:
                lr_data.append(event.value)
                steps.append(event.step)
    
    if lr_data:
        plt.figure(figsize=(10, 6))
        plt.plot(steps, lr_data, '-', linewidth=2)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.margins(x=0.02)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.savefig('learning_rate_schedule.png', dpi=300, bbox_inches='tight')
        plt.close()

def train_model(epochs=None, steps=None, ckpt_path=None, interupt_steps=DeepSeekConfig.max_steps):
    """
    Train the model for specified number of epochs or steps
    :param epochs: Number of epochs to train (optional)
    :param steps: Number of steps to train (optional)
    :param ckpt_path: Path to checkpoint for resuming training
    :param interupt_steps: Number of steps after which to interrupt training
    Note: Provide either epochs or steps, not both
    """
    # Set compilation mode for PyTorch 2.0+
    if hasattr(torch, 'compile'):
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.verbose = False
    
    torch.set_float32_matmul_precision('high')
    
    # Initialize data module with reduced workers and batch size
    data_module = CosmopediaDataModule(
        batch_size=DeepSeekConfig.batch_size,
        num_workers=DeepSeekConfig.num_workers,
        shuffle_buffer_size=DeepSeekConfig.shuffle_buffer_size,
        max_length=DeepSeekConfig.block_size
    )
    
    # Initialize model
    model = LitDeepSeekLM(total_epochs=epochs, total_steps=steps, interupt_steps=interupt_steps)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='deepseeklm-{step:05d}-{val_loss:.2f}',
        save_top_k=CheckpointConfig.save_top_k,
        monitor=CheckpointConfig.monitor,
        mode=CheckpointConfig.mode,
        save_last=CheckpointConfig.save_last,
        every_n_train_steps=CheckpointConfig.checkpoint_every,
        save_on_train_epoch_end=CheckpointConfig.save_on_train_epoch_end
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup logger
    logger = TensorBoardLogger("lightning_logs", name="deepseeklm", log_graph=True)
    
    # Initialize trainer with mixed precision training
    trainer_kwargs = {
        'accelerator': TrainerConfig.accelerator,
        'devices': TrainerConfig.devices,
        'callbacks': [checkpoint_callback, lr_monitor],
        'logger': logger,
        'precision': '16-mixed',  # Use mixed precision training
        'log_every_n_steps': TrainerConfig.log_every_n_steps,
        'strategy': TrainerConfig.strategy,
        'deterministic': TrainerConfig.deterministic,
        'benchmark': TrainerConfig.benchmark,
        'enable_progress_bar': TrainerConfig.enable_progress_bar,
        'enable_model_summary': TrainerConfig.enable_model_summary,
        'profiler': TrainerConfig.profiler,
        'gradient_clip_val': TrainerConfig.gradient_clip_val,
        'accumulate_grad_batches': TrainerConfig.accumulate_grad_batches,
        'val_check_interval': TrainerConfig.val_check_interval,
        'check_val_every_n_epoch': TrainerConfig.check_val_every_n_epoch
    }
    
    # Add either max_epochs or max_steps
    if epochs is not None:
        trainer_kwargs['max_epochs'] = epochs
    else:
        trainer_kwargs['max_steps'] = steps
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train with performance monitoring
    print("\nStarting training with performance monitoring...")
    print("Format: step | loss | iteration time | tokens per second | GPU memory\n")
    
    # Enable garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        trainer.fit(model, data_module, ckpt_path=ckpt_path)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        trainer.save_checkpoint("checkpoints/interrupted_training.ckpt")
        print("Checkpoint saved. Exiting...")
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e
    
    return checkpoint_callback.best_model_path

def get_latest_checkpoint():
    """
    Find the latest checkpoint in the checkpoints directory
    """
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    if not checkpoints:
        return None
    
    latest_checkpoint = max(
        [os.path.join(checkpoint_dir, f) for f in checkpoints],
        key=os.path.getmtime
    )
    return latest_checkpoint

def main(interupt_steps=DeepSeekConfig.max_steps):
    """
    Main function to handle training workflow
    """
    # Ask user for training mode
    mode = input("Train by epochs or steps? (e/s): ").lower()
    
    if mode == 'e':
        total_epochs = int(input("Enter number of epochs: "))
        steps = None
    else:
        steps = int(input("Enter number of steps: "))
        total_epochs = None
    
    try:
        latest_checkpoint = get_latest_checkpoint()
        
        if latest_checkpoint and os.path.exists(latest_checkpoint):
            print(f"\nFound existing checkpoint: {latest_checkpoint}")
            user_input = input("Resume training from checkpoint? (y/n): ").lower()
            
            if user_input == 'y':
                print(f"\nResuming training from checkpoint: {latest_checkpoint}")
                train_model(epochs=total_epochs, steps=steps, ckpt_path=latest_checkpoint, interupt_steps=interupt_steps)
            else:
                print("\nStarting fresh training...")
                best_model_path = train_model(epochs=total_epochs, steps=steps, interupt_steps=interupt_steps)
        else:
            print("\nNo checkpoints found. Starting fresh training...")
            best_model_path = train_model(epochs=total_epochs, steps=steps, interupt_steps=interupt_steps)
        
        print("\nGenerating learning rate plot...")
        plot_learning_rate("lightning_logs")
        print("Learning rate plot saved as 'learning_rate_schedule.png'")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()