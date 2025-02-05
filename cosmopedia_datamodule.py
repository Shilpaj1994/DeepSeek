#!/usr/bin/env python
"""
Data module for Cosmopedia dataset
Author: Shilpaj Bhalerao
Date: 2025-01-20
"""
# Standard Library Imports
from typing import Optional

# Third-Party Imports
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer

# Local Imports
from deepseek_config import DataConfig


class CosmopediaDataModule(pl.LightningDataModule):
    """
    Data module for Cosmopedia dataset
    """
    def __init__(
        self,
        batch_size: int = DataConfig.batch_size,
        num_workers: int = DataConfig.num_workers,
        shuffle_buffer_size: int = DataConfig.shuffle_buffer_size,
        max_length: int = DataConfig.max_length,
    ):
        """
        Constructor
        :param batch_size: Batch size for dataloaders
        :param num_workers: Number of workers for dataloaders
        :param shuffle_buffer_size: Size of buffer for shuffling streaming data
        :param max_length: Maximum sequence length for tokenized text
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_length = max_length
        
        # Dataset path on HuggingFace
        self.dataset_path = DataConfig.dataset_path
        self.dataset_name = DataConfig.dataset_name
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(DataConfig.tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training and validation
        """
        # Load dataset in streaming mode
        self.dataset = load_dataset(
            self.dataset_path,
            self.dataset_name,
            split="train",  # Only train split is available
            streaming=DataConfig.streaming
        )
        
        # Shuffle the streaming dataset
        self.dataset = self.dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        
        # Create train/val split using configured validation split
        val_size = int(DataConfig.validation_split * self.shuffle_buffer_size)
        self.train_dataset = self.dataset.skip(val_size)
        self.val_dataset = self.dataset.take(val_size)

    def collate_fn(self, batch):
        """
        Tokenize and pad the texts in the batch
        """
        texts = [item['text'] for item in batch]
        
        # Tokenize all texts in the batch
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Prepare inputs and labels for language modeling
        input_ids = encodings['input_ids'][:, :-1]
        labels = encodings['input_ids'][:, 1:]
        attention_mask = encodings['attention_mask'][:, :-1]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }

    def train_dataloader(self):
        """
        Return train dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=DataConfig.pin_memory,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        """
        Return validation dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=DataConfig.pin_memory,
            collate_fn=self.collate_fn
        )
