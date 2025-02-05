#! /usr/bin/env python
"""
Inference script for SmollmV2 model
Author: Shilpaj Bhalerao
Date: 2025-01-25
"""
# Third-Party Imports
import torch
from transformers import GPT2Tokenizer

# Local Imports
from deepseek_lightning import LitDeepSeekLM
from deepseek_config import DeepSeekConfig, DataConfig


def load_model(checkpoint_path):
    """
    Load the trained model from checkpoint.
    """
    model = LitDeepSeekLM.load_from_checkpoint(
        checkpoint_path,
        model_config=DeepSeekConfig,
        strict=False
    )
    model.eval()
    return model


def generate_text(model, prompt, max_new_tokens=100, temperature=0.8, top_p=0.9):
    """
    Generate text using the loaded model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize tokenizer the same way as in CosmopediaDataModule
    tokenizer = GPT2Tokenizer.from_pretrained(DataConfig.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate tokens one at a time
    for _ in range(max_new_tokens):
        # Get the model's predictions
        with torch.no_grad():
            logits, _ = model.model(input_ids)
        
        # Get the next token probabilities
        logits = logits[:, -1, :] / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Sample from the distribution
        if top_p > 0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_keep = cumsum_probs <= top_p
            sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
            sorted_indices_to_keep[..., 0] = 1
            indices_to_keep = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, sorted_indices, sorted_indices_to_keep)
            probs = torch.where(indices_to_keep, probs, torch.zeros_like(probs))
            probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to input_ids
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # Stop if we generate an EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


def main():
    # Path to your checkpoint
    checkpoint_path = "./checkpoints/last.ckpt"
    
    # Load the model
    model = load_model(checkpoint_path)
    print("Model loaded successfully!")
    
    # Example prompts for generation
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In the distant galaxy"
    ]
    
    # Generate text for each prompt
    for prompt in prompts:
        print("\nPrompt:", prompt)
        generated = generate_text(prompt=prompt, model=model)
        print("Generated:", generated)
        print("-" * 50)

if __name__ == "__main__":
    main() 