import os
import torch
import pandas as pd

from typing import Dict, Any
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase

class GPReviewDataset(Dataset):
    """
    A PyTorch Dataset class for processing Google Play Store reviews.
    """
    def __init__(self, reviews: pd.Series, targets: torch.Tensor, tokenizer: PreTrainedTokenizerBase, max_len: int):
        """
        Initializes the dataset.

        Args:
            reviews (pd.Series): Pandas Series containing review texts.
            targets (torch.Tensor): Tensor containing sentiment labels.
            tokenizer (PreTrainedTokenizerBase): Hugging Face tokenizer.
            max_len (int): Maximum token length for tokenization.
        """
        self.reviews = reviews.tolist()
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.reviews)
    
    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        Retrieves a single data sample and tokenizes it.

        Args:
            item (int): Index of the data sample.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'review_text' (str): Original review text.
                - 'input_ids' (torch.Tensor): Tokenized input IDs.
                - 'attention_mask' (torch.Tensor): Attention mask for BERT.
                - 'targets' (torch.Tensor): Sentiment label.
        """
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "targets": target
        }

def create_data_loader(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_len: int, batch_size: int) -> DataLoader:
    """
    Creates a PyTorch DataLoader for batching review data.

    Args:
        df (pd.DataFrame): DataFrame containing 'content' (review text) and 'label' (sentiment score).
        tokenizer (PreTrainedTokenizerBase): Hugging Face tokenizer for text processing.
        max_len (int): Maximum length for tokenized inputs.
        batch_size (int): Batch size for DataLoader.

    Returns:
        DataLoader: PyTorch DataLoader for efficient batch processing.
    """
    if df.empty:  # ✅ Check if dataset is empty
        return DataLoader([], batch_size=batch_size)  # Return empty DataLoader safely
    
    # Convert labels to a tensor for efficiency
    targets = torch.tensor(df["score"].to_numpy(), dtype=torch.long)

    # Create dataset instance
    dataset = GPReviewDataset(
        reviews=df["content"],
        targets=targets,
        tokenizer=tokenizer,
        max_len=max_len
    )

    # Create DataLoader with optimized settings
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=os.cpu_count() - 1,  # Enable multiprocessing for better performance
        shuffle=True
    )
