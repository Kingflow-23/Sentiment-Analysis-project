import re
import pandas as pd

from config import *

from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# Load tokenizer for BERT
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def clean_text(text: str) -> str:
    """
    Cleans text by removing special characters, HTML tags, and extra spaces.
    Converts text to lowercase.

    Args:
        text (str): Raw text input.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""  # Handle missing or non-string values

    text = text.lower()  # Convert to lowercase
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces

    return text


def tokenize_texts(texts: list, max_length: int = 128) -> dict:
    """
    Tokenizes text using the BERT tokenizer.

    Args:
        texts (list): List of text strings.
        max_length (int): Maximum token length.

    Returns:
        dict: Tokenized representation of input texts.
    """
    tokenized = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )

    # Convert PyTorch tensors to lists so they work with Pandas
    return {
        "input_ids": tokenized["input_ids"].tolist(),
        "attention_mask": tokenized["attention_mask"].tolist(),
    }


def preprocess_data(
    df: pd.DataFrame, test_size: float = 0.2, max_length: int = 128
) -> tuple:
    """
    Cleans, tokenizes, and splits the dataset into training and validation sets.

    Args:
        df (pd.DataFrame): Input dataset with 'content', 'score' and 'label' columns.
        test_size (float): Proportion of data for validation.
        max_length (int): Maximum token length.

    Returns:
        tuple: (train_df, val_df)
    """
    # Ensure content column is cleaned
    df["content"] = df["content"].apply(clean_text)

    # Tokenize the cleaned text
    tokenized_data = tokenize_texts(df["content"].tolist(), max_length)

    # Add tokenized columns back to the dataframe
    df["input_ids"] = tokenized_data["input_ids"]
    df["attention_mask"] = tokenized_data["attention_mask"]

    # Check if stratification is possible
    if df["score"].value_counts().min() < 2:
        stratify_param = None
    else:
        stratify_param = df["score"]

    # Split data into training and validation sets
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=stratify_param, random_state=42
    )

    return train_df, val_df
