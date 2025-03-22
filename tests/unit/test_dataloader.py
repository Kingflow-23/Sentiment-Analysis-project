import pytest
import torch
import pandas as pd

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from config import *
from dataset_config import FAKE_DATASET
from src.dataloader import GPReviewDataset, create_data_loader

# Load tokenizer (use the same one as in training)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


@pytest.fixture
def sample_dataframe():
    """Fixture to create a sample dataset for testing."""
    return pd.DataFrame(FAKE_DATASET)


@pytest.fixture
def dataset(sample_dataframe):
    """Fixture to create a dataset instance."""
    return GPReviewDataset(
        reviews=sample_dataframe["content"],
        targets=torch.tensor(sample_dataframe["score"].to_numpy(), dtype=torch.long),
        tokenizer=tokenizer,
        max_len=128,
    )


@pytest.fixture
def data_loader(sample_dataframe):
    """Fixture to create a DataLoader instance."""
    return create_data_loader(sample_dataframe, tokenizer, max_len=128, batch_size=2)


# -------------------------------
# ✅ Dataset Unit Tests
# -------------------------------
def test_dataset_length(dataset):
    """Test if dataset length matches input data."""
    assert len(dataset) == 75, "Dataset length does not match input data."


def test_dataset_getitem(dataset):
    """Test if dataset returns correctly formatted tokenized data."""
    sample = dataset[0]

    assert isinstance(sample, dict), "Dataset __getitem__ must return a dictionary."
    assert "input_ids" in sample and isinstance(
        sample["input_ids"], torch.Tensor
    ), "Missing or invalid 'input_ids'."
    assert "attention_mask" in sample and isinstance(
        sample["attention_mask"], torch.Tensor
    ), "Missing or invalid 'attention_mask'."
    assert "targets" in sample and isinstance(
        sample["targets"], torch.Tensor
    ), "Missing or invalid 'targets'."

    # Ensure tensor sizes are correct
    assert sample["input_ids"].shape == (128,), "Incorrect shape for input_ids."
    assert sample["attention_mask"].shape == (
        128,
    ), "Incorrect shape for attention_mask."


def test_dataset_targets_type(dataset):
    """Test if targets are stored as correct tensor type."""
    assert dataset.targets.dtype == torch.long, "Targets should be dtype=torch.long"


# -------------------------------
# ✅ DataLoader Unit Tests
# -------------------------------
def test_data_loader_batching(data_loader):
    """Test if DataLoader batches data correctly."""
    batch = next(iter(data_loader))

    assert isinstance(batch, dict), "DataLoader batch must return a dictionary."
    assert "input_ids" in batch, "Batch is missing 'input_ids'."
    assert "attention_mask" in batch, "Batch is missing 'attention_mask'."
    assert "targets" in batch, "Batch is missing 'targets'."

    # Ensure batch size is correct
    assert batch["input_ids"].shape[0] == 2, "Batch size mismatch."


def test_data_loader_empty():
    """Test if DataLoader handles empty datasets properly."""
    empty_df = pd.DataFrame({"content": [], "score": []})
    empty_loader = create_data_loader(empty_df, tokenizer, max_len=128, batch_size=2)

    assert len(list(empty_loader)) == 0, "Empty DataLoader should have zero batches."


def test_long_review_processing():
    """Test if DataLoader correctly truncates long reviews."""
    long_text = "This is a very long review " * 100  # Simulating an overly long review
    df = pd.DataFrame({"content": [long_text], "score": [3]})
    loader = create_data_loader(df, tokenizer, max_len=128, batch_size=1)

    batch = next(iter(loader))

    assert batch["input_ids"].shape == (
        1,
        128,
    ), "Long review was not truncated correctly."
    assert batch["attention_mask"].shape == (1, 128), "Attention mask shape incorrect."


def test_tokenizer_padding():
    """Test if DataLoader correctly pads short reviews."""
    short_text = "Okay"
    df = pd.DataFrame({"content": [short_text], "score": [3]})
    loader = create_data_loader(df, tokenizer, max_len=128, batch_size=1)

    batch = next(iter(loader))

    assert batch["input_ids"].shape == (
        1,
        128,
    ), "Short review was not padded correctly."
    assert batch["attention_mask"].shape == (1, 128), "Attention mask shape incorrect."
