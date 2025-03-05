import pytest
import pandas as pd

from config import SENTIMENT_MAPPING
from dataset_config import FAKE_DATASET
from src.data_processing import clean_text, tokenize_texts, preprocess_data


# Fixture for FAKE_DATASET (non-stratified)
@pytest.fixture
def sample_data_fake():
    """Creates a sample dataset for testing."""
    return pd.DataFrame(FAKE_DATASET)


# Fixture for a dataset with a single class for stratify=None testing
@pytest.fixture
def sample_data_single_label():
    """Creates a dataset where stratification is not possible (all labels are the same)."""
    return pd.DataFrame(
        {
            "content": [
                "I love this",
                "I hate this",
                "Neutral feeling",
                "So-so experience",
            ],
            "score": [
                5,
                1,
                3,
                3,
            ],  # No that much data, so stratify should be None.
            "label": [
                "positive",
                "negative",
                "neutral",
                "neutral",
            ],  # You can ignore this column
        }
    )


def test_clean_text():
    """Test text cleaning function."""
    assert clean_text("I love this app! 😍") == "i love this app"
    assert clean_text("<p>This is terrible!</p>") == "this is terrible"
    assert clean_text("Check this out: www.example.com") == "check this out"
    assert clean_text("No special characters!!!") == "no special characters"
    assert clean_text(None) == ""


def test_tokenize_texts():
    """Test tokenization output structure."""
    texts = ["Hello world!", "This is a test sentence."]
    tokens = tokenize_texts(texts, max_length=10)

    assert "input_ids" in tokens
    assert "attention_mask" in tokens
    assert len(tokens["input_ids"][0]) > 0  # Ensure tokens are created


# Test for non-stratified Fake Dataset
def test_preprocess_data_fake(sample_data_fake: pd.DataFrame):
    """Test dataset preprocessing (cleaning and splitting) with FAKE_DATASET (non-stratified)."""
    train_df, val_df = preprocess_data(sample_data_fake, test_size=0.2)

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)

    # Ensure 'content' and 'score' columns are present
    assert "content" in train_df.columns
    assert "score" in train_df.columns
    assert "content" in val_df.columns
    assert "score" in val_df.columns

    # Ensure tokenization was done (check for 'input_ids' and 'attention_mask' columns)
    assert "input_ids" in train_df.columns
    assert "attention_mask" in train_df.columns
    assert "input_ids" in val_df.columns
    assert "attention_mask" in val_df.columns

    # Ensure the correct number of samples in training and validation sets
    assert len(train_df) > len(val_df)  # Training set should be larger
    assert len(train_df) + len(val_df) == len(sample_data_fake)  # No samples lost

    # Verify that the labels are of the correct type (float/int)
    assert all(train_df["score"].apply(lambda x: isinstance(x, (int, float))))
    assert all(val_df["score"].apply(lambda x: isinstance(x, (int, float))))


# Test for stratified=None scenario (single label in the dataset)
def test_preprocess_data_stratify_none(sample_data_single_label: pd.DataFrame):
    """Test dataset preprocessing when stratify=None (no stratification)."""
    train_df, val_df = preprocess_data(sample_data_single_label, test_size=0.2)

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)

    # Ensure 'content' and 'score' columns are present
    assert "content" in train_df.columns
    assert "score" in train_df.columns
    assert "content" in val_df.columns
    assert "score" in val_df.columns

    # Ensure tokenization was done (check for 'input_ids' and 'attention_mask' columns)
    assert "input_ids" in train_df.columns
    assert "attention_mask" in train_df.columns
    assert "input_ids" in val_df.columns
    assert "attention_mask" in val_df.columns

    # Ensure the correct number of samples in training and validation sets
    assert len(train_df) > len(val_df)  # Training set should be larger
    assert len(train_df) + len(val_df) == len(
        sample_data_single_label
    )  # No samples lost

    # Verify that the labels are of the correct type (float/int)
    assert all(train_df["score"].apply(lambda x: isinstance(x, (int, float))))
    assert all(val_df["score"].apply(lambda x: isinstance(x, (int, float))))


def test_label_distribution(sample_data_fake: pd.DataFrame):
    """Test that label distribution is preserved after stratified splitting."""
    train_df, val_df = preprocess_data(sample_data_fake, test_size=0.2)

    # Get unique labels from the original dataset, training set, and validation set
    original_labels = set(sample_data_fake["score"].unique())
    train_labels = set(train_df["score"].unique())
    val_labels = set(val_df["score"].unique())

    # ✅ Ensure no label is completely lost in the train or val set
    assert original_labels.issubset(
        train_labels.union(val_labels)
    ), "Some labels are missing after split"
    assert len(val_labels) > 0, "Validation set is empty"


def test_sentiment_mapping(sample_data_fake: pd.DataFrame):
    """Test sentiment mapping after processing."""
    train_df, val_df = preprocess_data(sample_data_fake, test_size=0.2)

    # Check that labels are mapped correctly using SENTIMENT_MAPPING
    for label in train_df["score"]:
        assert SENTIMENT_MAPPING[label] in [
            "Really Negative",
            "Negative",
            "Neutral",
            "Positive",
            "Really Positive",
        ]

    for label in val_df["score"]:
        assert SENTIMENT_MAPPING[label] in [
            "Really Negative",
            "Negative",
            "Neutral",
            "Positive",
            "Really Positive",
        ]
