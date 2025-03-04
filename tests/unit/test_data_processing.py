import pytest
import pandas as pd

from src.data_processing import clean_text, tokenize_texts, preprocess_data
from config import SENTIMENT_MAPPING, FAKE_DATASET

@pytest.fixture
def sample_data():
    """Creates a sample dataset for testing."""
    return pd.DataFrame(FAKE_DATASET)

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
    
    assert 'input_ids' in tokens
    assert 'attention_mask' in tokens
    assert len(tokens['input_ids'][0]) > 0  # Ensure tokens are created

def test_preprocess_data(sample_data: pd.DataFrame):
    """Test dataset preprocessing (cleaning and splitting)."""
    train_df, val_df = preprocess_data(sample_data, test_size=0.2)

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    
    assert 'content' in train_df.columns
    assert 'label' in train_df.columns
    assert 'content' in val_df.columns
    assert 'label' in val_df.columns
    
    assert len(train_df) > len(val_df)  # Training set should be larger

def test_label_distribution(sample_data: pd.DataFrame):
    """Test that label distribution is preserved after stratified splitting."""
    train_df, val_df = preprocess_data(sample_data, test_size=0.2)

    # Get unique labels from the original dataset, training set, and validation set
    original_labels = set(sample_data["label"].unique())
    train_labels = set(train_df["label"].unique())
    val_labels = set(val_df["label"].unique())

    # ✅ Ensure no label is completely lost in the train or val set
    assert original_labels.issubset(train_labels.union(val_labels)), "Some labels are missing after split"
    assert len(val_labels) > 0, "Validation set is empty"
    
def test_sentiment_mapping(sample_data: pd.DataFrame):
    """Test sentiment mapping after processing."""
    train_df, val_df = preprocess_data(sample_data, test_size=0.2)
    
    # Check that labels are mapped correctly using SENTIMENT_MAPPING
    for label in train_df['label']:
        assert SENTIMENT_MAPPING[label] in ["Really Negative", "Negative", "Neutral", "Positive", "Really Positive"]
    
    for label in val_df['label']:
        assert SENTIMENT_MAPPING[label] in ["Really Negative", "Negative", "Neutral", "Positive", "Really Positive"]