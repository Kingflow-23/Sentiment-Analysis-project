import pytest
import pandas as pd

from src.data_processing import clean_text, tokenize_texts, preprocess_data
from config import FAKE_DATASET

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
    pass

def test_preprocess_data(sample_data: pd.DataFrame):
    """Test dataset preprocessing (cleaning and splitting)."""
    pass
