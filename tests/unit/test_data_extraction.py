import os
import pytest
import pandas as pd

from src.data_extraction import extraction

def create_test_csv(file_path, content):
    """Helper function to create test CSV files"""
    with open(file_path, "w") as f:
        f.write(content)

@pytest.fixture(scope="module")
def setup_test_files():
    """Setup test files before tests and cleanup after"""
    valid_file = "dataset/test_valid.csv"
    empty_file = "dataset/test_empty.csv"
    invalid_file = "dataset/test_invalid.csv"
    
    # Create a valid CSV file
    valid_content = "content,score\nThis is a review,5\nAnother review,4\nYet another review,3"
    create_test_csv(valid_file, valid_content)
    
    # Create an empty CSV file
    create_test_csv(empty_file, "")
    
    # Create an invalid CSV file
    create_test_csv(invalid_file, "content\nThis is a review\nAnother review")
    
    yield valid_file, empty_file, invalid_file
    
    # Cleanup
    for file in [valid_file, empty_file, invalid_file]:
        if os.path.exists(file):
            os.remove(file)

def test_load_valid_data(setup_test_files):
    """Test if valid data is loaded correctly"""
    valid_file, _, _ = setup_test_files
    df = extraction(valid_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 2)  # Check number of rows and columns

def test_expected_columns(setup_test_files):
    """Test if the loaded DataFrame has the expected columns"""
    valid_file, _, _ = setup_test_files
    df = extraction(valid_file)
    expected_columns = ["content", "score"]
    assert list(df.columns) == expected_columns

def test_load_empty_file(setup_test_files):
    """Test if an empty file is handled properly"""
    _, empty_file, _ = setup_test_files
    df = extraction(empty_file)
    assert df is None

def test_load_invalid_format(setup_test_files):
    """Test if an improperly formatted CSV file is handled"""
    _, _, invalid_file = setup_test_files
    df = extraction(invalid_file)
    assert df is None
