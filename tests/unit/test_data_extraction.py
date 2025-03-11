import os
import pytest
import pandas as pd

from pytest import raises
from config import *
from src.data_extraction import load_data, load_file_by_type
from dataset.test_datasets.generate_test_data import generate_test_files


@pytest.fixture(scope="module")
def generate_files():
    """Fixture to generate test files before tests run."""
    generate_test_files()
    yield
    cleanup_test_files()  # Comment this line to keep test files after tests run


def cleanup_test_files():
    """Deletes all generated test files after tests are complete."""
    expected_files = [
        "data.csv",
        "data.json",
        "data.xlsx",
        "empty.csv",
        "missing_columns.csv",
        "invalid_score.csv",
        "data.txt",
        "data.xml",
    ]

    for file_name in expected_files:
        file_path = os.path.join(TEST_DATA_DIR, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑 Deleted: {file_path}")


def test_generated_files(generate_files):
    """Test that the files created by generate_test_files() exist."""
    expected_files = [
        "data.csv",
        "data.json",
        "data.xlsx",
        "empty.csv",
        "missing_columns.csv",
        "invalid_score.csv",
        "data.txt",
        "data.xml",
    ]

    for file_name in expected_files:
        file_path = os.path.join(TEST_DATA_DIR, file_name)
        assert os.path.exists(
            file_path
        ), f"File {file_name} was not created as expected."


def test_load_data():
    df = load_data(DATASET_PATH)
    assert isinstance(df, pd.DataFrame)
    assert "content" in df.columns
    assert "score" in df.columns
    assert "label" in df.columns
    assert not df["label"].isna().any(), "Some labels were not mapped correctly"


def test_load_data_missing_file():
    """Test that loading a missing file raises an error or handles it properly."""
    with raises(FileNotFoundError):
        load_data(NON_EXISTING_DATASET_PATH)


def test_load_data_empty_file():
    """Test that an empty file is handled properly."""
    with raises(ValueError, match=r"Error: File .* is empty\."):
        load_data(EMPTY_DATASET_PATH)


def test_load_data_non_empty_data():
    df = load_data(DATASET_PATH)
    assert not df.empty, "The dataset is unexpectedly empty."


def test_load_data_missing_columns():
    """Test that missing required columns ('content' or 'score') raises an error."""
    with raises(
        ValueError, match="Dataset must contain 'content' and 'score' columns."
    ):
        load_data(MISSING_COLUMNS_DATASET_PATH)


def test_load_data_invalid_score():
    """Test that invalid score values raise an error."""
    with raises(ValueError, match=r"Dataset contains invalid score values.*"):
        load_data(INVALID_SCORE_DATASET_PATH)


def test_load_file_csv():
    """Test that a CSV file loads correctly."""
    df = load_file_by_type(CSV_FILE_PATH)
    assert isinstance(df, pd.DataFrame)


def test_load_file_json():
    """Test that a JSON file loads correctly."""
    df = load_file_by_type(JSON_FILE_PATH)
    assert isinstance(df, pd.DataFrame)


def test_load_file_xlsx():
    """Test that an XLSX file loads correctly."""
    df = load_file_by_type(XLSX_FILE_PATH)
    assert isinstance(df, pd.DataFrame)


def test_load_file_unsupported_format():
    """Test that unsupported file formats raise a ValueError."""
    with raises(ValueError, match=r"Unsupported file format: .*\.txt.*"):
        load_file_by_type(TXT_FILE_PATH)

    with raises(ValueError, match=r"Unsupported file format: .*\.xml.*"):
        load_file_by_type(XML_FILE_PATH)


def test_load_data_with_merge_labels():
    """Test that the 'merge_labels' feature is working correctly."""
    # Load the data with merge_labels set to True
    df = load_data(DATASET_PATH, merge_labels=True)

    # Check if the sentiment labels are within the range of 0, 1, 2 (Negative, Neutral, Positive)
    assert all(
        df["score"].isin([0, 1, 2])
    ), "The 'score' column values are incorrect after merging labels."

    # Check if 'label' column is using the 3-label version mapping
    assert all(
        df["label"].isin(SENTIMENT_MAPPING_3_LABEL_VERSION.values())
    ), "The 'label' values are not correct after merging."


def test_load_data_without_merge_labels():
    """Test that loading data without merge_labels works with the 5-class mapping."""
    # Load data without merge_labels
    df = load_data(DATASET_PATH, merge_labels=False)

    # Check if the 'score' column contains values within the 0-4 range after mapping (5-class range)
    assert all(
        df["score"].isin([0, 1, 2, 3, 4])
    ), "The 'score' column values are incorrect without merging labels."

    # Check if the 'label' column is using the 5-class sentiment mapping
    assert all(
        df["label"].isin(SENTIMENT_MAPPING.values())
    ), "The 'label' values are not correct without merging."
