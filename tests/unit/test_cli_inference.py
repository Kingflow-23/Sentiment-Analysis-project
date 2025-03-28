import os
import pytest
import sys
from unittest.mock import patch
from io import StringIO

from config import *
from src.cli import main


# Mock os.path.exists to prevent "Model path does not exist" errors
@pytest.fixture
def mock_valid_model_path():
    with patch("os.path.exists", return_value=True):
        yield


# Test for empty text input (empty string)
@pytest.mark.parametrize("text_input", [""])
def test_empty_text_input(text_input, mock_valid_model_path):
    test_args = ["cli.py", text_input, "--model", "3-class"]

    with patch.object(sys, "argv", test_args):
        with pytest.raises(ValueError) as excinfo:
            main()

        assert "Input text cannot be empty." in str(
            excinfo.value
        ), f"Unexpected error message: {excinfo.value}"


# Test when neither a trained model nor a path to the model is provided
def test_no_model_or_path(mock_valid_model_path):
    test_args = ["cli.py", "I love this product!"]

    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 2, f"Unexpected exit code: {excinfo.value.code}"


# Test for single text input
def test_single_text_prediction(mock_valid_model_path):
    test_args = ["cli.py", "I love this product!", "--model", "3-class"]

    with patch.object(sys, "argv", test_args):
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            main()
            output = mock_stdout.getvalue()

            assert "Predicted Sentiment:" in output  # Check sentiment label
            assert "%" in output  # Confidence should be formatted as percentage

            # Extract confidence value
            confidence = float(output.split("(")[-1].split("%")[0])
            assert 0 <= confidence <= 100  # Confidence score should be valid


# Test for multiple texts input
def test_multiple_texts_prediction(mock_valid_model_path):
    test_args = [
        "cli.py",
        "I love this product! || This is so bad!",
        "--model",
        "3-class",
    ]

    with patch.object(sys, "argv", test_args):
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            main()
            output = mock_stdout.getvalue()

            assert "Text 0: I love this product!" in output
            assert "Predicted Sentiment:" in output
            assert "%" in output  # Confidence format check
            confidence_values = [
                float(conf.split("%")[0]) for conf in output.split("(") if "%" in conf
            ]
            assert all(0 <= conf <= 100 for conf in confidence_values)

            assert "Text 1: This is so bad!" in output
            assert "Predicted Sentiment:" in output


# Test with 5-class model
def test_5_class_model(mock_valid_model_path):
    test_args = [
        "cli.py",
        "I love this product! || This is so bad!",
        "--model",
        "5-class",
    ]

    with patch.object(sys, "argv", test_args):
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            main()
            output = mock_stdout.getvalue()

            assert "Text 0: I love this product!" in output
            assert "Predicted Sentiment:" in output
            assert "%" in output  # Confidence format check
            confidence_values = [
                float(conf.split("%")[0]) for conf in output.split("(") if "%" in conf
            ]
            assert all(0 <= conf <= 100 for conf in confidence_values)

            assert "Text 1: This is so bad!" in output
            assert "Predicted Sentiment:" in output
