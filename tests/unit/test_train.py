import os
import json
import shutil
import pytest
import pandas as pd
import torch
import torch.nn as nn

from transformers import AutoTokenizer, get_scheduler

from src.evaluate import evaluate, evaluate_and_plot
from src.model import SentimentClassifier
from src.dataloader import create_data_loader
from src.train import train_epoch, train_model

from dataset_config import FAKE_DATASET
from config import (
    SENTIMENT_MAPPING,
    MODEL_NAME,
    DEVICE,
    MODEL_TRAINING_OUTPUT_DIR,
    MODEL_EVALUATION_OUTPUT_DIR,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture
def sample_dataloader():
    """Creates a sample DataLoader for testing."""
    df = pd.DataFrame(FAKE_DATASET)
    return create_data_loader(df, tokenizer, max_len=128, batch_size=2)


@pytest.fixture
def model():
    """Creates a SentimentClassifier model instance."""
    return SentimentClassifier(n_classes=5).to(DEVICE)


@pytest.fixture
def loss_fn():
    """Returns CrossEntropyLoss function."""
    return nn.CrossEntropyLoss()


@pytest.fixture
def optimizer(model):
    """Creates an AdamW optimizer for testing."""
    return torch.optim.AdamW(model.parameters(), lr=2e-5)


@pytest.fixture
def scheduler(optimizer, sample_dataloader):
    """Creates a learning rate scheduler."""
    return get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(sample_dataloader) * 2,
    )


def get_most_recent_subfolder(test_run_dir):
    """Get the most recently created subfolder in a given directory based on timestamp."""
    # List all subfolders in the test_run_dir
    dirs = [
        d
        for d in os.listdir(test_run_dir)
        if os.path.isdir(os.path.join(test_run_dir, d))
    ]

    # Sort directories by creation time (newest first)
    dirs.sort(
        key=lambda x: os.path.getctime(os.path.join(test_run_dir, x)), reverse=True
    )

    # Get the most recent subfolder
    timestamp_subfolder = os.path.join(test_run_dir, dirs[0]) if dirs else None

    assert timestamp_subfolder is not None, "Timestamp subfolder does not exist."

    return timestamp_subfolder


# -------------------------------
# ✅ Train Function Tests
# -------------------------------


def test_train_epoch(model, sample_dataloader, loss_fn, optimizer, scheduler):
    """Test if train_epoch runs without errors."""
    train_loss, train_acc = train_epoch(
        model, sample_dataloader, loss_fn, optimizer, scheduler, DEVICE
    )

    assert isinstance(train_loss, float), "Train loss should be a float."
    assert isinstance(train_acc, float), "Train accuracy should be a float."
    assert 0 <= train_acc <= 1, "Train accuracy should be between 0 and 1."


def test_data_loader_empty():
    """Test if DataLoader handles empty datasets properly."""
    empty_df = pd.DataFrame({"content": [], "score": []})
    empty_loader = create_data_loader(empty_df, tokenizer, max_len=128, batch_size=2)

    assert len(list(empty_loader)) == 0, "Empty DataLoader should have zero batches."


def test_train_model(model, sample_dataloader):
    """Test if train_model runs end-to-end."""
    try:
        # ✅ Create test output directory
        test_run_dir = f"{MODEL_TRAINING_OUTPUT_DIR}/test_run"
        os.makedirs(test_run_dir, exist_ok=True)

        trained_model = train_model(
            model,
            sample_dataloader,
            sample_dataloader,
            DEVICE,
            epochs=1,
            run_folder=test_run_dir,
        )

        assert isinstance(
            trained_model, SentimentClassifier
        ), "Returned object should be a SentimentClassifier instance."
        assert hasattr(trained_model, "bert"), "Model should contain a BERT backbone."
        assert hasattr(
            trained_model, "fc"
        ), "Model should contain a fully connected layer."

        # ✅ Get the most recently created subfolder
        timestamp_subfolder = get_most_recent_subfolder(test_run_dir)

        ## ✅ Check if the model file was saved correctly
        model_save_path = os.path.join(timestamp_subfolder, "best_model.pth")
        assert os.path.exists(model_save_path), "Model file was not saved."

        # ✅ Check if training history JSON file was created
        history_path = os.path.join(timestamp_subfolder, "training_history.json")
        assert os.path.exists(history_path), "Training history JSON file was not saved."

        # ✅ Check if training history is a valid JSON
        with open(history_path, "r") as f:
            history = json.load(f)
            assert isinstance(history, dict), "Training history should be a dictionary."
            assert (
                "train_loss" in history and "train_acc" in history
            ), "History should contain training loss and accuracy."
            assert (
                "val_loss" in history and "val_acc" in history
            ), "History should contain validation loss and accuracy."

        # ✅ Check if the combined plot exists
        combined_plot_path = os.path.join(
            timestamp_subfolder, "accuracy_and_loss_plot.png"
        )
        assert os.path.exists(
            combined_plot_path
        ), "Combined accuracy and loss plot was not saved."

        # ✅ Check that the file is not empty
        assert os.path.getsize(combined_plot_path) > 0, "Combined plot is empty."

    finally:
        # ✅ Cleanup test directories
        # shutil.rmtree(test_run_dir)
        pass


# -------------------------------
# ✅ Evaluation Function Tests
# -------------------------------


def test_evaluate(model, sample_dataloader, loss_fn):
    """Test if evaluate runs without errors."""
    val_loss, val_acc, y_true, y_pred, confidences = evaluate(
        model, sample_dataloader, loss_fn, DEVICE
    )

    assert isinstance(val_loss, float), "Validation loss should be a float."
    assert isinstance(val_acc, float), "Validation accuracy should be a float."
    assert 0 <= val_acc <= 1, "Validation accuracy should be between 0 and 1."

    assert isinstance(y_true, list) and isinstance(
        y_pred, list
    ), "y_true and y_pred should be lists."
    assert isinstance(confidences, list), "Confidences should be a list."


def test_evaluate_and_plot(model, sample_dataloader, loss_fn):
    """Test full evaluation and plotting pipeline."""
    try:
        # ✅ Create test output directory
        test_run_dir = os.path.join(MODEL_EVALUATION_OUTPUT_DIR, "test_run")
        os.makedirs(test_run_dir, exist_ok=True)

        # ✅ Run evaluation & plotting
        class_names = list(SENTIMENT_MAPPING.values())
        evaluate_and_plot(
            model, sample_dataloader, loss_fn, DEVICE, class_names, test_run_dir
        )

        # ✅ Get the most recently created subfolder based on timestamp
        timestamp_subfolder = get_most_recent_subfolder(test_run_dir)

        # ✅ Check if evaluation files exist
        metrics_path = os.path.join(timestamp_subfolder, "metrics.txt")
        confusion_matrix_path = os.path.join(
            timestamp_subfolder, "confusion_matrix.png"
        )
        classification_report_path = os.path.join(
            timestamp_subfolder, "classification_report.png"
        )
        confidence_histogram_path = os.path.join(
            timestamp_subfolder, "confidence_histogram.png"
        )

        assert os.path.exists(metrics_path), "Metrics file was not saved."
        assert os.path.exists(
            confusion_matrix_path
        ), "Confusion matrix plot was not saved."
        assert os.path.exists(
            classification_report_path
        ), "Classification report plot was not saved."
        assert os.path.exists(
            confidence_histogram_path
        ), "Confidence histogram plot was not saved."

    finally:
        # ✅ Cleanup test directories
        # shutil.rmtree(test_run_dir)
        pass
