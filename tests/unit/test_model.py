import pytest
import torch
from transformers import AutoTokenizer
from src.model import SentimentClassifier
from config import N_CLASSES, MODEL_NAME

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture
def sample_input():
    """Fixture to create a sample tokenized input for testing."""
    text = "This is a great app! I love using it every day."
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    return inputs


@pytest.fixture
def model():
    """Fixture to create a SentimentClassifier model instance."""
    return SentimentClassifier(n_classes=N_CLASSES)


def test_model_initialization(model):
    """Test if the model initializes correctly."""
    assert isinstance(model, SentimentClassifier), "Model is not an instance of SentimentClassifier."
    assert model.fc.out_features == N_CLASSES, "Output layer size does not match number of classes."


def test_model_forward_pass(model, sample_input):
    """Test if the model processes a forward pass correctly."""
    input_ids = sample_input["input_ids"]
    attention_mask = sample_input["attention_mask"]

    with torch.no_grad():  # No gradient needed for this test
        output = model(input_ids, attention_mask)

    assert isinstance(output, torch.Tensor), "Model output is not a tensor."
    assert output.shape == (1, N_CLASSES), f"Expected output shape (1, {N_CLASSES}), but got {output.shape}."


def test_model_requires_grad(model):
    """Test if model parameters require gradients (important for training)."""
    for param in model.parameters():
        assert param.requires_grad, "Some model parameters are frozen unexpectedly."


def test_model_backward_pass(model, sample_input):
    """Test if backpropagation works properly."""
    input_ids = sample_input["input_ids"]
    attention_mask = sample_input["attention_mask"]
    targets = torch.tensor([2])  # Example label (Negative sentiment)

    output = model(input_ids, attention_mask)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(output, targets)

    loss.backward()  # Backpropagation

    # Check if gradients were computed for model parameters
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None, "Gradient not computed for some parameters."
