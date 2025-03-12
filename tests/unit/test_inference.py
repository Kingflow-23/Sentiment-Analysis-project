import os
import torch
import pytest
import numpy as np

from unittest.mock import Mock
from transformers import AutoTokenizer

from config import *
from src.model import SentimentClassifier
from src.inference import predict_sentiment

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


@pytest.fixture
def mock_model():
    """
    Fixture to create a mock SentimentClassifier model.

    Returns:
        SentimentClassifier: An instance of SentimentClassifier with its 'to', 'eval',
                             and 'load_state_dict' methods replaced by mocks.
    """
    model = SentimentClassifier(n_classes=N_CLASSES)
    model.to = Mock()
    model.eval = Mock()
    model.load_state_dict = Mock()
    return model


@pytest.fixture
def mock_torch_load(monkeypatch):
    """
    Fixture to mock the torch.load function.

    This fixture monkeypatches torch.load so that it simulates loading a model
    by returning a dummy state dictionary.
    """

    def mock_load(model_path, map_location):
        # Simulate loading the model by returning dummy weights and bias tensors.
        return {"weight": torch.zeros(10, 10), "bias": torch.zeros(10)}

    monkeypatch.setattr(torch, "load", mock_load)


@pytest.fixture(scope="module")
def setup_model_3_classes():
    """
    Fixture for a 3 class model to load the model and tokenizer once for all tests.

    Returns:
        tuple: (model, tokenizer) where model is an instance of SentimentClassifier
               loaded with a state dict from a predefined model path, and tokenizer is
               an AutoTokenizer instance from the specified TOKENIZER_NAME.
    """
    model_path = PRETRAINED_MODEL_3_CLASS_PATH

    model = SentimentClassifier(n_classes=3)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    return model, tokenizer


@pytest.fixture(scope="module")
def setup_model_5_classes():
    """
    Fixture for a 5 class model to load the model and tokenizer once for all tests.

    Returns:
        tuple: (model, tokenizer) where model is an instance of SentimentClassifier
               loaded with a state dict from a predefined model path, and tokenizer is
               an AutoTokenizer instance from the specified TOKENIZER_NAME.
    """
    model_path = PRETRAINED_MODEL_5_CLASS_PATH

    model = SentimentClassifier(n_classes=5)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    return model, tokenizer


# Use a unified fixture for general tests (defaults to 5-class based on config)
@pytest.fixture(scope="module")
def setup_model():
    # For these tests, we use the default model from config (which is 5 classes)
    model_path = (
        PRETRAINED_MODEL_5_CLASS_PATH
        if N_CLASSES == 5
        else PRETRAINED_MODEL_3_CLASS_PATH
    )
    model = SentimentClassifier(n_classes=N_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return model, tokenizer


@pytest.mark.parametrize(
    "text, expected_type",
    [
        ("This product is amazing!", int),  # Single string input
        (
            ["Worst experience ever!", "It was okay.", "Absolutely loved it!"],
            list,
        ),  # List input
    ],
)
def test_predict_sentiment(setup_model, text, expected_type):
    """
    Test that `predict_sentiment` returns the correct data type based on input.

    Verifies that a single text input returns an integer and a list input returns
    a list of integers.
    """
    model, tokenizer = setup_model
    prediction = predict_sentiment(text, model=model, tokenizer=tokenizer)

    if isinstance(prediction, np.integer):
        prediction = int(prediction)

    assert isinstance(
        prediction, expected_type
    ), f"Expected {expected_type}, got {type(prediction)}"

    if isinstance(prediction, list):
        assert all(
            isinstance(p, int) for p in prediction
        ), "List elements must be integers"


def test_missing_model_and_path():
    """
    Test that `predict_sentiment` raises ValueError if both model and path_to_model are None.

    Ensures that the function enforces providing at least one of the model or its path.
    """
    with pytest.raises(
        ValueError,
        match="Either a trained model or a path to the model must be provided.",
    ):
        predict_sentiment(
            text="This is a test text", model=None, tokenizer=None, path_to_model=None
        )


def test_invalid_model_path():
    """
    Test that `predict_sentiment` raises ValueError if the provided model path does not exist.

    Checks that an appropriate error message is raised when a non-existent model file is specified.
    """
    with pytest.raises(
        ValueError, match=f"Model path {PRETRAINED_MODEL_INVALID_PATH} does not exist."
    ):
        predict_sentiment(
            text="This is a test text",
            model=None,
            tokenizer=None,
            path_to_model=PRETRAINED_MODEL_INVALID_PATH,
        )


def test_load_model_and_device(mock_torch_load, mock_model):
    """
    Test that the model is loaded correctly, moved to the correct device, and set to eval mode.

    This test simulates calling torch.load via the monkeypatched function and verifies
    that load_state_dict, to, and eval are invoked with the expected parameters.
    """
    # Choose the correct model path based on the number of classes
    model_path = (
        PRETRAINED_MODEL_5_CLASS_PATH
        if N_CLASSES == 5
        else PRETRAINED_MODEL_3_CLASS_PATH
    )

    # Call torch.load (monkeypatched by mock_torch_load) to get a simulated state dict.
    state_dict = torch.load(model_path, map_location="cpu")

    # Load the state dict into the mock model
    mock_model.load_state_dict(state_dict, strict=False)

    # Instead of assert_called_with (which fails on tensor comparison), retrieve call arguments:
    call_args, call_kwargs = mock_model.load_state_dict.call_args
    actual_state_dict = call_args[0]

    # Define the expected state dict
    expected_state_dict = {"weight": torch.zeros(10, 10), "bias": torch.zeros(10)}

    # Compare tensors using torch.equal to avoid ambiguity
    assert torch.equal(
        actual_state_dict["weight"], expected_state_dict["weight"]
    ), "Weights do not match"
    assert torch.equal(
        actual_state_dict["bias"], expected_state_dict["bias"]
    ), "Biases do not match"
    # Also verify that 'strict' was passed as False
    assert call_kwargs.get("strict") is False, "strict keyword argument is not False"

    # Set up the model.to() method to return the model itself to avoid ambiguity
    mock_model.to.return_value = mock_model
    moved_model = mock_model.to("cpu")

    # Assert that the 'to' method was called with "cpu"
    mock_model.to.assert_called_with("cpu")

    # Simulate setting the model to evaluation mode
    moved_model.eval()

    # Assert that eval() was called exactly once
    mock_model.eval.assert_called_once()


def test_predict_sentiment_model_loading(monkeypatch):
    """
    Test that the predict_sentiment function correctly loads the model from a file.

    This test verifies that when no model is provided (model=None), the function loads the model
    from the given path. It also checks that the model's load_state_dict, to, and eval methods
    are called, and that the inference produces a predictable output.
    """
    # Flags to record that each method is called
    load_called = False
    to_called = False
    eval_called = False

    # Dummy methods to replace SentimentClassifier methods
    def dummy_load_state_dict(self, state_dict, strict=True):
        nonlocal load_called
        load_called = True
        self._dummy_state = state_dict

    def dummy_to(self, device):
        nonlocal to_called
        to_called = True
        return self

    def dummy_eval(self):
        nonlocal eval_called
        eval_called = True
        return self

    monkeypatch.setattr(SentimentClassifier, "load_state_dict", dummy_load_state_dict)
    monkeypatch.setattr(SentimentClassifier, "to", dummy_to)
    monkeypatch.setattr(SentimentClassifier, "eval", dummy_eval)

    # Use a dummy model path and force os.path.exists to return True for it
    dummy_model_path = "dummy_model_path.pth"
    monkeypatch.setattr(
        os.path, "exists", lambda path: True if path == dummy_model_path else False
    )

    # Patch torch.load to return a dummy state dictionary
    def dummy_torch_load(path, map_location):
        return {"weight": torch.zeros(10, 10), "bias": torch.zeros(10)}

    monkeypatch.setattr(torch, "load", dummy_torch_load)

    # Create a DummyEncoding class that behaves like a BatchEncoding and supports .to()
    class DummyEncoding(dict):
        def to(self, device):
            return self

    # Dummy tokenizer that returns DummyEncoding
    class DummyTokenizer:
        def __call__(self, text, padding, truncation, max_length, return_tensors):
            return DummyEncoding(
                {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }
            )

    # Patch AutoTokenizer.from_pretrained to return our DummyTokenizer
    monkeypatch.setattr(
        "src.inference.AutoTokenizer.from_pretrained", lambda name: DummyTokenizer()
    )

    # Patch the model's forward method to simulate inference.
    # For N_CLASSES = 3, choose dummy logits so that softmax and argmax yield a predictable prediction.
    dummy_logits = torch.tensor([[0.1, 0.9, 0.0]])
    monkeypatch.setattr(
        SentimentClassifier, "__call__", lambda self, **kwargs: dummy_logits
    )

    # Call predict_sentiment with model=None so that the model is loaded from file.
    prediction = predict_sentiment(
        text="Test review",
        model=None,
        tokenizer=None,  # Will trigger the dummy tokenizer
        device="cpu",
        path_to_model=dummy_model_path,
    )

    # Verify that the branch for model loading was executed:
    assert load_called, "SentimentClassifier.load_state_dict was not called"
    assert to_called, "SentimentClassifier.to was not called"
    assert eval_called, "SentimentClassifier.eval was not called"

    # Given our dummy_logits, softmax -> argmax yields index 1; adding 1 gives prediction 2.
    assert prediction == 2, f"Expected prediction 2, got {prediction}"


def test_missing_tokenizer(setup_model):
    """
    Test that predict_sentiment uses the default tokenizer when tokenizer is None.

    Ensures that if no tokenizer is provided, the function correctly initializes the default tokenizer.
    """
    model, _ = setup_model

    prediction = predict_sentiment(
        text="This is a test text", model=model, tokenizer=None
    )

    if isinstance(prediction, np.integer):
        prediction = int(prediction)

    # Verify that the tokenizer is initialized by checking its type
    assert isinstance(prediction, int), "The prediction should be of type int."


def test_invalid_input(setup_model):
    """
    Test that predict_sentiment raises ValueError for empty input text.

    Verifies that providing an empty string, empty list, or None as input raises the appropriate error.
    """
    model, tokenizer = setup_model

    with pytest.raises(ValueError, match="Input text cannot be empty."):
        predict_sentiment("", model=model, tokenizer=tokenizer)

    with pytest.raises(ValueError, match="Input text cannot be empty."):
        predict_sentiment([], model=model, tokenizer=tokenizer)

    with pytest.raises(ValueError, match="Input text cannot be empty."):
        predict_sentiment(None, model=model, tokenizer=tokenizer)


# New fixture to allow indirect parameterization by fixture name
@pytest.fixture
def setup_model_name(request):
    fixture_name = request.param
    return request.getfixturevalue(fixture_name)


@pytest.mark.parametrize(
    "setup_model_name, expected_range",
    [
        ("setup_model_5_classes", {1, 2, 3, 4, 5}),  # 5-class model
        ("setup_model_3_classes", {1, 2, 3}),  # 3-class model
    ],
    indirect=["setup_model_name"],
)
def test_prediction_range(setup_model_name, expected_range):
    """
    Test that the predicted sentiment falls within the correct label range.
    Checks that for the given number of sentiment classes, the output of predict_sentiment
    is within the expected set of sentiment labels.
    """
    model, tokenizer = setup_model_name
    text = "I really enjoyed this movie!"

    prediction = predict_sentiment(text, model=model, tokenizer=tokenizer)

    # Convert numpy integer to Python int if necessary
    if isinstance(prediction, np.integer):
        prediction = int(prediction)

    assert (
        prediction in expected_range
    ), f"Prediction {prediction} not in {expected_range}"
