import os
import torch
from typing import Union, List
from transformers import AutoTokenizer

from src.model import SentimentClassifier
from config import *


def load_model(path_to_model: str, n_classes: int, device: str) -> SentimentClassifier:
    """
    Loads the trained sentiment classifier model from disk.

    Args:
        path_to_model (str): File path to the pre-trained model.
        n_classes (int): Number of sentiment classes.
        device (str): Device to run inference on.
    """
    if not os.path.exists(path_to_model):
        raise ValueError(f"Model path {path_to_model} does not exist.")
    model = SentimentClassifier(n_classes=n_classes)
    model.load_state_dict(torch.load(path_to_model, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def predict_sentiment(
    text: Union[str, List[str]] = None,
    model: SentimentClassifier = None,
    tokenizer: AutoTokenizer = None,
    device: Union[str, torch.device] = "cpu",
    path_to_model: str = None,
    n_classes: int = N_CLASSES,
) -> Union[tuple[int, float], tuple[List[int], List[float]]]:
    """
    Predicts the sentiment class for one or more text inputs.

    Args:
        text (Union[str, List[str]]): Input review text(s) to classify.
        model (SentimentClassifier, optional): A preloaded model instance.
        tokenizer (AutoTokenizer, optional): Tokenizer to use; if None, the default is loaded.
        device (Union[str, torch.device]): Device to run inference on.
        path_to_model (str, optional): File path to the pre-trained model.
        n_classes (int, optional): Number of sentiment classes.

    Returns:
        (List[int], List[float]): Lists of predicted labels and corresponding confidence scores.
    """
    # Validate and standardize input text
    if (
        text is None
        or (isinstance(text, list) and len(text) == 0)
        or (isinstance(text, str) and len(text) == 0)
    ):
        raise ValueError("Input text cannot be empty.")
    if isinstance(text, str):
        text = [text]

    # Load model if not provided
    if model is None:
        if path_to_model is None:
            raise ValueError(
                "Either a trained model or a path to the model must be provided."
            )
        model = load_model(path_to_model, n_classes, device)

    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Tokenize input text
    inputs = tokenizer(
        text, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
    ).to(device)

    # Perform inference
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence_scores, predicted_classes = probabilities.max(dim=1)

        confidence_scores = (confidence_scores.cpu().numpy() * 100).tolist()
        predictions = (predicted_classes.cpu().numpy() + 1).tolist()

    return predictions, confidence_scores
