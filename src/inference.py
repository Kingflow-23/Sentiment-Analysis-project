import os
import torch

from typing import Union
from transformers import AutoTokenizer

from src.model import SentimentClassifier
from config import *


def predict_sentiment(
    text: Union[str, list] = None,  # Can handle both single string and list of strings
    model: SentimentClassifier = None,
    tokenizer: AutoTokenizer = None,
    device: Union[str, torch.device] = "cpu",  # Default to cpu if not specified
    path_to_model: str = None,
    n_classes: int = N_CLASSES,
) -> Union[tuple[int, float], tuple[list, list]]:
    """
    Predicts the sentiment class for one or more text inputs.

    Args:
        text (Union[str, list]): Input review text(s) to classify. Can be a single string or a list of strings.
        model (SentimentClassifier, optional): The trained BERT-based sentiment model.
        tokenizer (AutoTokenizer, optional): Hugging Face tokenizer for text processing.
        device (Union[str, torch.device]): Device to run inference on ("cpu" or "cuda").
        path_to_model (str, optional): Path to the pre-trained model if model is not passed.
        n_classes (int, optional): Number of sentiment classes. Defaults to N_CLASSES.


    Returns:
        Union[int, list]: Predicted sentiment label:
            - If a single text is provided:
                - (int) Predicted sentiment label.
                - (float) Confidence score (percentage).

            - If a list of texts is provided:
                - (list[int]) List of predicted sentiment labels.
                - (list[float]) List of confidence scores (percentages).

        Classification Labels:
            - If `N_CLASSES = 5`:
                - 1 = Really Negative
                - 2 = Negative
                - 3 = Neutral
                - 4 = Positive
                - 5 = Really Positive

            - If `N_CLASSES = 3`:
                - 1 = Negative
                - 2 = Neutral
                - 3 = Positive
    """
    if (
        text is None
        or (isinstance(text, list) and len(text) == 0)
        or (isinstance(text, str) and len(text) == 0)
    ):
        raise ValueError("Input text cannot be empty.")

    if isinstance(text, str):
        text = [text]  # Convert a single string to a list

    if model is None:
        if path_to_model is None:
            raise ValueError(
                "Either a trained model or a path to the model must be provided."
            )

        # Check if model path exists
        if not os.path.exists(path_to_model):
            raise ValueError(f"Model path {path_to_model} does not exist.")

        # Load the model from the provided path
        model = SentimentClassifier(n_classes=n_classes)
        model.load_state_dict(torch.load(path_to_model, map_location=device))
        model.to(device)
        model.eval()  # Ensure model is in evaluation mode

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)  # Default tokenizer

    # Tokenize input text
    inputs = tokenizer(
        text, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
    ).to(device)

    # Perform inference
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        probabilities = torch.nn.functional.softmax(
            logits, dim=1
        )  # Get confidence scores
        confidence_scores, predicted_classes = probabilities.max(
            dim=1
        )  # Get max probability and corresponding class
        confidence_scores = (
            confidence_scores.cpu().numpy() * 100
        )  # Convert to percentage
        predictions = predicted_classes.cpu().numpy() + 1  # Map to 1-5 scale

    # Return a single prediction if one input text was provided, or a list if multiple texts were provided
    if len(text) == 1:
        return predictions[0], confidence_scores[0]
    else:
        return predictions.tolist(), confidence_scores.tolist()
