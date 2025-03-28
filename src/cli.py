import argparse

from config import *
from src.inference import predict_sentiment


def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis CLI")
    parser.add_argument(
        "text",
        type=str,
        help="Input text for sentiment analysis. Use '||' to separate multiple texts.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use (e.g., '3-class', '5-class')",
    )
    args = parser.parse_args()

    # Check if the input contains '||' to determine if multiple texts were provided
    if "||" in args.text:
        sample_text = [t.strip() for t in args.text.split("||") if t.strip()]
    else:
        sample_text = args.text  # single text remains unchanged

    # Select model path and label mapping based on the chosen model type
    if args.model == "3-class":
        model_path = PRETRAINED_MODEL_3_CLASS_PATH
        sentiment_labels = SENTIMENT_MAPPING_3_LABEL_VERSION
        n_classes = 3
    else:
        model_path = PRETRAINED_MODEL_5_CLASS_PATH
        sentiment_labels = SENTIMENT_MAPPING
        n_classes = 5

    # Get predictions using the unified inference function
    predictions, confidences = predict_sentiment(
        text=sample_text, path_to_model=model_path, n_classes=n_classes, device=DEVICE
    )

    # Display results depending on whether multiple texts were provided
    if isinstance(sample_text, str):
        sentiment = sentiment_labels.get(predictions, "Unknown")
        print(f"\nText: {sample_text}")
        print(f"Predicted Sentiment: {sentiment} ({confidences:.2f}%)\n")
    else:
        print("\nMultiple texts detected:\n")
        cpt = 0
        for text_item, pred, conf in zip(sample_text, predictions, confidences):
            sentiment = sentiment_labels.get(pred, "Unknown")
            print(f"{'-' * 45}")
            print(f"Text {cpt}: {text_item}")
            print(f"Predicted Sentiment: {sentiment} ({conf:.2f}%)\n")
            cpt += 1


if __name__ == "__main__":
    main()
