import argparse

from config import *
from src.db_logger import log_prediction
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

    # Handle empty text input
    if not args.text.strip():
        raise ValueError("Input text cannot be empty.")

    # Check for multiple texts using '||'
    sample_texts = [t.strip() for t in args.text.split("||") if t.strip()]

    if args.model not in ["3-class", "5-class"]:
        print("❌ Error: Invalid model choice. Use '3-class' or '5-class'.")
        raise ValueError("Invalid model choice")

    # Select model path and label mapping based on the chosen model type
    if args.model == "3-class":
        model_path = PRETRAINED_MODEL_3_CLASS_PATH
        sentiment_labels = SENTIMENT_MAPPING_3_LABEL_VERSION
        n_classes = 3
    else:
        model_path = PRETRAINED_MODEL_5_CLASS_PATH
        sentiment_labels = SENTIMENT_MAPPING
        n_classes = 5

    try:
        predictions, confidences = predict_sentiment(
            text=sample_texts,
            path_to_model=model_path,
            n_classes=n_classes,
            device=DEVICE,
        )
    except Exception as e:
        print(f"❌ Error during sentiment analysis: {e}")
        return

    # Display and log results
    print("\nSentiment Analysis Results:\n")
    for idx, (text, pred, conf) in enumerate(
        zip(sample_texts, predictions, confidences)
    ):
        sentiment = sentiment_labels.get(pred, "Unknown")
        print(f"{'-' * 45}")
        print(f"Text {idx + 1}: {text}")
        print(f"Predicted Sentiment: {sentiment} ({conf:.2f}%)\n")

        # Log each prediction
        log_prediction(text, pred, conf)


if __name__ == "__main__":
    main()
