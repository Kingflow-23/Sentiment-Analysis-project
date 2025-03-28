from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import *
from src.inference import predict_sentiment

app = FastAPI()


class TextRequest(BaseModel):
    text: str
    model_type: str = "3-class"  # Default model type is 3-class


@app.post("/predict/")
def predict_sentiment_api(request: TextRequest):
    # Check for empty or whitespace-only input text
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # Check if the input text contains "||" to determine if multiple texts were provided.
    if "||" in request.text:
        texts = [t.strip() for t in request.text.split("||") if t.strip()]
    else:
        texts = request.text  # Single text remains unchanged.

    # Choose the model and label mapping based on model_type.
    if request.model_type == "3-class":
        model_path = PRETRAINED_MODEL_3_CLASS_PATH
        sentiment_labels = SENTIMENT_MAPPING_3_LABEL_VERSION
        n_classes = 3
    else:
        model_path = PRETRAINED_MODEL_5_CLASS_PATH
        sentiment_labels = SENTIMENT_MAPPING
        n_classes = 5

    # Call the unified inference function.
    prediction, confidence = predict_sentiment(
        text=texts, path_to_model=model_path, n_classes=n_classes, device=DEVICE
    )

    # If a single text was provided, return a single prediction; otherwise, return a list.
    if isinstance(texts, str):
        sentiment = sentiment_labels.get(prediction, "Unknown")
        return {
            "text": texts,
            "predicted_sentiment": sentiment,
            "confidence": f"{confidence:.2f}%",
        }
    else:
        results = []
        for text_item, pred, conf in zip(texts, prediction, confidence):
            sentiment = sentiment_labels.get(pred, "Unknown")
            results.append(
                {
                    "text": text_item,
                    "predicted_sentiment": sentiment,
                    "confidence": f"{conf:.2f}%",
                }
            )
        return results


# To run the API, use:
# uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
