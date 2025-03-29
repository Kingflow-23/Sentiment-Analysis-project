import logging

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from config import *
from src.db_logger import log_prediction
from src.inference import predict_sentiment

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TextRequest(BaseModel):
    text: str
    model_type: str = "3-class"  # Default model type is 3-class


@app.get("/")
def health_check():
    return {"status": "API is running 🚀"}


@app.post("/predict/")
def predict_sentiment_api(request: TextRequest):
    logger.info(
        f"📩 Received request: text={request.text}, model_type={request.model_type}"
    )

    # Validate model type
    if request.model_type not in ["3-class", "5-class"]:
        raise HTTPException(
            status_code=400, detail="Invalid model_type. Use '3-class' or '5-class'."
        )

    # Check for empty or whitespace-only input text
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # Split input into multiple texts if "||" is present
    texts = [t.strip() for t in request.text.split("||") if t.strip()]

    # Choose model and label mapping
    model_path = (
        PRETRAINED_MODEL_3_CLASS_PATH
        if request.model_type == "3-class"
        else PRETRAINED_MODEL_5_CLASS_PATH
    )
    sentiment_labels = (
        SENTIMENT_MAPPING_3_LABEL_VERSION
        if request.model_type == "3-class"
        else SENTIMENT_MAPPING
    )
    n_classes = 3 if request.model_type == "3-class" else 5

    # Perform sentiment analysis
    try:
        predictions, confidences = predict_sentiment(
            text=texts, path_to_model=model_path, n_classes=n_classes, device=DEVICE
        )

        # Convert NumPy types to native Python types
        predictions = [int(pred) for pred in predictions]
        confidences = [float(conf) for conf in confidences]

    except FileNotFoundError:
        logger.error("❌ Model file not found!")
        raise HTTPException(status_code=500, detail="Model file is missing.")
    except RuntimeError as e:
        logger.error(f"🔥 GPU Error: {str(e)}")
        raise HTTPException(
            status_code=500, detail="GPU memory issue. Try running on CPU."
        )
    except Exception as e:
        logger.error(f"🔥 Error during sentiment analysis: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error during sentiment analysis: {str(e)}"
        )

    # Log and format results
    results = []

    for text, pred, conf in zip(texts, predictions, confidences):
        log_prediction(text, pred, conf, "FastAPI")
        sentiment = sentiment_labels.get(pred, "Unknown")

        # Optional: Flag low-confidence predictions
        # Confidence level classification
        if conf < 50.0:
            confidence_status = "🔴 Low confidence"
        elif 50.0 <= conf <= 85.0:
            confidence_status = "🟠 Medium confidence"
        else:
            confidence_status = "🟢 High confidence"

        results.append(
            {
                "text": text,
                "predicted_sentiment": sentiment,
                "confidence": round(conf, 2),
                "confidence_status": confidence_status,
            }
        )

    logger.info(f"✅ Sentiment analysis completed: {results}")

    return {"results": results}


# To run the API, use:
# uvicorn src.api:app --reload
