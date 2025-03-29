import pytest

from fastapi.testclient import TestClient

from src.api import app

# Create a TestClient instance to test the FastAPI app
client = TestClient(app)


# Test for health check (basic status check)
def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running 🚀"}


# Test for empty text input
def test_empty_text_input():
    response = client.post(
        "/predict/",
        json={
            "text": "   ",
            "model_type": "3-class",
        },  # Input text is empty or just whitespace
    )

    assert response.status_code == 400  # Bad Request status code
    assert response.json() == {"detail": "Input text cannot be empty."}


# Test for single text input
def test_single_text_prediction():
    response = client.post(
        "/predict/",
        json={"text": "I love this product!", "model_type": "3-class"},
    )

    assert response.status_code == 200
    result = response.json()
    assert "results" in result
    assert isinstance(result["results"], list)
    assert len(result["results"]) == 1

    res = result["results"][0]
    assert "text" in res
    assert "predicted_sentiment" in res
    assert "confidence" in res
    assert "confidence_status" in res

    assert isinstance(res["confidence"], (int, float))
    assert 0 <= res["confidence"] <= 100


# Test for multiple texts input
def test_multiple_texts_prediction():
    response = client.post(
        "/predict/",
        json={
            "text": "I love this product! || This is so bad!",
            "model_type": "3-class",
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert "results" in result
    assert isinstance(result["results"], list)
    assert len(result["results"]) == 2  # Two texts were provided

    for res in result["results"]:
        assert "text" in res
        assert "predicted_sentiment" in res
        assert "confidence" in res
        assert "confidence_status" in res

        assert isinstance(res["confidence"], (int, float))
        assert 0 <= res["confidence"] <= 100


# Test for confidence thresholds
def test_confidence_threshold():
    # Low confidence test (below 50%)
    response = client.post(
        "/predict/",
        json={"text": "Something neutral", "model_type": "3-class"},
    )
    result = response.json()
    assert response.status_code == 200
    assert len(result["results"]) == 1
    assert result["results"][0]["confidence_status"] == "🔴 Low confidence"

    # Medium confidence test (50% to 85%)
    response = client.post(
        "/predict/",
        json={"text": "Good app", "model_type": "3-class"},
    )
    result = response.json()
    assert response.status_code == 200
    assert len(result["results"]) == 1
    assert result["results"][0]["confidence_status"] == "🟠 Medium confidence"

    # High confidence test (above 85%)
    response = client.post(
        "/predict/",
        json={"text": "I absolutely love this!", "model_type": "3-class"},
    )
    result = response.json()
    assert response.status_code == 200
    assert len(result["results"]) == 1
    assert result["results"][0]["confidence_status"] == "🟢 High confidence"


# Test with a different model (5-class model)
def test_5_class_model():
    response = client.post(
        "/predict/",
        json={
            "text": "I love this product! || This is so bad!",
            "model_type": "5-class",
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert "results" in result
    assert isinstance(result["results"], list)
    assert len(result["results"]) == 2

    for res in result["results"]:
        assert "text" in res
        assert "predicted_sentiment" in res
        assert "confidence" in res
        assert "confidence_status" in res

        assert isinstance(res["confidence"], (int, float))
        assert 0 <= res["confidence"] <= 100


# Test for invalid model type
def test_invalid_model_type():
    response = client.post(
        "/predict/",
        json={"text": "I love this product!", "model_type": "invalid-model"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Invalid model_type. Use '3-class' or '5-class'."
    }


# Test for missing text field
def test_missing_text_field():
    response = client.post("/predict/", json={"model_type": "3-class"})

    assert response.status_code == 422  # Unprocessable Entity (missing required field)
