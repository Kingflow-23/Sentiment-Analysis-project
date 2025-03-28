import pytest

from fastapi.testclient import TestClient

from src.api import app

# Create a TestClient instance to test the FastAPI app
client = TestClient(app)


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

    assert "text" in result
    assert "predicted_sentiment" in result
    assert "confidence" in result

    confidence = float(result["confidence"].strip("%"))
    assert 0 <= confidence <= 100


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

    assert isinstance(result, list)
    assert len(result) == 2  # Two texts were provided

    for res in result:
        assert "text" in res
        assert "predicted_sentiment" in res
        assert "confidence" in res

        confidence = float(res["confidence"].strip("%"))
        assert 0 <= confidence <= 100


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

    assert isinstance(result, list)
    assert len(result) == 2  # Two texts were provided

    for res in result:
        assert "text" in res
        assert "predicted_sentiment" in res
        assert "confidence" in res

        confidence = float(res["confidence"].strip("%"))
        assert 0 <= confidence <= 100
