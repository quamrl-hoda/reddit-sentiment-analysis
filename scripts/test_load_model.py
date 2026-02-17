import joblib
import os
import sys

def test_load_model():
    # Define paths
    model_path = os.path.join("artifacts", "models", "lgbm_model.pkl")
    vectorizer_path = os.path.join("artifacts", "models", "tfidf_vectorizer.pkl")

    # Check if files exist
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    assert os.path.exists(vectorizer_path), f"Vectorizer file not found at {vectorizer_path}"

    # Try to load them
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("Model and vectorizer loaded successfully")
    except Exception as e:
        import pytest
        pytest.fail(f"Failed to load model or vectorizer: {e}")

if __name__ == "__main__":
    test_load_model()
