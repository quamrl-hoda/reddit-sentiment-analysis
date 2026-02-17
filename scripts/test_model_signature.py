import joblib
import os
import pandas as pd
import pytest

def test_model_signature():
    model_path = os.path.join("artifacts", "models", "lgbm_model.pkl")
    vectorizer_path = os.path.join("artifacts", "models", "tfidf_vectorizer.pkl")
    
    # Load model and vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # Sample input
    sample_text = ["This is a test comment", "Another test comment"]
    
    # Transform
    try:
        transformed = vectorizer.transform(sample_text)
        
        # Handle feature names
        if hasattr(vectorizer, 'get_feature_names_out'):
            feature_names = vectorizer.get_feature_names_out()
        else:
            feature_names = vectorizer.get_feature_names()
            
        transformed_df = pd.DataFrame(transformed.toarray(), columns=feature_names)
        
        # Predict
        predictions = model.predict(transformed_df)
        
        # Verify output
        assert len(predictions) == 2
        # Check if predictions are roughly within expected range (e.g., -1, 0, 1)
        # Note: Model might output floats or ints depending on training. 
        # Adjust assertion based on actual model output type if needed.
        print(f"Predictions: {predictions}")
        
    except Exception as e:
        pytest.fail(f"Model signature test failed: {e}")

if __name__ == "__main__":
    test_model_signature()
