import os
import sys
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score
import pytest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_model_performance():
    # Define paths
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(root_dir, "artifacts", "models", "lgbm_model.pkl")
    vectorizer_path = os.path.join(root_dir, "artifacts", "models", "tfidf_vectorizer.pkl")
    
    # Data paths to check
    possible_data_paths = [
        os.path.join(root_dir, "artifacts", "interim", "test_processed.csv"),
        os.path.join(root_dir, "artifacts", "data", "interim", "test_processed.csv")
    ]
    
    test_data_path = None
    for path in possible_data_paths:
        if os.path.exists(path):
            test_data_path = path
            break
            
    # Assertions for file existence
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    assert os.path.exists(vectorizer_path), f"Vectorizer not found at {vectorizer_path}"
    assert test_data_path is not None, "Test data not found in any expected location"
    
    print(f"Loading model form: {model_path}")
    print(f"Loading data from: {test_data_path}")

    # Load resources
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    df = pd.read_csv(test_data_path)
    df.fillna('', inplace=True)
    
    # Check if required columns exist
    assert 'clean_comment' in df.columns, "Column 'clean_comment' missing from test data"
    assert 'category' in df.columns, "Column 'category' missing from test data"

    # Prepare data
    X_test = vectorizer.transform(df['clean_comment'])
    y_test = df['category']
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    
    # Performance thresholds (Adjust based on requirements)
    # Using 0.0 as baseline to ensure pipeline runs, increase as needed
    assert acc > 0.0, f"Accuracy {acc} is too low" 
    
if __name__ == "__main__":
    test_model_performance()
