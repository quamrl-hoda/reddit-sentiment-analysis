import logging
import os
import pickle
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# Logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise
    
def apply_tfidf(train_data, max_features, ngram_range):
    """Apply TF-IDF vectorization to the training data."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import joblib
        import os
        
        # Create the models directory if it doesn't exist
        os.makedirs('artifacts/models', exist_ok=True)
        logger.debug(f"Created directory: artifacts/models")
        
        # Initialize TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features, 
                                           ngram_range=ngram_range)
        
        # Assuming 'clean_comment' or 'processed_text' column contains the text
        text_column = 'clean_comment' if 'clean_comment' in train_data.columns else 'processed_text'
        
        # Fit and transform the training data
        X_train = tfidf_vectorizer.fit_transform(train_data[text_column])
        
        # Extract target variable (assuming 'sentiment' or 'label' column)
        y_train = train_data['category'] if 'category' in train_data.columns else train_data['label']
        
        # Save the vectorizer
        vectorizer_path = os.path.join('artifacts/models', 'tfidf_vectorizer.pkl')
        joblib.dump(tfidf_vectorizer, vectorizer_path)
        logger.debug(f"TF-IDF vectorizer saved to {vectorizer_path}")
        
        return X_train, y_train
        
    except Exception as e:
        logger.error(f"Failed to apply TF-IDF feature engineering: {e}")
        raise

def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int):
    """Train a LightGBM model using the provided training data and hyperparameters."""
    try:
        best_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric='multi_logloss',
            is_unbalance=True,
            class_weight='balanced',
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        best_model.fit(X_train, y_train)
        logger.debug('LightGBM model trained successfully')
        return best_model
    except Exception as e:
        logger.error('Failed to train the LightGBM model: %s', e)
        raise
def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug('Model saved successfully to %s', file_path)
    except Exception as e:
        logger.error('Failed to save the model: %s', e)
        raise
    
def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from a YAML file."""
    try:
        # If you want to always look in project root, you could do:
        import os
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        full_path = os.path.join(root_dir, params_path)
        
        with open(full_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', full_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', full_path)
        raise

def main():
   try:
    # Load parameters from the root directory
    params = load_params('params.yaml')
    
    max_features = params['model_building']['max_features']
    ngram_range = tuple(params['model_building']['ngram_range'])
    learning_rate = params['model_building']['learning_rate']
    max_depth = params['model_building']['max_depth']
    n_estimators = params['model_building']['n_estimators']
    
    logger.debug(f"Model parameters loaded: max_features={max_features}, ngram_range={ngram_range}")
    
    # Load the preprocessed training data from interim directory
    train_data = load_data('artifacts/interim/train_processed.csv')
    
    # Apply TF-IDF feature engineering on training data
    X_train_tfidf, y_train = apply_tfidf(train_data, max_features, ngram_range)
    
    # Train the LightGBM model using hyperparameters from params.yaml
    best_model = train_lgbm(X_train_tfidf, y_train, learning_rate, max_depth, n_estimators)
    
    # Create models directory if it doesn't exist
    os.makedirs('artifacts/models', exist_ok=True)
    
    # Save the trained model
    save_model(best_model, 'artifacts/models/lgbm_model.pkl')
    logger.debug('Model saved to artifacts/models/lgbm_model.pkl')

   except KeyError as e:
    logger.error(f"Missing key in params.yaml: {e}")
    logger.error("Please ensure params.yaml contains all required model_building parameters")
    raise
   
if __name__ == "__main__":
    main()