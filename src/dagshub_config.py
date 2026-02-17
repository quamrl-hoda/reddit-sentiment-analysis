import os
from dotenv import load_dotenv
import dagshub
import mlflow

def setup_dagshub():
    """Initialize DagsHub and MLflow tracking."""
    load_dotenv()
    
    # Get credentials from environment variables
    DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME')
    DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN')
    REPO_NAME = os.getenv('REPO_NAME', 'reddit-sentiment-analysis')
    
    # Check if credentials exist
    if not DAGSHUB_USERNAME or not DAGSHUB_TOKEN:
        print("⚠️  DagsHub credentials not found. Running without MLflow tracking.")
        return None
    
    # Set token for authentication
    os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
    
    try:
        # Initialize DagsHub
        dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=REPO_NAME, mlflow=True)
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(f'https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow')
        
        print(f"✅ DagsHub initialized for {DAGSHUB_USERNAME}/{REPO_NAME}")
        return mlflow
        
    except Exception as e:
        print(f"❌ Failed to initialize DagsHub: {e}")
        print("Continuing without MLflow tracking...")
        return None

def set_experiment(experiment_name='dvc-pipeline-runs'):
    """Set or create an MLflow experiment."""
    try:
        mlflow.set_experiment(experiment_name)
        print(f"✅ Experiment set: {experiment_name}")
    except Exception as e:
        print(f"❌ Failed to set experiment: {e}")
        raise