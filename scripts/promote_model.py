import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dagshub_config import setup_dagshub

def promote_model():
    print("Initializing DagsHub/MLflow...")
    setup_dagshub()
    
    model_name = "lgbm_model"
    client = MlflowClient()
    
    try:
        # Get all versions
        # Note: DagsHub MLflow might behave slightly differently, so we get all and filter
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print(f"No versions found for model {model_name}")
            return

        # Sort by version number (descending) to get the latest
        versions.sort(key=lambda x: int(x.version), reverse=True)
        latest_version = versions[0]
        
        print(f"Found latest version: {latest_version.version} (Stage: {latest_version.current_stage})")
        
        if latest_version.current_stage == "Production":
            print(f"Version {latest_version.version} is already in Production.")
            return

        print(f"Promoting model {model_name} version {latest_version.version} to Production...")
        
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        print("✅ Model promotion successful!")
        
    except Exception as e:
        print(f"❌ Error promoting model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    promote_model()
