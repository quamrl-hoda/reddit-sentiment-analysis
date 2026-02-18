import os
import sys
import json
import yaml
import pickle
import logging

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.metrics import classification_report, confusion_matrix
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# Add project root to PYTHONPATH
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

from src.dagshub_config import setup_dagshub, set_experiment

# -------------------------------------------------
# Logging configuration
# -------------------------------------------------
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# -------------------------------------------------
# Utility functions
# -------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    df.fillna("", inplace=True)
    logger.debug(f"Data loaded from {path}, shape={df.shape}")
    return df


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.debug(f"Model loaded from {path}")
    return model


def load_vectorizer(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vectorizer file not found: {path}")
    vectorizer = joblib.load(path)
    logger.debug(f"Vectorizer loaded from {path}")
    return vectorizer


def load_params(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError("params.yaml not found")
    with open(path, "r") as f:
        params = yaml.safe_load(f)
    logger.debug("Parameters loaded")
    return params


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)
    return report, cm


def log_confusion_matrix(cm, name: str):
    file_name = f"confusion_matrix_{name.replace(' ', '_')}.png"
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(file_name)
    plt.close()

    mlflow.log_artifact(file_name)
    os.remove(file_name)

# -------------------------------------------------
# Main evaluation pipeline
# -------------------------------------------------
def main():
    try:
        # 🔐 Authenticate + init MLflow FIRST
        setup_dagshub()
        set_experiment("dvc-pipeline-runs")

        with mlflow.start_run() as run:
            logger.info("🚀 Starting model evaluation")

            # ----------------------------
            # Load params
            # ----------------------------
            params_path = os.path.join(PROJECT_ROOT, "params.yaml")
            params = load_params(params_path)

            for k, v in params.items():
                mlflow.log_param(k, str(v) if isinstance(v, (dict, list)) else v)

            # ----------------------------
            # Load model & vectorizer
            # ----------------------------
            model_path = os.path.join("artifacts", "models", "lgbm_model.pkl")
            vectorizer_path = os.path.join("artifacts", "models", "tfidf_vectorizer.pkl")

            model = load_model(model_path)
            vectorizer = load_vectorizer(vectorizer_path)

            # ----------------------------
            # Load test data
            # ----------------------------
            possible_paths = [
                "artifacts/interim/test_processed.csv",
                "artifacts/data/interim/test_processed.csv",
                os.path.join(PROJECT_ROOT, "artifacts/interim/test_processed.csv"),
            ]

            test_df = None
            for p in possible_paths:
                if os.path.exists(p):
                    test_df = load_data(p)
                    break

            if test_df is None:
                raise FileNotFoundError("test_processed.csv not found")

            X_test = vectorizer.transform(test_df["clean_comment"].values)
            y_test = test_df["category"].values

            # ----------------------------
            # Infer signature (CI-safe)
            # ----------------------------
            signature = infer_signature(
                X_test[:5],
                model.predict(X_test[:5])
            )

            # ----------------------------
            # 🔥 REGISTER MODEL (CRITICAL)
            # ----------------------------
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="lgbm_model",
                signature=signature
            )

            # ----------------------------
            # Evaluation
            # ----------------------------
            report, cm = evaluate_model(model, X_test, y_test)

            mlflow.log_metric("test_accuracy", report.get("accuracy", 0.0))

            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metric(f"{label}_f1", metrics.get("f1-score", 0))

            log_confusion_matrix(cm, "Test Data")

            # ----------------------------
            # Tags
            # ----------------------------
            mlflow.set_tags({
                "model_type": "LightGBM",
                "task": "Sentiment Analysis",
                "dataset": "Reddit Comments",
                "run_id": run.info.run_id
            })

            # Save run info
            with open("experiment_info.json", "w") as f:
                json.dump(
                    {
                        "run_id": run.info.run_id,
                        "model_name": "lgbm_model"
                    },
                    f,
                    indent=4
                )
            mlflow.log_artifact("experiment_info.json")

            logger.info("✅ Model evaluation completed successfully")
            logger.info(f"📈 Test accuracy: {report.get('accuracy', 0.0):.4f}")

    except Exception as e:
        logger.exception("❌ Model evaluation failed")
        raise


if __name__ == "__main__":
    main()
