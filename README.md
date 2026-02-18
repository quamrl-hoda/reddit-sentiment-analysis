# Reddit Sentiment Analysis

A machine learning project to analyze sentiment in Reddit comments. This project uses DVC for data versioning and pipeline management.

## Folder Structure

- `src/`: Source code for data processing and model training.
- `backend/`: Flask backend for the application.
- `scripts/`: Helper scripts for testing and automation.
- `artifacts/`: Generated artifacts (data, models, etc.).
- `models/`: Registered models.

## Data Import

The data ingestion process downloads the dataset from a remote source and saves it locally.

- **Source URL**: `https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv`
- **Destination**: The data is split and saved to `artifacts/data/` as `train.csv` and `test.csv`.

To run the data ingestion step manually:
```bash
python src/data/data_ingestion.py
```

## DVC Pipeline

This project uses DVC (Data Version Control) to manage the machine learning pipeline.

To reproduce the entire pipeline:
```bash
dvc repro
```

### Pipeline Stages

1. **Data Ingestion** (`data_ingestion`): Downloads and splits the data.
2. **Data Preprocessing** (`data_preprocessing`): Cleans and preprocesses the text data.
3. **Model Building** (`model_building`): Trains the machine learning model.
4. **Model Evaluation** (`model_evaluation`): Evaluates the model's performance.
5. **Model Registration** (`model_registration`): Registers the trained model for tracking.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the DVC pipeline:
   ```bash
   dvc repro
   ```