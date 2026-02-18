FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY backend/requirements.txt /app/requirements.txt

# Upgrade pip & install deps (more stable)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy app code
COPY backend/ /app/

# Copy vectorizer and model
COPY artifacts/models/tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl
COPY artifacts/models/lgbm_model.pkl /app/lgbm_model.pkl

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["python", "app.py"]
