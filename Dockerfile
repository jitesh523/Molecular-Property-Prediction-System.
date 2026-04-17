FROM python:3.11-slim AS builder

LABEL maintainer="Molecular Property Prediction System"
LABEL version="1.0.0"
LABEL description="Production inference API for molecular property prediction"

WORKDIR /app
COPY pyproject.toml .
# Create a dummy structure to install the package structure properly
RUN mkdir -p src/molprop && touch src/molprop/__init__.py
RUN pip install --no-cache-dir build && \
    python -m build --wheel

FROM python:3.11-slim

WORKDIR /app

# Install system utilities needed by RDKit
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxtst6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install specific Python requirements needed for ML inference
RUN pip install --no-cache-dir \
    torch==2.2.1 \
    torch-geometric==2.5.2 \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn[standard]==0.27.1 \
    pydantic==2.6.3 \
    rdkit==2023.9.5 \
    pandas==2.2.1 \
    scikit-learn==1.4.1.post1 \
    numpy==1.26.4 \
    joblib>=1.3.0 \
    shap>=0.44.0 \
    streamlit>=1.31.0 \
    plotly>=5.18.0

# Copy built package and install it
COPY --from=builder /app/dist/*.whl ./
RUN pip install --no-cache-dir *.whl && rm *.whl

# Copy models and source
COPY best_model_*.pt ./
COPY src ./src

ENV PYTHONPATH=/app/src
ENV MODEL_TYPE=gcn
ENV MODEL_WEIGHTS=best_model_gcn_bbbp.pt
ENV MODEL_DATASET=bbbp
ENV MODEL_TASK=classification

EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "molprop.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
