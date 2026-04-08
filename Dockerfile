FROM python:3.11-slim as builder

WORKDIR /app
COPY pyproject.toml .
# Create a dummy structure to install the package structure properly
RUN mkdir -p src/molprop && touch src/molprop/__init__.py
RUN pip install --no-cache-dir build && \
    python -m build --wheel

FROM python:3.11-slim

WORKDIR /app

# Install system utilities needed by RDKit or PyTorch if any
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxtst6 \
    && rm -rf /var/lib/apt/lists/*

# Install specific Python requirements needed for ML inference
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torch-geometric==2.7.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.6 \
    pydantic==2.9.2 \
    rdkit==2025.09.6 \
    pandas==2.2.2 \
    scikit-learn==1.8.0 \
    numpy==1.26.4

# Copy built package and install it
COPY --from=builder /app/dist/*.whl ./
RUN pip install --no-cache-dir *.whl && rm *.whl

# Copy models and source
COPY best_model_*.pt ./
COPY src ./src

ENV PYTHONPATH=/app/src
ENV MODEL_TYPE=gcn
ENV MODEL_WEIGHTS=best_model_gcn_bbbp.pt

EXPOSE 8000

CMD ["uvicorn", "molprop.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
