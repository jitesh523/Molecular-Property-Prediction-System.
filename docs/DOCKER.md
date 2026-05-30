# Docker Setup and Deployment Guide

This guide covers building, testing, and deploying the molprop application using Docker and Docker Compose.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- At least 8GB of available disk space (for images and models)

## Quick Start

### Build and Run with Docker Compose

```bash
# Start all services
docker compose up -d

# Check logs
docker compose logs -f api

# Stop services
docker compose down

# View running containers
docker compose ps
```

### Build Docker Image

```bash
# Build the image
docker build -t molprop:latest .

# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.12 -t molprop:py312 .

# Build with build cache disabled
docker build --no-cache -t molprop:latest .
```

### Run Container

```bash
# Run API server
docker run -p 8000:8000 \
  -e MODEL_TYPE=gcn \
  -e MODEL_DATASET=bbbp \
  molprop:latest

# Run with GPU support
docker run --gpus all -p 8000:8000 \
  -e CUDA_VISIBLE_DEVICES=0 \
  molprop:latest

# Run with volume mounts
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  molprop:latest

# Run with environment file
docker run -p 8000:8000 \
  --env-file .env.production \
  molprop:latest
```

## Docker Compose

### Configuration

The `docker-compose.yml` file includes:

- **API Service**: FastAPI server on port 8000
- **Redis (Optional)**: Caching backend on port 6379
- **Postgres (Optional)**: Database on port 5432

### Environment Variables

Create `.env.docker` for environment configuration:

```bash
# Model Configuration
MODEL_TYPE=gcn
MODEL_WEIGHTS=best_model_gcn_bbbp.pt
MODEL_DATASET=bbbp
MODEL_TASK=classification

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_DEBUG=false

# Cache Configuration
CACHE_ENABLED=true
CACHE_MAX_SIZE=512
CACHE_TTL_SECONDS=600
REDIS_URL=redis://redis:6379/0

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_CAPACITY=120
RATE_LIMIT_WINDOW=60

# Database
LIBRARY_DB_PATH=/app/data/library.db
VECTOR_DB_ENABLED=true
VECTOR_DB_PATH=/app/data/vectors

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=text  # or 'json' for production
```

### Multi-Stage Build

The Dockerfile uses multi-stage builds to reduce image size:

1. **Builder Stage**: Installs build tools and compiles packages
2. **Runtime Stage**: Copies only compiled artifacts, reducing final size

## Production Deployment

### Image Size Optimization

```bash
# Check image size
docker images molprop:latest

# Build optimized image
docker build --target runtime -t molprop:prod .
```

### Security Best Practices

1. **Use non-root user:**
   ```dockerfile
   RUN useradd -m -u 1000 molprop
   USER molprop
   ```

2. **Keep base image updated:**
   ```dockerfile
   FROM python:3.11-slim-bullseye
   RUN apt-get update && apt-get upgrade -y
   ```

3. **Scan for vulnerabilities:**
   ```bash
   # Using Docker Scout
   docker scout cves molprop:latest

   # Using Trivy
   trivy image molprop:latest
   ```

4. **Use secrets for sensitive data:**
   ```bash
   docker run --secret api_key --secret db_password \
     molprop:latest
   ```

### Resource Limits

Set memory and CPU limits in production:

```bash
docker run -m 4g --cpus 2 \
  -p 8000:8000 \
  molprop:latest
```

Or in docker-compose.yml:

```yaml
services:
  api:
    image: molprop:latest
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Debugging

### Access container shell

```bash
# Using docker
docker exec -it <container_id> /bin/bash

# Using docker compose
docker compose exec api bash
```

### View logs

```bash
# Last 100 lines
docker logs -n 100 <container_id>

# Follow logs in real-time
docker logs -f <container_id>

# With timestamps
docker logs --timestamps <container_id>

# Specific time range
docker logs --since 2024-01-15T10:00:00 <container_id>
```

### Health checks

```bash
# Define health check in docker-compose.yml
services:
  api:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

# Check health status
docker ps --format "table {{.Names}}\t{{.Status}}"
```

## Testing

### Build and test locally

```bash
# Build image
docker build -t molprop:test .

# Run tests inside container
docker run --rm molprop:test pytest tests/

# Run with coverage
docker run --rm -v $(pwd):/app molprop:test \
  pytest tests/ --cov=molprop --cov-report=html
```

### Integration tests

```bash
# Start services
docker compose up -d

# Wait for API to be ready
sleep 5

# Run integration tests
docker compose exec api pytest tests/test_api_integration.py -v

# Clean up
docker compose down
```

## Kubernetes Deployment

### Create Kubernetes manifests

Save as `k8s/api.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: molprop-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: molprop-api
  template:
    metadata:
      labels:
        app: molprop-api
    spec:
      containers:
      - name: api
        image: molprop:latest
        ports:
        - containerPort: 8000
        env:
        - name: API_WORKERS
          value: "4"
        - name: MODEL_TYPE
          value: "gcn"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: molprop-api
spec:
  selector:
    app: molprop-api
  type: LoadBalancer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

### Deploy to Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/api.yaml

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get svc

# View logs
kubectl logs deployment/molprop-api

# Scale deployments
kubectl scale deployment molprop-api --replicas=5
```

## Performance Tuning

### Enable GPU acceleration

```bash
# Dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 as base
# ... rest of Dockerfile

# Run with GPU
docker run --gpus all -p 8000:8000 molprop:gpu
```

### Optimize number of workers

```bash
# For CPU-bound workload
API_WORKERS=<number_of_cores>

# For I/O-bound workload
API_WORKERS=<number_of_cores * 2>
```

### Enable caching

```bash
docker compose -f docker-compose.yml -f docker-compose.cache.yml up -d
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs <container_id>

# Run with interactive terminal
docker run -it molprop:latest /bin/bash

# Check environment
docker run -it molprop:latest env
```

### Out of memory errors

```bash
# Increase memory limit
docker run -m 8g molprop:latest

# Check memory usage
docker stats <container_id>
```

### Port already in use

```bash
# Find what's using port 8000
lsof -i :8000

# Use different port
docker run -p 8001:8000 molprop:latest
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/docker.yml
name: Docker Build and Push

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v2
      - uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      - uses: docker/build-push-action@v4
        with:
          push: true
          tags: ${{ secrets.DOCKER_USERNAME }}/molprop:latest
          cache-from: type=registry,ref=${{ secrets.DOCKER_USERNAME }}/molprop:buildcache
          cache-to: type=registry,ref=${{ secrets.DOCKER_USERNAME }}/molprop:buildcache,mode=max
```

## References

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Best Practices for Python Docker Images](https://docs.docker.com/language/python/build-images/)
