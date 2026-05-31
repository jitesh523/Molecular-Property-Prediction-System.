# Advanced Usage Guide

This guide covers advanced topics for power users and developers.

## Performance Optimization

### Using Performance Monitoring

Monitor function execution times:

```python
from molprop.utils.performance import profile, performance_threshold, measure_time

# Method 1: Using @profile decorator
@profile(collect_metrics=True)
def process_molecules(smiles_list):
    return [validate_smiles(s) for s in smiles_list]

# Method 2: Using @performance_threshold decorator
@performance_threshold(threshold_ms=1000)
def critical_operation():
    pass

# Method 3: Using measure_time context manager
from molprop.utils.performance import get_metrics_collector

with measure_time("custom_operation"):
    result = expensive_operation()

# Get metrics
collector = get_metrics_collector()
stats = collector.stats()
print(f"Total operations: {stats['total_operations']}")
for name, metrics in stats['metrics'].items():
    print(f"{name}: avg={metrics['avg_time_ms']:.1f}ms, max={metrics['max_time_ms']:.1f}ms")
```

### Batch Processing

Use batch processors for efficient concurrent processing:

```python
from molprop.utils.async_utils import ThreadPoolBatchProcessor

# Process items concurrently
processor = ThreadPoolBatchProcessor(batch_size=32, max_workers=4)
smiles_list = ["CCO", "CCN", "c1ccccc1", ...]

results = processor.process(smiles_list, process_molecule)
```

### Async Operations

For async-native code:

```python
import asyncio
from molprop.utils.async_utils import AsyncBatchProcessor, async_retry

# Batch process with async
processor = AsyncBatchProcessor(batch_size=32, max_concurrent=4)

async def predict_batch(smiles_batch):
    # Make async predictions
    pass

results = await processor.process(smiles_list, predict_batch)

# Retry with exponential backoff
async def fetch_data():
    return await async_retry(
        lambda: api_call(),
        max_attempts=3,
        delay_seconds=1.0,
        backoff_factor=2.0,
    )
```

## Security

### Input Sanitization

Always sanitize user input:

```python
from molprop.utils.security import sanitize_string, safe_filename

# Sanitize string input
user_input = sanitize_string(request.get("query"), max_length=1000)

# Create safe filenames
filename = safe_filename(user_uploaded_name)
```

### Password Management

Securely handle passwords:

```python
from molprop.utils.security import hash_password, verify_password

# Create user account
password = "user_password"
hashed, salt = hash_password(password)
# Store hashed and salt in database

# Verify login
if verify_password(user_input_password, stored_hash, stored_salt):
    print("Login successful")
```

### API Key Generation

Generate and validate API keys:

```python
from molprop.utils.security import generate_api_key, validate_api_key

# Generate for new user
api_key = generate_api_key(prefix="sk", length=32)

# Validate incoming keys
if validate_api_key(request_api_key):
    process_request()
```

### Rate Limiting Keys

Generate rate limit keys:

```python
from molprop.utils.security import rate_limit_key

# Generate key for rate limit check
key = rate_limit_key(client_id, operation="predict", window=60)

# Use with cache/Redis
if cache.get(key):
    # Rate limit exceeded
    pass
```

## Response Formatting

### Standardized Responses

Use consistent response formatting:

```python
from molprop.utils.response import success_response, error_response, paginated_response

# Success response
@app.get("/predictions")
def get_predictions():
    results = db.query_predictions()
    return success_response(results, message="Predictions retrieved")

# Error response
@app.post("/predict")
def predict(smiles: str):
    try:
        result = model.predict(smiles)
        return success_response({"prediction": result})
    except ValidationError as e:
        return error_response(e, status_code=400)

# Paginated response
@app.get("/molecules")
def list_molecules(page: int = 1, page_size: int = 20):
    molecules = db.get_molecules(page, page_size)
    total = db.count_molecules()
    return paginated_response(molecules, total, page, page_size)

# Batch response
@app.post("/predict/batch")
def batch_predict(smiles_list: List[str]):
    results = []
    errors = []
    
    for i, smiles in enumerate(smiles_list):
        try:
            pred = model.predict(smiles)
            results.append({"smiles": smiles, "prediction": pred})
        except Exception as e:
            errors.append((i, str(e)))
    
    return batch_response(results, errors=errors)
```

### JSON Serialization

Handle complex types:

```python
from molprop.utils.response import to_json, from_json
from datetime import datetime
from decimal import Decimal
from uuid import UUID

# Serialize complex data
data = {
    "id": UUID("12345678-1234-5678-1234-567812345678"),
    "timestamp": datetime.now(),
    "score": Decimal("0.95"),
    "items": [1, 2, 3],
}

json_str = to_json(data, pretty=True)
print(json_str)

# Deserialize back
recovered = from_json(json_str)
```

## Logging & Debugging

### Structured Logging

Use structured logging:

```python
from molprop.logger import get_logger

log = get_logger(__name__)

# Debug-level logging for development
log.debug("Processing molecule: %s", smiles)

# Info-level for important events
log.info("Successfully processed %d molecules", count)

# Warning-level for recoverable issues
log.warning("Deprecated function used: use new_function instead")

# Error-level with exception info
try:
    process()
except Exception as e:
    log.error("Processing failed", exc_info=True)
```

### Sensitive Data Sanitization

Prevent sensitive data in logs:

```python
from molprop.utils.security import sanitize_log_data

user_data = {
    "username": "john",
    "password": "secret123",
    "api_key": "sk_123456",
    "profile": {"email": "john@example.com"}
}

# Sanitize before logging
safe_data = sanitize_log_data(user_data)
log.info("User operation: %s", safe_data)
# Output: {'username': 'john', 'password': '***REDACTED***', 'api_key': '***REDACTED***', ...}
```

## Configuration Management

### Dynamic Configuration

Access configuration throughout your code:

```python
from molprop.config import get_config

config = get_config()

# Access different config sections
print(config.api.port)  # 8000
print(config.model.type)  # 'gcn'
print(config.cache.enabled)  # True

# Use configuration in functions
def create_database():
    db_path = config.database.library_db_path
    return create_connection(db_path)
```

### Environment-Specific Configuration

Set configuration via environment variables:

```bash
# Development
export LOG_LEVEL=DEBUG
export CACHE_ENABLED=true
export API_DEBUG=true

# Production
export LOG_LEVEL=WARNING
export CACHE_ENABLED=true
export CACHE_MAX_SIZE=2048
export API_WORKERS=8
export RATE_LIMIT_CAPACITY=1000
```

## Testing

### Testing with Fixtures

Use pytest fixtures for testing:

```python
import pytest
from molprop.client import MolpropClient

@pytest.fixture
def api_client():
    """Provide API client for tests."""
    return MolpropClient("http://localhost:8000")

@pytest.fixture
def sample_molecules():
    """Provide test molecules."""
    return {
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "benzene": "c1ccccc1",
    }

def test_predict_aspirin(api_client, sample_molecules):
    """Test prediction on aspirin."""
    result = api_client.predict(sample_molecules["aspirin"])
    assert "prediction" in result
    assert 0 <= result["prediction"] <= 1

def test_batch_predict(api_client, sample_molecules):
    """Test batch prediction."""
    smiles_list = list(sample_molecules.values())
    results = api_client.predict_batch(smiles_list)
    assert len(results["predictions"]) == len(smiles_list)
```

### Performance Testing

Benchmark critical operations:

```python
import pytest
from molprop.utils.performance import profile

@profile(collect_metrics=True)
def slow_operation():
    result = expensive_computation()
    return result

def test_performance_acceptable():
    """Performance should meet threshold."""
    from molprop.utils.performance import get_metrics_collector
    
    collector = get_metrics_collector()
    
    for _ in range(100):
        slow_operation()
    
    metrics = collector.get_metrics("slow_operation")
    assert metrics.avg_time_ms < 100  # Average < 100ms
    assert metrics.max_time_ms < 500  # Max < 500ms
```

## Production Deployment

### Health Checks

Implement custom health checks:

```python
from molprop.config import get_config

def health_check() -> dict:
    """Comprehensive health check."""
    config = get_config()
    
    checks = {
        "api": check_api_status(),
        "model": check_model_loaded(),
        "cache": check_cache_connection(),
        "database": check_database_connection(),
    }
    
    is_healthy = all(checks.values())
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "checks": checks,
        "config": {
            "model": config.model.type,
            "cache_enabled": config.cache.enabled,
        }
    }
```

### Metrics Collection

Export metrics for monitoring:

```python
from molprop.utils.performance import get_metrics_collector

def export_metrics() -> dict:
    """Export metrics for Prometheus/monitoring."""
    collector = get_metrics_collector()
    stats = collector.stats()
    
    return {
        "molprop_operations_total": stats["total_operations"],
        "molprop_errors_total": stats["total_errors"],
        "molprop_processing_time_ms": stats["total_time_ms"],
        "molprop_operation_avg_time_ms": {
            name: m["avg_time_ms"] 
            for name, m in stats["metrics"].items()
        },
    }
```

## Troubleshooting

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
export API_DEBUG=true
```

```python
from molprop.config import get_config

config = get_config()
if config.api.debug:
    log.debug("Debug mode enabled")
    # Additional debug logging
```

### Performance Debugging

Identify slow operations:

```python
from molprop.utils.performance import get_metrics_collector

collector = get_metrics_collector()
stats = collector.stats()

# Find slowest operations
slowest = sorted(
    stats["metrics"].items(),
    key=lambda x: x[1]["avg_time_ms"],
    reverse=True
)

for name, metrics in slowest[:5]:
    print(f"{name}: avg={metrics['avg_time_ms']:.1f}ms")
```

### Memory Profiling

Profile memory usage:

```bash
pip install memory-profiler
python -m memory_profiler script.py
```

```python
from memory_profiler import profile

@profile
def process_large_dataset():
    molecules = load_molecules()
    results = [predict(mol) for mol in molecules]
    return results
```

## Best Practices

### Error Handling

Always use specific exceptions:

```python
from molprop.exceptions import (
    InvalidSMILESError,
    ValidationError,
    ModelLoadError,
)

try:
    smiles = validate_smiles(user_input)
    predictions = model.predict(smiles)
except InvalidSMILESError as e:
    log.error(f"Invalid SMILES: {e.smiles}")
    return error_response(e, status_code=400)
except ValidationError as e:
    return error_response(e, status_code=400)
except ModelLoadError as e:
    return error_response(e, status_code=503)
```

### Resource Management

Use context managers:

```python
from contextlib import contextmanager

@contextmanager
def database_session():
    """Manage database connection lifecycle."""
    db = connect_database()
    try:
        yield db
    finally:
        db.close()

# Usage
with database_session() as db:
    results = db.query("SELECT * FROM molecules")
```

### Caching Strategy

Implement intelligent caching:

```python
from functools import lru_cache
from molprop.utils.performance import cache_stats

@cache_stats
@lru_cache(maxsize=1000)
def expensive_calculation(param):
    """Cache results of expensive operations."""
    return complex_computation(param)

# Monitor cache effectiveness
if expensive_calculation.cache_info().hits / (expensive_calculation.cache_info().hits + expensive_calculation.cache_info().misses) < 0.5:
    log.warning("Cache hit rate is low, consider optimizing")
```
