# Troubleshooting Guide

Common issues and solutions for molprop.

## Installation Issues

### ImportError: No module named 'molprop'

**Problem:** Python can't find the molprop package.

**Solutions:**
```bash
# Install in development mode
pip install -e .

# Or install from source
pip install -e . --upgrade

# Verify installation
python -c "import molprop; print(molprop.__version__)"
```

### Module not found: 'rdkit'

**Problem:** RDKit dependency not installed.

**Solutions:**
```bash
# Install RDKit (may require conda)
conda install -c conda-forge rdkit

# Or via pip
pip install rdkit

# Verify
python -c "from rdkit import Chem; print('RDKit OK')"
```

### CUDA/GPU Issues

**Problem:** GPU not detected or CUDA errors.

**Solutions:**
```bash
# Check CUDA installation
nvidia-smi

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
python script.py
```

## Runtime Issues

### ConnectionError: Failed to connect to localhost:8000

**Problem:** Can't connect to API server.

**Solutions:**
```bash
# Check if server is running
curl http://localhost:8000/health

# Start the server
molprop-serve

# Or manually
python -m uvicorn molprop.serving.api:app --port 8000

# Check port usage
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows
```

### InvalidSMILESError: Invalid SMILES

**Problem:** SMILES string is invalid or malformed.

**Solutions:**
```python
from molprop.client import MolpropClient

client = MolpropClient()

# Valid SMILES examples
valid_smiles = [
    "CCO",           # Ethanol
    "CCN",           # Ethylamine
    "c1ccccc1",      # Benzene
    "CC(C)O",        # Isopropanol
    "C1=CC=CC=C1",   # Benzene (alternative)
]

# Common mistakes to avoid
invalid_smiles = [
    "INVALID",       # Not a valid SMILES
    "CC(O",          # Unbalanced parentheses
    "c1ccccc",       # Ring not closed
]

# Test SMILES
for smiles in valid_smiles:
    result = client.predict(smiles)
    print(f"{smiles}: OK")
```

### Rate limit exceeded

**Problem:** Too many requests error.

**Solutions:**
```bash
# Check rate limit settings
export RATE_LIMIT_CAPACITY=200  # Increase limit
export RATE_LIMIT_WINDOW=60     # Adjust window

# Implement client-side throttling
import time
from molprop.client import MolpropClient

client = MolpropClient()
smiles_list = [...]

results = []
for smiles in smiles_list:
    try:
        result = client.predict(smiles)
        results.append(result)
    except Exception as e:
        if "rate limit" in str(e).lower():
            print("Rate limited, waiting...")
            time.sleep(5)
            # Retry
            result = client.predict(smiles)
            results.append(result)
```

### Memory errors

**Problem:** Out of memory (OOM) errors.

**Solutions:**
```python
# Reduce batch size
batch_size = 16  # Default is 32

result = client.predict_batch(smiles_list, batch_size=batch_size)

# Process in smaller chunks
def process_large_dataset(smiles_list, chunk_size=100):
    results = []
    for i in range(0, len(smiles_list), chunk_size):
        chunk = smiles_list[i:i+chunk_size]
        chunk_result = client.predict_batch(chunk)
        results.extend(chunk_result["predictions"])
    return results
```

## Performance Issues

### Slow predictions

**Problem:** API responses are slow.

**Solutions:**
```bash
# Check system resources
# CPU/memory usage
top -p $(pgrep -f uvicorn)

# Increase workers
export API_WORKERS=8  # Increase from default 4

# Enable caching
export CACHE_ENABLED=true
export CACHE_MAX_SIZE=1024

# Check batch size
export BATCH_SIZE=64  # Larger batches may be faster
```

```python
# Profile your code
from molprop.utils.performance import profile, get_metrics_collector

@profile
def process_molecules(smiles_list):
    return [predict(s) for s in smiles_list]

process_molecules(["CCO", "CCN", "c1ccccc1"] * 100)

collector = get_metrics_collector()
stats = collector.stats()
print(stats)
```

### Cache not working

**Problem:** Cache hits are low.

**Solutions:**
```bash
# Verify cache is enabled
export CACHE_ENABLED=true

# Check cache size
export CACHE_MAX_SIZE=512

# Monitor cache stats
curl http://localhost:8000/cache/stats

# Clear cache if corrupted
curl -X POST http://localhost:8000/cache/clear
```

## Testing Issues

### Tests failing

**Problem:** Pytest tests are failing.

**Solutions:**
```bash
# Run with verbose output
pytest tests/ -vv

# Run specific test
pytest tests/test_validators.py::TestSMILESValidation::test_validate_smiles -vv

# Show print statements
pytest tests/ -s

# Show local variables on failure
pytest tests/ -l

# Stop at first failure
pytest tests/ -x
```

### Import errors in tests

**Problem:** Can't import molprop in tests.

**Solutions:**
```bash
# Ensure pythonpath includes src
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Run tests with proper path
python -m pytest tests/

# Or install in development mode
pip install -e .
```

### Fixture not found

**Problem:** Pytest fixture not recognized.

**Solutions:**
```python
# Ensure conftest.py is in tests directory
# tests/conftest.py

import pytest
from molprop.client import MolpropClient

@pytest.fixture
def api_client():
    return MolpropClient("http://localhost:8000")

# Then use in tests
def test_api(api_client):
    result = api_client.health()
    assert result["status"] == "ok"
```

## Docker Issues

### Container fails to start

**Problem:** Docker container exits immediately.

**Solutions:**
```bash
# Check logs
docker logs <container_id>

# Run with interactive terminal
docker run -it molprop:latest bash

# Check Dockerfile for errors
docker build --progress=plain .
```

### Port already in use

**Problem:** Port 8000 is already in use.

**Solutions:**
```bash
# Use different port
docker run -p 8001:8000 molprop:latest

# Find what's using the port
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill the process
kill -9 <PID>
```

### Out of memory in container

**Problem:** Container killed due to memory limit.

**Solutions:**
```bash
# Increase memory limit
docker run -m 4g molprop:latest

# Check current usage
docker stats <container_id>

# Optimize image
docker system prune -a
```

## Database Issues

### Library database errors

**Problem:** Issues with compound library database.

**Solutions:**
```bash
# Check database path
export LIBRARY_DB_PATH=~/.molprop/library.db

# Backup database
cp ~/.molprop/library.db ~/.molprop/library.db.backup

# Reset database
rm ~/.molprop/library.db
# Restart API to recreate

# Check database integrity
python -c "
import sqlite3
db = sqlite3.connect('~/.molprop/library.db')
cursor = db.cursor()
cursor.execute('PRAGMA integrity_check')
print(cursor.fetchone())
"
```

## API Issues

### Invalid API response

**Problem:** API returning unexpected format.

**Solutions:**
```python
# Check API version
client = MolpropClient()
version = client.version()
print(version)

# Check model info
model_info = client.model_info()
print(f"Model: {model_info['model_type']}")
print(f"Task: {model_info['task']}")

# Test basic endpoint
health = client.health()
assert health["status"] == "ok"
```

### CORS errors in browser

**Problem:** JavaScript requests blocked by CORS.

**Solutions:**
```python
# Ensure CORS is enabled in FastAPI app
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Configuration Issues

### Configuration not loaded

**Problem:** Environment variables not being used.

**Solutions:**
```bash
# Verify environment variables are set
echo $MODEL_TYPE
echo $API_PORT

# Check current config
python -c "
from molprop.config import get_config
config = get_config()
print(f'Model: {config.model.type}')
print(f'Port: {config.api.port}')
"

# Set and verify
export API_PORT=9000
python -c "from molprop.config import get_config; print(get_config().api.port)"
```

### Invalid configuration values

**Problem:** Configuration validation failing.

**Solutions:**
```python
from molprop.config import get_config
from molprop.exceptions import ConfigurationError

try:
    config = get_config()
except ConfigurationError as e:
    print(f"Configuration error: {e.message}")
    # Fix the issue and try again
```

## Logging Issues

### Logs not appearing

**Problem:** Log output not visible.

**Solutions:**
```bash
# Set log level to DEBUG
export LOG_LEVEL=DEBUG

# Check log file
export LOG_FILE=app.log
# Run application
python app.py
# Check file
cat app.log
```

### Sensitive data in logs

**Problem:** Passwords/tokens appearing in logs.

**Solutions:**
```python
# Use sanitize_log_data before logging
from molprop.utils.security import sanitize_log_data

sensitive_data = {"username": "user", "password": "secret"}
safe_data = sanitize_log_data(sensitive_data)
log.info("User data: %s", safe_data)
```

## Getting Help

### Debug Information

Collect debug info for support:

```bash
# System info
python --version
pip list | grep molprop
pip list | grep rdkit
pip list | grep torch

# Configuration
python -c "
from molprop.config import get_config
from molprop import __version__
config = get_config()
print(f'molprop version: {__version__}')
print(f'Model: {config.model.type}')
print(f'Task: {config.model.task}')
"

# Save debug info
python -c "
import sys, platform
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
" > debug_info.txt
```

### Report Issues

When reporting bugs, include:

1. **Error message** - Full traceback
2. **Steps to reproduce** - Minimal example
3. **Environment** - OS, Python version, package versions
4. **Debug info** - Output from commands above
5. **Logs** - Relevant log output (sanitize sensitive data)

Example issue report:

```
**Title:** InvalidSMILESError on valid SMILES "c1ccccc1"

**Description:**
Getting InvalidSMILESError when predicting on benzene.

**Steps to reproduce:**
```python
from molprop.client import MolpropClient
client = MolpropClient()
result = client.predict("c1ccccc1")
```

**Error:**
```
InvalidSMILESError: Invalid SMILES: 'c1ccccc1'
```

**Environment:**
- Python 3.11.0
- molprop 2.45.0
- rdkit 2023.9.5

**Debug info:**
[output from debug commands]
```

## Additional Resources

- **Documentation:** See [docs/](docs/)
- **Examples:** Check [examples/](examples/)
- **Tests:** Reference [tests/](tests/)
- **API Reference:** Read [docs/API.md](docs/API.md)
- **Advanced Guide:** See [ADVANCED.md](ADVANCED.md)
