# tests/conftest.py (shared test fixtures)
import pytest
import tempfile
from dentalvision.config import Config

@pytest.fixture
def sample_config():
    return Config()

@pytest.fixture  
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir