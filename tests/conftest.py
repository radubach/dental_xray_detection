# tests/conftest.py
# Shared test fixtures and configuration

import pytest
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import shutil

# Import your project modules
from dentalvision.config import Config


# =============================================================================
# Path and File Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def tests_dir() -> Path:
    """Path to the tests directory."""
    return Path(__file__).parent


@pytest.fixture(scope="session") 
def sample_data_dir(tests_dir) -> Path:
    """Path to sample data directory."""
    sample_dir = tests_dir / "sample_data"
    if not sample_dir.exists():
        pytest.skip("Sample data not found. Run scripts/create_sample_data.py first.")
    return sample_dir


@pytest.fixture(scope="session")
def sample_images_dir(sample_data_dir) -> Path:
    """Path to sample images directory."""
    return sample_data_dir / "images"


@pytest.fixture(scope="session")
def sample_annotations_file(sample_data_dir) -> Path:
    """Path to sample annotations JSON file."""
    annotations_file = sample_data_dir / "annotations.json"
    if not annotations_file.exists():
        pytest.skip("Sample annotations not found.")
    return annotations_file


# =============================================================================
# Data Loading Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def sample_annotations_data(sample_annotations_file) -> Dict[str, Any]:
    """Load sample annotations JSON data."""
    with open(sample_annotations_file, 'r') as f:
        return json.load(f)


@pytest.fixture(scope="session")
def sample_images_list(sample_annotations_data) -> List[Dict[str, Any]]:
    """List of sample image metadata."""
    return sample_annotations_data["images"]


@pytest.fixture(scope="session")
def sample_annotations_list(sample_annotations_data) -> List[Dict[str, Any]]:
    """List of sample annotations."""
    return sample_annotations_data["annotations"]


@pytest.fixture(scope="session")
def sample_categories(sample_annotations_data) -> Dict[str, List[Dict[str, Any]]]:
    """All category information from sample data."""
    return {
        "categories": sample_annotations_data.get("categories", []),
        "categories_1": sample_annotations_data.get("categories_1", []),  # Quadrants
        "categories_2": sample_annotations_data.get("categories_2", []),  # Tooth numbers
        "categories_3": sample_annotations_data.get("categories_3", [])   # Diseases
    }


# =============================================================================
# Individual Sample Fixtures
# =============================================================================

@pytest.fixture
def sample_image_path(sample_images_dir, sample_images_list) -> Path:
    """Path to a single sample image (first one)."""
    if not sample_images_list:
        pytest.skip("No sample images available.")
    first_image = sample_images_list[0]
    image_path = sample_images_dir / first_image["file_name"]
    if not image_path.exists():
        pytest.skip(f"Sample image not found: {image_path}")
    return image_path


@pytest.fixture
def sample_image_info(sample_images_list) -> Dict[str, Any]:
    """Metadata for a single sample image."""
    if not sample_images_list:
        pytest.skip("No sample images available.")
    return sample_images_list[0]


@pytest.fixture
def sample_annotation(sample_annotations_list) -> Dict[str, Any]:
    """A single sample annotation."""
    if not sample_annotations_list:
        pytest.skip("No sample annotations available.")
    return sample_annotations_list[0]


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def test_config(sample_data_dir) -> Config:
    """Test configuration that uses sample data."""
    return Config.create_test_config(str(sample_data_dir))


@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# =============================================================================
# Dataset Fixtures (for your specific use cases)
# =============================================================================

@pytest.fixture
def tooth_dataset_config(test_config, sample_data_dir):
    """Configuration for tooth detection dataset tests."""
    # You can modify Config for testing if needed
    config = test_config
    # Override paths to use sample data
    config.annotation_file = sample_data_dir / "annotations.json"
    config.image_dir = sample_data_dir / "images"
    return config


@pytest.fixture 
def disease_dataset_config(test_config, sample_data_dir):
    """Configuration for disease classification dataset tests."""
    config = test_config
    config.annotation_file = sample_data_dir / "annotations.json"
    config.image_dir = sample_data_dir / "images"
    return config


# =============================================================================
# Data Validation Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def sample_data_stats(sample_annotations_data) -> Dict[str, int]:
    """Basic statistics about sample data."""
    return {
        "num_images": len(sample_annotations_data.get("images", [])),
        "num_annotations": len(sample_annotations_data.get("annotations", [])),
        "num_quadrant_categories": len(sample_annotations_data.get("categories_1", [])),
        "num_tooth_categories": len(sample_annotations_data.get("categories_2", [])),
        "num_disease_categories": len(sample_annotations_data.get("categories_3", []))
    }


@pytest.fixture(autouse=True, scope="session")
def validate_sample_data(sample_data_stats):
    """Automatically validate sample data before running tests."""
    stats = sample_data_stats
    
    # Basic validation
    assert stats["num_images"] > 0, "No sample images found"
    assert stats["num_annotations"] > 0, "No sample annotations found"
    
    # Check that we have the expected categories
    assert stats["num_quadrant_categories"] > 0, "No quadrant categories found"
    assert stats["num_tooth_categories"] > 0, "No tooth categories found"
    assert stats["num_disease_categories"] > 0, "No disease categories found"
    
    print(f"✅ Sample data validation passed: {stats}")


# =============================================================================
# Pytest Hooks (optional customizations)
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers if they're not already defined
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests") 
    config.addinivalue_line("markers", "data: Tests requiring sample data")


def pytest_collection_modifyitems(config, items):
    """Modify test collection - add markers automatically."""
    for item in items:
        # Auto-mark tests that use sample data fixtures
        if any(fixture in item.fixturenames for fixture in [
            "sample_data_dir", "sample_annotations_file", "sample_images_dir"
        ]):
            item.add_marker(pytest.mark.data)
        
        # Auto-mark slow tests (you can customize this logic)
        if "comprehensive" in item.nodeid or "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)


# =============================================================================
# Custom Assertions (optional)
# =============================================================================

def assert_valid_coco_format(annotation_data):
    """Custom assertion for validating COCO format (supports multi-category)."""
    required_keys = ["images", "annotations", "categories"]
    for key in required_keys:
        assert key in annotation_data, f"Missing required key: {key}"
    
    # Validate images format
    for img in annotation_data["images"]:
        assert "id" in img, "Image missing ID"
        assert "file_name" in img, "Image missing file_name"
        assert "width" in img, "Image missing width"
        assert "height" in img, "Image missing height"
    
    # Validate annotations format  
    for ann in annotation_data["annotations"]:
        assert "id" in ann, "Annotation missing ID"
        assert "image_id" in ann, "Annotation missing image_id"
        # Accept either standard COCO or multi-category format
        if "category_id" not in ann:
            assert all(k in ann for k in ["category_id_1", "category_id_2", "category_id_3"]), (
                "Annotation missing category_id or multi-category keys (category_id_1, category_id_2, category_id_3)"
            )


# Make custom assertions available to all tests
pytest.assert_valid_coco_format = assert_valid_coco_format