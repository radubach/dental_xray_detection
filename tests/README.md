# Tests for dental_xray_detection

This directory contains comprehensive tests for the `dentalvision.data.datasets.base` module.

## Test Files

All dataset-related tests are now in the `tests/datasets/` subdirectory:

- `datasets/test_datasets_base.py` - Main test file covering BaseDataset and COCODataset classes
- `datasets/test_simple.py` - Simple tests to verify the testing setup works
- `datasets/test_dataset_edge_cases.py` - Edge cases and error scenarios
- `datasets/test_datasets_comprehensive.py` - Comprehensive test coverage

Other files:

- `run_tests.py` - Test runner script
- `requirements-test.txt` - Testing dependencies
- `README.md` - This file

## Running Tests

### Prerequisites

Install the testing dependencies:

```bash
pip install -r tests/requirements-test.txt
```

### Running All Tests

```bash
# Using the test runner script
python tests/run_tests.py

# Using unittest directly (from project root)
python -m unittest discover tests

# Using pytest (if installed)
pytest tests/
```

### Running Specific Test Files

```bash
# Run only the simple tests
python -m unittest tests.datasets.test_simple

# Run only the main dataset tests
python -m unittest tests.datasets.test_datasets_base

# Run with verbose output
python -m unittest tests.datasets.test_simple -v
```

### Running Individual Test Classes

```bash
# Run only BaseDataset tests
python -m unittest tests.datasets.test_datasets_base.TestBaseDataset

# Run only COCODataset tests
python -m unittest tests.datasets.test_datasets_base.TestCOCODataset
```

## Test Coverage

The tests cover:

### BaseDataset Class

- ✅ Initialization with different splits (train/val/test)
- ✅ Configuration validation
- ✅ Path validation helpers
- ✅ Image loading functionality
- ✅ Dataset information retrieval
- ✅ Abstract method enforcement

### COCODataset Class

- ✅ Initialization for different task types (teeth/disease/quadrant)
- ✅ Multi-category annotation handling
- ✅ Standard COCO format support
- ✅ Image loading and annotation retrieval
- ✅ Unified category ID calculation
- ✅ Error handling for missing dependencies
- ✅ Dataset validation

### Edge Cases

- ✅ Missing or corrupted files
- ✅ Invalid configurations
- ✅ Empty datasets
- ✅ Missing category information
- ✅ Image dimension mismatches
- ✅ Invalid task types

### Utility Functions

- ✅ Collate function testing
- ✅ Error handling scenarios

## Mocking Strategy

The tests use extensive mocking to avoid requiring actual data files:

- **COCO Library**: Mocked to avoid requiring `pycocotools`
- **File System**: Mocked to avoid requiring actual image files
- **PIL Image**: Mocked for image loading tests
- **Configuration**: Uses test configurations

## Adding New Tests

When adding new tests:

1. Place new dataset-related tests in `tests/datasets/`
2. Follow the existing naming convention: `test_*.py`
3. Use descriptive test method names
4. Include docstrings explaining what each test does
5. Use appropriate mocking to avoid external dependencies
6. Test both success and failure scenarios
7. Include edge cases and error conditions

## Example Test Structure

```python
class TestNewFeature(unittest.TestCase):
    """Test description."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_basic_functionality(self):
        """Test basic functionality."""
        # Test implementation
        pass

    def test_error_handling(self):
        """Test error handling."""
        # Test error scenarios
        pass
```

## Continuous Integration

These tests are designed to run in CI/CD environments without requiring:

- Actual data files
- GPU resources
- External dependencies (beyond standard Python packages)

The tests use mocking extensively to ensure they can run in any environment.
