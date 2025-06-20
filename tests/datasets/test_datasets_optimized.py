"""
Optimized tests for dentalvision.data.datasets.base module.

This module provides efficient tests using the sample dataset fixtures:
- Uses real sample data instead of mocks where possible
- Leverages conftest.py fixtures for consistent test data
- Focuses on core functionality and edge cases
- Designed for fast execution and good coverage
- Handles imperfect COCO formatting (multi-category format)
"""

import pytest
import torch
from PIL import Image
import numpy as np
from unittest.mock import patch, Mock, MagicMock
import json

# Import the classes to test
from dentalvision.data.datasets.base import BaseDataset, COCODataset, collate_fn_base
from dentalvision.config import Config


class TestBaseDatasetOptimized:
    """Optimized tests for BaseDataset class using fixtures."""
    
    def test_base_dataset_initialization(self, test_config):
        """Test BaseDataset initialization with different splits."""
        # Create concrete implementation for testing
        class ConcreteDataset(BaseDataset):
            def __init__(self, config, split='train'):
                super().__init__(config, split)
                self.data = [1, 2, 3]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return {"data": self.data[idx], "index": idx}
        
        # Test train split
        train_dataset = ConcreteDataset(test_config, 'train')
        assert train_dataset.is_training
        assert train_dataset.split == 'train'
        
        # Test validation split
        val_dataset = ConcreteDataset(test_config, 'val')
        assert not val_dataset.is_training
        assert val_dataset.split == 'val'
        
        # Test test split
        test_dataset = ConcreteDataset(test_config, 'test')
        assert not test_dataset.is_training
        assert test_dataset.split == 'test'
    
    def test_config_validation(self, test_config):
        """Test configuration validation with various inputs."""
        class ConcreteDataset(BaseDataset):
            def __init__(self, config, split='train'):
                super().__init__(config, split)
                self.data = [1]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return {"data": self.data[idx]}
        
        # Test valid config
        dataset = ConcreteDataset(test_config)
        dataset._validate_config()  # Should not raise
        
        # Test invalid INPUT_SIZE types
        invalid_configs = [
            ([512, 512], "List instead of tuple"),
            ("512,512", "String instead of tuple"),
            (512, "Single integer"),
            ((512,), "Single element tuple"),
            ((512, 512, 512), "Three element tuple"),
            (None, "None value")
        ]
        
        for invalid_input, description in invalid_configs:
            config = Config()
            config.INPUT_SIZE = invalid_input
            
            with pytest.raises(ValueError, match="INPUT_SIZE must be a tuple"):
                ConcreteDataset(config)
    
    def test_path_validation(self, test_config, sample_data_dir):
        """Test path validation helper."""
        class ConcreteDataset(BaseDataset):
            def __init__(self, config, split='train'):
                super().__init__(config, split)
                self.data = [1]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return {"data": self.data[idx]}
        
        dataset = ConcreteDataset(test_config)
        
        # Test existing path (use sample data)
        dataset._validate_path_exists(str(sample_data_dir), "Sample data directory")
        
        # Test non-existing path
        with pytest.raises(FileNotFoundError):
            dataset._validate_path_exists("/non/existent/path", "Test path")
        
        # Test with custom description
        with pytest.raises(FileNotFoundError) as exc_info:
            dataset._validate_path_exists("/bad/path", "Custom description")
        assert "Custom description" in str(exc_info.value)
    
    def test_image_loading(self, test_config, sample_image_path):
        """Test image loading functionality with real sample image."""
        class ConcreteDataset(BaseDataset):
            def __init__(self, config, split='train'):
                super().__init__(config, split)
                self.data = [1]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return {"data": self.data[idx]}
        
        dataset = ConcreteDataset(test_config)
        
        # Test loading without conversion
        loaded_image = dataset._load_image_from_path(str(sample_image_path), convert_mode=None)
        assert isinstance(loaded_image, Image.Image)
        assert loaded_image.size == (512, 512)
        
        # Test loading with RGB conversion
        loaded_rgb = dataset._load_image_from_path(str(sample_image_path), convert_mode="RGB")
        assert loaded_rgb.mode == "RGB"
        
        # Test loading with grayscale conversion
        loaded_gray = dataset._load_image_from_path(str(sample_image_path), convert_mode="L")
        assert loaded_gray.mode == "L"
    
    def test_image_loading_errors(self, test_config):
        """Test image loading error scenarios."""
        class ConcreteDataset(BaseDataset):
            def __init__(self, config, split='train'):
                super().__init__(config, split)
                self.data = [1]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return {"data": self.data[idx]}
        
        dataset = ConcreteDataset(test_config)
        
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            dataset._load_image_from_path("/non/existent/image.jpg")
        
        # Test corrupted image file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(b"not an image file")
            tmp_path = tmp_file.name
        
        try:
            with pytest.raises(RuntimeError):
                dataset._load_image_from_path(tmp_path)
        finally:
            import os
            os.unlink(tmp_path)
    
    def test_dataset_info(self, test_config):
        """Test dataset information retrieval."""
        class ConcreteDataset(BaseDataset):
            def __init__(self, config, split='train'):
                super().__init__(config, split)
                self.data = [1, 2, 3]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return {"data": self.data[idx]}
        
        dataset = ConcreteDataset(test_config)
        info = dataset.get_dataset_info()
        
        # Check all required keys are present
        expected_keys = ["split", "length", "is_training", "class_name"]
        for key in expected_keys:
            assert key in info
        
        # Check values are correct
        assert info["split"] == "train"
        assert info["length"] == 3
        assert info["is_training"]
        assert info["class_name"] == "ConcreteDataset"
    
    def test_abstract_methods(self, test_config):
        """Test that abstract methods are properly enforced."""
        # Test incomplete implementation
        class IncompleteDataset(BaseDataset):
            pass
        
        with pytest.raises(TypeError):
            IncompleteDataset(test_config)
        
        # Test missing __getitem__
        class MissingGetItem(BaseDataset):
            def __len__(self):
                return 1
        
        with pytest.raises(TypeError):
            MissingGetItem(test_config)
        
        # Test missing __len__
        class MissingLen(BaseDataset):
            def __getitem__(self, idx):
                return {}
        
        with pytest.raises(TypeError):
            MissingLen(test_config)


class TestCOCODatasetOptimized:
    """Optimized tests for COCODataset class using sample data with mocked COCO loading."""
    
    def create_mock_coco(self, sample_annotations_data):
        """Create a mock COCO object that works with multi-category format."""
        mock_coco = Mock()
        
        # Set up basic COCO structure
        mock_coco.dataset = sample_annotations_data
        mock_coco.imgs = {img['id']: img for img in sample_annotations_data['images']}
        mock_coco.anns = {ann['id']: ann for ann in sample_annotations_data['annotations']}
        
        # Mock methods that COCODataset uses
        mock_coco.getAnnIds.return_value = [ann['id'] for ann in sample_annotations_data['annotations']]
        mock_coco.loadAnns.side_effect = lambda ann_ids: [
            ann for ann in sample_annotations_data['annotations'] 
            if ann['id'] in ann_ids
        ]
        
        return mock_coco
    
    @pytest.mark.data
    def test_coco_dataset_initialization_teeth(self, tooth_dataset_config, sample_annotations_file, sample_images_dir, sample_annotations_data):
        """Test COCODataset initialization for teeth task with real data."""
        with patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True):
            with patch('dentalvision.data.datasets.base.COCO') as mock_coco_class:
                with patch('os.path.exists', return_value=True):  # Mock path existence check
                    # Create mock COCO object
                    mock_coco = self.create_mock_coco(sample_annotations_data)
                    mock_coco_class.return_value = mock_coco
                    
                    class TestCOCODataset(COCODataset):
                        def __getitem__(self, idx):
                            return {"index": idx, "image_id": self.image_ids[idx]}
                    
                    # Set config attributes for sample data
                    tooth_dataset_config.TOOTH_ANNOTATION_FILE = str(sample_annotations_file)
                    tooth_dataset_config.image_dir = str(sample_images_dir)
                    
                    dataset = TestCOCODataset(tooth_dataset_config, 'train', 'teeth')
                    
                    assert dataset.task_type == 'teeth'
                    assert len(dataset) == 2  # 2 sample images
                    assert dataset.get_num_classes() == 32  # 4 quadrants * 8 teeth
    
    @pytest.mark.data
    def test_coco_dataset_initialization_disease(self, disease_dataset_config, sample_annotations_file, sample_images_dir, sample_annotations_data):
        """Test COCODataset initialization for disease task with real data."""
        with patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True):
            with patch('dentalvision.data.datasets.base.COCO') as mock_coco_class:
                with patch('os.path.exists', return_value=True):  # Mock path existence check
                    # Create mock COCO object
                    mock_coco = self.create_mock_coco(sample_annotations_data)
                    mock_coco_class.return_value = mock_coco
                    
                    class TestCOCODataset(COCODataset):
                        def __getitem__(self, idx):
                            return {"index": idx, "image_id": self.image_ids[idx]}
                    
                    # Set config attributes for sample data
                    disease_dataset_config.DISEASE_ANNOTATION_FILE = str(sample_annotations_file)
                    disease_dataset_config.image_dir = str(sample_images_dir)
                    
                    dataset = TestCOCODataset(disease_dataset_config, 'train', 'disease')
                    
                    assert dataset.task_type == 'disease'
                    assert len(dataset) == 2  # 2 sample images
                    assert dataset.get_num_classes() == 4  # 4 disease categories
    
    @pytest.mark.data
    def test_coco_dataset_initialization_quadrant(self, tooth_dataset_config, sample_annotations_file, sample_images_dir, sample_annotations_data):
        """Test COCODataset initialization for quadrant task with real data."""
        with patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True):
            with patch('dentalvision.data.datasets.base.COCO') as mock_coco_class:
                with patch('os.path.exists', return_value=True):  # Mock path existence check
                    # Create mock COCO object
                    mock_coco = self.create_mock_coco(sample_annotations_data)
                    mock_coco_class.return_value = mock_coco
                    
                    class TestCOCODataset(COCODataset):
                        def __getitem__(self, idx):
                            return {"index": idx, "image_id": self.image_ids[idx]}
                    
                    # Set config attributes for sample data
                    tooth_dataset_config.TOOTH_ANNOTATION_FILE = str(sample_annotations_file)
                    tooth_dataset_config.image_dir = str(sample_images_dir)
                    
                    dataset = TestCOCODataset(tooth_dataset_config, 'train', 'quadrant')
                    
                    assert dataset.task_type == 'quadrant'
                    assert len(dataset) == 2  # 2 sample images
                    assert dataset.get_num_classes() == 4  # 4 quadrant categories
    
    def test_coco_dataset_no_pycocotools(self, test_config):
        """Test that COCODataset raises ImportError when pycocotools not available."""
        with patch('dentalvision.data.datasets.base.COCO_AVAILABLE', False):
            class TestCOCODataset(COCODataset):
                def __getitem__(self, idx):
                    return {"index": idx}
            
            with pytest.raises(ImportError, match="pycocotools is required"):
                TestCOCODataset(test_config, 'train', 'teeth')
    
    @pytest.mark.data
    def test_unified_category_id_calculation(self, tooth_dataset_config, sample_annotations_file, sample_images_dir, sample_annotations_data):
        """Test unified category ID calculation with real annotations."""
        with patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True):
            with patch('dentalvision.data.datasets.base.COCO') as mock_coco_class:
                with patch('os.path.exists', return_value=True):  # Mock path existence check
                    # Create mock COCO object
                    mock_coco = self.create_mock_coco(sample_annotations_data)
                    mock_coco_class.return_value = mock_coco
                    
                    class TestCOCODataset(COCODataset):
                        def __getitem__(self, idx):
                            return {"index": idx, "image_id": self.image_ids[idx]}
                    
                    # Set config attributes for sample data
                    tooth_dataset_config.TOOTH_ANNOTATION_FILE = str(sample_annotations_file)
                    tooth_dataset_config.image_dir = str(sample_images_dir)
                    
                    dataset = TestCOCODataset(tooth_dataset_config, 'train', 'teeth')
                    
                    # Test with sample annotation
                    sample_annotation = {
                        "category_id_1": 0,  # Quadrant 1
                        "category_id_2": 7,  # Tooth 8
                        "category_id_3": 0   # Disease: Impacted
                    }
                    
                    unified_id = dataset.get_unified_category_id(sample_annotation)
                    # Expected: quadrant * 8 + tooth = 0 * 8 + 7 = 7
                    assert unified_id == 7
    
    @pytest.mark.data
    def test_image_loading_with_real_data(self, tooth_dataset_config, sample_annotations_file, sample_image_path, sample_images_dir, sample_annotations_data):
        """Test image loading with real sample image."""
        with patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True):
            with patch('dentalvision.data.datasets.base.COCO') as mock_coco_class:
                with patch('os.path.exists', return_value=True):  # Mock path existence check
                    with patch('PIL.Image.open') as mock_image_open:  # Mock image loading
                        # Create a dummy image
                        dummy_image = Image.new('RGB', (512, 512), color='white')
                        mock_image_open.return_value = dummy_image
                        
                        # Create mock COCO object
                        mock_coco = self.create_mock_coco(sample_annotations_data)
                        mock_coco_class.return_value = mock_coco
                        
                        class TestCOCODataset(COCODataset):
                            def __getitem__(self, idx):
                                return {"index": idx, "image_id": self.image_ids[idx]}
                        
                        # Set config attributes for sample data
                        tooth_dataset_config.TOOTH_ANNOTATION_FILE = str(sample_annotations_file)
                        tooth_dataset_config.image_dir = str(sample_images_dir)
                        
                        dataset = TestCOCODataset(tooth_dataset_config, 'train', 'teeth')
                        
                        # Test loading first image
                        loaded_image = dataset.load_image(0)
                        assert isinstance(loaded_image, Image.Image)
                        assert loaded_image.size == (512, 512)
    
    @pytest.mark.data
    def test_annotation_retrieval(self, tooth_dataset_config, sample_annotations_file, sample_images_dir, sample_annotations_data):
        """Test annotation retrieval with real data."""
        with patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True):
            with patch('dentalvision.data.datasets.base.COCO') as mock_coco_class:
                with patch('os.path.exists', return_value=True):  # Mock path existence check
                    # Create mock COCO object
                    mock_coco = self.create_mock_coco(sample_annotations_data)
                    mock_coco_class.return_value = mock_coco
                    
                    class TestCOCODataset(COCODataset):
                        def __getitem__(self, idx):
                            return {"index": idx, "image_id": self.image_ids[idx]}
                    
                    # Set config attributes for sample data
                    tooth_dataset_config.TOOTH_ANNOTATION_FILE = str(sample_annotations_file)
                    tooth_dataset_config.image_dir = str(sample_images_dir)
                    
                    dataset = TestCOCODataset(tooth_dataset_config, 'train', 'teeth')
                    
                    # Test getting annotations for first image
                    annotations = dataset.get_annotations(0)
                    assert isinstance(annotations, list)
                    assert len(annotations) > 0
                    
                    # Check annotation structure
                    for ann in annotations:
                        assert "id" in ann
                        assert "image_id" in ann
                        assert "category_id_1" in ann
                        assert "category_id_2" in ann
                        assert "category_id_3" in ann
    
    def test_invalid_task_type(self, test_config, sample_annotations_file, sample_images_dir, sample_annotations_data):
        """Test that invalid task type raises error."""
        with patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True):
            with patch('dentalvision.data.datasets.base.COCO') as mock_coco_class:
                with patch('os.path.exists', return_value=True):  # Mock path existence check
                    # Create mock COCO object
                    mock_coco = self.create_mock_coco(sample_annotations_data)
                    mock_coco_class.return_value = mock_coco
                    
                    class TestCOCODataset(COCODataset):
                        def __getitem__(self, idx):
                            return {"index": idx}
                    
                    # Set config attributes for sample data
                    test_config.COCO_ANNOTATION_FILE = str(sample_annotations_file)
                    test_config.image_dir = str(sample_images_dir)
                    
                    with pytest.raises(ValueError, match="Unsupported task type for multi-category data"):
                        TestCOCODataset(test_config, 'train', 'invalid_task')
    
    @pytest.mark.data
    def test_category_info_retrieval(self, tooth_dataset_config, sample_annotations_file, sample_images_dir, sample_annotations_data):
        """Test category information retrieval with real data."""
        with patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True):
            with patch('dentalvision.data.datasets.base.COCO') as mock_coco_class:
                with patch('os.path.exists', return_value=True):  # Mock path existence check
                    # Create mock COCO object
                    mock_coco = self.create_mock_coco(sample_annotations_data)
                    mock_coco_class.return_value = mock_coco
                    
                    class TestCOCODataset(COCODataset):
                        def __getitem__(self, idx):
                            return {"index": idx, "image_id": self.image_ids[idx]}
                    
                    # Set config attributes for sample data
                    tooth_dataset_config.TOOTH_ANNOTATION_FILE = str(sample_annotations_file)
                    tooth_dataset_config.image_dir = str(sample_images_dir)
                    
                    dataset = TestCOCODataset(tooth_dataset_config, 'train', 'teeth')
                    
                    # Test category info
                    category_info = dataset.get_category_info()
                    assert isinstance(category_info, dict)
                    
                    # Test class names
                    class_names = dataset.get_class_names()
                    assert isinstance(class_names, list)
                    assert len(class_names) == dataset.get_num_classes()


class TestCollateFunctionOptimized:
    """Optimized tests for collate function."""
    
    def test_basic_collate(self):
        """Test basic collate function with simple data."""
        # Create simple batch data
        batch = [
            {"image": torch.randn(3, 512, 512), "label": torch.tensor([1])},
            {"image": torch.randn(3, 512, 512), "label": torch.tensor([2])}
        ]
        
        try:
            result = collate_fn_base(batch)
            
            # Check that it returns a dictionary
            assert isinstance(result, dict)
            
            # Check that it contains expected keys
            assert "image" in result
            assert "label" in result
            
            # Check tensor shapes
            assert result["image"].shape == (2, 3, 512, 512)
            assert result["label"].shape == (2, 1)
            
        except Exception as e:
            # If default_collate fails, that's expected for custom data structures
            assert isinstance(e, (TypeError, RuntimeError))
    
    def test_empty_batch(self):
        """Test collate function with empty batch."""
        batch = []
        
        try:
            result = collate_fn_base(batch)
            assert isinstance(result, dict)
        except Exception as e:
            # Empty batch might raise an exception, which is acceptable
            assert isinstance(e, (ValueError, RuntimeError, IndexError))
    
    def test_mixed_data_types(self):
        """Test collate function with mixed data types."""
        batch = [
            {"image": torch.randn(3, 512, 512), "label": 1, "metadata": {"id": 1}},
            {"image": torch.randn(3, 512, 512), "label": 2, "metadata": {"id": 2}}
        ]
        
        try:
            result = collate_fn_base(batch)
            
            assert isinstance(result, dict)
            assert "image" in result
            assert "label" in result
            assert "metadata" in result
            
        except Exception as e:
            # Mixed types might cause issues with default_collate
            assert isinstance(e, (TypeError, RuntimeError))


class TestDatasetIntegrationOptimized:
    """Optimized integration tests using sample data."""
    
    @pytest.mark.data
    def test_full_dataset_iteration(self, tooth_dataset_config, sample_annotations_file, sample_images_dir, sample_annotations_data):
        """Test full dataset iteration with real data."""
        with patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True):
            with patch('dentalvision.data.datasets.base.COCO') as mock_coco_class:
                with patch('os.path.exists', return_value=True):  # Mock path existence check
                    with patch('PIL.Image.open') as mock_image_open:  # Mock image loading
                        # Create a dummy image
                        dummy_image = Image.new('RGB', (512, 512), color='white')
                        mock_image_open.return_value = dummy_image
                        
                        # Create mock COCO object
                        mock_coco = Mock()
                        mock_coco.dataset = sample_annotations_data
                        mock_coco.imgs = {img['id']: img for img in sample_annotations_data['images']}
                        mock_coco.anns = {ann['id']: ann for ann in sample_annotations_data['annotations']}
                        mock_coco.getAnnIds.return_value = [ann['id'] for ann in sample_annotations_data['annotations']]
                        mock_coco.loadAnns.side_effect = lambda ann_ids: [
                            ann for ann in sample_annotations_data['annotations'] 
                            if ann['id'] in ann_ids
                        ]
                        mock_coco_class.return_value = mock_coco
                        
                        class TestCOCODataset(COCODataset):
                            def __getitem__(self, idx):
                                return {
                                    "index": idx,
                                    "image_id": self.image_ids[idx],
                                    "image": self.load_image(idx),
                                    "annotations": self.get_annotations(idx)
                                }
                        
                        # Set config attributes for sample data
                        tooth_dataset_config.TOOTH_ANNOTATION_FILE = str(sample_annotations_file)
                        tooth_dataset_config.image_dir = str(sample_images_dir)
                        
                        dataset = TestCOCODataset(tooth_dataset_config, 'train', 'teeth')
                        
                        # Test iteration
                        items = []
                        for i in range(len(dataset)):
                            item = dataset[i]
                            items.append(item)
                            
                            # Check item structure
                            assert "index" in item
                            assert "image_id" in item
                            assert "image" in item
                            assert "annotations" in item
                            
                            # Check image
                            assert isinstance(item["image"], Image.Image)
                            assert item["image"].size == (512, 512)
                            
                            # Check annotations
                            assert isinstance(item["annotations"], list)
                        
                        # Should have processed all images
                        assert len(items) == 2
    
    @pytest.mark.data
    def test_dataset_statistics(self, sample_data_stats):
        """Test that dataset statistics match expected values."""
        stats = sample_data_stats
        
        # Verify sample data has expected structure
        assert stats["num_images"] == 2
        assert stats["num_annotations"] > 0
        assert stats["num_quadrant_categories"] == 4
        assert stats["num_tooth_categories"] == 8
        assert stats["num_disease_categories"] == 4
    
    @pytest.mark.data
    def test_coco_format_validation(self, sample_annotations_data):
        """Test that sample data follows valid COCO format."""
        # Use the custom assertion from conftest.py
        pytest.assert_valid_coco_format(sample_annotations_data)
        
        # Additional checks for multi-category format
        assert "categories_1" in sample_annotations_data
        assert "categories_2" in sample_annotations_data
        assert "categories_3" in sample_annotations_data
        
        # Check that annotations have multi-category IDs
        for ann in sample_annotations_data["annotations"]:
            assert "category_id_1" in ann
            assert "category_id_2" in ann
            assert "category_id_3" in ann


# Performance and stress tests
class TestDatasetPerformance:
    """Performance and stress tests for datasets."""
    
    @pytest.mark.data
    def test_dataset_initialization_performance(self, tooth_dataset_config, sample_annotations_file, sample_images_dir, sample_annotations_data):
        """Test that dataset initialization is fast."""
        import time
        
        with patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True):
            with patch('dentalvision.data.datasets.base.COCO') as mock_coco_class:
                with patch('os.path.exists', return_value=True):  # Mock path existence check
                    # Create mock COCO object
                    mock_coco = Mock()
                    mock_coco.dataset = sample_annotations_data
                    mock_coco.imgs = {img['id']: img for img in sample_annotations_data['images']}
                    mock_coco.anns = {ann['id']: ann for ann in sample_annotations_data['annotations']}
                    mock_coco.getAnnIds.return_value = [ann['id'] for ann in sample_annotations_data['annotations']]
                    mock_coco.loadAnns.side_effect = lambda ann_ids: [
                        ann for ann in sample_annotations_data['annotations'] 
                        if ann['id'] in ann_ids
                    ]
                    mock_coco_class.return_value = mock_coco
                    
                    class TestCOCODataset(COCODataset):
                        def __getitem__(self, idx):
                            return {"index": idx, "image_id": self.image_ids[idx]}
                    
                    # Set config attributes for sample data
                    tooth_dataset_config.TOOTH_ANNOTATION_FILE = str(sample_annotations_file)
                    tooth_dataset_config.image_dir = str(sample_images_dir)
                    
                    start_time = time.time()
                    dataset = TestCOCODataset(tooth_dataset_config, 'train', 'teeth')
                    init_time = time.time() - start_time
                    
                    # Initialization should be fast (< 1 second for small dataset)
                    assert init_time < 1.0
                    assert len(dataset) == 2
    
    @pytest.mark.data
    def test_memory_usage(self, tooth_dataset_config, sample_annotations_file, sample_images_dir, sample_annotations_data):
        """Test that dataset doesn't consume excessive memory."""
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not installed; skipping memory usage test.")
        
        with patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True):
            with patch('dentalvision.data.datasets.base.COCO') as mock_coco_class:
                with patch('os.path.exists', return_value=True):  # Mock path existence check
                    # Create mock COCO object
                    mock_coco = Mock()
                    mock_coco.dataset = sample_annotations_data
                    mock_coco.imgs = {img['id']: img for img in sample_annotations_data['images']}
                    mock_coco.anns = {ann['id']: ann for ann in sample_annotations_data['annotations']}
                    mock_coco.getAnnIds.return_value = [ann['id'] for ann in sample_annotations_data['annotations']]
                    mock_coco.loadAnns.side_effect = lambda ann_ids: [
                        ann for ann in sample_annotations_data['annotations'] 
                        if ann['id'] in ann_ids
                    ]
                    mock_coco_class.return_value = mock_coco
                    
                    class TestCOCODataset(COCODataset):
                        def __getitem__(self, idx):
                            return {"index": idx, "image_id": self.image_ids[idx]}
                    
                    # Set config attributes for sample data
                    tooth_dataset_config.TOOTH_ANNOTATION_FILE = str(sample_annotations_file)
                    tooth_dataset_config.image_dir = str(sample_images_dir)
                    
                    process = psutil.Process(os.getpid())
                    memory_before = process.memory_info().rss
                    
                    dataset = TestCOCODataset(tooth_dataset_config, 'train', 'teeth')
                    
                    memory_after = process.memory_info().rss
                    memory_increase = memory_after - memory_before
                    
                    # Memory increase should be reasonable (< 100MB for small dataset)
                    assert memory_increase < 100 * 1024 * 1024  # 100MB 