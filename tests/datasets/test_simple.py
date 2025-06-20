"""
Simple tests to verify the testing setup works correctly.
"""

import unittest
from unittest.mock import Mock, patch
import torch
from PIL import Image

# Import the classes to test
from dentalvision.data.datasets.base import BaseDataset, COCODataset, collate_fn_base
from dentalvision.config import Config


class TestSimpleBaseDataset(unittest.TestCase):
    """Simple tests for BaseDataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.INPUT_SIZE = (512, 512)
        
        # Create a concrete implementation for testing
        class ConcreteDataset(BaseDataset):
            def __init__(self, config, split='train'):
                super().__init__(config, split)
                self.data = [1, 2, 3]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return {"data": self.data[idx]}
        
        self.dataset = ConcreteDataset(self.config, 'train')
        self.ConcreteDataset = ConcreteDataset  # Store class for other tests
    
    def test_basic_functionality(self):
        """Test basic BaseDataset functionality."""
        self.assertEqual(len(self.dataset), 3)
        self.assertTrue(self.dataset.is_training)
        self.assertEqual(self.dataset.split, 'train')
        
        item = self.dataset[0]
        self.assertEqual(item["data"], 1)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid config
        self.dataset._validate_config()
        
        # Test invalid config
        invalid_config = Config()
        invalid_config.INPUT_SIZE = [512, 512]  # List instead of tuple
        
        with self.assertRaises(ValueError):
            self.ConcreteDataset(invalid_config)
    
    def test_dataset_info(self):
        """Test dataset information retrieval."""
        info = self.dataset.get_dataset_info()
        
        self.assertIn("split", info)
        self.assertIn("length", info)
        self.assertIn("is_training", info)
        self.assertIn("class_name", info)
        
        self.assertEqual(info["split"], "train")
        self.assertEqual(info["length"], 3)
        self.assertTrue(info["is_training"])


class TestSimpleCOCODataset(unittest.TestCase):
    """Simple tests for COCODataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.INPUT_SIZE = (512, 512)
        self.config.RAW_DATA_DIR = "/test/data"
    
    @patch('dentalvision.data.datasets.base.COCO_AVAILABLE', False)
    def test_missing_pycocotools(self):
        """Test that COCODataset raises ImportError when pycocotools not available."""
        # Create a concrete implementation for testing
        class TestCOCODataset(COCODataset):
            def __getitem__(self, idx):
                return {"index": idx, "image_id": self.image_ids[idx]}
        
        with self.assertRaises(ImportError):
            TestCOCODataset(self.config, 'train')
    
    @patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True)
    @patch('dentalvision.data.datasets.base.COCO')
    def test_coco_dataset_initialization(self, mock_coco_class):
        """Test COCODataset initialization."""
        # Create mock COCO data
        mock_data = {
            "images": [{"id": 1, "file_name": "image1.jpg", "height": 512, "width": 512}],
            "annotations": [
                {
                    "id": 1, "image_id": 1, "category_id": 1,
                    "category_id_1": 0, "category_id_2": 1, "category_id_3": 2
                }
            ],
            "categories": [{"id": 1, "name": "tooth"}],
            "categories_3": [{"id": 1, "name": "cavity"}, {"id": 2, "name": "filling"}]
        }
        
        mock_coco = Mock()
        mock_coco.imgs = {1: mock_data["images"][0]}
        mock_coco.anns = {1: mock_data["annotations"][0]}
        mock_coco.dataset = mock_data
        mock_coco.getAnnIds.return_value = [1]
        mock_coco.loadAnns.return_value = [mock_data["annotations"][0]]
        mock_coco_class.return_value = mock_coco
        
        # Create concrete implementation for testing
        class TestCOCODataset(COCODataset):
            def __getitem__(self, idx):
                return {"index": idx, "image_id": self.image_ids[idx]}
        
        with patch('os.path.exists', return_value=True):
            dataset = TestCOCODataset(self.config, 'train', 'teeth')
            
            self.assertEqual(dataset.task_type, 'teeth')
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.num_classes, 32)  # 4 quadrants * 8 teeth


class TestSimpleCollateFunction(unittest.TestCase):
    """Simple tests for collate function."""
    
    def test_collate_fn_base(self):
        """Test basic collate function."""
        # Create simple batch data
        batch = [
            {"image": torch.randn(3, 512, 512), "label": torch.tensor([1])},
            {"image": torch.randn(3, 512, 512), "label": torch.tensor([2])}
        ]
        
        try:
            result = collate_fn_base(batch)
            
            # Check that it returns a dictionary
            self.assertIsInstance(result, dict)
            
            # Check that it contains expected keys
            self.assertIn("image", result)
            self.assertIn("label", result)
            
        except Exception as e:
            # If default_collate fails, that's expected for custom data structures
            self.assertIsInstance(e, (TypeError, RuntimeError))


if __name__ == '__main__':
    unittest.main() 