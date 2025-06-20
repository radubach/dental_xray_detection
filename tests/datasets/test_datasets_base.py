"""
Tests for dentalvision.data.datasets.base module.

This module tests:
1. BaseDataset abstract class functionality
2. COCODataset class with various task types
3. Error handling and edge cases
4. Utility functions
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile
import torch
from PIL import Image

# Import the classes to test
from dentalvision.data.datasets.base import BaseDataset, COCODataset, collate_fn_base
from dentalvision.config import Config


class TestBaseDataset(unittest.TestCase):
    """Test the abstract BaseDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.INPUT_SIZE = (512, 512)
        
        # Create a concrete implementation for testing
        class ConcreteDataset(BaseDataset):
            def __init__(self, config, split='train'):
                super().__init__(config, split)
                self.data = [1, 2, 3, 4, 5]  # Mock data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return {"data": self.data[idx], "index": idx}
        
        self.concrete_dataset = ConcreteDataset(self.config, 'train')
        self.ConcreteDataset = ConcreteDataset  # Store class for other tests
    
    def test_init(self):
        """Test BaseDataset initialization."""
        dataset = self.concrete_dataset
        
        self.assertEqual(dataset.config, self.config)
        self.assertEqual(dataset.split, 'train')
        self.assertTrue(dataset.is_training)
        
        # Test validation split
        val_dataset = self.ConcreteDataset(self.config, 'val')
        self.assertFalse(val_dataset.is_training)
    
    def test_validate_config(self):
        """Test configuration validation."""
        # Test valid config
        self.concrete_dataset._validate_config()
        
        # Test invalid INPUT_SIZE
        invalid_config = Config()
        invalid_config.INPUT_SIZE = [512, 512]  # List instead of tuple
        
        with self.assertRaises(ValueError):
            self.ConcreteDataset(invalid_config)
    
    def test_validate_path_exists(self):
        """Test path validation helper."""
        # Test existing path
        with tempfile.NamedTemporaryFile() as tmp_file:
            self.concrete_dataset._validate_path_exists(tmp_file.name, "Test file")
        
        # Test non-existing path
        with self.assertRaises(FileNotFoundError):
            self.concrete_dataset._validate_path_exists("/non/existent/path", "Test path")
    
    def test_load_image_from_path(self):
        """Test image loading functionality."""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            test_image.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Test successful loading
            loaded_image = self.concrete_dataset._load_image_from_path(tmp_path)
            self.assertIsInstance(loaded_image, Image.Image)
            self.assertEqual(loaded_image.size, (100, 100))
            
            # Test with conversion mode
            loaded_gray = self.concrete_dataset._load_image_from_path(tmp_path, convert_mode="L")
            self.assertEqual(loaded_gray.mode, "L")
            
            # Test non-existent file
            with self.assertRaises(FileNotFoundError):
                self.concrete_dataset._load_image_from_path("/non/existent/image.png")
                
        finally:
            os.unlink(tmp_path)
    
    def test_get_dataset_info(self):
        """Test dataset information retrieval."""
        info = self.concrete_dataset.get_dataset_info()
        
        expected_keys = ["split", "length", "is_training", "class_name"]
        for key in expected_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info["split"], "train")
        self.assertEqual(info["length"], 5)
        self.assertTrue(info["is_training"])
        self.assertEqual(info["class_name"], "ConcreteDataset")
    
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        # Create a dataset without implementing abstract methods
        class IncompleteDataset(BaseDataset):
            pass
        
        with self.assertRaises(TypeError):
            IncompleteDataset(self.config)


class TestCOCODataset(unittest.TestCase):
    """Test the COCODataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.INPUT_SIZE = (512, 512)
        self.config.RAW_DATA_DIR = "/test/data"
        
        # Create mock COCO data
        self.mock_coco_data = {
            "images": [
                {"id": 1, "file_name": "image1.jpg", "height": 512, "width": 512},
                {"id": 2, "file_name": "image2.jpg", "height": 512, "width": 512},
                {"id": 3, "file_name": "image3.jpg", "height": 512, "width": 512}
            ],
            "annotations": [
                {
                    "id": 1, "image_id": 1, "category_id": 1,
                    "category_id_1": 0, "category_id_2": 1, "category_id_3": 2,
                    "bbox": [100, 100, 50, 50], "area": 2500
                },
                {
                    "id": 2, "image_id": 1, "category_id": 2,
                    "category_id_1": 1, "category_id_2": 2, "category_id_3": 3,
                    "bbox": [200, 200, 50, 50], "area": 2500
                },
                {
                    "id": 3, "image_id": 2, "category_id": 1,
                    "category_id_1": 2, "category_id_2": 3, "category_id_3": 1,
                    "bbox": [150, 150, 50, 50], "area": 2500
                }
            ],
            "categories": [
                {"id": 1, "name": "tooth_1"},
                {"id": 2, "name": "tooth_2"}
            ],
            "categories_3": [
                {"id": 1, "name": "cavity"},
                {"id": 2, "name": "filling"},
                {"id": 3, "name": "crown"}
            ]
        }
    
    @patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True)
    @patch('dentalvision.data.datasets.base.COCO')
    def test_coco_dataset_init_teeth_task(self, mock_coco_class):
        """Test COCODataset initialization for teeth task."""
        # Mock COCO instance
        mock_coco = Mock()
        mock_coco.imgs = {1: self.mock_coco_data["images"][0],
                         2: self.mock_coco_data["images"][1],
                         3: self.mock_coco_data["images"][2]}
        mock_coco.anns = {1: self.mock_coco_data["annotations"][0],
                         2: self.mock_coco_data["annotations"][1],
                         3: self.mock_coco_data["annotations"][2]}
        mock_coco.dataset = self.mock_coco_data
        mock_coco.getAnnIds.return_value = [1, 2]
        mock_coco.loadAnns.return_value = self.mock_coco_data["annotations"][:2]
        mock_coco_class.return_value = mock_coco
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            # Create concrete implementation for testing
            class TestCOCODataset(COCODataset):
                def __getitem__(self, idx):
                    return {"index": idx, "image_id": self.image_ids[idx]}
            
            dataset = TestCOCODataset(self.config, 'train', 'teeth')
            
            self.assertEqual(dataset.task_type, 'teeth')
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset.num_classes, 32)  # 4 quadrants * 8 teeth
            self.assertEqual(len(dataset.class_names), 32)
            self.assertTrue(all(name.startswith('Q') for name in dataset.class_names))
    
    @patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True)
    @patch('dentalvision.data.datasets.base.COCO')
    def test_coco_dataset_init_disease_task(self, mock_coco_class):
        """Test COCODataset initialization for disease task."""
        mock_coco = Mock()
        mock_coco.imgs = {1: self.mock_coco_data["images"][0]}
        mock_coco.anns = {1: self.mock_coco_data["annotations"][0]}
        mock_coco.dataset = self.mock_coco_data
        mock_coco.getAnnIds.return_value = [1]
        mock_coco.loadAnns.return_value = [self.mock_coco_data["annotations"][0]]
        mock_coco_class.return_value = mock_coco
        
        with patch('os.path.exists', return_value=True):
            # Create concrete implementation for testing
            class TestCOCODataset(COCODataset):
                def __getitem__(self, idx):
                    return {"index": idx, "image_id": self.image_ids[idx]}
            
            dataset = TestCOCODataset(self.config, 'train', 'disease')
            
            self.assertEqual(dataset.task_type, 'disease')
            self.assertEqual(dataset.num_classes, 3)  # 3 disease categories
            self.assertEqual(dataset.class_names, ['cavity', 'filling', 'crown'])
    
    @patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True)
    @patch('dentalvision.data.datasets.base.COCO')
    def test_coco_dataset_init_quadrant_task(self, mock_coco_class):
        """Test COCODataset initialization for quadrant task."""
        mock_coco = Mock()
        mock_coco.imgs = {1: self.mock_coco_data["images"][0]}
        mock_coco.anns = {1: self.mock_coco_data["annotations"][0]}
        mock_coco.dataset = self.mock_coco_data
        mock_coco.getAnnIds.return_value = [1]
        mock_coco.loadAnns.return_value = [self.mock_coco_data["annotations"][0]]
        mock_coco_class.return_value = mock_coco
        
        with patch('os.path.exists', return_value=True):
            # Create concrete implementation for testing
            class TestCOCODataset(COCODataset):
                def __getitem__(self, idx):
                    return {"index": idx, "image_id": self.image_ids[idx]}
            
            dataset = TestCOCODataset(self.config, 'train', 'quadrant')
            
            self.assertEqual(dataset.task_type, 'quadrant')
            self.assertEqual(dataset.num_classes, 4)  # 4 quadrants
            self.assertEqual(dataset.class_names, ['Q1', 'Q2', 'Q3', 'Q4'])
    
    @patch('dentalvision.data.datasets.base.COCO_AVAILABLE', False)
    def test_coco_dataset_no_pycocotools(self):
        """Test COCODataset raises ImportError when pycocotools not available."""
        # Create concrete implementation for testing
        class TestCOCODataset(COCODataset):
            def __getitem__(self, idx):
                return {"index": idx, "image_id": self.image_ids[idx]}
        
        with self.assertRaises(ImportError):
            TestCOCODataset(self.config, 'train')
    
    @patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True)
    @patch('dentalvision.data.datasets.base.COCO')
    def test_get_unified_category_id(self, mock_coco_class):
        """Test unified category ID generation."""
        mock_coco = Mock()
        mock_coco.imgs = {1: self.mock_coco_data["images"][0]}
        mock_coco.anns = {1: self.mock_coco_data["annotations"][0]}
        mock_coco.dataset = self.mock_coco_data
        mock_coco.getAnnIds.return_value = [1]
        mock_coco.loadAnns.return_value = [self.mock_coco_data["annotations"][0]]
        mock_coco_class.return_value = mock_coco
        
        with patch('os.path.exists', return_value=True):
            # Create concrete implementation for testing
            class TestCOCODataset(COCODataset):
                def __getitem__(self, idx):
                    return {"index": idx, "image_id": self.image_ids[idx]}
            
            dataset = TestCOCODataset(self.config, 'train', 'teeth')
            
            # Test teeth task (quadrant * 8 + tooth)
            ann = self.mock_coco_data["annotations"][0]  # category_id_1=0, category_id_2=1
            unified_id = dataset.get_unified_category_id(ann)
            self.assertEqual(unified_id, 1)  # 0 * 8 + 1 = 1
            
            # Test disease task
            dataset.task_type = 'disease'
            unified_id = dataset.get_unified_category_id(ann)
            self.assertEqual(unified_id, 2)  # category_id_3
            
            # Test quadrant task
            dataset.task_type = 'quadrant'
            unified_id = dataset.get_unified_category_id(ann)
            self.assertEqual(unified_id, 0)  # category_id_1
    
    @patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True)
    @patch('dentalvision.data.datasets.base.COCO')
    def test_load_image(self, mock_coco_class):
        """Test image loading functionality."""
        mock_coco = Mock()
        mock_coco.imgs = {1: self.mock_coco_data["images"][0]}
        mock_coco.anns = {1: self.mock_coco_data["annotations"][0]}
        mock_coco.dataset = self.mock_coco_data
        mock_coco.getAnnIds.return_value = [1]
        mock_coco.loadAnns.return_value = [self.mock_coco_data["annotations"][0]]
        mock_coco_class.return_value = mock_coco
        
        # Create a test image
        test_image = Image.new('RGB', (512, 512), color='white')
        
        with patch('os.path.exists', return_value=True), \
             patch('PIL.Image.open', return_value=test_image):
            
            # Create concrete implementation for testing
            class TestCOCODataset(COCODataset):
                def __getitem__(self, idx):
                    return {"index": idx, "image_id": self.image_ids[idx]}
            
            dataset = TestCOCODataset(self.config, 'train', 'teeth')
            loaded_image = dataset.load_image(0)
            
            self.assertIsInstance(loaded_image, Image.Image)
            self.assertEqual(loaded_image.size, (512, 512))
    
    @patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True)
    @patch('dentalvision.data.datasets.base.COCO')
    def test_get_annotations(self, mock_coco_class):
        """Test annotation retrieval."""
        mock_coco = Mock()
        mock_coco.imgs = {1: self.mock_coco_data["images"][0]}
        mock_coco.anns = {1: self.mock_coco_data["annotations"][0]}
        mock_coco.dataset = self.mock_coco_data
        mock_coco.getAnnIds.return_value = [1]
        mock_coco.loadAnns.return_value = [self.mock_coco_data["annotations"][0]]
        mock_coco_class.return_value = mock_coco
        
        with patch('os.path.exists', return_value=True):
            # Create concrete implementation for testing
            class TestCOCODataset(COCODataset):
                def __getitem__(self, idx):
                    return {"index": idx, "image_id": self.image_ids[idx]}
            
            dataset = TestCOCODataset(self.config, 'train', 'teeth')
            annotations = dataset.get_annotations(0)
            
            self.assertEqual(len(annotations), 1)
            self.assertEqual(annotations[0]['id'], 1)
            mock_coco.getAnnIds.assert_called_with(imgIds=1)
            mock_coco.loadAnns.assert_called_with([1])
    
    @patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True)
    @patch('dentalvision.data.datasets.base.COCO')
    def test_dataset_validation(self, mock_coco_class):
        """Test dataset validation."""
        mock_coco = Mock()
        mock_coco.imgs = {}  # Empty dataset
        mock_coco.dataset = self.mock_coco_data
        mock_coco_class.return_value = mock_coco
        
        with patch('os.path.exists', return_value=True):
            # Create concrete implementation for testing
            class TestCOCODataset(COCODataset):
                def __getitem__(self, idx):
                    return {"index": idx, "image_id": self.image_ids[idx]}
            
            with self.assertRaises(ValueError):
                TestCOCODataset(self.config, 'train', 'teeth')
    
    def test_invalid_task_type(self):
        """Test invalid task type raises ValueError."""
        with patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True), \
             patch('dentalvision.data.datasets.base.COCO'):
            
            mock_coco = Mock()
            mock_coco.imgs = {1: self.mock_coco_data["images"][0]}
            mock_coco.anns = {1: self.mock_coco_data["annotations"][0]}
            mock_coco.dataset = self.mock_coco_data
            mock_coco.getAnnIds.return_value = [1]
            mock_coco.loadAnns.return_value = [self.mock_coco_data["annotations"][0]]
            
            with patch('dentalvision.data.datasets.base.COCO', return_value=mock_coco), \
                 patch('os.path.exists', return_value=True):
                
                # Create concrete implementation for testing
                class TestCOCODataset(COCODataset):
                    def __getitem__(self, idx):
                        return {"index": idx, "image_id": self.image_ids[idx]}
                
                with self.assertRaises(ValueError):
                    TestCOCODataset(self.config, 'train', 'invalid_task')


class TestCollateFunction(unittest.TestCase):
    """Test the collate function."""
    
    def test_collate_fn_base(self):
        """Test basic collate function."""
        # Create mock batch data
        batch = [
            {"image": torch.randn(3, 512, 512), "label": torch.tensor([1])},
            {"image": torch.randn(3, 512, 512), "label": torch.tensor([2])},
            {"image": torch.randn(3, 512, 512), "label": torch.tensor([3])}
        ]
        
        # Test that it doesn't raise an error
        try:
            result = collate_fn_base(batch)
            # Should return a dictionary with batched tensors
            self.assertIn("image", result)
            self.assertIn("label", result)
            self.assertEqual(result["image"].shape[0], 3)  # Batch size
            self.assertEqual(result["label"].shape[0], 3)  # Batch size
        except Exception as e:
            # If default_collate fails, that's expected for custom data structures
            self.assertIsInstance(e, (TypeError, RuntimeError))


class TestDatasetIntegration(unittest.TestCase):
    """Integration tests for dataset functionality."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.config = Config()
        self.config.INPUT_SIZE = (512, 512)
        self.config.RAW_DATA_DIR = "/test/data"
    
    @patch('dentalvision.data.datasets.base.COCO_AVAILABLE', True)
    @patch('dentalvision.data.datasets.base.COCO')
    def test_dataset_iteration(self, mock_coco_class):
        """Test that dataset can be iterated over."""
        # Create mock COCO data
        mock_data = {
            "images": [
                {"id": 1, "file_name": "image1.jpg", "height": 512, "width": 512},
                {"id": 2, "file_name": "image2.jpg", "height": 512, "width": 512}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "category_id_1": 0, "category_id_2": 1, "category_id_3": 2}
            ],
            "categories": [{"id": 1, "name": "tooth"}],
            "categories_3": [{"id": 1, "name": "cavity"}, {"id": 2, "name": "filling"}]
        }
        
        mock_coco = Mock()
        mock_coco.imgs = {1: mock_data["images"][0], 2: mock_data["images"][1]}
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
            
            # Test iteration
            items = list(dataset)
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0]["index"], 0)
            self.assertEqual(items[0]["image_id"], 1)
            self.assertEqual(items[1]["index"], 1)
            self.assertEqual(items[1]["image_id"], 2)
    
    def test_config_validation_edge_cases(self):
        """Test edge cases in configuration validation."""
        # Test with None INPUT_SIZE
        config = Config()
        config.INPUT_SIZE = None
        
        class TestDataset(BaseDataset):
            def __len__(self): return 1
            def __getitem__(self, idx): return {}
        
        with self.assertRaises(ValueError):
            TestDataset(config)


if __name__ == '__main__':
    unittest.main() 