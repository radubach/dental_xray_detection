"""
Base dataset classes for dental image analysis.

This module provides:
1. BaseDataset: Generic dataset functionality
2. COCODataset: COCO-specific functionality for most computer vision models
"""

from abc import ABC, abstractmethod
import os
from typing import Dict, Any, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

try:
    from pycocotools.coco import COCO
    COCO_AVAILABLE = True
except ImportError:
    COCO_AVAILABLE = False
    print("Warning: pycocotools not available. COCO functionality disabled.")

from dentalvision.config import Config


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all datasets.
    
    Provides common functionality that applies to any dataset regardless of format:
    - Path validation
    - Basic error handling
    - Common utilities
    - Standard interface
    
    This class is framework-agnostic and can be extended for COCO, YOLO, 
    custom formats, or any other data structure.
    """
    
    def __init__(self, config: Config, split: str = 'train'):
        """
        Initialize base dataset.
        
        Args:
            config: Configuration object
            split: Dataset split ('train', 'val', 'test')
        """
        self.config = config
        self.split = split
        self.is_training = split == 'train'
        
        # Validate configuration
        self._validate_config()
        
        # Set up paths
        self.setup_paths()
        
        print(f"Initializing {self.__class__.__name__} for {split} split")
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if not isinstance(self.config.INPUT_SIZE, tuple) or len(self.config.INPUT_SIZE) != 2:
            raise ValueError("INPUT_SIZE must be a tuple of (height, width)")
        
    def _validate_path_exists(self, path: str, description: str = "Path"):
        """Helper to validate paths exist."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"{description} not found: {path}")
    
    def setup_paths(self):
        """Set up data paths. Can be overridden by subclasses."""
        # Base implementation - override in subclasses
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return dataset length. Override if you have a different length source."""
        if hasattr(self, 'image_ids'):
            return len(self.image_ids)
        elif hasattr(self, 'data'):
            return len(self.data)
        else:
            raise NotImplementedError("Subclass must implement __len__ or provide image_ids/data attribute")
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return item at index. Must be implemented by subclasses."""
        pass
    
    def _load_image_from_path(self, image_path: str, convert_mode: str = "L") -> Image.Image:
        """
        Load image from file path with error handling.
        
        Args:
            image_path: Path to image file
            convert_mode: PIL conversion mode ("RGB", "L", etc.)
            
        Returns:
            PIL Image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            image = Image.open(image_path)
            if convert_mode:
                image = image.convert(convert_mode)
            return image
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return information about the dataset."""
        return {
            "split": self.split,
            "length": len(self),
            "is_training": self.is_training,
            "class_name": self.__class__.__name__
        }


class COCODataset(BaseDataset):
    """
    Base class for datasets using COCO format annotations.
    
    This class handles all COCO-specific functionality:
    - Loading COCO annotation files
    - Image loading from COCO metadata
    - Annotation parsing
    - COCO utilities

    Supports multi-category annotations (category_id_1, category_id_2, category_id_3)
    and automatically creates unified category mappings for different tasks.
    
    Child classes (UNet, Mask R-CNN, etc.) inherit this and implement
    model-specific __getitem__ methods for different mask formats.
    """
    
    def __init__(self, config: Config, split: str = 'train', task_type: str = 'teeth'):
        """
        Initialize COCO dataset.
        
        Args:
            config: Configuration object containing COCO-specific paths
            split: Dataset split
            task_type: Type of task - 'teeth', 'disease', 'quadrant', or 'combined'
                - 'teeth': Uses combined category_id_1 + category_id_2 (32 tooth classes)
                - 'disease': Uses category_id_3 (disease classes)
                - 'quadrant': Uses category_id_1 (quadrant classes)
                - 'combined': Uses all categories for multi-task learning
        """
        if not COCO_AVAILABLE:
            raise ImportError("pycocotools is required for COCO datasets. Install with: pip install pycocotools")
        
        self.task_type = task_type
        super().__init__(config, split)
        
        # Load COCO annotation file
        self.coco = self._load_coco_file()
        
        # Process categories based on task type
        self._setup_categories()
        
        # Get image IDs and validate
        self.image_ids = list(sorted(self.coco.imgs.keys()))
        self._validate_dataset()
        
        print(f"COCO dataset initialized with {len(self.image_ids)} images for task: {task_type}")
        print(f"Number of classes: {self.get_num_classes()}")
    
    def setup_paths(self):
        """Set up paths based on task type using explicit config paths."""
        
        # Get paths from config based on task type
        self.annotation_file = self.config.get_annotation_file(self.task_type)
        self.image_dir = self.config.get_image_dir(self.task_type)
        
        # Validate paths exist (optional - could be skipped for testing)
        if not os.path.exists(self.annotation_file):
            print(f"Warning: Annotation file not found: {self.annotation_file}")
        
        if not os.path.exists(self.image_dir):
            print(f"Warning: Image directory not found: {self.image_dir}")
        
        print(f"Dataset setup for task '{self.task_type}':")
        print(f"  Annotation file: {self.annotation_file}")
        print(f"  Image directory: {self.image_dir}")
    
    def _load_coco_file(self) -> COCO:
        """Load COCO annotation file."""
        if not os.path.exists(self.annotation_file):
            raise FileNotFoundError(f"COCO annotation file not found: {self.annotation_file}")
        
        try:
            return COCO(self.annotation_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load COCO file {self.annotation_file}: {str(e)}")
    
    def _validate_dataset(self):
        """Validate dataset integrity."""
        if len(self.image_ids) == 0:
            raise ValueError("No images found in COCO dataset")
        
        # Check if image directory exists
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Validate a few sample images exist
        sample_size = min(5, len(self.image_ids))
        for i in range(sample_size):
            image_id = self.image_ids[i]
            image_info = self.coco.imgs[image_id]
            image_path = os.path.join(self.image_dir, image_info['file_name'])
            if not os.path.exists(image_path):
                print(f"Warning: Sample image not found: {image_path}")
    
    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.image_ids)
    
    def load_image(self, idx: int) -> Image.Image:
        """
        Load image by dataset index.
        
        Args:
            idx: Dataset index (not image_id)
            
        Returns:
            PIL Image in RGB format
        """
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        
        # Load image (RGB for most models, can be overridden)
        image = self._load_image_from_path(image_path, convert_mode="RGB")
        
        # Optional: Validate dimensions match COCO metadata
        if (image.height != image_info['height'] or 
            image.width != image_info['width']):
            print(f"Warning: Image {image_id} dimensions mismatch. "
                  f"COCO metadata: {image_info['height']}x{image_info['width']}, "
                  f"Actual: {image.height}x{image.width}")
        
        return image
    
    def get_image_info(self, idx: int) -> Dict[str, Any]:
        """Get COCO image metadata."""
        image_id = self.image_ids[idx]
        return self.coco.imgs[image_id]
    
    def get_annotations(self, idx: int) -> List[Dict[str, Any]]:
        """
        Get all annotations for image at dataset index.
        
        Args:
            idx: Dataset index
            
        Returns:
            List of COCO annotation dictionaries
        """
        image_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        return self.coco.loadAnns(ann_ids)
    
    def _setup_categories(self):
        """
        Set up category mappings based on task type.
        
        Handles multi-category annotations by creating appropriate mappings
        for different tasks without requiring data preprocessing.
        """
        # Check if we have multi-category data
        sample_anns = list(self.coco.anns.values())[:5]  # Check first 5 annotations
        has_multi_categories = any('category_id_1' in ann for ann in sample_anns)
        
        if has_multi_categories:
            self._setup_multi_categories()
        else:
            self._setup_standard_categories()

    def _setup_multi_categories(self):
        """Handle multi-category annotations (category_id_1, category_id_2, category_id_3)."""
        print(f"Detected multi-category annotations. Setting up for task: {self.task_type}")
        
        if self.task_type == 'teeth':
            # Combine quadrant (0-3) and tooth (0-7) into single tooth class (0-31)
            self.category_mapping = {}
            self.class_names = []
            
            # Create 32 tooth classes: quadrant * 8 + tooth_in_quad
            for quad in range(4):
                for tooth in range(8):
                    tooth_id = quad * 8 + tooth
                    tooth_name = f"Q{quad+1}T{tooth+1}"  # Q1T1, Q1T2, etc.
                    self.category_mapping[(quad, tooth)] = tooth_id
                    self.class_names.append(tooth_name)
            
            self.num_classes = 32
            
        elif self.task_type == 'disease':
            # Use category_id_3 for disease classification
            if 'categories_3' in self.coco.dataset:
                disease_cats = self.coco.dataset['categories_3']
                self.class_names = [cat['name'] for cat in sorted(disease_cats, key=lambda x: x['id'])]
                self.num_classes = len(disease_cats)
            else:
                # Fallback: extract from annotations
                disease_ids = set()
                for ann in self.coco.anns.values():
                    if 'category_id_3' in ann:
                        disease_ids.add(ann['category_id_3'])
                self.num_classes = len(disease_ids)
                self.class_names = [f"Disease_{i}" for i in sorted(disease_ids)]
                
        elif self.task_type == 'quadrant':
            # Use category_id_1 for quadrant classification
            self.num_classes = 4
            self.class_names = ["Q1", "Q2", "Q3", "Q4"]
            
        else:
            raise ValueError(f"Unsupported task type for multi-category data: {self.task_type}")
        
    def _setup_standard_categories(self):
        """Handle standard COCO categories."""
        if 'categories' in self.coco.dataset:
            categories = sorted(self.coco.dataset['categories'], key=lambda x: x['id'])
            self.class_names = [cat['name'] for cat in categories]
            self.num_classes = len(categories)
        else:
            # Fallback: extract from annotations
            category_ids = set(ann['category_id'] for ann in self.coco.anns.values())
            self.num_classes = len(category_ids)
            self.class_names = [f"Class_{i}" for i in sorted(category_ids)]

    def get_unified_category_id(self, annotation: Dict[str, Any]) -> int:
        """
        Get unified category ID based on task type and annotation format.
        
        Args:
            annotation: COCO annotation dictionary
            
        Returns:
            Unified category ID for the current task
        """
        # Handle multi-category annotations
        if 'category_id_1' in annotation:
            if self.task_type == 'teeth':
                quad = annotation['category_id_1']
                tooth = annotation['category_id_2']
                return self.category_mapping[(quad, tooth)]
            elif self.task_type == 'disease':
                return annotation['category_id_3']
            elif self.task_type == 'quadrant':
                return annotation['category_id_1']
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")
        else:
            # Standard COCO format
            return annotation['category_id']
    
    def get_category_info(self) -> Dict[int, Dict[str, Any]]:
        """Get mapping of category IDs to category information."""
        if hasattr(self, 'category_mapping') and self.category_mapping:
            # For multi-category data, return the computed mapping
            return self.category_mapping
        else:
            # Fallback for standard COCO format
            return {cat['id']: cat for cat in self.coco.dataset.get('categories', [])}

    def get_class_names(self) -> List[str]:
        """Get list of class names in order of category IDs."""
        # Use the pre-computed class names from setup_multi_categories()
        return self.class_names if hasattr(self, 'class_names') else []

    def get_num_classes(self) -> int:
        """Get number of classes (categories) in dataset."""
        # Use the pre-computed num_classes from setup_multi_categories()
        return self.num_classes if hasattr(self, 'num_classes') else 0
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        This must be implemented by model-specific subclasses.
        
        Different models need different formats:
        - UNet: Single semantic mask
        - Mask R-CNN: Multiple instance masks + bboxes
        - etc.
        """
        pass


# Utility functions for common operations
def collate_fn_base(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Basic collate function that can be extended by model-specific datasets.
    
    Args:
        batch: List of dataset items
        
    Returns:
        Batched data
    """
    # This is a placeholder - implement based on your needs
    # or let individual datasets define their own collate functions
    return torch.utils.data.dataloader.default_collate(batch)