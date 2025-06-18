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
    
    Child classes (UNet, Mask R-CNN, etc.) inherit this and implement
    model-specific __getitem__ methods for different mask formats.
    """
    
    def __init__(self, config: Config, split: str = 'train'):
        """
        Initialize COCO dataset.
        
        Args:
            config: Configuration object containing COCO-specific paths
            split: Dataset split
        """
        if not COCO_AVAILABLE:
            raise ImportError("pycocotools is required for COCO datasets. Install with: pip install pycocotools")
        
        super().__init__(config, split)
        
        # Load COCO annotation file
        self.coco = self._load_coco_file()
        
        # Get image IDs and validate
        self.image_ids = list(sorted(self.coco.imgs.keys()))
        self._validate_dataset()
        
        print(f"COCO dataset initialized with {len(self.image_ids)} images")
    
    def setup_paths(self):
        """Set up COCO-specific paths."""
        # These should come from your config
        # For now, using placeholder paths - update based on your config structure
        self.image_dir = getattr(self.config, 'RAW_DATA_DIR', '/content/drive/MyDrive/Dentex_raw/images')
        self.annotation_file = getattr(self.config, 'COCO_ANNOTATION_FILE', '/content/drive/MyDrive/Dentex_raw/annotations.json')
        self.mask_dir = getattr(self.config, 'MASK_DIR', None)  # Optional
    
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
    
    def get_category_info(self) -> Dict[int, Dict[str, Any]]:
        """Get mapping of category IDs to category information."""
        return {cat['id']: cat for cat in self.coco.dataset['categories']}
    
    def get_class_names(self) -> List[str]:
        """Get list of class names in order of category IDs."""
        categories = sorted(self.coco.dataset['categories'], key=lambda x: x['id'])
        return [cat['name'] for cat in categories]
    
    def get_num_classes(self) -> int:
        """Get number of classes (categories) in dataset."""
        return len(self.coco.dataset['categories'])
    
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