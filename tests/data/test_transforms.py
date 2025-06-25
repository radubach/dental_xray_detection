"""
Comprehensive tests for dentalvision.data.transforms module.

This module tests the SegmentationTransforms class which provides:
- Training transforms for semantic and instance segmentation
- Inference transforms
- Resize transforms
- Sliding window transforms
- Error handling and validation
"""

import pytest
import numpy as np
import torch
from PIL import Image
import albumentations as A

# Import the module to test
from dentalvision.data.transforms import SegmentationTransforms
from dentalvision.config import Config


class TestSegmentationTransforms:
    """Test suite for SegmentationTransforms class."""
    
    @pytest.fixture
    def valid_config(self):
        """Create a valid config for testing."""
        config = Config()
        config.input_size = (512, 512)
        config.normalize_mean = [0.485, 0.456, 0.406]
        config.normalize_std = [0.229, 0.224, 0.225]
        config.augment = True
        config.augment_prob = 0.5
        config.brightness_limit = 0.2
        config.contrast_limit = 0.2
        config.noise_limit = (10.0, 50.0)
        config.rotation_limit = 15
        return config
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_mask(self):
        """Create a sample mask for testing."""
        return np.random.randint(0, 32, (256, 256), dtype=np.uint8)
    
    @pytest.fixture
    def sample_bboxes(self):
        """Create sample bounding boxes for testing."""
        return [[10, 10, 100, 100], [50, 50, 150, 150]]
    
    @pytest.fixture
    def sample_labels(self):
        """Create sample labels for testing."""
        return [1, 2]
    
    def test_get_training_transforms_semantic(self, valid_config):
        """Test training transforms for semantic segmentation."""
        transforms = SegmentationTransforms.get_training_transforms(
            valid_config, is_instance_segmentation=False
        )
        
        # Check that it's a Compose object
        assert isinstance(transforms, A.Compose)
        
        # Check that it has the expected transforms
        transform_names = [type(t).__name__ for t in transforms.transforms]
        assert 'Resize' in transform_names
        assert 'Normalize' in transform_names
        assert 'ToTensorV2' in transform_names
        
        # Check that augmentation transforms are included
        assert 'RandomBrightnessContrast' in transform_names
        assert 'GaussNoise' in transform_names
        assert 'Rotate' in transform_names
        assert 'ShiftScaleRotate' in transform_names
    
    def test_get_training_transforms_instance(self, valid_config):
        """Test training transforms for instance segmentation."""
        transforms = SegmentationTransforms.get_training_transforms(
            valid_config, is_instance_segmentation=True
        )
        
        # Check that it's a Compose object
        assert isinstance(transforms, A.Compose)
        
        # Check that it has bbox parameters
        assert transforms.bbox_params is not None
        assert transforms.bbox_params.format == 'pascal_voc'
        
        # Check that it has additional targets for masks
        assert 'masks' in transforms.additional_targets
    
    def test_get_training_transforms_no_augmentation(self, valid_config):
        """Test training transforms without augmentation."""
        valid_config.augment = False
        transforms = SegmentationTransforms.get_training_transforms(
            valid_config, is_instance_segmentation=False
        )
        
        # Check that augmentation transforms are not included
        transform_names = [type(t).__name__ for t in transforms.transforms]
        assert 'RandomBrightnessContrast' not in transform_names
        assert 'GaussNoise' not in transform_names
        assert 'Rotate' not in transform_names
        assert 'ShiftScaleRotate' not in transform_names
        
        # Check that basic transforms are still there
        assert 'Resize' in transform_names
        assert 'Normalize' in transform_names
        assert 'ToTensorV2' in transform_names
    
    def test_get_inference_transforms_semantic(self, valid_config):
        """Test inference transforms for semantic segmentation."""
        transforms = SegmentationTransforms.get_inference_transforms(
            valid_config, is_instance_segmentation=False
        )
        
        # Check that it's a Compose object
        assert isinstance(transforms, A.Compose)
        
        # Check that it has only basic transforms (no augmentation)
        transform_names = [type(t).__name__ for t in transforms.transforms]
        assert 'Resize' in transform_names
        assert 'Normalize' in transform_names
        assert 'ToTensorV2' in transform_names
        
        # Check that augmentation transforms are not included
        assert 'RandomBrightnessContrast' not in transform_names
        assert 'GaussNoise' not in transform_names
        assert 'Rotate' not in transform_names
    
    def test_get_inference_transforms_instance(self, valid_config):
        """Test inference transforms for instance segmentation."""
        transforms = SegmentationTransforms.get_inference_transforms(
            valid_config, is_instance_segmentation=True
        )
        
        # Check that it's a Compose object
        assert isinstance(transforms, A.Compose)
        
        # Check that it has bbox parameters
        assert transforms.bbox_params is not None
        assert transforms.bbox_params.format == 'pascal_voc'
        
        # Check that it has additional targets for masks
        assert 'masks' in transforms.additional_targets
    
    def test_get_resize_transform(self):
        """Test resize transform creation."""
        transforms = SegmentationTransforms.get_resize_transform(1024, 768)
        
        # Check that it's a Compose object
        assert isinstance(transforms, A.Compose)
        
        # Check that it has a Resize transform
        transform_names = [type(t).__name__ for t in transforms.transforms]
        assert 'Resize' in transform_names
        
        # Check that the resize transform has correct parameters
        resize_transform = transforms.transforms[0]
        assert resize_transform.height == 1024
        assert resize_transform.width == 768
    
    def test_get_sliding_window_transforms(self):
        """Test sliding window transforms creation."""
        transforms = SegmentationTransforms.get_sliding_window_transforms()
        
        # Check that it's a Compose object
        assert isinstance(transforms, A.Compose)
        
        # Check that it has the expected transforms
        transform_names = [type(t).__name__ for t in transforms.transforms]
        assert 'Normalize' in transform_names
        assert 'ToTensorV2' in transform_names
        
        # Check that normalization uses the expected values
        normalize_transform = transforms.transforms[0]
        assert normalize_transform.mean == [0.5]
        assert normalize_transform.std == [0.5]
    
    def test_invalid_input_size_tuple(self, valid_config):
        """Test error handling for invalid input_size."""
        # Test with list instead of tuple
        valid_config.input_size = [512, 512]
        with pytest.raises(ValueError, match="input_size must be a tuple"):
            SegmentationTransforms.get_training_transforms(valid_config)
        
        # Test with single integer
        valid_config.input_size = 512
        with pytest.raises(ValueError, match="input_size must be a tuple"):
            SegmentationTransforms.get_training_transforms(valid_config)
        
        # Test with wrong length tuple
        valid_config.input_size = (512, 512, 512)
        with pytest.raises(ValueError, match="input_size must be a tuple"):
            SegmentationTransforms.get_training_transforms(valid_config)
    
    def test_missing_normalization_params(self, valid_config):
        """Test error handling for missing normalization parameters."""
        # Test with None mean
        valid_config.normalize_mean = None
        with pytest.raises(ValueError, match="normalize_mean and normalize_std must be specified"):
            SegmentationTransforms.get_training_transforms(valid_config)
        
        # Test with None std
        valid_config.normalize_mean = [0.485, 0.456, 0.406]
        valid_config.normalize_std = None
        with pytest.raises(ValueError, match="normalize_mean and normalize_std must be specified"):
            SegmentationTransforms.get_training_transforms(valid_config)
    
    def test_transform_application_semantic(self, valid_config, sample_image, sample_mask):
        """Test that transforms can be applied to semantic segmentation data."""
        transforms = SegmentationTransforms.get_training_transforms(
            valid_config, is_instance_segmentation=False
        )
        
        # Apply transforms
        result = transforms(image=sample_image, masks=[sample_mask])
        
        # Check that image is transformed
        assert 'image' in result
        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape == (3, 512, 512)  # (C, H, W)
        
        # Check that mask is transformed
        assert 'masks' in result
        assert len(result['masks']) == 1
        assert result['masks'][0].dtype == np.float32
    
    def test_transform_application_instance(self, valid_config, sample_image, sample_mask, sample_bboxes, sample_labels):
        """Test that transforms can be applied to instance segmentation data."""
        transforms = SegmentationTransforms.get_training_transforms(
            valid_config, is_instance_segmentation=True
        )
        
        # Apply transforms
        result = transforms(
            image=sample_image,
            masks=[sample_mask],
            bboxes=sample_bboxes,
            labels=sample_labels
        )
        
        # Check that image is transformed
        assert 'image' in result
        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape == (3, 512, 512)
        
        # Check that mask is transformed
        assert 'masks' in result
        assert len(result['masks']) == 1
        assert result['masks'][0].dtype == np.float32
        
        # Check that bboxes are transformed
        assert 'bboxes' in result
        assert len(result['bboxes']) == len(sample_bboxes)
        
        # Check that labels are transformed
        assert 'labels' in result
        assert len(result['labels']) == len(sample_labels)
    
    def test_inference_transform_application(self, valid_config, sample_image, sample_mask):
        """Test that inference transforms can be applied."""
        transforms = SegmentationTransforms.get_inference_transforms(
            valid_config, is_instance_segmentation=False
        )
        
        # Apply transforms
        result = transforms(image=sample_image, masks=[sample_mask])
        
        # Check that image is transformed
        assert 'image' in result
        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape == (3, 512, 512)
        
        # Check that mask is transformed
        assert 'masks' in result
        assert len(result['masks']) == 1
        assert result['masks'][0].dtype == np.float32
    
    def test_resize_transform_application(self, sample_image):
        """Test that resize transform can be applied."""
        transforms = SegmentationTransforms.get_resize_transform(1024, 768)
        
        # Apply transforms
        result = transforms(image=sample_image)
        
        # Check that image is resized
        assert 'image' in result
        assert result['image'].shape == (1024, 768, 3)
    
    def test_sliding_window_transform_application(self, sample_image):
        """Test that sliding window transforms can be applied."""
        transforms = SegmentationTransforms.get_sliding_window_transforms()
        
        # Apply transforms
        result = transforms(image=sample_image)
        
        # Check that image is transformed
        assert 'image' in result
        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape == (3, 256, 256)  # Original size maintained
    
    def test_different_input_sizes(self):
        """Test transforms with different input sizes."""
        config = Config()
        config.normalize_mean = [0.485, 0.456, 0.406]
        config.normalize_std = [0.229, 0.224, 0.225]
        config.augment = False
        
        # Test with different sizes
        sizes = [(256, 256), (512, 512), (1024, 1024), (512, 768)]
        
        for height, width in sizes:
            config.input_size = (height, width)
            transforms = SegmentationTransforms.get_training_transforms(config)
            
            # Apply to a test image
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = transforms(image=test_image)
            
            # Check output size
            assert result['image'].shape == (3, height, width)
    
    def test_augmentation_probability(self, valid_config):
        """Test that augmentation probability is correctly set."""
        # Test with different probabilities
        probabilities = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for prob in probabilities:
            valid_config.augment_prob = prob
            transforms = SegmentationTransforms.get_training_transforms(valid_config)
            
            # Check that augmentation transforms have the correct probability
            for transform in transforms.transforms:
                if hasattr(transform, 'p'):
                    if transform.p is not None:  # Some transforms might not have p set
                        assert transform.p == prob
    
    def test_augmentation_parameters(self, valid_config):
        """Test that augmentation parameters are correctly set."""
        # Test with different augmentation parameters
        valid_config.brightness_limit = 0.3
        valid_config.contrast_limit = 0.3
        valid_config.noise_limit = (20.0, 60.0)
        valid_config.rotation_limit = 30
        
        transforms = SegmentationTransforms.get_training_transforms(valid_config)
        
        # Check that parameters are correctly set in transforms
        for transform in transforms.transforms:
            if isinstance(transform, A.RandomBrightnessContrast):
                assert transform.brightness_limit == 0.3
                assert transform.contrast_limit == 0.3
            elif isinstance(transform, A.GaussNoise):
                assert transform.var_limit == (20.0, 60.0)
            elif isinstance(transform, A.Rotate):
                assert transform.limit == 30
    
    def test_mask_dtype_conversion(self, valid_config, sample_image, sample_mask):
        """Test that masks are converted to float32 for semantic segmentation."""
        transforms = SegmentationTransforms.get_training_transforms(
            valid_config, is_instance_segmentation=False
        )
        
        # Apply transforms
        result = transforms(image=sample_image, masks=[sample_mask])
        
        # Check that mask is converted to float32
        assert result['masks'][0].dtype == np.float32
        
        # Test with different input mask dtypes
        for dtype in [np.uint8, np.uint16, np.int32, np.float64]:
            mask = sample_mask.astype(dtype)
            result = transforms(image=sample_image, masks=[mask])
            assert result['masks'][0].dtype == np.float32
    
    def test_bbox_format_validation(self, valid_config, sample_image, sample_bboxes, sample_labels):
        """Test that bbox format is correctly set for instance segmentation."""
        transforms = SegmentationTransforms.get_training_transforms(
            valid_config, is_instance_segmentation=True
        )
        
        # Check bbox parameters
        assert transforms.bbox_params.format == 'pascal_voc'
        assert transforms.bbox_params.min_area == 0
        assert transforms.bbox_params.min_visibility == 0
        assert 'labels' in transforms.bbox_params.label_fields
        
        # Apply transforms
        result = transforms(
            image=sample_image,
            bboxes=sample_bboxes,
            labels=sample_labels
        )
        
        # Check that bboxes are preserved
        assert 'bboxes' in result
        assert len(result['bboxes']) == len(sample_bboxes)
    
    def test_additional_targets_configuration(self, valid_config):
        """Test that additional targets are correctly configured."""
        # Test semantic segmentation
        semantic_transforms = SegmentationTransforms.get_training_transforms(
            valid_config, is_instance_segmentation=False
        )
        assert semantic_transforms.additional_targets == {}
        
        # Test instance segmentation
        instance_transforms = SegmentationTransforms.get_training_transforms(
            valid_config, is_instance_segmentation=True
        )
        assert instance_transforms.additional_targets == {'masks': 'masks'}
    
    def test_transform_chain_order(self, valid_config):
        """Test that transforms are applied in the correct order."""
        transforms = SegmentationTransforms.get_training_transforms(valid_config)
        
        # Check that resize is first
        assert isinstance(transforms.transforms[0], A.Resize)
        
        # Check that normalization is before ToTensorV2
        normalize_indices = [i for i, t in enumerate(transforms.transforms) 
                           if isinstance(t, A.Normalize)]
        to_tensor_indices = [i for i, t in enumerate(transforms.transforms) 
                           if isinstance(t, A.pytorch.ToTensorV2)]
        
        assert all(n < to_tensor_indices[0] for n in normalize_indices)
    
    def test_error_handling_edge_cases(self):
        """Test error handling for edge cases."""
        config = Config()
        
        # Test with completely invalid config
        with pytest.raises(ValueError):
            SegmentationTransforms.get_training_transforms(config)
        
        # Test with empty tuple
        config.input_size = ()
        with pytest.raises(ValueError):
            SegmentationTransforms.get_training_transforms(config)
        
        # Test with negative dimensions
        config.input_size = (-512, 512)
        config.normalize_mean = [0.485, 0.456, 0.406]
        config.normalize_std = [0.229, 0.224, 0.225]
        # This should work (albumentations handles negative values)
        transforms = SegmentationTransforms.get_training_transforms(config)
        assert isinstance(transforms, A.Compose)


class TestTransformsIntegration:
    """Integration tests for transforms with real data scenarios."""
    
    @pytest.fixture
    def realistic_config(self):
        """Create a realistic config for integration testing."""
        config = Config()
        config.input_size = (512, 512)
        config.normalize_mean = [0.485, 0.456, 0.406]
        config.normalize_std = [0.229, 0.224, 0.225]
        config.augment = True
        config.augment_prob = 0.5
        config.brightness_limit = 0.2
        config.contrast_limit = 0.2
        config.noise_limit = (10.0, 50.0)
        config.rotation_limit = 15
        return config
    
    def test_full_training_pipeline_semantic(self, realistic_config):
        """Test full training transform pipeline for semantic segmentation."""
        # Create realistic data
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        mask = np.random.randint(0, 32, (1024, 1024), dtype=np.uint8)
        
        # Get transforms
        transforms = SegmentationTransforms.get_training_transforms(
            realistic_config, is_instance_segmentation=False
        )
        
        # Apply transforms
        result = transforms(image=image, masks=[mask])
        
        # Validate results
        assert 'image' in result
        assert 'masks' in result
        
        # Check image
        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape == (3, 512, 512)
        assert result['image'].dtype == torch.float32
        
        # Check mask
        assert len(result['masks']) == 1
        assert result['masks'][0].shape == (512, 512)
        assert result['masks'][0].dtype == np.float32
    
    def test_full_training_pipeline_instance(self, realistic_config):
        """Test full training transform pipeline for instance segmentation."""
        # Create realistic data
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        mask = np.random.randint(0, 32, (1024, 1024), dtype=np.uint8)
        bboxes = [[100, 100, 300, 300], [400, 400, 600, 600]]
        labels = [1, 2]
        
        # Get transforms
        transforms = SegmentationTransforms.get_training_transforms(
            realistic_config, is_instance_segmentation=True
        )
        
        # Apply transforms
        result = transforms(
            image=image, 
            masks=[mask], 
            bboxes=bboxes, 
            labels=labels
        )
        
        # Validate results
        assert 'image' in result
        assert 'masks' in result
        assert 'bboxes' in result
        assert 'labels' in result
        
        # Check image
        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape == (3, 512, 512)
        
        # Check mask
        assert len(result['masks']) == 1
        assert result['masks'][0].shape == (512, 512)
        assert result['masks'][0].dtype == np.float32
        
        # Check bboxes (should be adjusted for resize)
        assert len(result['bboxes']) == 2
        
        # Check labels
        assert len(result['labels']) == 2
    
    def test_inference_pipeline(self, realistic_config):
        """Test inference transform pipeline."""
        # Create realistic data
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        mask = np.random.randint(0, 32, (1024, 1024), dtype=np.uint8)
        
        # Get transforms
        transforms = SegmentationTransforms.get_inference_transforms(
            realistic_config, is_instance_segmentation=False
        )
        
        # Apply transforms
        result = transforms(image=image, masks=[mask])
        
        # Validate results
        assert 'image' in result
        assert 'masks' in result
        
        # Check image
        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape == (3, 512, 512)
        
        # Check mask
        assert len(result['masks']) == 1
        assert result['masks'][0].shape == (512, 512)
        assert result['masks'][0].dtype == np.float32
    
    def test_resize_pipeline(self):
        """Test resize transform pipeline."""
        # Create realistic data
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Get transforms
        transforms = SegmentationTransforms.get_resize_transform(1024, 768)
        
        # Apply transforms
        result = transforms(image=image)
        
        # Validate results
        assert 'image' in result
        assert result['image'].shape == (1024, 768, 3)
    
    def test_sliding_window_pipeline(self):
        """Test sliding window transform pipeline."""
        # Create realistic data
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        # Get transforms
        transforms = SegmentationTransforms.get_sliding_window_transforms()
        
        # Apply transforms
        result = transforms(image=image)
        
        # Validate results
        assert 'image' in result
        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape == (3, 1024, 1024)  # Original size maintained


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"]) 