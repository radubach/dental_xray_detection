# dentalvision/config.py

import os
from pathlib import Path


class Config:
    """Configuration class for DentalVision project."""
    
    # =============================================================================
    # BASE PATHS
    # =============================================================================
    RAW_DATA_DIR = "/content/drive/MyDrive/Dentex_raw"
    
    # =============================================================================  
    # TASK-SPECIFIC PATHS (Explicit and testable)
    # =============================================================================
    
    # Tooth detection/segmentation paths
    TOOTH_ANNOTATION_FILE = os.path.join(RAW_DATA_DIR, 'DENTEX', 'train', 'training_data', 
                                         'quadrant_enumeration', 'train_quadrant_enumeration.json')
    TOOTH_IMAGE_DIR = os.path.join(RAW_DATA_DIR, 'DENTEX', 'train', 'training_data',
                                   'quadrant_enumeration', 'xrays')
    
    # Disease classification paths  
    DISEASE_ANNOTATION_FILE = os.path.join(RAW_DATA_DIR, 'DENTEX', 'train', 'training_data',
                                           'quadrant-enumeration-disease', 'train_quadrant_enumeration_disease.json')
    DISEASE_IMAGE_DIR = os.path.join(RAW_DATA_DIR, 'DENTEX', 'train', 'training_data',
                                     'quadrant-enumeration-disease', 'xrays')
    
    # =============================================================================
    # DATASET SETTINGS
    # =============================================================================
    DATASET_NAME = "ibrahimhamamci/DENTEX"
    TASKS = ["tooth_detection", "disease_detection"]
    
    # Data splits
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1
    
    # Data loading
    BATCH_SIZE = 16
    NUM_WORKERS = 2  # Lower for Colab
    PIN_MEMORY = True
    
    # =============================================================================
    # IMAGE TRANSFORMS & AUGMENTATION
    # =============================================================================
    # Image dimensions
    INPUT_SIZE = (512, 512)  # (height, width)
    
    # Normalization
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet standard
    NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet standard
    
    # Augmentation settings
    AUGMENT = True
    AUGMENT_PROB = 0.5
    
    # Augmentation parameters
    BRIGHTNESS_LIMIT = 0.2
    CONTRAST_LIMIT = 0.2
    NOISE_LIMIT = (10.0, 50.0)
    ROTATION_LIMIT = 15
    
    # =============================================================================
    # MODEL SETTINGS
    # =============================================================================
    # Model architecture
    MODEL_TYPE = "unet"  # "unet" or "maskrcnn"
    IN_CHANNELS = 3      # RGB images
    OUT_CHANNELS = 33    # 32 teeth + background
    PRETRAINED = True
    
    # UNet specific
    UNET_FEATURES_START = 64
    UNET_NUM_LAYERS = 4
    
    # Mask R-CNN specific
    MASKRCNN_BACKBONE = "resnet50"
    MASKRCNN_TRAINABLE_LAYERS = 5
    MASKRCNN_RPN_SCORE_THRESH = 0.05
    MASKRCNN_BOX_SCORE_THRESH = 0.05
    
    # =============================================================================
    # TRAINING SETTINGS
    # =============================================================================
    # Training parameters
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP_VAL = 1.0
    
    # Optimizer
    OPTIMIZER_TYPE = "adamw"  # "adam", "sgd", "adamw"
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.999
    SGD_MOMENTUM = 0.9
    
    # Training features
    MIXED_PRECISION = True
    DEVICE = "cuda"
    
    # =============================================================================
    # VALIDATION & CHECKPOINTING
    # =============================================================================
    # Validation
    VAL_FREQUENCY = 1  # Validate every N epochs
    EARLY_STOPPING_PATIENCE = 10
    
    # Checkpointing
    SAVE_FREQUENCY = 5  # Save checkpoint every N epochs
    KEEP_LAST_K_CHECKPOINTS = 3
    SAVE_BEST_MODEL = True
    
    # =============================================================================
    # LOGGING & MONITORING
    # =============================================================================
    EXPERIMENT_NAME = "dental_segmentation"
    USE_WANDB = False
    WANDB_PROJECT = "dental-xray-detection"
    LOG_INTERVAL = 10  # Log every N batches
    
    # =============================================================================
    # EVALUATION SETTINGS
    # =============================================================================
    # Metrics to track
    TRACK_METRICS = ["dice", "iou", "pixel_accuracy"]
    
    # Inference settings
    TEST_TIME_AUGMENTATION = False
    SLIDING_WINDOW_INFERENCE = False
    

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    @classmethod
    def create_test_config(cls, sample_data_dir: str):
        """Create a test configuration that uses sample data."""
        test_config = cls()
        
        # Override paths to use sample data
        test_config.TOOTH_ANNOTATION_FILE = os.path.join(sample_data_dir, 'annotations.json')
        test_config.TOOTH_IMAGE_DIR = os.path.join(sample_data_dir, 'images')
        test_config.DISEASE_ANNOTATION_FILE = os.path.join(sample_data_dir, 'annotations.json')  
        test_config.DISEASE_IMAGE_DIR = os.path.join(sample_data_dir, 'images')
        test_config.COCO_ANNOTATION_FILE = os.path.join(sample_data_dir, 'annotations.json')
        test_config.COCO_IMAGE_DIR = os.path.join(sample_data_dir, 'images')
        
        return test_config
    
    @classmethod  
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary (useful for testing)."""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def get_annotation_file(self, task_type: str) -> str:
        """Get annotation file path for specific task."""
        task_mapping = {
            'teeth': self.TOOTH_ANNOTATION_FILE,
            'tooth': self.TOOTH_ANNOTATION_FILE, 
            'quadrant': self.TOOTH_ANNOTATION_FILE,
            'disease': self.DISEASE_ANNOTATION_FILE,
            'coco': self.COCO_ANNOTATION_FILE
        }
        return task_mapping.get(task_type, self.COCO_ANNOTATION_FILE)
    
    def get_image_dir(self, task_type: str) -> str:
        """Get image directory path for specific task."""
        task_mapping = {
            'teeth': self.TOOTH_IMAGE_DIR,
            'tooth': self.TOOTH_IMAGE_DIR,
            'quadrant': self.TOOTH_IMAGE_DIR, 
            'disease': self.DISEASE_IMAGE_DIR,
            'coco': self.COCO_IMAGE_DIR
        }
        return task_mapping.get(task_type, self.COCO_IMAGE_DIR)


    @classmethod
    def get_model_config(cls):
        """Get model-specific configuration as a dict."""
        if cls.MODEL_TYPE == "unet":
            return {
                "in_channels": cls.IN_CHANNELS,
                "out_channels": cls.OUT_CHANNELS,
                "features_start": cls.UNET_FEATURES_START,
                "num_layers": cls.UNET_NUM_LAYERS,
                "pretrained": cls.PRETRAINED
            }
        elif cls.MODEL_TYPE == "maskrcnn":
            return {
                "backbone": cls.MASKRCNN_BACKBONE,
                "trainable_backbone_layers": cls.MASKRCNN_TRAINABLE_LAYERS,
                "rpn_score_thresh": cls.MASKRCNN_RPN_SCORE_THRESH,
                "box_score_thresh": cls.MASKRCNN_BOX_SCORE_THRESH,
                "pretrained": cls.PRETRAINED
            }
        else:
            raise ValueError(f"Unknown model type: {cls.MODEL_TYPE}")
    
    @classmethod
    def get_optimizer_config(cls):
        """Get optimizer configuration as a dict."""
        base_config = {
            "lr": cls.LEARNING_RATE,
            "weight_decay": cls.WEIGHT_DECAY
        }
        
        if cls.OPTIMIZER_TYPE == "adam":
            base_config.update({
                "betas": (cls.ADAM_BETA1, cls.ADAM_BETA2)
            })
        elif cls.OPTIMIZER_TYPE == "sgd":
            base_config.update({
                "momentum": cls.SGD_MOMENTUM
            })
        
        return base_config
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("=" * 60)
        print("CURRENT CONFIGURATION")
        print("=" * 60)
        
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and not callable(getattr(cls, attr_name)):
                value = getattr(cls, attr_name)
                print(f"{attr_name}: {value}")
        print("=" * 60)