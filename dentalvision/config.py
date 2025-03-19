class Config:
    """Configuration class for the dental_xray_detection project."""
    
    # Default Google Drive directories
    RAW_DATA_DIR = "/content/drive/MyDrive/Dentex_raw"
    PROCESSED_DATA_DIR = "/content/drive/MyDrive/Dentex_processed"
    
    # Dataset-specific settings
    DATASET_NAME = "ibrahimhamamci/DENTEX"  # Hugging Face dataset identifier
    TASKS = ["tooth_detection", "disease_detection"]  # Tasks to process
    SPLITS = ["train", "val"]  # Data splits
    
    # Add more configs as needed (e.g., model parameters, file extensions)


    