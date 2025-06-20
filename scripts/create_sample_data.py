# scripts/create_sample_data.py
# Create sample test data from full dataset

import json
import os
import zipfile
from PIL import Image
import random


def create_sample_data(
    source_images_dir: str,
    source_annotations: str,
    output_dir: str,
    num_samples: int = 2,
    target_size: tuple = (512, 512),
    specific_image_ids: list = None
):
    """
    Create sample data for testing.
    
    Args:
        source_images_dir: Path to source images 
        source_annotations: Path to COCO annotations
        output_dir: Where to save sample data
        num_samples: Number of sample images (ignored if specific_image_ids provided)
        target_size: Resize images to this size
        specific_image_ids: Optional list of specific image IDs to use
        
    Returns:
        tuple: (output_dir, zip_path)
    """
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    print("Loading annotations...")
    with open(source_annotations, 'r') as f:
        full_data = json.load(f)
    
    print(f"Found {len(full_data['images'])} total images")
    print(f"Found {len(full_data['annotations'])} total annotations")
    
    # Select sample images
    if specific_image_ids is not None:
        sample_image_ids = specific_image_ids
        print(f"Using specific image IDs: {sample_image_ids}")
    else:
        all_image_ids = [img['id'] for img in full_data['images']]    
        sample_image_ids = random.sample(all_image_ids, min(num_samples, len(all_image_ids)))
        print(f"Randomly selected image IDs: {sample_image_ids}")
    
    # Filter data for selected images
    sample_images = [img for img in full_data['images'] if img['id'] in sample_image_ids]
    sample_annotations = [ann for ann in full_data['annotations'] if ann['image_id'] in sample_image_ids]
    
    print(f"Processing {len(sample_images)} images with {len(sample_annotations)} annotations")
    
    # Process each image
    for img_info in sample_images:
        source_path = os.path.join(source_images_dir, img_info['file_name'])
        target_path = os.path.join(output_dir, "images", img_info['file_name'])
        
        print(f"Processing: {img_info['file_name']}")
        
        try:
            # Load, resize, and save image
            with Image.open(source_path) as img:
                # Convert to grayscale if not already
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Resize to keep file small
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                img_resized.save(target_path, optimize=True)
                
                # Update image dimensions in metadata
                img_info['width'] = target_size[0]
                img_info['height'] = target_size[1]
                
                print(f"  Saved {target_path} ({target_size[0]}x{target_size[1]})")
                
        except Exception as e:
            print(f"  Error processing {img_info['file_name']}: {e}")
    
    # Create single annotation file (covers both tooth and disease tasks)
    sample_data = {
        "images": sample_images,
        "annotations": sample_annotations,
        "categories": full_data.get('categories', []),
        "categories_1": full_data.get('categories_1', []),  # Quadrants
        "categories_2": full_data.get('categories_2', []),  # Tooth numbers
        "categories_3": full_data.get('categories_3', [])   # Diseases
    }
    
    # Save single annotation file
    annotation_file = os.path.join(output_dir, "annotations.json")
    with open(annotation_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    print(f"Created: annotations.json")
    
    # Create README
    readme_content = f"""# Sample Data for Testing

Created from dental X-ray dataset for automated testing.

## Contents:
- `images/`: {len(sample_images)} sample dental X-ray images ({target_size[0]}x{target_size[1]}, grayscale)
- `annotations.json`: COCO annotations with multi-category format

## Data Description:
- Images are resized versions of real dental X-rays
- Annotations include multi-category format:
  - `category_id_1`: Quadrant (Q1, Q2, Q3, Q4)
  - `category_id_2`: Tooth number (T1-T8)
  - `category_id_3`: Disease classification
- Sample image IDs: {sample_image_ids}
- **Supports both tooth and disease tasks** with single dataset

## Statistics:
- Total images: {len(sample_images)}
- Total annotations: {len(sample_annotations)}
- Quadrant categories: {len(sample_data.get('categories_1', []))}
- Tooth categories: {len(sample_data.get('categories_2', []))}
- Disease categories: {len(sample_data.get('categories_3', []))}

## Usage in Tests:
```python
# Works for both tooth and disease tasks
import os
sample_dir = "tests/sample_data"
annotation_file = os.path.join(sample_dir, "annotations.json")
image_dir = os.path.join(sample_dir, "images")

# Test tooth detection
tooth_dataset = ToothDataset(config, annotation_file=annotation_file)

# Test disease classification
disease_dataset = DiseaseDataset(config, annotation_file=annotation_file)
```
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    # Create zip file for easy transfer
    zip_path = os.path.join(output_dir, "sample_data.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files_list in os.walk(output_dir):
            for file in files_list:
                if file != "sample_data.zip":  # Don't include the zip file itself
                    file_path = os.path.join(root, file)
                    arc_path = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arc_path)
    
    print(f"\n✅ Sample data created successfully!")
    print(f"📁 Output directory: {output_dir}")
    
    # Show file sizes
    print(f"\n📊 File sizes:")
    for root, dirs, files_list in os.walk(output_dir):
        for file in files_list:
            file_path = os.path.join(root, file)
            size_kb = os.path.getsize(file_path) / 1024
            print(f"  {file}: {size_kb:.1f} KB")
    
    print(f"\n💡 Next steps:")
    print(f"1. Download/copy files from: {output_dir}")
    print(f"2. Extract to local: tests/sample_data/")
    print(f"3. Commit to repository")
    
    return output_dir, zip_path


# Example usage functions for different scenarios
def create_random_sample(source_images_dir: str, source_annotations: str, output_dir: str):
    """Create sample with randomly selected images."""
    return create_sample_data(
        source_images_dir=source_images_dir,
        source_annotations=source_annotations,
        output_dir=output_dir,
        num_samples=3
    )


def create_specific_sample(source_images_dir: str, source_annotations: str, output_dir: str):
    """Create sample with manually selected images for consistent testing."""
    # Replace with actual image IDs from your dataset that provide good test coverage
    specific_ids = [1, 15, 42]  # Example IDs - update these based on your data
    
    return create_sample_data(
        source_images_dir=source_images_dir,
        source_annotations=source_annotations,
        output_dir=output_dir,
        specific_image_ids=specific_ids
    )


if __name__ == "__main__":
    print("Sample data creation script.")
    print("Import and use create_sample_data() or create_random_sample()/create_specific_sample()")
    print("Designed to run in environment with access to source data (e.g., Google Colab)")