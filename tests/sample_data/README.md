# Sample Data for Testing

Created from dental X-ray dataset for automated testing.

## Contents:
- `images/`: 2 sample dental X-ray images (512x512, grayscale)
- `annotations.json`: COCO annotations with multi-category format

## Data Description:
- Images are resized versions of real dental X-rays
- Annotations include multi-category format:
  - `category_id_1`: Quadrant (Q1, Q2, Q3, Q4)
  - `category_id_2`: Tooth number (T1-T8)
  - `category_id_3`: Disease classification
- Sample image IDs: [19, 417]
- **Supports both tooth and disease tasks** with single dataset

## Statistics:
- Total images: 2
- Total annotations: 15
- Quadrant categories: 4
- Tooth categories: 8
- Disease categories: 4

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
