# Sample Data for Testing

This directory contains minimal sample data for automated testing.

## Contents:

- `images/`: 2 sample dental X-ray images (512x512, grayscale)
- `tooth_annotations.json`: COCO annotations for tooth detection/segmentation
- `disease_annotations.json`: COCO annotations for disease classification

## Data Description:

- Images are resized versions of real dental X-rays
- Annotations include multi-category format (category_id_1, category_id_2, category_id_3)
- Covers basic test cases: multiple teeth, different quadrants, various diseases

## Usage in Tests:

```python
import os
sample_dir = os.path.join(os.path.dirname(__file__), "sample_data")
annotation_file = os.path.join(sample_dir, "tooth_annotations.json")
image_dir = os.path.join(sample_dir, "images")
```

## File Sizes:

- Total size: < 500KB
- Safe to include in Git repository
- Automatically used by CI/CD testing
