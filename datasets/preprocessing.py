import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image

# Paths
data_dir = "datasets/Images"  # Replace with the path to your main images folder
output_dir = "datasets"  # Where to store the split data
categories = ["glasses", "no_glasses"]

# Create directories for train, validation, and test sets
for split in ['train', 'valid', 'test']:
    for category in categories:
        os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)

# Split ratios
train_ratio = 0.7
validation_ratio = 0.2
test_ratio = 0.1

def resize_image(image_path, output_path, size=(224, 224)):
    """Resize an image to the given size and save it to the output path."""
    with Image.open(image_path) as img:
        img_resized = img.resize(size, Image.Resampling.LANCZOS)
        img_resized.save(output_path)

# Loop through each category and split the data
for category in categories:
    category_path = os.path.join(data_dir, category)
    images = os.listdir(category_path)
    
    # Ensure proper paths for each image
    images = [os.path.join(category_path, img) for img in images]
    
    # Split into train and temp (validation + test)
    train_images, temp_images = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
    # Split temp into validation and test
    validation_images, test_images = train_test_split(temp_images, test_size=(test_ratio / (test_ratio + validation_ratio)), random_state=42)
    
    # Move files to respective folders
    for img_path in train_images:
        output_path = os.path.join(output_dir, 'train', category, os.path.basename(img_path))
        resize_image(img_path, output_path)
    for img_path in validation_images:
        output_path = os.path.join(output_dir, 'valid', category, os.path.basename(img_path))
        resize_image(img_path, output_path)
    for img_path in test_images:
        output_path = os.path.join(output_dir, 'test', category, os.path.basename(img_path))
        resize_image(img_path, output_path)

print("Data successfully split and resized to train, validation, and test sets.")
