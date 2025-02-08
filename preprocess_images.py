from datasets import load_dataset
import os

# Define the dataset name and the splits you want to download
dataset_name = 'StanfordAIMI/mimic-cxr-images-512'
splits = ['train']  # You can add more splits if available
# Define the output directory
output_dir = 'mimic-cxr-images-512'

# Iterate over each split
for split in splits:
    print(f"Processing split: {split}")
    # Load the dataset split
    dataset = load_dataset(dataset_name, split=split)
    # Iterate over each item in the dataset
    for idx, item in enumerate(dataset):
        # Extract the path and image
        path = item['path']  
        image = item['image']  # PIL Image
        # Create the full output path
        full_path = os.path.join(output_dir, path)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        # Save the image as JPEG
        image.save(full_path, format='JPEG', quality=100, subsampling=0)
        # Optional: Print progress every 1000 images
        if idx % 1000 == 0:
            print(f"Saved {idx} images from split {split}")
