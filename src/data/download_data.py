"""
Data download script for Cats vs Dogs dataset (Assignment 2).
"""

import os
from pathlib import Path


def download_cats_vs_dogs_data(output_dir: str = "data/raw") -> str:
    """
    Download the Cats vs Dogs classification dataset from Kaggle using kagglehub.

    Dataset:
        https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

    Requirements (one-time Kaggle API setup):
        1. Go to https://www.kaggle.com/settings/account
        2. Click "Create New API Token" (downloads kaggle.json)
        3. Place kaggle.json in the ~/.kaggle/ directory

    Args:
        output_dir: Directory under which the dataset will be stored.

    Returns:
        Path to the downloaded dataset directory, or None on failure.
    """
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub library not installed. Install with: pip install kagglehub")
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Downloading 'dog-and-cat-classification-dataset' from Kaggle...")
    try:
        dataset_path = kagglehub.dataset_download(
            "bhavikjikadara/dog-and-cat-classification-dataset"
        )
        print("Dataset downloaded successfully!")
        print(f"Dataset path: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"ERROR: Failed to download dataset: {e}")
        print("Please ensure Kaggle API credentials are set up (see docstring).")
        return None


if __name__ == "__main__":
    dataset_dir = download_cats_vs_dogs_data()
    if dataset_dir and os.path.exists(dataset_dir):
        print("\nCats vs Dogs dataset download complete!")
    else:
        print("Failed to download Cats vs Dogs dataset. Please check your Kaggle setup.")
