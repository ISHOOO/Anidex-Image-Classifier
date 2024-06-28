import os
import shutil
import random

def split_dataset(base_dir, train_dir, valid_dir, split_ratio=0.75):
    # Create train and validation directories if they do not exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    # Iterate over each sub-directory in the base directory
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            # Create corresponding sub-directories in train and valid directories
            train_subdir = os.path.join(train_dir, subdir)
            valid_subdir = os.path.join(valid_dir, subdir)
            os.makedirs(train_subdir, exist_ok=True)
            os.makedirs(valid_subdir, exist_ok=True)

            # List all files in the current sub-directory
            files = os.listdir(subdir_path)
            random.shuffle(files)  # Shuffle the files to ensure random split

            # Calculate the split point
            split_point = int(len(files) * split_ratio)

            # Copy files to train and valid sub-directories
            for file in files[:split_point]:
                shutil.copy(os.path.join(subdir_path, file), os.path.join(train_subdir, file))
            for file in files[split_point:]:
                shutil.copy(os.path.join(subdir_path, file), os.path.join(valid_subdir, file))

# Define directories
base_dir = 'animals data'
train_dir = 'train'
valid_dir = 'valid'

# Split the dataset
split_dataset(base_dir, train_dir, valid_dir)