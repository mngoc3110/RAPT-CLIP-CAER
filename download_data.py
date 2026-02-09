import kagglehub
import shutil
import os

# Download latest version
print("Downloading CK+ dataset...")
path = kagglehub.dataset_download("shawon10/ckplus")

print("Path to dataset files:", path)

# Define target directory
target_dir = os.path.join(os.getcwd(), "ckplus")

# Move the downloaded files to the target directory
print(f"Moving dataset to {target_dir}...")
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
shutil.move(path, target_dir)

print("Download and setup complete.")
