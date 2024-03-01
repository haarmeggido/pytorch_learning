"""
Gets the pizza Steak Sushi dataset from the web and saves it to disk.
"""

import os
import requests
import zipfile
from pathlib import Path

# Setup path to data directory
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

if image_path.is_dir():
  print(f'directory {image_path} already exists')
else:
  print(f'directory {image_path} does not exist')
  image_path.mkdir(parents=True, exist_ok=True)

#Download data, if not already downloaded
if (len(os.listdir(data_path / "pizza_steak_sushi"))):
  print(f'folder {data_path / "pizza_steak_sushi"} already exists and is not empty. Skipping download.')
else:
  with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading pizza, steak, sushi data...")
    f.write(request.content)

  # Unzip data
  with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    zip_ref.extractall(image_path)
    print("Unzipping pizza, steak, sushi data...")

  # Remove zip file
  os.remove(data_path / "pizza_steak_sushi.zip")