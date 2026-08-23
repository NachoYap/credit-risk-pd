
import shutil
import kagglehub
from pathlib import Path

# Set the path to the file you'd like to load
file_path = Path(__file__).resolve().parent.parent / "data" / "data_raw"
file_path.mkdir(parents=True, exist_ok=True)

# Download the latest version (kagglehub caches it under its own directory
# and returns that path — there is no built-in way to point it at a
# destination, so copy the files into data_raw ourselves).
cache_path = Path(kagglehub.dataset_download("megancrenshaw/home-credit-default-risk"))
shutil.copytree(cache_path, file_path, dirs_exist_ok=True)
print(f"Copied dataset from {cache_path} to {file_path}")
