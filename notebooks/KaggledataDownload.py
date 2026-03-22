
import kagglehub
from kagglehub import KaggleDatasetAdapter
from pathlib import Path
# Set the path to the file you'd like to load
file_path = Path(Path.cwd().parent/"data/data_raw")
# Load the latest version
kagglehub.dataset_download("megancrenshaw/home-credit-default-risk", output_dir=file_path)
