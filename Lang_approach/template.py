import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# List of files to be created
list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompts.py",
    ".env",
    "requirements.txt",
    "setup.py",
    "app.py",
    "research/trials.ipynb"
]

for file_path in list_of_files:
    file = Path(file_path)
    dir_path = file.parent

    # Create the directory if it doesn't exist
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")

    # Create the file if it doesn't exist or is empty
    if not file.exists() or file.stat().st_size == 0:
        file.touch()
        logging.info(f"Created file: {file}")
    else:
        logging.info(f"File already exists: {file}")
