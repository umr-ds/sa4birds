"""
Download and extract pretrained model checkpoints.

This script downloads multiple checkpoint archives (ZIP files)
from remote URLs and extracts them into local directories.

Checkpoints are grouped into:
    - DT
    - MT
    - LT

Behavior:
    - The first 8 URLs (DT checkpoints) are extracted into:
        checkpoints/DT/
    - Remaining checkpoints (LT and MT) are extracted into:
        checkpoints/

Requirements:
    - requests
    - zipfile (standard library)
    - os (standard library)

Note:
    - Each file is temporarily saved as "file.zip" and overwritten
      for each download.
    - Ensure sufficient disk space before running.
"""

import requests
import zipfile
import os


# Downstream task checkpoints
dt_checkpoints = [
    "https://next.hessenbox.de/index.php/s/KR92DHDjYCSMREc/download",
    "https://next.hessenbox.de/index.php/s/fYKk7FDG446jgxD/download",
    "https://next.hessenbox.de/index.php/s/7YRQ2NopSGmsxFX/download",
    "https://next.hessenbox.de/index.php/s/dKjJgk3WEpFGpf4/download",
    "https://next.hessenbox.de/index.php/s/GHMrTbregzZ66CE/download",
    "https://next.hessenbox.de/index.php/s/Jr3KWKMMJyF4Zgb/download",
    "https://next.hessenbox.de/index.php/s/qKmMDPyQSzRRzoo/download",
    "https://next.hessenbox.de/index.php/s/GPk5MdGsLikmHKa/download",
]

# Multi-task checkpoint
mt_checkpoints = [
    "https://next.hessenbox.de/index.php/s/ck8J8A95DdssSo4/download"
]

# Long-term checkpoint
lt_checkpoints = [
    "https://next.hessenbox.de/index.php/s/xxE5XTaNcHCXidy/download"
]


for i, url in enumerate(dt_checkpoints + lt_checkpoints + mt_checkpoints):
    """
    Download and extract each checkpoint archive.

    Args:
        i (int): Index of the current checkpoint in the combined list.
        url (str): Direct download URL of the ZIP archive.

    Extraction logic:
        - Indices 0–7 (DT checkpoints) → "checkpoints/DT/"
        - Remaining checkpoints → "checkpoints/"
    """

    zip_filename = "file.zip"
    extract_dir = "checkpoints/DT"

    # Non-DT checkpoints are extracted to a different directory
    if i > 7:
        extract_dir = "checkpoints/"

    # -----------------------
    # Download ZIP file
    # -----------------------
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print("Download complete.")

    # -----------------------
    # Extract ZIP file
    # -----------------------
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Unzipped to '{extract_dir}'")