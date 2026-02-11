import requests
import zipfile
import os


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

mt_checkpoints = ["https://next.hessenbox.de/index.php/s/ck8J8A95DdssSo4/download"]

lt_checkpoints = ["https://next.hessenbox.de/index.php/s/xxE5XTaNcHCXidy/download"]

for i, url in enumerate(dt_checkpoints + lt_checkpoints + mt_checkpoints):
    zip_filename = "file.zip"
    extract_dir = "checkpoints/DT"
    if i > 7:
        extract_dir = "checkpoints/"

    # Download
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print("Download complete.")

    # Unzip
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Unzipped to '{extract_dir}'")
