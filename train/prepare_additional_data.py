import os
import zipfile

import librosa
import soundfile as sf
import requests
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# Configuration
# --------------------------------------------------

# Directory where additional datasets will be stored
extract_dir = "../additional_data"

# (Download URL, dataset folder name)
urls = [("https://archive.org/download/ff1010bird/ff1010bird_wav.zip", "freefield1010"),
        ("https://zenodo.org/record/1208080/files/BirdVox-DCASE-20k.zip", "BirdVox-DCASE-20k"),
        ("https://github.com/karoldvl/ESC-50/archive/master.zip", "ESC-50")
        ]

# ==================================================
# 1. Download + Selective Extraction
# ==================================================
for i, (url, dataset) in enumerate(urls):
    # -----------------------
    # Download ZIP file
    # -----------------------
    print("downloading dataset {}...".format(dataset),)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        zip_filename = f"../additional_data/{dataset}.zip"

        with open(zip_filename, "wb") as f, tqdm(
            desc=zip_filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


    # -----------------------
    # Extract ZIP file
    # -----------------------
    os.makedirs(extract_dir + f'/{dataset}', exist_ok=True)

    no_call_list = open('../additional_data/no_call_samples.txt', 'r').read().splitlines()

    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        members = zip_ref.infolist()

        for member in tqdm(members, desc="Extracting", unit="file"):
            if member.filename.split('/')[-1] in no_call_list:
                zip_ref.extract(member, extract_dir + f'/{dataset}')

    print(f"Unzipped to '{extract_dir}'")




# ==================================================
# 2. Crawl iNaturalist Audio Data
# ==================================================
target_dir = '../additional_data/iNat'
os.makedirs('../additional_data/iNat_insects', exist_ok=True)
obs_ids = open('../additional_data/inaturalist_insect_observation_ids.txt').read().splitlines()
for obs_id in tqdm(obs_ids, desc="Crawl iNaturalist"):
    url = f"https://api.inaturalist.org/v1/observations/{obs_id}"

    data = requests.get(url).json()
    if len(data["results"]) == 0:
        print(f"obs:{obs_id} failed")
        continue
    obs = data["results"][0]

    # audio
    for i, sound in enumerate(obs["sounds"]):
        audio = requests.get(sound["file_url"]).content

        with open(f"{target_dir}/{obs_id}.mp3", "wb") as f:
            f.write(audio)


# ==================================================
# 3. Resample All Audio to 32 kHz (In-Place)
# ==================================================
target_sr = 32_000
dirs = ["../additional_data/freefield1010", "../additional_data/BirdVox-DCASE-20k/", "../additional_data/ESC-50/", "../additional_data/iNat/"]
# dirs = ["../additional_data/iNat/"]

wav_files = []

for d in dirs:
    for root, _, files in os.walk(d):
        for f in files:
            if f.lower().endswith(".wav") or f.lower().endswith(".mp3"):
                wav_files.append(os.path.join(root, f))

    for path in tqdm(wav_files, desc="Resampling", unit="file"):
        audio, sr = librosa.load(path, sr=None)

        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sf.write(path, audio, target_sr)

