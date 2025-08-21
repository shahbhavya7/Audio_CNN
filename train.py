from pathlib import Path
import sys
import pandas as pd
import numpy as np

import modal
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import AudioCNN

app = modal.App("cnn-aud")

image = (modal.Image.debian_slim() # defines the base image for the Modal app, starts from a lightweight Debian base image.
         .pip_install_from_requirements("requirements.txt") # installs Python dependencies from a requirements.txt file
         .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"]) # installs system packages like wget, ffmpeg, libsndfile1 (needed for audio preprocessing).
         .run_commands([ # runs shell commands inside the image (here: downloads and unzips the ESC-50 dataset, copies it into /opt/esc50-data).
             "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
             "cd /tmp && unzip esc50.zip",
             "mkdir -p /opt/esc50-data",
             "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
             "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
         ])
         .add_local_python_source("model")) # adds the local Python source code (model.py) to the image, so it can be used in the Modal app.
 
# image defines the environment (system + Python + dataset + your code) in which your Modal app will run for training and inference.

volume = modal.Volume.from_name("esc50-data", create_if_missing=True) # defines a volume to store the ESC-50 dataset, creating it if it doesn't exist.
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True) # defines a volume to store the trained model, creating it if it doesn't exist.

class ESC50Dataset(Dataset): # This class defines the ESC-50 dataset for audio classification.
    # It inherits from torch.utils.data.Dataset, which is a standard way to create custom datasets
    # here we make this class to load audio files and their corresponding labels from the ESC-50 dataset.
    def __init__(self, data_dir, metadata_file, split="train", transform=None): 
        super().__init__()
        self.data_dir = Path(data_dir) # data_dir is the directory where the audio files are stored.
        self.metadata = pd.read_csv(metadata_file) # metadata_file is a CSV file containing metadata about the audio files (like labels, fold, etc.).
        self.split = split # split indicates whether this dataset is for training or testing.
        self.transform = transform # transform is an optional transformation to apply to the audio data (like spectrogram conversion).

        if split == 'train': # if the split is 'train', we exclude fold 5 (used for testing).
            self.metadata = self.metadata[self.metadata['fold'] != 5]
        else:
            self.metadata = self.metadata[self.metadata['fold'] == 5]

        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(
            self.class_to_idx)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row['filename']

        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        return spectrogram, row['label']

@app.function(image=image, gpu="A10G", volumes={"/data": volume, "/models": model_volume}, timeout=60 * 60 * 3)
# runs on A10G GPU, mounts the dataset and model volumes, and sets a timeout of 3 hours for training.
def train():
    print("Starting training...")

@app.local_entrypoint()
def main():
    train.remote()