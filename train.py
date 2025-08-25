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
            self.metadata = self.metadata[self.metadata['fold'] != 5] # storing fold information in the metadata dataframe.
            # fold means dataset is divided into 5 parts, 4 for training and 1 for testing. 
        else:
            self.metadata = self.metadata[self.metadata['fold'] == 5]

        self.classes = sorted(self.metadata['category'].unique()) # get unique classes from the metadata and sort them according to alphabetical order.
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)} # for every class we give it unique index.
        # like "dog" -> 0, "cat" -> 1, etc and store this mapping in a dictionary called class_to_idx.
        self.metadata['label'] = self.metadata['category'].map( # now make a new column in metadata dataframe called label 
            # which contains the index of the class just generated above. in dataframe each category is assigned a number in label column using class_to_idx dictionary.
            self.class_to_idx)

    def __len__(self):
        return len(self.metadata) # returns the total number of samples in the dataset.

    def __getitem__(self, idx): # retrieves a single sample from the dataset at the specified index idx.
        import torchaudio
        row = self.metadata.iloc[idx] 
        audio_path = self.data_dir / "audio" / row['filename'] # constructs the full path to the audio file using the filename from the metadata.

        waveform, sample_rate = torchaudio.load(audio_path) # loads the audio file using torchaudio, which returns the waveform (audio signal) and sample rate.
        # waveform is a tensor of shape (channels, time), where channels is usually 1 for mono audio and time is the number of samples i.e length of audio in samples.

        if waveform.shape[0] > 1: # if the audio has more than one channel (i.e stereo), we convert it to mono by averaging the channels.
            waveform = torch.mean(waveform, dim=0, keepdim=True) # averaging across the channel dimension to get a single channel.

        if self.transform: # if a transform is provided (like converting to spectrogram), we apply it to the waveform.
            spectrogram = self.transform(waveform)
        else: # if no transform is provided, we just return the raw waveform as the spectrogram.
            spectrogram = waveform

        return spectrogram, row['label'] # returns the spectrogram (or waveform) and the corresponding label for the audio file.

@app.function(image=image, gpu="A10G", volumes={"/data": volume, "/models": model_volume}, timeout=60 * 60 * 3)
# runs on A10G GPU, mounts the dataset and model volumes, and sets a timeout of 3 hours for training.
def train():
    import torchaudio
    from torchaudio import transforms as T
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'/models/tensorboard_logs/run_{timestamp}'
    writer = SummaryWriter(log_dir)

    esc50_dir = Path("/opt/esc50-data") # directory where the ESC-50 dataset is stored inside the Modal container.

    train_transform = nn.Sequential( # defining a sequence of transformations to apply to the audio data during training.
        T.MelSpectrogram( # converts the waveform to a Mel spectrogram, which is a common representation for audio data in machine learning.
            sample_rate=22050, # sample rate of the audio i.e number of samples per second.
            n_fft=1024, # size of the FFT window i.e number of frequency bins.
            hop_length=512, # number of samples between successive frames. this controls the overlap between frames.
            n_mels=128, # number of Mel bands to generate. more Mel bands means higher frequency resolution.
            f_min=0, # minimum frequency to include in the Mel spectrogram. here we include all frequencies from 0 Hz.
            f_max=11025 # maximum frequency to include in the Mel spectrogram. here we include frequencies up to 11025 Hz (half of sample rate, Nyquist frequency).
        ),
        T.AmplitudeToDB(), # converts the amplitude of the spectrogram to decibels (dB) for better visualization and learning.
        T.FrequencyMasking(freq_mask_param=30), # applies frequency masking as a form of data augmentation, randomly masks out some frequency bands to make the model robust to frequency variations.
        T.TimeMasking(time_mask_param=80) # applies time masking as a form of data augmentation, randomly masks out some time segments to make the model robust to temporal variations.
    )

    val_transform = nn.Sequential( # defining transformations for validation data, similar to training but without data augmentation as we want to evaluate on clean data and not augmented data.
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB()
    )

    train_dataset = ESC50Dataset( # creating the training dataset using the ESC50Dataset class defined above.
        data_dir=esc50_dir, metadata_file=esc50_dir / "meta" / "esc50.csv", split="train", transform=train_transform)

    val_dataset = ESC50Dataset( # creating the validation dataset using the ESC50Dataset class defined above.
        data_dir=esc50_dir, metadata_file=esc50_dir / "meta" / "esc50.csv", split="test", transform=val_transform)

    print(f"Training samples: {len(train_dataset)}") # printing the number of samples in the training and validation datasets.
    print(f"Val samples: {len(val_dataset)}")

    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = AudioCNN(num_classes=len(train_dataset.classes))
    # model.to(device)

    # num_epochs = 100
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    # scheduler = OneCycleLR(
    #     optimizer,
    #     max_lr=0.002,
    #     epochs=num_epochs,
    #     steps_per_epoch=len(train_dataloader),
    #     pct_start=0.1
    # )

    # best_accuracy = 0.0

    # print("Starting training")
    # for epoch in range(num_epochs):
    #     model.train()
    #     epoch_loss = 0.0

    #     progress_bar = tqdm(
    #         train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
    #     for data, target in progress_bar:
    #         data, target = data.to(device), target.to(device)

    #         if np.random.random() > 0.7:
    #             data, target_a, target_b, lam = mixup_data(data, target)
    #             output = model(data)
    #             loss = mixup_criterion(
    #                 criterion, output, target_a, target_b, lam)
    #         else:
    #             output = model(data)
    #             loss = criterion(output, target)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()

    #         epoch_loss += loss.item()
    #         progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

    #     avg_epoch_loss = epoch_loss / len(train_dataloader)
    #     writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
    #     writer.add_scalar(
    #         'Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

    #     # Validation after each epoch
    #     model.eval()

    #     correct = 0
    #     total = 0
    #     val_loss = 0

    #     with torch.no_grad():
    #         for data, target in test_dataloader:
    #             data, target = data.to(device), target.to(device)
    #             outputs = model(data)
    #             loss = criterion(outputs, target)
    #             val_loss += loss.item()

    #             _, predicted = torch.max(outputs.data, 1)
    #             total += target.size(0)
    #             correct += (predicted == target).sum().item()

    #     accuracy = 100 * correct / total
    #     avg_val_loss = val_loss / len(test_dataloader)

    #     writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
    #     writer.add_scalar('Accuracy/Validation', accuracy, epoch)

    #     print(
    #         f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

    #     if accuracy > best_accuracy:
    #         best_accuracy = accuracy
    #         torch.save({
    #             'model_state_dict': model.state_dict(),
    #             'accuracy': accuracy,
    #             'epoch': epoch,
    #             'classes': train_dataset.classes
    #         }, '/models/best_model.pth')
    #         print(f'New best model saved: {accuracy:.2f}%')

    # writer.close()
    # print(f'Training completed! Best accuracy: {best_accuracy:.2f}%')

@app.local_entrypoint()
def main():
    train.remote()