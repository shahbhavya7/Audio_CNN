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
    
def mixup_data(x, y): # it takes a batch of inputs x and their corresponding labels y, and returns mixed inputs and labels creating new training samples for data augmentation.
    # data augmentation technique that creates new training samples by combining pairs of examples and their labels.
    lam = np.random.beta(0.2, 0.2) # lambda is sampled from a Beta distribution with parameters alpha=0.2 and beta=0.2, which controls the degree of mixing between two samples.

    batch_size = x.size(0)  # batch_size is the number of samples in the batch extracted from the first dimension of x.
    index = torch.randperm(batch_size).to(x.device) # index is a random permutation of indices from 0 to batch_size-1, used to shuffle the batch.

    mixed_x = lam * x + (1 - lam) * x[index, :] # mixed_x is the new input created by linearly combining each sample in x with another randomly selected 
    # sample from the batch, weighted by lambda. like (0.7 * sample1 + 0.3 * sample2) i.e 70% of sample1 and 30% of sample2 in the new sample.
    y_a, y_b = y, y[index] # y_a is the original label and y_b is the label of the randomly selected sample via index.
    return mixed_x, y_a, y_b, lam # returns the mixed inputs, the two sets of labels, and the mixing coefficient lambda.


def mixup_criterion(criterion, pred, y_a, y_b, lam): # computes the loss for the mixed inputs created by mixup_data, using the provided loss function criterion.
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b) # computes the loss as a weighted sum of the losses for the two sets of labels, weighted by lambda and (1-lambda) respectively.

@app.function(image=image, gpu="A10G", volumes={"/data": volume, "/models": model_volume}, timeout=60 * 60 * 3)
# runs on A10G GPU, mounts the dataset and model volumes, and sets a timeout of 3 hours for training.
def train():
    import torchaudio
    from torchaudio import transforms as T
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # generates a timestamp string to uniquely identify the training run.
    log_dir = f'/models/tensorboard_logs/run_{timestamp}' # directory to store TensorBoard logs for this training run.
    writer = SummaryWriter(log_dir) # creates a SummaryWriter object to log training metrics for visualization in TensorBoard.

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

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) # creating data loaders for training and validation datasets, 
    # which will load the data in batches and shuffle the training data for better learning.
    test_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # setting the device to GPU if available, else CPU.
    model = AudioCNN(num_classes=len(train_dataset.classes)) # creating an instance of the AudioCNN model defined in model.py, with the number of output classes equal to the number of unique classes in the dataset.
    model.to(device)

    num_epochs = 100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # defining the loss function as cross-entropy loss with label smoothing of 0.1 to prevent overfitting.
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01) # defining the optimizer as AdamW with a learning rate of 0.0005 and weight decay of 0.01 to prevent overfitting.
    # AdamW is a variant of Adam optimizer that decouples weight decay from the gradient

    scheduler = OneCycleLR( # learning rate scheduler that adjusts the learning rate during training using the One Cycle Policy.
        # this policy starts with a low learning rate, increases it to a maximum value, and then decreases it back to a low value over the course of training.
        optimizer,
        max_lr=0.002, # maximum learning rate to reach during the cycle.
        epochs=num_epochs, # total number of epochs for training.
        steps_per_epoch=len(train_dataloader), # number of steps (batches) per epoch
        pct_start=0.1 # percentage of the cycle spent increasing the learning rate (10% here) i.e for the first 10 epochs it will increase the learning rate from initial to max_lr,
    )

    best_accuracy = 0.0

    print("Starting training")
    for epoch in range(num_epochs): # loop over the dataset multiple times equal to num_epochs
        model.train()
        epoch_loss = 0.0 # to keep track of the loss for the epoch

        progress_bar = tqdm( # tqdm shows a live progress bar in the terminal. train_dataloader → gives data in mini-batches (like giving the students 10 questions at a time instead of 1000 at once).
            train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}') # creates a progress bar for the training dataloader, with a description showing the current epoch out of total epochs.
        for data, target in progress_bar: # iterating over the training dataloader in progress bar to get batches of data(inputs) and their corresponding labels (targets).
            data, target = data.to(device), target.to(device) # moving the data and targets to the specified device (GPU or CPU).
            # tqdm  is just a layer on top of the dataloader to show progress bar in terminal.
            # it does not change the data or how we get it from the dataloader.
            # train_dataloader is the actual dataloader that loads the data in batches from the dataset in each epoch.
            if np.random.random() > 0.7: # if random number > 0.7 (30% chance), we apply mixup data augmentation to the batch
                data, target_a, target_b, lam = mixup_data(data, target)
                output = model(data)
                loss = mixup_criterion(
                    criterion, output, target_a, target_b, lam)
            else: # else (70% chance), we do normal training without mixup.
                output = model(data)
                loss = criterion(output, target)
                # When I said “with 30% chance”, I meant: On average, in 30 out of 100 batches, the code will use Mixup. 
                # In the other 70 out of 100 batches, it will use normal training.
                

            optimizer.zero_grad() # zero the gradients before backpropagation to prevent accumulation from previous batches.
            loss.backward() # backpropagation to compute the gradients of the loss with respect to the model parameters.
            optimizer.step()  # update the model parameters using the computed gradients.
            scheduler.step()    # update the learning rate according to the One Cycle Policy.

            epoch_loss += loss.item() # adding the loss for this batch to the total epoch loss.
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'}) # updating the progress bar to show the loss for the current batch.

        avg_epoch_loss = epoch_loss / len(train_dataloader) # after all batches in the epoch, we compute the average loss for the epoch by dividing the total epoch loss by the number of batches.
        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch) # logging the average training loss for the epoch to TensorBoard.
        writer.add_scalar( # logging the current learning rate to TensorBoard for monitoring.
            'Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Validation after each epoch
        model.eval() # set the model to evaluation mode for validation (disables dropout, batchnorm, etc.)

        correct = 0 # to keep track of the number of correct predictions
        total = 0 # to keep track of the total number of samples evaluated
        val_loss = 0 # to keep track of the total validation loss 

        with torch.no_grad(): # disables gradient computation for validation to save memory and computation as we don't need gradients for evaluation.
            for data, target in test_dataloader: # iterating over the validation dataloader to get batches of validation data and their corresponding labels.
                data, target = data.to(device), target.to(device) # moving the validation data and targets to the specified device (GPU or CPU).
                outputs = model(data) # getting the model's predictions for the validation data.
                loss = criterion(outputs, target) # computing the loss for the validation batch using the same criterion as training.
                val_loss += loss.item() # adding the loss for this validation batch to the total validation loss.

                _, predicted = torch.max(outputs.data, 1) # getting best predicted class for each sample in the batch.
                # outputs.data → gives the raw output scores (logits) from the model for each class.
                # torch.max(..., 1) → computes the maximum value along dimension 1 i.e for each sample in the batch, it finds the class with the highest score.
                # the _ variable captures the maximum values (not needed here as we only care about the predicted class indices),
                # and predicted captures the indices of the maximum values (the predicted class labels).
                total += target.size(0) # updating the total number of samples evaluated by adding the batch size (number of samples in the current batch).
                correct += (predicted == target).sum().item() # updating the number of correct predictions by comparing the predicted labels with the true labels (target)
                # (predicted == target) → creates a boolean tensor where each element is True if the prediction is correct and False otherwise.
                # .sum() → counts the number of True values (correct predictions) in the boolean 

        accuracy = 100 * correct / total # computing the accuracy as the percentage of correct predictions out of the total samples evaluated.
        avg_val_loss = val_loss / len(test_dataloader) # computing the average validation loss by dividing the total validation loss by the number of validation batches.

        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)

        print( # printing the epoch summary with average training loss, average validation loss, and validation accuracy.
            f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        if accuracy > best_accuracy: # if the validation accuracy for this epoch is better than the best accuracy seen so far, we save the model checkpoint.
            best_accuracy = accuracy # update the best accuracy
            torch.save({ # save the model state dictionary, accuracy, epoch, and class mapping to a file named best_model.pth in the /models directory.
                'model_state_dict': model.state_dict(), # model.state_dict() contains the model parameters (weights and biases).
                'accuracy': accuracy,
                'epoch': epoch,
                'classes': train_dataset.classes
            }, '/models/best_model.pth')
            print(f'New best model saved: {accuracy:.2f}%') # print a message indicating that a new best model has been saved.

    writer.close()
    print(f'Training completed! Best accuracy: {best_accuracy:.2f}%') # print a message indicating that training is complete and showing the best accuracy achieved.

@app.local_entrypoint()
def main():
    train.remote()