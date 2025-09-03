import base64
import io
import modal
import numpy as np
import requests
import torch.nn as nn
import torchaudio.transforms as T
import torch
from pydantic import BaseModel
import soundfile as sf
import librosa

from model import AudioCNN

app = modal.App("audio-cnn-inference") # create a new modal app named "audio-cnn-inference"

image = (modal.Image.debian_slim() # create a debian slim image with required dependencies for audio processing and model inference
         .pip_install_from_requirements("requirements.txt")  # install python packages from requirements.txt
         .apt_install(["libsndfile1"]) # install libsndfile1 for reading and writing sound files 
         .add_local_python_source("model")) # add local python source code from the model.py file

model_volume = modal.Volume.from_name("esc-model") # create a volume to store the pre-trained model weights

class AudioProcessor: # class to handle audio processing tasks
    def __init__(self):
        self.transform = nn.Sequential( # define a sequential model for audio transformations to convert raw audio to mel spectrogram
                                       # it comes from torchaudio.transforms
            T.MelSpectrogram(
                sample_rate=22050,
                n_fft=1024,
                hop_length=512,
                n_mels=128,
                f_min=0,
                f_max=11025
            ),
            T.AmplitudeToDB() # convert amplitude to decibels for better representation of the spectrogram
        )

    def process_audio_chunk(self, audio_data): # function to process a chunk of audio data and return the mel spectrogram
        waveform = torch.from_numpy(audio_data).float() # convert numpy array to torch tensor and ensure it's of type float

        waveform = waveform.unsqueeze(0) # add a batch dimension to the waveform tensor

        spectrogram = self.transform(waveform) # apply the defined transformations to get the mel spectrogram

        return spectrogram.unsqueeze(0) # add another dimension to match the input shape expected by the model


class InferenceRequest(BaseModel): # define the structure of the inference request using Pydantic for data validation
    audio_data: str
     
@app.cls(image=image, gpu="A10G", volumes={"/models": model_volume}, scaledown_window=15) # app.cls decorator to define a class that will run in the modal environment
# scaledown_window=15 means the instance will be scaled down after 15 minutes of inactivity
class AudioClassifier: 
    @modal.enter() # modal.enter decorator to define a method that runs when the instance starts
    def load_model(self): # This method loads the pre-trained model and prepares it for inference
        print("Loading models on enter") # log message to indicate model loading
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load('/models/best_model.pth',
                                map_location=self.device) # load the model checkpoint from the mounted volume, checkpoint contains model weights and other info
        self.classes = checkpoint['classes'] # list of class names for classification

        self.model = AudioCNN(num_classes=len(self.classes)) # initialize the AudioCNN model with the number of classes
        self.model.load_state_dict(checkpoint['model_state_dict']) # load the model weights from the checkpoint
        self.model.to(self.device) # move the model to the appropriate device (GPU or CPU)
        self.model.eval() # set the model to evaluation mode, this is important for layers like dropout and batchnorm to behave correctly during inference

        self.audio_processor = AudioProcessor() # initialize the audio processor
        print("Model loaded on enter")

    @modal.fastapi_endpoint(method="POST") # define a FastAPI endpoint for inference, it will handle POST requests made to this endpoint by clients 
    def inference(self, request: InferenceRequest): # method to handle inference requests, it takes an InferenceRequest object as input
        audio_bytes = base64.b64decode(request.audio_data) # decode the base64 encoded audio data from the request

        audio_data, sample_rate = sf.read( # read the audio data using soundfile library and get the audio samples and sample rate
            io.BytesIO(audio_bytes), dtype="float32")

        if audio_data.ndim > 1: # if the audio has more than one channel (stereo), convert it to mono by averaging the channels
            audio_data = np.mean(audio_data, axis=1)

        if sample_rate != 44100: # if the sample rate is not 44100 Hz, resample the audio to 44100 Hz using librosa 
            audio_data = librosa.resample(
                y=audio_data, orig_sr=sample_rate, target_sr=44100)

        spectrogram = self.audio_processor.process_audio_chunk(audio_data) # process the audio data to get the mel spectrogram
        spectrogram = spectrogram.to(self.device) # move the spectrogram tensor to the appropriate device (GPU or CPU)

        with torch.no_grad(): # disable gradient calculation for inference to save memory and computation
            output, feature_maps = self.model(
                spectrogram, return_feature_maps=True) # pass the spectrogram through the model to get the output logits and feature maps 
                # feature maps are intermediate outputs from various layers of the model useful for visualization

            output = torch.nan_to_num(output) # replace NaNs with zero and inf with large finite numbers to ensure numerical stability
            probabilities = torch.softmax(output, dim=1) # apply softmax to get class probabilities
            top3_probs, top3_indicies = torch.topk(probabilities[0], 3) # get the top 3 predicted class probabilities and their indices

            predictions = [{"class": self.classes[idx.item()], "confidence": prob.item()}
                           for prob, idx in zip(top3_probs, top3_indicies)] # format the predictions as a list of dictionaries with class names and confidence scores

            viz_data = {} # dictionary to hold visualization data for feature maps
            for name, tensor in feature_maps.items(): # iterate over the feature maps returned by the model
                if tensor.dim() == 4:  # [batch_size, channels, height, width]
                    aggregated_tensor = torch.mean(tensor, dim=1)
                    squeezed_tensor = aggregated_tensor.squeeze(0)
                    numpy_array = squeezed_tensor.cpu().numpy()
                    clean_array = np.nan_to_num(numpy_array)
                    viz_data[name] = {
                        "shape": list(clean_array.shape),
                        "values": clean_array.tolist()
                    }

            spectrogram_np = spectrogram.squeeze(0).squeeze(0).cpu().numpy()
            clean_spectrogram = np.nan_to_num(spectrogram_np)

            max_samples = 8000
            waveform_sample_rate = 44100
            if len(audio_data) > max_samples:
                step = len(audio_data) // max_samples
                waveform_data = audio_data[::step]
            else:
                waveform_data = audio_data

        response = {
            "predictions": predictions,
            "visualization": viz_data,
            "input_spectrogram": {
                "shape": list(clean_spectrogram.shape),
                "values": clean_spectrogram.tolist()
            },
            "waveform": {
                "values": waveform_data.tolist(),
                "sample_rate": waveform_sample_rate,
                "duration": len(audio_data) / waveform_sample_rate
            }
        }

        return response