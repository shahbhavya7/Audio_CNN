import base64
import io
import modal
import numpy as np
import requests
from pydantic import BaseModel


from model import AudioCNN

app = modal.App("audio-cnn-inference") # create a new modal app named "audio-cnn-inference"

image = (modal.Image.debian_slim() # create a debian slim image with required dependencies for audio processing and model inference
         .pip_install_from_requirements("requirements.txt")  # install python packages from requirements.txt
         .apt_install(["libsndfile1"]) # install libsndfile1 for reading and writing sound files 
         .add_local_python_source("model")) # add local python source code from the model.py file

model_volume = modal.Volume.from_name("esc-model") # create a volume to store the pre-trained model weights

class AudioProcessor: # class to handle audio processing tasks
    def __init__(self):
        import torch.nn as nn
        import torchaudio.transforms as T
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
        import torch
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
        import torch
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
        import torch
        import soundfile as sf
        import librosa
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
                # output is usually the raw class scores (logits) before applying softmax, each value corresponds to a class
                # output eg - tensor([[-1.2345, 0.5678, 2.3456, ...]]) for a batch size of 1 and multiple classes

            output = torch.nan_to_num(output) # replace NaNs with zero and inf with large finite numbers to ensure numerical stability
            probabilities = torch.softmax(output, dim=1) # apply softmax to get class probabilities for raw output scores
            top3_probs, top3_indicies = torch.topk(probabilities[0], 3) # get the top 3 predicted class probabilities and their indices based on the probabilities tensor

            predictions = [{"class": self.classes[idx.item()], "confidence": prob.item()} # 
                           for prob, idx in zip(top3_probs, top3_indicies)] # putting the top 3 predictions into a list of dictionaries with class names and confidence scores
            # like dog: 0.95, cat: 0.03, bird: 0.02, .zip is used to pair each probability with its corresponding index i.e joining two lists element-wise
            # like top3_probs = [0.95, 0.03, 0.02], top3_indicies = [5, 2, 8] => zip(top3_probs, top3_indicies) => [(0.95, 5), (0.03, 2), (0.02, 8)]

            viz_data = {} # dictionary to hold visualization data for feature maps returned by the model in heatmap format
            for name, tensor in feature_maps.items(): # iterate over the feature maps returned by the model in tensor format
                if tensor.dim() == 4:  # [batch_size, channels, height, width] # check if the tensor has 4 dimensions (batch size, channels, height, width)
                    aggregated_tensor = torch.mean(tensor, dim=1) # aggregate the feature maps across the channel dimension by taking the mean like there are 64 channels, we take mean across all 64 channels to get a single 2D feature map
                    # resulting shape will be [batch_size, height, width]
                    squeezed_tensor = aggregated_tensor.squeeze(0) # remove the batch dimension since batch size is 1 for all inference requests
                    # resulting shape will be [height, width]
                    numpy_array = squeezed_tensor.cpu().numpy() # convert the tensor to a numpy array and move it to CPU if it's on GPU as numpy doesn't support GPU tensors
                    clean_array = np.nan_to_num(numpy_array) # replace NaNs with zero and inf with large finite numbers to ensure numerical stability
                    viz_data[name] = { # store the shape and values of the feature map in the viz_data dictionary for visualization with key as the layer name
                        "shape": list(clean_array.shape), # convert the shape tuple to a list for JSON serialization
                        "values": clean_array.tolist() # store the feature map values as a list 
                    }

            spectrogram_np = spectrogram.squeeze(0).squeeze(0).cpu().numpy() # convert the input spectrogram tensor to a numpy array for visualization
            # it had shape [batch,channel,height,width] => we remove batch and channel dimensions since both are 1 for inference requests so no need to keep them
            # resulting shape will be [height, width]
            clean_spectrogram = np.nan_to_num(spectrogram_np) # replace NaNs with zero and inf with large finite numbers to ensure numerical stability

            max_samples = 8000 # limit the number of samples in the waveform to 8000 for visualization purposes
            waveform_sample_rate = 44100 # set the sample rate for the waveform visualization
            if len(audio_data) > max_samples: # if the audio data has more than 8000 samples, downsample it to fit within the limit as we don't want to send too much data back to the client
                # for visualization purposes
                step = len(audio_data) // max_samples # calculate the step size for downsampling
                waveform_data = audio_data[::step] # downsample the audio data by taking every 'step'th sample
            else:
                waveform_data = audio_data # if the audio data is within the limit, use it as is

        response = { # construct the response dictionary to be returned to the client
            "predictions": predictions, # list of top 3 predictions with class names and confidence scores
            "visualization": viz_data, # dictionary of feature maps for visualization
            "input_spectrogram": { # input spectrogram data for visualization
                "shape": list(clean_spectrogram.shape),
                "values": clean_spectrogram.tolist()
            },
            "waveform": { # waveform data for visualization
                "values": waveform_data.tolist(), # convert the waveform numpy array to a list for JSON serialization
                "sample_rate": waveform_sample_rate, # sample rate of the waveform
                "duration": len(audio_data) / waveform_sample_rate # duration of the audio clip in seconds
            }
        }

        return response # return the response dictionary to the client

@app.local_entrypoint() # decorator to define the main entry point for local execution
def main():
    import soundfile as sf
    
    import requests
    audio_data, sample_rate = sf.read("cb.wav") # read a local audio file using soundfile library

    buffer = io.BytesIO() # create an in-memory bytes buffer to hold the audio data
    sf.write(buffer, audio_data, sample_rate, format="WAV") # write the audio data to the buffer in WAV format
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8") # encode the audio data in base64 format for transmission over HTTP
    payload = {"audio_data": audio_b64} # create the payload dictionary with the base64 encoded audio data

    server = AudioClassifier() # create an instance of the AudioClassifier class which will start the modal instance if not already running
    url = server.inference.get_web_url() # get the URL of the inference endpoint to send requests to
    response = requests.post(url, json=payload) # send a POST request to the inference endpoint with the payload as JSON
    response.raise_for_status() # raise an error if the request was not successful (status code not in 200-299 range)

    result = response.json() # parse the JSON response from the server

    waveform_info = result.get("waveform", {}) # get the waveform information from the response if available
    if waveform_info: # if waveform information is present, print some details about it
        values = waveform_info.get("values", {}) # get the waveform values from the waveform info
        print(f"First 10 values: {[round(v, 4) for v in values[:10]]}...") # print the first 10 values of the waveform rounded to 4 decimal places
        print(f"Duration: {waveform_info.get("duration", 0)}")  # print the duration of the audio clip

    print("Top predictions:") # print the top predictions returned by the model
    for pred in result.get("predictions", []): # iterate over the predictions in the response,each prediction is a dictionary with class name and confidence score
        print(f"  -{pred["class"]} {pred["confidence"]:0.2%}") # print each class name and its confidence score formatted as a percentage
    