import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import librosa # Import librosa for resampling

# Load Wav2Vec2 Processor and Model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Ensure the model is in evaluation mode
model.eval()

def speech_file_to_array_fn(filepath):
    """Read speech file and convert to array."""
    speech_array, sampling_rate = sf.read(filepath)
    # Resample to 16000 Hz if necessary
    if sampling_rate != 16000:
        speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000 # Update the sampling rate
    return speech_array, sampling_rate

def predict(filepath):
    # Load audio file
    speech, sampling_rate = speech_file_to_array_fn(filepath)

    # Preprocess the audio
    input_values = processor(speech, return_tensors="pt", sampling_rate=sampling_rate).input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the logits to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription

# Example usage
audio_file = "./recordings/recording.wav"
transcription = predict(audio_file)
print(f"Transcription: {transcription}")