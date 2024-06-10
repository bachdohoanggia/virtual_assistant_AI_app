import sounddevice as sd
import wavio
import numpy as np
import os

# Parameters for the audio recording
sample_rate = 44100  # Hz
folder_path = "recordings"
silence_threshold = 500  # Adjust based on your microphone sensitivity
silence_duration = 2  # seconds

# Create the recordings folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)


def record_audio(sample_rate, second):
    print("Recording... Press Ctrl+C to stop.")

    audio_data = sd.rec(sample_rate * second, samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    
    return audio_data

def save_audio(audio_data, sample_rate, folder_path):
    file_name = "recording.wav"
    file_path = os.path.join(folder_path, file_name)
    wavio.write(file_path, audio_data, sample_rate, sampwidth=2)
    print(f"Recording saved as: {file_path}")
    return file_path

# Record the audio
audio_data = record_audio(sample_rate, 5)

# Save the audio file
if audio_data.size > 0:
    file_path = save_audio(audio_data, sample_rate, folder_path)
    print(f"Audio file path: {file_path}")
else:
    print("No audio recorded.")
