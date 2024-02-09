import tkinter as tk
from tkinter import messagebox
import soundfile as sf
import pyaudio
import librosa
import numpy as np
from keras.models import load_model
import tensorflow as tf

# Disable eager execution
tf.compat.v1.disable_v2_behavior()

# Functions for recording and classifying sound
def record_sound(filename, duration=5):
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []

    print("Recording...")

    for i in range(0, int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with sf.SoundFile(filename, 'w', samplerate=fs, channels=channels) as f:
        f.write(np.concatenate([np.frombuffer(frame, dtype=np.int16) for frame in frames]))

def classify_sound(filename):
    # Load the pre-trained model
    model = load_model(r'C:\Users\User\PycharmProjects\pythonProject4\car_fault_model.pth')

    # Preprocess the audio
    audio, _ = librosa.load(filename, sr=44100)
    audio = audio.reshape((1, audio.shape[0], 1))

    # Classification
    prediction = model.predict(audio)
    predicted_class = np.argmax(prediction)

    # Match with labels
    labels = {
        0: "Wear of elements",
        1: "Fluid ingress",
        2: "Low belt quality",
        3: "Worn tensioner pulleys"
    }

    result = labels.get(predicted_class, "No problem detected")
    return result

# Record button handler
def on_record_button_click():
    filename = "recorded_sound.wav"
    record_sound(filename)
    problem = classify_sound(filename)

    if problem == "No problem detected":
        messagebox.showinfo("Result", "No problem detected")
    else:
        messagebox.showinfo("Result", f"Problem detected: {problem}")

# Create a graphical interface
if __name__ == "__main__":
    window = tk.Tk()
    window.title("Automotive Diagnostic Assistant")

    # Add a button for sound recording
    record_button = tk.Button(window, text="Record Sound (5 seconds)", command=on_record_button_click)
    record_button.pack(pady=20)

    # Run the main event loop
    window.mainloop()
