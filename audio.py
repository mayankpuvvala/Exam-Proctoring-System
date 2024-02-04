import sounddevice as sd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Choose your microphone (if you have multiple)
mic = 1

# Set the volume threshold
threshold = 0.02

# Set up the stream
stream = sd.InputStream(device=mic, channels=1, samplerate=44100, blocksize=1024)

# Start the stream
stream.start()

active = False
start_time = None
total_time = 0

# Set up matplotlib
fig, ax = plt.subplots()
time_text = ax.text(0.5, 0.5, '', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

def update(frame):
    global active, start_time, total_time

    # Read audio data from the microphone
    data, overflowed = stream.read(1024)
    
    # Calculate the RMS volume
    rms = np.sqrt(np.mean(data**2))
    
    if rms > threshold and not active:
        # Start of voice activity
        active = True
        start_time = time.time()
    elif rms <= threshold and active:
        # End of voice activity
        active = False
        end_time = time.time()
        total_time += end_time - start_time
    
    time_text.set_text(f"Active voice time: {total_time} seconds")
    return time_text,

# Create an animation
ani = FuncAnimation(fig, update, interval=50, blit=True)

plt.show()

# Don't forget to stop the stream
stream.stop()
stream.close()
