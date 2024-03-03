import cv2
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

# Function to preprocess frames
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (256, 256))
    frame = frame / 255.0
    return frame

# Function to perform video processing using JAX
@jit
def process_video(frames):
    # Example operation: calculate the mean of all frames
    mean_frame = jnp.mean(frames, axis=0)
    return mean_frame

# Load video file
cap = cv2.VideoCapture('input_video.mp4')

# Read the first frame to initialize dimensions
ret, frame = cap.read()
frame_height, frame_width, _ = frame.shape

# Initialize buffer to hold frames
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_buffer = np.zeros((num_frames, frame_height, frame_width))

# Read all frames into the buffer
for i in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break
    frame = preprocess_frame(frame)
    frame_buffer[i] = frame

# Close the video file
cap.release()

# Convert frame buffer to JAX array
frame_buffer_jax = jnp.array(frame_buffer)

# Process video using JAX
processed_frame = process_video(frame_buffer_jax)

# Convert processed frame back to numpy array for visualization
processed_frame_numpy = processed_frame.get()

# Display or save the processed frame
cv2.imshow('Processed Frame', processed_frame_numpy)
cv2.waitKey(0)
cv2.destroyAllWindows()
