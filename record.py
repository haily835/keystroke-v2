import cv2
import time
import os
import sys
import re

# Check if video name is provided as command line argument
if not os.path.exists('videos'):
    os.makedirs('videos')  # Create the videos directory

# Get the video file name from command line arguments
if len(sys.argv) < 2:
    # Scan the videos folder for existing video files
    existing_files = os.listdir('videos')
    video_files = [f for f in existing_files if re.match(r'video_\d+\.mp4', f)]
    
    # Determine the next index for the new video file
    if video_files:
        indices = [int(re.search(r'(\d+)', f).group(1)) for f in video_files]
        next_index = max(indices) + 1
    else:
        next_index = 0
    video_filename = f'video_{next_index}'  # Generate new video filename
else:
    video_filename = f'video_{sys.argv[1]}'
timestamp = None
filename = f"videos/{video_filename}.mp4"
print(filename)
# Create videos directory if it doesn't exist
os.makedirs("videos", exist_ok=True)

# Create a corresponding txt file for the timestamp
txt_filename = f"videos/{video_filename}_info.txt"

# Start the webcam capture
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
frame_count = 0  # Add manual frame counter

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the frame width, height, and fps from the camera
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the width from the camera
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
print("this", frame_width, frame_height)
fps = cap.get(cv2.CAP_PROP_FPS)  # Get the FPS of the camera

# If the FPS value is not returned correctly (it may vary depending on the camera), default to 30
if fps == 0:
    fps = 30
    print("Warning: Unable to get FPS, defaulting to 30.")

# Video writer setup with mp4v codec for .mp4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 files
out = cv2.VideoWriter(filename, fourcc, fps, (640, 480))

print(f"Recording started at {fps} FPS. Press Ctrl+C to stop recording.")

# Remove window display and keyboard check
while True:
    try:
        # Capture each frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break
        else:
            # Record timestamp of first frame
            if timestamp is None:
                timestamp = int(time.time_ns())
                print(timestamp)
            # print(frame.shape)
            # Add frame number to the frame using manual counter
            cv2.putText(frame, f'Frame: {frame_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame_count += 1  # Increment frame counter
            frame = cv2.resize(frame, (640, 480))
            # Write the frame to the output file
            out.write(frame)
            # print(frame.shape)

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Write the timestamp to file after recording is complete
txt_filename = f"videos/{video_filename}_info.txt"
with open(txt_filename, 'w') as f:
    f.write(str(timestamp))

print(f"Recording saved as {filename}.")
