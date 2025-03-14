import cv2
import numpy as np
import os
import pandas as pd

def extract_and_merge_frames(video_path, 
                             csv_path, 
                             output_dir,
                             n_samples=10,
                             frames_before=3, 
                             frames_after=4):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Define the frames per second (fps) for the output video
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    count = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for index, row in df.iterrows():
        key = row['Key']
        # frame_number = round((row['Time (ms)'] * fps) / 1000)  # Use the 'Frame' column directly
        frame_number =  row['Frame']
        frames = []
        for i in range(-frames_before, frames_after + 1):
            frame_index = frame_number + i
            if 0 <= frame_index < total_frames:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)

        if frames:
            # Stack frames horizontally
            merged_frame = np.stack(frames)
            
            output_filename = f"{frame_number}_{key}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # Define video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width = frames[0].shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Write frames
            for frame in merged_frame:
                out.write(frame)
            
            out.release()
            # print(f"Saved: {output_filename}")
        count += 1
        if count == n_samples:
            break
    
    # Release video capture
    cap.release()

if __name__ == "__main__":
    for i in range(0, 44):
        extract_and_merge_frames(
            f"/Users/lyhai/Documents/GitHub/keystroke-recognition/datasets/tablet3/labels/video_{i}.mp4",
            f"/Users/lyhai/Documents/GitHub/keystroke-recognition/datasets/tablet3/labels/video_{i}.csv",
            f"/Users/lyhai/Documents/GitHub/keystroke-recognition/datasets/tablet3/output/video_{i}",
            10
        )