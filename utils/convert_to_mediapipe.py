# Mediapip
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import glob
import torchvision
import torchvision.transforms.functional
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import rotate
import cv2

import argparse


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)


def process_image(pil_img):
    data = np.asarray(pil_img)
    media_pipe_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=data
    )

    detection_result = detector.detect(media_pipe_img)
    hand_landmarks_list = detection_result.hand_landmarks

    coords = [[], []]  # coordinates of 21 points of 2 hands
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        for landmark in hand_landmarks:
            coords[idx].append((landmark.x, landmark.y, landmark.z))

    # Ensure each hand has 21 landmarks, fill with zeros if necessary
    for idx in range(2):
        while len(coords[idx]) < 21:
            coords[idx].append((0.0, 0.0, 0.0))

    return torch.tensor(coords)

# Usage in the main loop.
# frames = []
# for i in range(len(jpgs)):
#     image_path = f"{src}/frame_{i}.jpg"
#     result = process_image(image_path, detector)
#     if result is not None:
#         frames.append(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert raw frames to landmarks')
    parser.add_argument('--src_dir', 
                        type=str, 
                        required=False, 
                        help='Source directory containing raw frames',
                        default='datasets/topview-test/raw_frames')
    parser.add_argument('--dest_dir', 
                        type=str, 
                        required=False, 
                        help='Destination directory for landmarks',
                        default='datasets/topview-test/landmarks'
    )
    args = parser.parse_args()

    src_dir = args.src_dir
    dest_dir = args.dest_dir
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for video in range(0, 15):
        video_name = f'video-{video}'
        print(f"Video {video_name}")
        src = f'{src_dir}/{video_name}.mp4'
        dest = f'{dest_dir}/{video_name}.pt'

        # Open the video file
        cap = cv2.VideoCapture(src)
        frames = []
        last_succeed = None
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_path = f"{src}/frame_{frame_idx}.jpg"  # Optional: Save frame as image if needed
            result = process_image(pil_img)
            if result is not None:
                last_succeed = result
                frames.append(result)
            else:
                if last_succeed is not None:
                    frames.append(last_succeed)
                print(f'Mediapipe failed at frame {frame_idx}')

            frame_idx += 1

        cap.release()

        frames = torch.stack(frames)
        print(f"Sucessed {len(frames)}, shape {frames.shape} in {frame_idx} frames")
        torch.save(frames, dest)