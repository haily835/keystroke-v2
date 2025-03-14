import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import glob
import os
from tqdm import tqdm
import torch
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                    num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)


src = f'./datasets/shortvideo-1/raw_frames_224'
dest = './datasets/shortvideo-1/landmarks'

videos = ['video_0']

for video in videos:
  print('video: ', video)
  image_paths = glob.glob(f"{src}/{video}/*.png")
  lms = []
  status = []
  for path in tqdm(image_paths):
    image = mp.Image.create_from_file(path)
    detection_result = detector.detect(image)
    if len(detection_result.hand_landmarks) != 2: 
      status.append(False)
      lms.append([])
      continue

    image = torch.Tensor(
                    [[(p.x, p.y, p.z) for p in detection_result.hand_landmarks[0]], 
                     [(p.x, p.y, p.z) for p in detection_result.hand_landmarks[1]]]
                     ).float()
    status.append(image.shape[1] == 21)
    lms.append(image)

  for i in range(len(lms)):
    if status[i]: continue
    print(f"Missing at {i}")

    for j in range(i + 1, len(image_paths)):
      if status[j]: 
        lms[i] = lms[j]
        continue

    for j in range(i-1, 0, -1):
      if status[j]: 
        lms[i] = lms[j]
        continue
  
  lms = torch.stack(lms)
  torch.save(lms, f'{dest}/{video}.pt')
  print(f"Shape: {lms.shape}")
