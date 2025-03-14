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
import torchvision.transforms.functional
import torchvision.transforms.v2
import torchvision


src = f'./datasets/longvideo-1/raw_frames'
dest = './datasets/longvideo-1/low_raw_frames'
if not os.path.exists(dest):
  os.mkdir(dest)
videos = ['video_0', 'video_1']

for video in videos:
  print('video: ', video)

  image_paths = glob.glob(f"{src}/{video}/*.png")

  for path in tqdm(image_paths):
    image = torchvision.io.read_image(path)
    h, w = image.shape[-2], image.shape[-1]
    image = torchvision.transforms.functional.resize(
        image, (720, 720), antialias=True,
    )
    image_name, _ = path.split('/')[-1].split('.')

    if not os.path.exists(f"{dest}/{video}"):
      os.mkdir(f"{dest}/{video}")
    torchvision.io.write_jpeg(image, f"{dest}/{video}/{image_name}.jpg", quality=60)