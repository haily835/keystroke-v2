from torchvision.transforms.v2 import Transform
import torchvision.transforms.v2 as v2
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
# Mediapipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
from PIL import Image

class LandmarkTransform(Transform):
    """Return the hand landmarks coordinate of the input images"""
    
    def __init__(self, model_asset_path: str) -> None:
        super().__init__()
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)
            

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        out = []
        for img in inpt:
            img = img.permute(1, 2, 0).numpy()
            pil_img = Image.fromarray(img)
            data = np.asarray(pil_img)
            media_pipe_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=data
            )

            detection_result = self.detector.detect(media_pipe_img)
            hand_landmarks_list = detection_result.hand_landmarks

            coords = [[], []]  # coordinates of 21 points of 2 hands
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                for landmark in hand_landmarks:
                    coords[idx].append((landmark.x, landmark.y, landmark.z))
            if len(coords[1]) == 21 and len(coords[0]) == 21:
                out.append(coords)
            
        out = torch.tensor(out).float()
        if len(out) != len(inpt): 
            return None

        out = out.permute(3, 0, 2, 1) # permute to channels, frames, points, hands
        return out
    
class VideoPermuteTransform(Transform):
    """Permute video frames to channels, frames, height, width"""
    
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt.permute(1, 0, 2, 3)
    
target_size = (60, 60)

rotation = v2.Compose([
    v2.RandomRotation(degrees=10),
    v2.Resize(size=target_size, antialias=True),
    VideoPermuteTransform()
])

resize = v2.Compose([
    v2.Resize(size=target_size, antialias=True),
    VideoPermuteTransform()
])

sharpness = v2.Compose([
    v2.RandomAdjustSharpness(sharpness_factor=2),
    v2.Resize(size=target_size, antialias=True),
    VideoPermuteTransform()
])

color = v2.Compose([
    v2.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),
    v2.Resize(size=target_size, antialias=True),
    VideoPermuteTransform(),
])

perspective = v2.Compose([
    v2.RandomPerspective(0.2),
    v2.Resize(size=target_size, antialias=True),
    VideoPermuteTransform()
])

zoom = v2.Compose([
    v2.RandomZoomOut(side_range=(1, 1.5)),
    v2.Resize(size=target_size, antialias=True),
    VideoPermuteTransform()
])