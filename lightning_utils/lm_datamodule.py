from typing import List
import torch
import lightning as L
import pandas as pd
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import lightning as L
import pandas as pd
from lightning_utils.lm_dataset import BaseStreamDataset

def get_dataloader(
        frames_dir,
        labels_dir,
        landmarks_dir,
        classes_path,
        videos,
        idle_gap=None,
        delay=10,
        batch_size=4,
        num_workers=4,
        windows=[(2, 2)],
        shuffle=False):

    key_counts = pd.DataFrame()
    datasets = []

    for window in windows:
        for video in videos:
            f_before, f_after = window
            datasets.append(BaseStreamDataset.create_dataset(
                video_path=f"{frames_dir}/video_{video}",
                landmark_path=f"{landmarks_dir}/video_{video}.pt",
                label_path=f"{labels_dir}/video_{video}.csv",
                classes_path=classes_path,
                gap=idle_gap,
                delay=delay,
                f_after=f_after,
                f_before=f_before,
            ))
    
    key_counts['label'] = datasets[0].get_class_counts()['label']
    for video, ds in zip(videos, datasets):
        key_counts[video] = ds.get_class_counts()['count']
    key_counts.to_csv("count.csv")
    merged = torch.utils.data.ConcatDataset(datasets)
    # print('Key counts: \n', key_counts)
    # print("Total samples: ", len(merged))

    loader = DataLoader(
        merged,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers,
        shuffle=shuffle
    )
    return loader


class LmKeyStreamModule(L.LightningDataModule):
    def __init__(self,
                 frames_dir: str,
                 landmarks_dir: str,
                 labels_dir: str,
                 classes_path: str,
                 train_videos: List[str | int] = [],
                 val_videos: List[str | int] = [],
                 test_videos: List[str | int] = [],
                 train_windows: List[List[int]] = [[2, 2], [3, 1], [1, 3]],
                 val_windows: List[List[int]] = [[2, 2]],
                 test_windows: List[List[int]] = [[2, 2]],
                 idle_gap: int = None,
                 delay: int = 10,
                 batch_size: int = 4,
                 num_workers: int = 4):
        """
        train_collate_fns: create multiple loaders for every collate functions and a loader without any collate function
        idle_gap=None: if None, the binary detect (idle or active segments) dataset will be used
        """
        super().__init__()
        
        self.frames_dir = frames_dir
        self.landmarks_dir = landmarks_dir
        self.labels_dir = labels_dir
        self.classes_path = classes_path
        self.train_videos = train_videos
        self.val_videos = val_videos
        self.test_videos = test_videos
        self.idle_gap = idle_gap
        self.delay = delay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_windows = train_windows
        self.val_windows = val_windows
        self.test_windows = test_windows

    def setup(self, stage) -> None:
        if stage == 'fit':
            self.train_loader = get_dataloader(self.frames_dir,
                                           self.labels_dir,
                                           self.landmarks_dir,
                                           self.classes_path,
                                           videos=self.train_videos,
                                           idle_gap=self.idle_gap,
                                           delay=self.delay,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           windows=self.train_windows,
                                           shuffle=True) if len(self.train_videos) else None

            self.val_loader = get_dataloader(
                self.frames_dir,
                self.labels_dir,
                self.landmarks_dir,
                self.classes_path,
                videos=self.val_videos,
                idle_gap=self.idle_gap,
                delay=self.delay,
                batch_size=self.batch_size,
                windows=self.val_windows,
                num_workers=self.num_workers,
            ) if len(self.val_videos) else None
        elif stage == 'test':
            self.test_loader = get_dataloader(
                self.frames_dir,
                self.labels_dir,
                self.landmarks_dir,
                self.classes_path,
                videos=self.test_videos,
                idle_gap=self.idle_gap,
                delay=self.delay,
                batch_size=self.batch_size,
                windows=self.test_windows,
                num_workers=self.num_workers,
            ) if len(self.test_videos) else None

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
    
    
