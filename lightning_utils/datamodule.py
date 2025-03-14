from typing import List, Tuple
import torch
import lightning as L
import pandas as pd
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from lightning.pytorch.utilities import CombinedLoader
import torchvision.transforms.v2 as v2
import lightning as L
import pandas as pd
from lightning_utils.dataset import BaseStreamDataset
from utils.transforms import *

def get_dataloader(
        frames_dir,
        labels_dir,
        videos,
        idle_gap=None,
        delay=10,
        batch_size=4,
        num_workers=4,
        transforms=[],
        windows=[(2, 2)],
        shuffle=False):

    key_counts = pd.DataFrame()
    datasets = []

    for window in windows:
        for t in transforms:
            for video in videos:
                f_before, f_after = window
                datasets.append(
                    BaseStreamDataset.create_dataset(
                        video_path=f"{frames_dir}/video_{video}",
                        label_path=f"{labels_dir}/video_{video}.csv",
                        gap=idle_gap,
                        delay=delay,
                        f_after=f_after,
                        f_before=f_before,
                        transforms=eval(t))
                )

    # count only the original data, not augmented
    key_counts['label'] = datasets[0].get_class_counts()['label']
    for video, ds in zip(videos, datasets):
        key_counts[video] = ds.get_class_counts()['count']

    merged = torch.utils.data.ConcatDataset(datasets)
    key_counts.to_csv('key_counts.csv')
    # print('Key counts: \n', key_counts)
    print("Total samples: ", len(merged))

    loader = DataLoader(
        merged,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers,
        shuffle=shuffle
    )
    return loader


class KeyStreamModule(L.LightningDataModule):
    def __init__(self,
                 frames_dir: str,
                 labels_dir: str,
                 train_videos: List[str] = [],
                 val_videos: List[str] = [],
                 test_videos: List[str] = [],
                 train_windows: List[List[int]] = [[2, 2], [3, 1], [1, 3]],
                 val_windows: List[List[int]] = [[2, 2]],
                 test_windows: List[List[int]] = [[2, 2]],
                 idle_gap: int = None,
                 delay: int = 4,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 train_transforms: List[str] = ['resize'],
                 val_transforms: List[str] = ['resize'],
                 test_transforms: List[str] = ['resize']):
        """
        train_collate_fns: create multiple loaders for every collate functions and a loader without any collate function
        idle_gap=None: if None, the binary detect (idle or active segments) dataset will be used
        """
        super().__init__()

        self.train_loader = get_dataloader(frames_dir,
                                           labels_dir,
                                           videos=train_videos,
                                           idle_gap=idle_gap,
                                           delay=delay,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           transforms=train_transforms,
                                           windows=train_windows,
                                           shuffle=True) if len(train_videos) else None

        self.val_loader = get_dataloader(
            frames_dir,
            labels_dir,
            videos=val_videos,
            idle_gap=idle_gap,
            delay=delay,
            batch_size=batch_size,
            transforms=val_transforms,
            windows=val_windows,
            num_workers=num_workers,
        ) if len(val_videos) else None

        self.test_loader = get_dataloader(
            frames_dir,
            labels_dir,
            videos=test_videos,
            idle_gap=idle_gap,
            delay=delay,
            batch_size=batch_size,
            transforms=test_transforms,
            windows=test_windows,
            num_workers=num_workers,
        ) if len(test_videos) else None

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader