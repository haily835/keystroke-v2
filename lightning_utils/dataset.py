import torch
import torch.utils
import torch.utils.data
import pandas as pd
import torchvision
import torchvision.transforms.functional
import torchvision.transforms.v2 as v2
import numpy as np
import os
import glob

# clf_id2label = ['comma', 'dot', 'delete', 'space',
#                          'a', 'b', 'c', 'd',
#                          'e', 'f', 'g', 'h',
#                          'i', 'j', 'k', 'l',
#                          'm', 'n', 'o', 'p',
#                          'q', 'r', 's', 't',
#                          'u', 'v', 'w', 'x',
#                          'y', 'z']

# clf_label2id = {
#     'comma': 0, 'dot': 1, 'delete': 2, 'space': 3,
#     'a': 4, 'b': 5, 'c': 6, 'd': 7,
#             'e': 8, 'f': 9, 'g': 10, 'h': 11,
#             'i': 12, 'j': 13,  'k': 14, 'l': 15,
#             'm': 16, 'n': 17, 'o': 18, 'p': 19,
#             'q': 20, 'r': 21, 's': 22, 't': 23,
#             'u': 24, 'v': 25, 'w': 26, 'x': 27,
#             'y': 28, 'z': 29
# }

# clf_id2label = ['comma', 'dot', 'delete', 'space', 'shift',
#                 '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
#                 'a', 'b', 'c', 'd',
#                 'e', 'f', 'g', 'h',
#                 'i', 'j', 'k', 'l',
#                 'm', 'n', 'o', 'p',
#                 'q', 'r', 's', 't',
#                 'u', 'v', 'w', 'x',
#                 'y', 'z']

clf_id2label = [
"a", "b", "c", "d", "e", 
"f", "g", "h", "i", "j", 
"k", "l", "m", "n", "o", 
"p", "q", "r", "s", "t", 
"u", "v", "w", "x", "y", "z", 
"comma", "period", "space", "backspace",
"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]

clf_label2id = {label: idx for idx, label in enumerate(clf_id2label)}

detect_id2label = ['idle', 'active']
detect_label2id = {'idle': 0, 'active': 1}


class BaseStreamDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs): pass

    @classmethod
    def create_dataset(cls, video_path: str, label_path: str,
                       gap: int, 
                       f_before: int = 3, 
                       f_after: int = 4, 
                       delay: int = 10,
                       transforms=None):
        if gap:
            return KeyDetectDataset(
                video_path=video_path,
                label_path=label_path,
                gap=gap,
                f_after=f_after,
                f_before=f_before,
                delay=delay,
                transforms=transforms)
        
        return KeyClfStreamDataset(
            video_path=video_path,
            label_path=label_path,
            f_after=f_after,
            f_before=f_before,
            delay=delay,
            transforms=transforms)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        (start, end), label = self.segments[idx]
        frames = []
        for i in range(start, end + 1):
            image = torchvision.io.read_image(
                f"{self.video_path}/frame_{i}.jpg")
            frames.append(image)

        frames = torch.stack(frames)
        if self.transforms:
            frames = self.transforms(frames)

        return frames.float(), self.label2id[label]

    def get_class_counts(self):
        labels = np.array([segment[1] for segment in self.segments])

        counts = []
        for label in self.id2label:
            count = np.sum(labels == label)
            counts.append(count)

        df = pd.DataFrame({'label': self.id2label, 'count': counts})
        return df

    def create_segment(self, idx, dest_folder=None, format='mp4', fps=3.0):
        """
        create video formats or copy the frames of a segment, apply transforms if any.
        """
        (start, end), label = self.segments[idx]
        
        # label = self.id2label[id]
        if dest_folder:
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)

        frames, id = self.__getitem__(idx)
        frames = frames.permute(1, 2, 3, 0)
        label = self.id2label[id]

        if format == 'dir':
            destination = f'{dest_folder}/{self.video_name}_f_{start}_{end}_{label}'
            if not os.path.exists(destination):
                os.makedirs(destination)

            for i in range(start, end + 1):
                image_name = f'frame_{i}.jpg'
                image_save_path = f'{destination}/{image_name}'
                frame = frames[i - start]
                torchvision.io.image.write_jpeg(image_save_path,
                                                frame,
                                                quality=60)

        elif format:
            video_name = f'{self.data_dir}/segments_{format}/{self.video_name}_{label}_f{start}_{end}.{format}'
            torchvision.io.video.write_video(
                filename=video_name, video_array=frames, fps=fps
            )

        else:
            return frames, label
class KeyClfStreamDataset(BaseStreamDataset):
    def __init__(self,
                 video_path: str,
                 label_path: str,
                 f_before=3,
                 f_after=4,
                 delay=10,
                 transforms=None):

        self.video_path = video_path
        self.video_name = video_path.split('/')[-1]
        self.data_dir = video_path.split('/')[-3]
        self.transforms = transforms
        self.id2label = clf_id2label
        self.label2id = clf_label2id

        df = pd.read_csv(label_path)
        total = f_after + f_before + 1
        last_frame = len(glob.glob(f"{video_path}/*.jpg")) - 1
        segments = []

        for index, row in df.iterrows():
            key_value = row['Key']  # Key pressed
            # Frame number where key was pressed
            key_frame = int(row['Frame']) + delay

            if key_value not in self.id2label:
                continue

            pos_start, pos_end = max(key_frame - f_before, 0), min(key_frame + f_after, last_frame)
            if (pos_end - pos_start + 1) == total:
                segments.append(([pos_start, pos_end], key_value))
        self.segments = segments

class KeyDetectDataset(BaseStreamDataset):
    def __init__(self,
                 video_path: str,
                 label_path: str,
                 gap,
                 f_before=3,
                 f_after=4,
                 delay=10,
                 transforms=None):

        self.video_path = video_path
        self.video_name = video_path.split('/')[-1]
        self.data_dir = video_path.split('/')[-3]
        self.transforms = transforms
        last_frame = len(glob.glob(f"{video_path}/*.jpg")) - 1

        df = pd.read_csv(label_path)
        total_window = f_before + f_after + 1

        segments = []

        self.id2label = detect_id2label
        self.label2id = detect_label2id

        for index, row in df.iterrows():
            # Frame number where key was pressed
            key_frame = int(row['Frame']) + delay
            key_value = row['Key']  # Key pressed

            pos_start, pos_end = max(key_frame - f_before, 0), min(key_frame + f_after, last_frame)

            # Current video with keystroke
            if (pos_end - pos_start + 1) == total_window:
                if key_value not in clf_id2label:
                    segments.append(([pos_start, pos_end], self.id2label[0]))
                else:
                    segments.append(([pos_start, pos_end], self.id2label[1]))

            # Infer idle frames.
            is_idle_before = False
            if index == 0:
                neg_start, neg_end = 0, pos_start - gap
                is_idle_before = True
            else:
                prev_key_frame = df.iloc[index - 1]['Frame']
                prev_pos_end = prev_key_frame + f_after
                if (pos_start - prev_pos_end) - 1 >= (f_after + f_before + 1 + gap * 2):
                    neg_start = prev_pos_end + gap
                    neg_end = pos_start - gap
                    is_idle_before = True

            # Negative class video segments before
            if is_idle_before:
                j = neg_start
                while (j + total_window - 1) <= neg_end:
                    segments.append(
                        ([j, j + total_window - 1], self.id2label[0]))
                    j += total_window
        self.segments = segments


if __name__ == "__main__":
    detect_ds = BaseStreamDataset.create_dataset(
        video_path='datasets/test-1/raw_frames/video_1',
        label_path='datasets/test-1/labels/video_1.csv',
        gap=2,
        delay=4,
    )

    video, label = detect_ds[0]
    print('label: ', label)
    print('video: ', video.shape)

    print(detect_ds.get_class_counts())

    clf_ds = BaseStreamDataset.create_dataset(
        video_path='datasets/test-1/raw_frames/video_1',
        label_path='datasets/test-1/labels/video_1.csv',
        gap=None,
        delay=4,
        transforms=v2.ColorJitter(brightness=(0.5, 1.5), contrast=(
            1), saturation=(0.5, 1.5), hue=(-0.1, 0.1)),
        resize_shape=(360, 640)
    )

    video, label = clf_ds[0]
    print('label: ', label)
    print('video: ', video.shape)

    torchvision.io.video.write_video(
        'sample.mp4', 
        video.permute(0, 2, 3, 1), 
        fps=3.0)

    print(clf_ds.get_class_counts())