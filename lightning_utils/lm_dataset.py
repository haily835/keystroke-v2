import torch
import torch.utils
import torch.utils.data
import pandas as pd
import numpy as np
import json

detect_id2label = ['idle', 'active']
detect_label2id = {'idle': 0, 'active': 1}

class BaseStreamDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs): pass

    @classmethod
    def create_dataset(cls, video_path: str, landmark_path:str, label_path: str,
                       gap: int, classes_path,  f_before: int = 3, f_after: int = 4, delay: int = 10):
        if gap:
            return KeyDetectDataset(
                video_path=video_path,
                label_path=label_path,
                landmark_path=landmark_path,
                classes_path=classes_path,
                gap=gap,
                f_after=f_after,
                f_before=f_before,
                delay=delay)
        
        return KeyClfStreamDataset(
            video_path=video_path,
            label_path=label_path,
            landmark_path=landmark_path,
            classes_path=classes_path,
            f_after=f_after,
            f_before=f_before,
            delay=delay)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        (start, end), label = self.segments[idx]
        
        frames = self.video[start: end+1]
        frames = frames.permute(3, 0, 2, 1) # permute to channels, frames, points, hands

        return frames.float(), self.label2id[label]

    def get_class_counts(self):
        labels = np.array([segment[1] for segment in self.segments])

        counts = []
        for label in self.id2label:
            count = np.sum(labels == label)
            counts.append(count)

        df = pd.DataFrame({'label': self.id2label, 'count': counts})
        return df

    
class KeyClfStreamDataset(BaseStreamDataset):
    def __init__(self,
                 video_path: str,
                 label_path: str,
                 classes_path: str,
                 landmark_path: str,
                 f_before=3,
                 f_after=4,
                 delay=4,
                 transforms=None):

        self.video_path = video_path
        self.landmark_path = landmark_path
        self.video_name = video_path.split('/')[-1]
        self.data_dir = video_path.split('/')[-3]
        self.transforms = transforms


        # Load the array of characters from the JSON file
        with open(classes_path, 'r') as f:
            self.id2label = json.load(f)

            self.label2id = {label: idx for idx, label in enumerate(self.id2label)}
        
        self.video = torch.load(self.landmark_path, weights_only=True)
        df = pd.read_csv(label_path)

        if not df['Frame'].is_unique:
            duplicated = df['Frame'].duplicated(keep=False)
            # print(f"Duplicate segment found {len(duplicated)}/ {len(df)}")
            # print(df[duplicated])
            df = df[~df['Frame'].duplicated()]
        
        segments = []
        total = f_after + f_before + 1
        last_frame = len(self.video) - 1
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
                 landmark_path: str,
                 classes_path: str,
                 gap: int,
                 f_before: str = 3,
                 f_after: str = 4,
                 delay: str = 4):

        self.video_path = video_path
        self.landmark_path = landmark_path
        self.video_name = video_path.split('/')[-1]
        self.data_dir = video_path.split('/')[-3]
        self.video = torch.load(self.landmark_path, weights_only=True)
        df = pd.read_csv(label_path)
        total_window = f_before + f_after + 1

        segments = []
        with open(classes_path, 'r') as f:
            clf_id2label = json.load(f)

        self.id2label = detect_id2label
        self.label2id = detect_label2id
        
        last_frame = len(self.video) - 1
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

