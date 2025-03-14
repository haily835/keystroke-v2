import argparse
import glob
from lightning.pytorch.cli import LightningCLI
import torch
import pandas as pd
import torchvision
import os
from lightning_utils.lm_module import LmKeyClf
from utils.import_by_modulepath import import_by_modulepath


device = (
    "cuda" if torch.cuda.is_available() else "cpu"
)

clf_id2label = [
"a", "b", "c", "d", "e", 
"f", "g", "h", "i", "j", 
"k", "l", "m", "n", "o", 
"p", "q", "r", "s", "t", 
"u", "v", "w", "x", "y", "z", 
"comma", "period", "space", "backspace",
"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]

print(f"Using {device} device")

def parse_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process videos with specified models.")

    # Add arguments
    parser.add_argument(
        '--videos',
        type=int,
        nargs='+',  # Accept one or more values
        help='List of video paths or a single video path.',
        default=[0],
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Dataset directory',
        default='datasets/topview/landmarks',
    )

    parser.add_argument(
        '--landmark',
        type=int,
        help='File is in image form or landmarks',
        default=1,
    )

    parser.add_argument(
        '--window_size',
        type=int,
        help='Window size to scan',
        default=8,
    )

    parser.add_argument(
        '--clf_ckpt',
        type=str,
        help='Path to the classifier checkpoint file.',
        default='ckpts/epoch=24-step=88100.ckpt',
        required=False
    )

    parser.add_argument(
        '--det_ckpt',
        type=str,
        help='Path to the detector checkpoint file.',
        default='/Users/haily/Documents/GitHub/Research Learning/ckpts/hf-tv2/detect-epoch=21-step=5478.ckpt',
        required=False
    )

    parser.add_argument(
        '--result_dir',
        default='./stream_results',
        type=str,
        help='Directory to save the results.',
        required=False
    )

    parser.add_argument(
        '--module_classpath',
        type=str,
        help='Lightning module class',
        default='lightning_utils.lm_module.LmKeyClf',
    )
    # Parse the arguments
    args = parser.parse_args()
    return args


def get_model_weight_from_ckpt(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
    model_weights = checkpoint['state_dict']
    for key in list(model_weights):
        model_weights[key.replace("model.", "")] = model_weights.pop(key)
    return model_weights

def main():
    args = parse_arguments()
    # Access the arguments
    videos = args.videos
    data_dir = args.data_dir
    clf_ckpt = args.clf_ckpt
    det_ckpt = args.det_ckpt
    result_dir = args.result_dir
    window_size = args.window_size
    landmark = args.landmark
    print('landmark: ', landmark)
    module = import_by_modulepath(args.module_classpath)

    print(f"Data: {data_dir}")
    print(f"Videos: {videos}")
    print(f"Window size: {window_size}")
    print(f"Classifier checkpoint: {clf_ckpt}")
    print(f"Detector checkpoint: {det_ckpt}")
    print(f"Results will be saved in: {result_dir}")
    clf_checkpoint = torch.load(clf_ckpt, map_location=lambda storage, loc: storage)
    det_checkpoint = torch.load(det_ckpt, map_location=lambda storage, loc: storage)
    clf_init_args = clf_checkpoint["hyper_parameters"]
    det_init_args = det_checkpoint["hyper_parameters"]
    
    clf = module.load_from_checkpoint(**clf_init_args, checkpoint_path=clf_ckpt).model
    det = module.load_from_checkpoint(**det_init_args, checkpoint_path=det_ckpt).model

    clf.to(device)
    det.to(device)
    clf.eval()
    det.eval()

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for video_name in videos:
        print(f"-----Video: {video_name}----")
        if landmark:
            video_path = f"{data_dir}/video_{video_name}.pt"
            video = torch.load(video_path, weights_only=True)
        else:
            video = glob.glob(f"{data_dir}/video_{video_name}/*.jpg")
        
        print('Total frames: ', len(video))
        clf.to(device)
        clf.eval()
        clf_record = []

        curr_frame = 0
        windows = []
        detect_record = []
        clf_record = []
        
        while curr_frame < len(video):
            frame = video[curr_frame]

            if not landmark:
                frame = torchvision.io.read_image(
                    f"{data_dir}/video_{video_name}/frame_{curr_frame}.jpg"
                )

            if len(windows) < window_size:
                windows.append(frame)
                curr_frame += 1
            else:
                frames = torch.stack(windows)
                if landmark:
                    frames = frames.permute(3, 0, 2, 1).float().unsqueeze(dim=0).to(device)
                else:
                    frames = frames.permute(1, 0, 2, 3).float().unsqueeze(dim=0).to(device)
                detect_logits = torch.nn.functional.softmax(det(frames).squeeze(), dim=0)
                detect_record.append([curr_frame] + detect_logits.tolist())
                clf_logits = torch.nn.functional.softmax(clf(frames).squeeze(), dim=0)
                pred_id = torch.argmax(clf_logits, dim=0).item()
                clf_label = clf_id2label[pred_id]
                clf_record.append([curr_frame, clf_label] + clf_logits.tolist())
                windows = windows[1:]

        record_dict = {
            'Start frame': [record[0] - window_size for record in detect_record],
            'Key prediction': [record[1] for record in clf_record],
            'Idle Prob': [record[1] for record in detect_record],
            'Active Prob': [record[2] for record in detect_record],
        }

        for i in range(30):
            record_dict[clf_id2label[i]] = [record[2 + i] for record in clf_record]

        df = pd.DataFrame(record_dict)

        df.to_csv(f'{result_dir}/{video_name}.csv', index=False)

if __name__ == "__main__":
    main()