import argparse
import glob
from lightning.pytorch.cli import LightningCLI
import torch
import pandas as pd
import torchvision
import os
from lightning_utils.dataset import clf_id2label, detect_id2label
from lightning_utils.lm_module import LmKeyClf
from utils.import_by_modulepath import import_by_modulepath
import numpy as np

def weighting_func(x):
    return (1 / (1 + np.exp(-0.2 * (x - 9))))

class Queue:
    # Constructor creates a list
    def __init__(self, max_size, n_classes):
        self.queue = list(np.zeros((max_size, n_classes), dtype=float).tolist())
        self.max_size = max_size
        self.median = None
        self.ma = None
        self.ewma = None

    # Adding elements to queue
    def enqueue(self, data):
        self.queue.insert(0, data)
        self.median = self._median()
        self.ma = self._ma()
        self.ewma = self._ewma()
        return True

    # Removing the last element from the queue
    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.pop()
        return ("Queue Empty!")

    # Getting the size of the queue
    def size(self):
        return len(self.queue)

    # printing the elements of the queue
    def printQueue(self):
        return self.queue

    # Average
    def _ma(self):
        return np.array(self.queue[:self.max_size]).mean(axis=0)

    # Median
    def _median(self):
        return np.median(np.array(self.queue[:self.max_size]), axis=0)

    # Exponential average
    def _ewma(self):
        weights = np.exp(np.linspace(-1., 0., self.max_size))
        weights /= weights.sum()
        average = weights.reshape(1, self.max_size).dot(np.array(self.queue[:self.max_size]))
        return average.reshape(average.shape[1], )


device = (
    "cuda" if torch.cuda.is_available() else "cpu"
)

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
        default=[3,4,10],
        required=False
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Dataset directory',
        default='datasets/tablet7/landmarks',
        required=False
    )

    parser.add_argument(
        '--landmark',
        type=int,
        help='File is in image form or landmarks',
        default=1,
        required=False
    )

    parser.add_argument(
        '--window_size',
        type=int,
        help='Window size to scan',
        default=8,
        required=False
    )

    parser.add_argument(
        '--det_counter',
        type=int,
        help='Window size to scan',
        default=2,
    )

    parser.add_argument(
        '--stride_len',
        type=int,
        default=1,
        required=False
    )



    parser.add_argument(
        '--sample_duration_clf',
        type=int,
        default=8,
        required=False
    )

    parser.add_argument(
        '--clf_threshold_final',
        type=float,
        default=0.05,
        required=False
    )

    parser.add_argument(
        '--clf_threshold_pre',
        type=float,
        default=1.0,
        required=False
    )
    
    parser.add_argument(
        '--n_classes_clf',
        type=int,
        default=30,
        required=False
    )



    parser.add_argument(
        '--n_classes_det',
        type=int,
        default=2,
        required=False
    )


    parser.add_argument(
        '--det_queue_size',
        type=int,
        default=16,
        required=False
    )

    parser.add_argument(
        '--clf_queue_size',
        type=int,
        default=4,
        required=False
    )

    parser.add_argument(
        '--det_strategy',
        type=str,
        default='median',
        required=False
    )

    parser.add_argument(
        '--clf_strategy',
        type=str,
        default='median',
        required=False
    )


    parser.add_argument(
        '--clf_ckpt',
        type=str,
        help='Path to the classifier checkpoint file.',
        default='ckpts/HyperGT-tablet/clf/epoch=34-step=1680.ckpt',
        required=False
    )

    parser.add_argument(
        '--det_ckpt',
        type=str,
        help='Path to the detector checkpoint file.',
        default='ckpts/HyperGT-tablet/det/epoch=20-step=1218.ckpt',
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
        
        clf_record = []
        curr_frame = 0
        windows = []
        detect_record = []
        clf_record = []
        
        active_index = 0
        passive_count = 0
        active = False
        prev_active = False
        finished_prediction = None
        pre_predict = False
        results = []
        prev_best1 = args.n_classes_clf
        cum_sum = np.zeros(args.n_classes_clf, )
        clf_selected_queue = np.zeros(args.n_classes_clf, )
        det_selected_queue = np.zeros(args.n_classes_det, )
        myqueue_det = Queue(args.det_queue_size, n_classes=args.n_classes_det)
        myqueue_clf = Queue(args.clf_queue_size, n_classes=args.n_classes_clf)

        while curr_frame < len(video):
            frame = video[curr_frame]

            if not landmark:
                frame = torchvision.io.read_image(
                    f"{data_dir}/video_{video_name}/frame_{curr_frame}.jpg"
                )

            if len(windows) < window_size:
                windows.append(frame)
                curr_frame += 1
                continue
     
            frames = torch.stack(windows)
            if landmark:
                frames = frames.permute(3, 0, 2, 1).float().unsqueeze(dim=0).to(device)
            else:
                frames = frames.permute(1, 0, 2, 3).float().unsqueeze(dim=0).to(device)
            
            detect_logits = torch.nn.functional.softmax(det(frames).squeeze(), dim=0)
            myqueue_det.enqueue(detect_logits.tolist())

            if args.det_strategy == 'raw':
                det_selected_queue = detect_logits.tolist()
            elif args.det_strategy == 'median':
                det_selected_queue = myqueue_det.median
            elif args.det_strategy == 'ma':
                det_selected_queue = myqueue_det.ma
            elif args.det_strategy == 'ewma':
                det_selected_queue = myqueue_det.ewma
            prediction_det = np.argmax(det_selected_queue)
            prob_det = det_selected_queue[prediction_det]

            #### State of the detector is checked here as detector act as a switch for the classifier
            if prediction_det == 1:
                clf_logits = torch.nn.functional.softmax(clf(frames).squeeze(), dim=0)
                # Push the probabilities to queue
                myqueue_clf.enqueue(clf_logits.tolist())
                passive_count = 0

                if args.clf_strategy == 'raw':
                    clf_selected_queue = outputs_clf
                elif args.clf_strategy == 'median':
                    clf_selected_queue = myqueue_clf.median
                elif args.clf_strategy == 'ma':
                    clf_selected_queue = myqueue_clf.ma
                elif args.clf_strategy == 'ewma':
                    clf_selected_queue = myqueue_clf.ewma

            else:
                outputs_clf = np.zeros(args.n_classes_clf, )
                # Push the probabilities to queue
                myqueue_clf.enqueue(outputs_clf.tolist())
                passive_count += 1

            if passive_count >= args.det_counter or curr_frame == (len(video) -2):
                active = False
            else:
                active = True

            # one of the following line need to be commented !!!!
            if active:
                active_index += 1
                cum_sum = ((cum_sum * (active_index - 1)) + (
                            weighting_func(active_index) * clf_selected_queue)) / active_index  # Weighted Aproach
                # cum_sum = ((cum_sum * (x-1)) + (1.0 * clf_selected_queue))/x #Not Weighting Aproach

                best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
                if float(cum_sum[best1] - cum_sum[best2]) > args.clf_threshold_pre:
                    finished_prediction = True
                    pre_predict = True

            else:
                active_index = 0

            if active == False and prev_active == True:
                finished_prediction = True
            elif active == True and prev_active == False:
                finished_prediction = False

            if finished_prediction == True:
                best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
                print(cum_sum[best1])
                
                if cum_sum[best1] > args.clf_threshold_final:
                    if pre_predict == True:
                        if best1 != prev_best1:
                            if cum_sum[best1] > args.clf_threshold_final:
                                results.append(((curr_frame * args.stride_len) + args.sample_duration_clf, best1))
                                print('Early Detected - class : {} with prob : {} at frame {}'.format(clf_id2label[best1], cum_sum[best1],
                                                                                                    (
                                                                                                                curr_frame * args.stride_len) + args.sample_duration_clf))
                    else:
                        if cum_sum[best1] > args.clf_threshold_final:
                            if best1 == prev_best1:
                                if cum_sum[best1] > 5:
                                    results.append(((curr_frame * args.stride_len) + args.sample_duration_clf, best1))
                                    print('Late Detected - class : {} with prob : {} at frame {}'.format(clf_id2label[best1],
                                                                                                        cum_sum[best1], 
                                                                                                        (curr_frame * args.stride_len) + args.sample_duration_clf))
                            else:
                                results.append(((curr_frame * args.stride_len) + args.sample_duration_clf, best1))

                                print('Late Detected - class : {} with prob : {} at frame {}'.format(
                                    clf_id2label[best1], cum_sum[best1],
                                    (curr_frame * args.stride_len) + args.sample_duration_clf)
                                )

                    finished_prediction = False
                    prev_best1 = best1

                cum_sum = np.zeros(args.n_classes_clf, )

            if active == False and prev_active == True:
                pre_predict = False

            prev_active = active
            windows = windows[1:]
        

if __name__ == "__main__":
    main()