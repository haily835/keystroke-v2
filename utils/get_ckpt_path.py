
import pandas as pd
import glob

def get_ckpt_path(log_folder):
    print(f"----- TRAIN METRICS {log_folder}-----")
    metric_path = f'./{log_folder}/lightning_logs/version_0/metrics.csv'
    metrics = pd.read_csv(f'./{log_folder}/lightning_logs/version_0/metrics.csv')
    print(metrics)
    ckpt_path = glob.glob(f'./{log_folder}/lightning_logs/version_0/checkpoints/*.ckpt')[0]

    config_path = f'./{log_folder}/lightning_logs/version_0/config.yaml'
    return ckpt_path, metric_path, config_path
