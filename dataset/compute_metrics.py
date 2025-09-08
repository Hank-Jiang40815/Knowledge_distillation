# -*- coding: utf-8 -*-

import sys
import os
import toml
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.append(os.getcwd())
from audio.metrics import SI_SDR, STOI, WB_PESQ, NB_PESQ, REGISTERED_METRICS


def calculate_metric(noisy_file, clean_file, sr=16000, metric_type="STOI", pre_load=False):
    # get noisy, clean
    if pre_load == False:
        noisy, _ = librosa.load(noisy_file, sr=sr)
        clean, _ = librosa.load(clean_file, sr=sr)
    else:
        noisy = noisy_file
        clean = clean_file
    
    # 检查数据类型，确保是数组而不是标量
    if isinstance(noisy, (float, np.float32, np.float64)) or isinstance(clean, (float, np.float32, np.float64)):
        print(f"Warning: Skipping scalar value in calculate_metric. Types: noisy={type(noisy)}, clean={type(clean)}")
        # 返回一个默认值
        if metric_type in ["SI_SDR"]:
            return 0.0
        elif metric_type in ["STOI"]:
            return 0.0
        elif metric_type in ["WB_PESQ"]:
            return 1.0  # PESQ的最小值
        elif metric_type in ["NB_PESQ"]:
            return 1.0  # PESQ的最小值
    
    # 确保两个数组长度相同
    if len(noisy) != len(clean):
        min_len = min(len(noisy), len(clean))
        noisy = noisy[:min_len]
        clean = clean[:min_len]

    # get metric score
    if metric_type in ["SI_SDR"]:
        return SI_SDR(noisy, clean)
    elif metric_type in ["STOI"]:
        return STOI(noisy, clean, sr=sr)
    elif metric_type in ["WB_PESQ"]:
        return WB_PESQ(noisy, clean)
    elif metric_type in ["NB_PESQ"]:
        return NB_PESQ(noisy, clean)


def compute_metric(noisy_files, clean_files, metrics, n_folds=1, n_jobs=8, pre_load=False):
    for metric_type, _ in metrics.items():
        assert metric_type in REGISTERED_METRICS

        # 过滤掉无效数据
        valid_indices = []
        for i, (noisy, clean) in enumerate(zip(noisy_files, clean_files)):
            if pre_load:
                # 确保数据有效且不是标量
                if (not isinstance(noisy, (float, np.float32, np.float64)) and 
                    not isinstance(clean, (float, np.float32, np.float64)) and
                    len(noisy) > 0 and len(clean) > 0):
                    valid_indices.append(i)
            else:
                # 文件路径模式，默认视为有效
                valid_indices.append(i)
        
        # 仅使用有效数据
        valid_noisy = [noisy_files[i] for i in valid_indices]
        valid_clean = [clean_files[i] for i in valid_indices]
        
        if len(valid_noisy) == 0:
            print(f"警告: 没有找到有效的音频数据用于计算 {metric_type}，返回默认值")
            if metric_type in ["SI_SDR"]:
                metrics[metric_type] = 0.0
            elif metric_type in ["STOI"]:
                metrics[metric_type] = 0.0
            elif metric_type in ["WB_PESQ"]:
                metrics[metric_type] = 1.0
            elif metric_type in ["NB_PESQ"]:
                metrics[metric_type] = 1.0
            continue

        split_num = len(valid_noisy) // n_folds
        score = []
        for n in range(n_folds):
            # 确保索引不越界
            start_idx = n * split_num
            end_idx = min((n + 1) * split_num, len(valid_noisy))
            
            if start_idx >= end_idx:
                continue
                
            metric_score = Parallel(n_jobs=n_jobs)(
                delayed(calculate_metric)(
                    valid_noisy[i],
                    valid_clean[i],
                    sr=8000 if metric_type in ["NB_PESQ"] else 16000,
                    metric_type=metric_type,
                    pre_load=pre_load,
                )
                for i in range(start_idx, end_idx)
            )
            
            # 过滤掉None值
            metric_score = [s for s in metric_score if s is not None]
            if metric_score:
                score.append(np.mean(metric_score))
        
        if score:
            metrics[metric_type] = np.mean(score)
        else:
            print(f"警告: 计算{metric_type}时没有有效分数，返回默认值")
            if metric_type in ["SI_SDR"]:
                metrics[metric_type] = 0.0
            elif metric_type in ["STOI"]:
                metrics[metric_type] = 0.0
            elif metric_type in ["WB_PESQ"]:
                metrics[metric_type] = 1.0
            elif metric_type in ["NB_PESQ"]:
                metrics[metric_type] = 1.0


if __name__ == "__main__":
    # get dataset path
    dataset_path = os.path.join(os.getcwd(), "dataset_csv")

    # get set path
    train_path = os.path.join(dataset_path, "train.csv")
    valid_path = os.path.join(dataset_path, "valid.csv")
    test_path = os.path.join(dataset_path, "test.csv")

    # get train files
    train_files = pd.read_csv(train_path).values
    train_noisy_files = train_files[:, 0].reshape(1, len(train_files))[0]
    train_clean_files = train_files[:, 1].reshape(1, len(train_files))[0]
    # get valid files
    valid_files = pd.read_csv(valid_path).values
    valid_noisy_files = valid_files[:, 0].reshape(1, len(valid_files))[0]
    valid_clean_files = valid_files[:, 1].reshape(1, len(valid_files))[0]
    # get test files
    test_files = pd.read_csv(test_path).values
    test_noisy_files = test_files[:, 0].reshape(1, len(test_files))[0]
    test_clean_files = test_files[:, 1].reshape(1, len(test_files))[0]

    # get compute metrics config
    toml_path = os.path.join(os.path.dirname(__file__), "compute_metrics_cfg.toml")
    config = toml.load(toml_path)
    # get n_jobs
    n_folds = config["ppl"]["n_folds"]
    n_jobs = config["ppl"]["n_jobs"]

    # get metrics
    metrics = {
        "SI_SDR": [],
        "STOI": [],
        "WB_PESQ": [],
        "NB_PESQ": [],
    }

    # compute train metrics
    compute_metric(
        train_noisy_files,
        train_clean_files,
        metrics,
        n_folds=n_folds,
        n_jobs=n_jobs,
        pre_load=False,
    )
    # save train metrics
    df = pd.DataFrame(metrics, index=["train"])
    df.to_csv(os.path.join(dataset_path, "train_metrics.csv"))

    # get metrics
    metrics = {
        "SI_SDR": [],
        "STOI": [],
        "WB_PESQ": [],
        "NB_PESQ": [],
    }

    # compute valid metrics
    compute_metric(
        valid_noisy_files,
        valid_clean_files,
        metrics,
        n_folds=n_folds,
        n_jobs=n_jobs,
        pre_load=False,
    )
    # save train metrics
    df = pd.DataFrame(metrics, index=["valid"])
    df.to_csv(os.path.join(dataset_path, "valid_metrics.csv"))

    # get metrics
    metrics = {
        "SI_SDR": [],
        "STOI": [],
        "WB_PESQ": [],
        "NB_PESQ": [],
    }

    # compute test metrics
    compute_metric(
        test_noisy_files,
        test_clean_files,
        metrics,
        n_folds=n_folds,
        n_jobs=n_jobs,
        pre_load=False,
    )
    # save train metrics
    df = pd.DataFrame(metrics, index=["test"])
    df.to_csv(os.path.join(dataset_path, "test_metrics.csv"))
