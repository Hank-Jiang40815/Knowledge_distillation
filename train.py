#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import toml
import torch

from trainer.base_trainer import BaseTrainer
from module.dc_crn import DCCRN
from dataset.dataset import DNS_Dataset
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description="DCCRN Model Training")
    parser.add_argument("-C", "--config", default="config/base_config.toml", type=str, help="Config file path (.toml)")
    args = parser.parse_args()

    # 確保模型使用單精度浮點數
    torch.set_default_dtype(torch.float32)
    
    # 配置設備 (CPU/GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 在 PyTorch 1.12.0+ 中才支持 MPS
    # PyTorch 1.10.2 不支持 MPS
    print(f"使用設備: {device}")

    # 讀取配置文件
    config = toml.load(args.config)

    # 設置數據集路徑
    dataset_path = os.path.join(os.getcwd(), "dataset_csv")

    # 獲取數據加載器參數
    batch_size = config["dataloader"]["batch_size"]
    num_workers = 0 if device == "cpu" else config["dataloader"]["num_workers"]
    drop_last = config["dataloader"]["drop_last"]
    pin_memory = config["dataloader"]["pin_memory"]

    # 創建訓練數據加載器
    train_set = DNS_Dataset(dataset_path, config, mode="train")
    train_iter = DataLoader(
        train_set,
        batch_size=batch_size[0],
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    # 創建驗證數據加載器
    valid_set = DNS_Dataset(dataset_path, config, mode="valid")
    valid_iter = DataLoader(
        valid_set,
        batch_size=batch_size[1],
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    # 創建模型
    model = DCCRN(
        n_fft=config["dataset"]["n_fft"],
        rnn_layers=config["model"]["rnn_layers"],
        rnn_units=config["model"]["rnn_units"],
        kernel_num=config["model"]["kernel_num"],
        kernel_size=config["model"]["kernel_size"],
    )
    
    # 確保模型使用單精度浮點數
    model = model.float()

    # 創建訓練器並開始訓練
    trainer = BaseTrainer(config, model, train_iter, valid_iter, device)
    trainer()

if __name__ == "__main__":
    main()