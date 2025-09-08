#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import toml
import pandas as pd
import soundfile as sf
import numpy as np
import time
import torch
import importlib
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from module.dc_crn import DCCRN
from module.ds_dc_crn import DSDCCRN
from dataset.dataset import DNS_Dataset
from dataset.compute_metrics import compute_metric
from audio.utils import prepare_empty_path
from audio.metrics import SI_SDR

def normalize_audio(audio):
    """对音频进行归一化处理，确保最大振幅为0.9"""
    if np.max(np.abs(audio)) > 0:  # 避免全零音频导致的除零错误
        # 将音频归一化到[-0.9, 0.9]范围，留出一点余量避免截断
        scale_factor = 0.9 / np.max(np.abs(audio))
        return audio * scale_factor
    return audio

def get_model_class(model_name):
    """
    根据模型名称动态获取模型类
    """
    # 预定义的模型映射
    model_mapping = {
        "DCCRN": DCCRN,
        "DSDCCRN": DSDCCRN,
        # 在这里可以添加更多模型类型
    }
    
    if model_name in model_mapping:
        return model_mapping[model_name]
    else:
        # 动态导入模型 (如果模型名称不在预定义列表中)
        try:
            module_path = f"module.{model_name.lower()}"
            module = importlib.import_module(module_path)
            return getattr(module, model_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"未找到模型: {model_name}. 错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="Student Model Inference")
    parser.add_argument("-C", "--config", default="config/student_config.toml", type=str, help="配置文件路径 (.toml)")
    parser.add_argument("--model", type=str, help="模型权重文件路径 (.pth)")
    parser.add_argument("--input", type=str, help="输入噪声音频文件路径 (单文件模式)")
    parser.add_argument("--output", type=str, help="输出增强音频文件路径 (单文件模式)")
    parser.add_argument("--batch_size", type=int, default=1, help="推理批次大小")
    parser.add_argument("--max_files", type=int, default=None, help="最大处理文件数量 (用于测试)")
    parser.add_argument("--disable_audio_save", action="store_true", help="禁用增强音频文件保存")
    parser.add_argument("--model_name", type=str, help="模型名称 (覆盖配置文件中的名称)")
    args = parser.parse_args()

    print(f"===== 学生模型推理脚本 - 启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    
    # 配置设备
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("使用 Apple MPS 加速推理")
    else:
        print("无法使用 GPU 加速，使用 CPU 推理")
        
    print(f"使用设备: {device}")

    # 读取配置文件
    config = toml.load(args.config)
    
    # 设置路径
    base_path = config["path"]["base"]
    model_path = args.model if args.model else config["path"].get("pre_model")
    if model_path == 0 or not model_path:  # 检查模型路径是否有效
        print("错误: 未指定有效的模型路径")
        sys.exit(1)
        
    # 目录名称基于模型名称
    model_name = args.model_name if args.model_name else config["model"]["name"]
    output_path = os.path.join(base_path, "enhanced", f"{model_name.lower()}")
    metrics_path = os.path.join(base_path, "metrics", f"{model_name.lower()}")
    
    # 确保输出目录存在
    prepare_empty_path([output_path, metrics_path])
    
    # 单文件处理模式
    single_file_mode = args.input is not None and args.output is not None
    
    if single_file_mode:
        print(f"单文件处理模式: 输入 = {args.input}, 输出 = {args.output}")
    else:
        print("批处理模式: 处理整个测试集")
    
    # 设置STFT参数
    sr = config["dataset"]["sr"]
    n_fft = config["dataset"]["n_fft"]
    win_len = config["dataset"]["win_len"]
    hop_len = config["dataset"]["hop_len"]
    window = torch.hann_window(win_len, periodic=False).to(device)
    
    # 创建模型
    print(f"创建模型: {model_name}...")
    try:
        # 动态获取模型类
        ModelClass = get_model_class(model_name)
        
        # 创建模型实例
        model = ModelClass(
            n_fft=config["dataset"]["n_fft"],
            rnn_layers=config["model"]["rnn_layers"],
            rnn_units=config["model"]["rnn_units"],
            kernel_num=config["model"]["kernel_num"],
            kernel_size=config["model"]["kernel_size"],
        )
        print(f"成功创建模型: {model_name}")
    except Exception as e:
        print(f"创建模型失败: {e}")
        sys.exit(1)
    
    # 加载模型权重
    try:
        print(f"加载模型权重: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)
    
    # 将模型移动到设备
    model = model.to(device)
    model.eval()
    
    # 单文件处理模式
    if single_file_mode:
        process_single_file(args.input, args.output, model, device, sr, n_fft, win_len, hop_len, window)
    else:
        # 批处理模式
        process_test_dataset(args, config, device, model, sr, n_fft, win_len, hop_len, window, output_path, metrics_path)
    
    print(f"===== 推理完成 - 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')} =====")

def process_single_file(input_file, output_file, model, device, sr, n_fft, win_len, hop_len, window):
    """处理单个音频文件"""
    print(f"加载音频文件: {input_file}")
    
    try:
        # 加载音频
        noisy, file_sr = sf.read(input_file)
        
        # 重采样（如果需要）
        if file_sr != sr:
            print(f"重采样音频从 {file_sr}Hz 到 {sr}Hz")
            import librosa
            noisy = librosa.resample(noisy, orig_sr=file_sr, target_sr=sr)
        
        # 确保音频是float32类型
        if noisy.dtype != np.float32:
            noisy = noisy.astype(np.float32)
            
        # 归一化音频音量
        noisy = normalize_audio(noisy)
        
        # 转换为torch张量并添加批次维度
        noisy = torch.FloatTensor(noisy).unsqueeze(0).to(device)
        
        # STFT
        noisy_spec = torch.stft(
            noisy,
            n_fft,
            hop_length=hop_len,
            win_length=win_len,
            window=window,
            return_complex=False,
        )
        
        # 前向传播
        with torch.no_grad():
            mask = model(noisy_spec)
        
        # 应用掩码
        mask_mags = (mask[:, 0, :, :] ** 2 + mask[:, 1, :, :] ** 2) ** 0.5
        phase_real = mask[:, 0, :, :] / (mask_mags + 1e-8)
        phase_imag = mask[:, 1, :, :] / (mask_mags + 1e-8)
        mask_phase = torch.atan2(phase_imag, phase_real)
        mask_mags = torch.tanh(mask_mags)
        enh_mags = mask_mags * torch.sqrt(noisy_spec[:, :, :, 0] ** 2 + noisy_spec[:, :, :, 1] ** 2)
        enh_phase = torch.atan2(noisy_spec[:, :, :, 1], noisy_spec[:, :, :, 0]) + mask_phase
        spec_real = enh_mags * torch.cos(enh_phase)
        spec_imag = enh_mags * torch.sin(enh_phase)
        cspec = spec_real + 1j * spec_imag
        
        # ISTFT
        enh = torch.istft(
            cspec,
            n_fft,
            hop_length=hop_len,
            win_length=win_len,
            window=window,
            return_complex=False,
        )
        enh = torch.clamp(enh, min=-1.0, max=1.0)
        
        # 转换为numpy数组
        enh_np = enh.cpu().numpy().squeeze()
        
        # 保存增强音频
        print(f"保存增强音频到: {output_file}")
        sf.write(output_file, enh_np, sr)
        print("处理完成!")
        
    except Exception as e:
        print(f"处理音频文件时出错: {e}")
        import traceback
        traceback.print_exc()

def process_test_dataset(args, config, device, model, sr, n_fft, win_len, hop_len, window, output_path, metrics_path):
    """处理整个测试数据集"""
    # 获取数据集路径
    dataset_path = os.path.join(os.getcwd(), "dataset_csv")
    
    # 创建测试数据加载器
    print("加载测试数据集...")
    test_set = DNS_Dataset(dataset_path, config, mode="test")
    
    # 如果指定了最大文件数，截取测试集
    if args.max_files is not None:
        test_set.length = min(test_set.length, args.max_files)
        test_set.noisy_files = test_set.noisy_files[:test_set.length]
        test_set.clean_files = test_set.clean_files[:test_set.length]
        print(f"已限制处理文件数量为: {test_set.length}")
    
    test_iter = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 使用单线程避免多进程问题
        drop_last=False,
        pin_memory=False,
    )
    
    # 准备存储评估指标的列表
    si_sdr_scores = []
    stoi_scores = []
    wb_pesq_scores = []
    
    # 处理批次
    print("开始推理...")
    
    # 创建保存增强音频文件的列表
    enh_file_list = []
    enh_data_list = []
    clean_data_list = []
    
    try:
        with torch.no_grad():
            for batch_idx, (noisy, clean, noisy_file) in enumerate(tqdm(test_iter, desc="推理进度")):
                try:
                    # 将数据移动到设备
                    noisy = noisy.to(device)
                    clean = clean.to(device)
                    
                    # 进行STFT
                    noisy_spec = torch.stft(
                        noisy,
                        n_fft,
                        hop_length=hop_len,
                        win_length=win_len,
                        window=window,
                        return_complex=False,
                    )
                    
                    # 前向传播
                    mask = model(noisy_spec)
                    
                    # 应用掩码并进行ISTFT以获得增强音频
                    mask_mags = (mask[:, 0, :, :] ** 2 + mask[:, 1, :, :] ** 2) ** 0.5
                    phase_real = mask[:, 0, :, :] / (mask_mags + 1e-8)
                    phase_imag = mask[:, 1, :, :] / (mask_mags + 1e-8)
                    mask_phase = torch.atan2(phase_imag, phase_real)
                    mask_mags = torch.tanh(mask_mags)
                    enh_mags = mask_mags * torch.sqrt(noisy_spec[:, :, :, 0] ** 2 + noisy_spec[:, :, :, 1] ** 2)
                    enh_phase = torch.atan2(noisy_spec[:, :, :, 1], noisy_spec[:, :, :, 0]) + mask_phase
                    spec_real = enh_mags * torch.cos(enh_phase)
                    spec_imag = enh_mags * torch.sin(enh_phase)
                    cspec = spec_real + 1j * spec_imag
                    
                    # ISTFT
                    enh = torch.istft(
                        cspec,
                        n_fft,
                        hop_length=hop_len,
                        win_length=win_len,
                        window=window,
                        return_complex=False,
                    )
                    enh = torch.clamp(enh, min=-1.0, max=1.0)
                    
                    # 确保长度一致
                    min_len = min(enh.size(-1), clean.size(-1))
                    enh = enh[..., :min_len]
                    clean = clean[..., :min_len]
                    
                    # 转换为numpy数组
                    noisy_np = noisy.cpu().numpy()
                    clean_np = clean.cpu().numpy()
                    enh_np = enh.cpu().numpy()
                    
                    # 挤压单批次维度
                    noisy_np = np.squeeze(noisy_np, axis=0) 
                    clean_np = np.squeeze(clean_np, axis=0)
                    enh_np = np.squeeze(enh_np, axis=0)
                    
                    # 确保长度一致
                    min_len = min(len(enh_np), len(clean_np))
                    clean_np = clean_np[:min_len]
                    enh_np = enh_np[:min_len]
                    
                    # 计算当前批次的SI-SDR分数
                    try:
                        si_sdr = SI_SDR(enh_np, clean_np)
                        si_sdr_scores.append(si_sdr)
                    except Exception as e:
                        print(f"[警告] 计算SI-SDR时出错: {e}")
                    
                    # 准备保存增强音频
                    for i in range(len(noisy_file)):
                        enh_file = os.path.join(output_path, os.path.basename(noisy_file[i]).replace("noisy", "enh_noisy"))
                        enh_file_list.append(enh_file)
                        enh_data_list.append(enh_np)
                        clean_data_list.append(clean_np)
                    
                    # 每处理10个批次打印一次进度
                    if (batch_idx + 1) % 10 == 0:
                        print(f"已处理 {batch_idx + 1} 个批次, 共 {len(test_iter)} 个批次")
                        
                except Exception as e:
                    print(f"[错误] 处理批次 {batch_idx} 时出错: {e}")
                    continue
    except Exception as e:
        print(f"[严重错误] 推理过程中断: {e}")
    
    print("推理完成, 开始保存结果...")
    
    # 计算平均分数
    avg_si_sdr = np.mean(si_sdr_scores) if si_sdr_scores else 0
    print(f"平均SI-SDR得分: {avg_si_sdr:.4f}")
    
    # 计算和保存STOI和PESQ指标
    try:
        print("计算STOI和PESQ指标...")
        metrics = {
            "SI_SDR": [],
            "STOI": [],
            "WB_PESQ": [],
            "NB_PESQ": [],
        }
        
        # 计算所有指标
        compute_metric(
            enh_data_list,
            clean_data_list,
            metrics,
            n_folds=1,
            n_jobs=1,  # 使用单线程避免多进程问题
            pre_load=True,
        )
        
        # 保存指标
        print("保存评估指标...")
        df = pd.DataFrame(metrics, index=["enh"])
        df.to_csv(os.path.join(metrics_path, "enh_metrics.csv"))
        print(f"评估指标已保存至: {os.path.join(metrics_path, 'enh_metrics.csv')}")
        
        # 显示评估结果
        print("\n===== 评估结果 =====")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
    except Exception as e:
        print(f"[错误] 计算或保存指标时出错: {e}")
    
    # 保存增强音频文件
    if not args.disable_audio_save:
        try:
            print("保存增强音频文件...")
            for i, (enh_file, enh_data) in enumerate(zip(enh_file_list, enh_data_list)):
                try:
                    sf.write(enh_file, enh_data, sr)
                    if (i + 1) % 10 == 0:
                        print(f"已保存 {i + 1}/{len(enh_file_list)} 个音频文件")
                except Exception as e:
                    print(f"[警告] 保存文件 {enh_file} 时出错: {e}")
                    continue
            print(f"所有增强音频文件已保存至: {output_path}")
        except Exception as e:
            print(f"[错误] 保存音频文件时出错: {e}")
    else:
        print("已禁用音频文件保存")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[严重错误] 程序崩溃: {e}")
        import traceback
        traceback.print_exc()