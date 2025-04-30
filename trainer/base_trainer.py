# -*- coding: utf-8 -*-

import sys
import os
import argparse
import toml
from tqdm import tqdm
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from dataset.dataset import DNS_Dataset
from module.dc_crn import DCCRN
from dataset.compute_metrics import compute_metric
from audio.utils import prepare_empty_path, print_networks
from audio.metrics import SI_SDR, transform_pesq_range
from audio.feature import EPS

plt.switch_backend("agg")


class BaseTrainer:
    def __init__(self, config, model, train_iter, valid_iter, device="cpu"):
        # get config
        self.config = config
        # get model
        self.model = model
        # get device
        self.device = device
        # get dataset iter
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        # get meta args
        self.use_cudnn = False if self.device == "cpu" else config["meta"]["use_cudnn"]
        self.use_amp = False if self.device == "cpu" else config["meta"]["use_amp"]
        # 在MPS設備上禁用AMP，因為MPS目前不完全支援
        if self.device == "mps":
            self.use_amp = False
        # get base path
        self.base_path = config["path"]["base"]
        # get pre model path
        self.pre_model_path = config["path"]["pre_model"]
        
        # 生成实验标识符，用于区分不同实验
        self.experiment_id = self._generate_experiment_id(config)
        
        # get ppl args
        self.n_folds = config["ppl"]["n_folds"]
        self.n_jobs = config["ppl"]["n_jobs"]
        # get visual args
        self.visual_samples = config["visual"]["samples"]
        # get checkpoint args
        self.save_checkpoint_interval = config["checkpoint"]["save_interval"]
        # get grad args
        self.clip_grad_norm_value = config["grad"]["clip_grad_norm_value"]
        # get dataset args
        self.sr = config["dataset"]["sr"]
        self.n_fft = config["dataset"]["n_fft"]
        self.win_len = config["dataset"]["win_len"]
        self.hop_len = config["dataset"]["hop_len"]
        self.audio_len = config["dataset"]["audio_len"]
        self.window = torch.hann_window(self.win_len, periodic=False).to(self.device)
        # get train args
        self.resume = config["train"]["resume"]
        self.epochs = config["train"]["epochs"]
        self.valid_start_epoch = config["train"]["valid_start_epoch"]
        self.valid_interval = config["train"]["valid_interval"]

        # init cudnn
        if self.device == "cuda":
            torch.backends.cudnn.enabled = self.use_cudnn
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # init common args
        self.start_epoch = 1
        self.best_score = 0.0

        # init path - 使用实验标识符创建唯一的保存路径
        self.checkpoints_path = os.path.join(self.base_path, "checkpoints", self.experiment_id)
        self.logs_path = os.path.join(self.base_path, "logs", "train", self.experiment_id)
        # mkdir path
        prepare_empty_path([self.checkpoints_path, self.logs_path], self.resume)

        # init amp
        if self.device == "cuda" and self.use_amp:
            self.scaler = GradScaler(enabled=self.use_amp)
        else:
            self.scaler = GradScaler(enabled=False)

        # init optimizer
        self.optimizer = getattr(torch.optim, config["optimizer"]["name"])(
            params=self.model.parameters(),
            lr=config["optimizer"]["lr"],
        )

        # init lr scheduler
        self.scheduler_metric = 0.0
        self.scheduler = getattr(torch.optim.lr_scheduler, config["lr_scheduler"]["name"])(
            self.optimizer,
            mode=config["lr_scheduler"]["mode"],
            factor=config["lr_scheduler"]["factor"],
            patience=config["lr_scheduler"]["patience"],
            threshold=config["lr_scheduler"]["threshold"],
            min_lr=config["lr_scheduler"]["min_lr"],
            verbose=config["lr_scheduler"]["verbose"],
        )

        # print params
        print_networks([self.model])

        # Add experiment logging
        self.log_experiment_info()

    def _generate_experiment_id(self, config):
        """根据实验配置生成唯一的实验标识符"""
        # 提取关键参数
        batch_size = config["dataloader"]["batch_size"][0]
        lr = config["optimizer"]["lr"]
        
        # 如果有模型特定参数，也包含进来
        rnn_units = config["model"]["rnn_units"]
        rnn_layers = config["model"]["rnn_layers"]
        
        # 生成实验ID，例如：bs16_lr0.001_u256_l2
        experiment_id = f"bs{batch_size}_lr{lr}_u{rnn_units}_l{rnn_layers}"
        
        # 记录实验ID
        print(f"实验ID: {experiment_id}")
        
        return experiment_id

    @staticmethod
    def loss(enh, clean):
        # 确保两个输入张量具有相同的形状
        min_len = min(enh.shape[-1], clean.shape[-1])
        enh = enh[..., :min_len]
        clean = clean[..., :min_len]
        return -(torch.mean(SI_SDR(enh, clean)))

    def load_pre_model(self):
        load_model = torch.load(self.pre_model_path, map_location="cpu")
        self.model.load_state_dict(load_model)

        print(f"Load pre model done...")

    def init_logs(self):
        # init logs
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.logs_path, f"start_epoch_{self.start_epoch}"),
            max_queue=5,
            flush_secs=60,
        )

        # logs config
        self.writer.add_text(
            tag="config",
            text_string=f"<pre>  \n{toml.dumps(self.config)}  \n</pre>",
            global_step=1,
        )

    def save_checkpoint(self, epoch, is_best_epoch=False):
        print(f"Saving {epoch} epoch checkpoint, {is_best_epoch} best checkpoint...")

        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "model": self.model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

        checkpoint_name_prefix = "best" if is_best_epoch else "latest"
        torch.save(state_dict, os.path.join(self.checkpoints_path, checkpoint_name_prefix + "_checkpoint.tar"))
        torch.save(state_dict["model"], os.path.join(self.checkpoints_path, checkpoint_name_prefix + "_model.pth"))

    def resume_checkpoint(self):
        checkpoint_path = os.path.join(self.checkpoints_path, "latest_checkpoint.tar")
        assert os.path.exists(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.model.load_state_dict(checkpoint["model"])
        self.scaler.load_state_dict(checkpoint["scaler"])

        print(f"Latest checkpoint loaded. Training will begin at {self.start_epoch} epoch.")

    def is_best_epoch(self, score):
        if score > self.best_score:
            self.best_score = score
            return True
        else:
            return False

    def audio_visualization(self, noisy, clean, enh, name, epoch):
        # 确保输入是一维的并且是 float32 类型
        if isinstance(noisy, np.ndarray):
            noisy = noisy.reshape(-1).astype(np.float32)
        if isinstance(clean, np.ndarray):
            clean = clean.reshape(-1).astype(np.float32)
        if isinstance(enh, np.ndarray):
            enh = enh.reshape(-1).astype(np.float32)
            
        try:
            # 确保音频数据在 [-1, 1] 范围内
            noisy = np.clip(noisy, -1.0, 1.0)
            clean = np.clip(clean, -1.0, 1.0)
            enh = np.clip(enh, -1.0, 1.0)
            
            self.writer.add_audio(f"audio/{name}/noisy", noisy, epoch, sample_rate=self.sr)
            self.writer.add_audio(f"audio/{name}/clean", clean, epoch, sample_rate=self.sr)
            self.writer.add_audio(f"audio/{name}/enh", enh, epoch, sample_rate=self.sr)

            # Visualize the spectrogram of noisy speech, clean speech, and enhanced speech
            noisy_mag, _ = librosa.magphase(librosa.stft(noisy, n_fft=320, hop_length=160, win_length=320))
            clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=320, hop_length=160, win_length=320))
            enh_mag, _ = librosa.magphase(librosa.stft(enh, n_fft=320, hop_length=160, win_length=320))
            fig, axes = plt.subplots(3, 1, figsize=(6, 6))
            for k, mag in enumerate([noisy_mag, clean_mag, enh_mag]):
                axes[k].set_title(
                    f"mean: {np.mean(mag):.3f}, "
                    f"std: {np.std(mag):.3f}, "
                    f"max: {np.max(mag):.3f}, "
                    f"min: {np.min(mag):.3f}"
                )
                librosa.display.specshow(
                    librosa.amplitude_to_db(mag, ref=np.max), cmap="magma", y_axis="linear", ax=axes[k], sr=16000
                )
            plt.tight_layout()
            self.writer.add_figure(f"spec/{name}", fig, epoch)
        except Exception as e:
            print(f"音频可视化过程出现错误: {e}")
            # 不让可视化错误影响整体运行

    def metrics_visualization(self, enh_list, clean_list, epoch, n_folds=1, n_jobs=8):
        # get metrics
        metrics = {
            "STOI": [],
            "WB_PESQ": [],
        }

        # 确保输入是数组而非标量
        if len(enh_list.shape) == 0:
            print("警告：增强音频列表是标量值，跳过指标计算")
            # 返回一个默认值
            return 0.5  # 默认值
        
        if len(clean_list.shape) == 0:
            print("警告：干净音频列表是标量值，跳过指标计算")
            # 返回一个默认值
            return 0.5  # 默认值

        # compute enh metrics
        compute_metric(
            enh_list,
            clean_list,
            metrics,
            n_folds=n_folds,
            n_jobs=n_jobs,
            pre_load=True,
        )

        self.writer.add_scalar("STOI/valid", metrics["STOI"], epoch)
        self.writer.add_scalar("WB_PESQ/valid", metrics["WB_PESQ"], epoch)
        
        # 为了在终端打印具体指标值，保存当前指标值
        self.current_stoi = metrics["STOI"]
        self.current_wb_pesq = metrics["WB_PESQ"]

        return ((metrics["STOI"]) + transform_pesq_range(metrics["WB_PESQ"])) / 2

    def update_scheduler(self, metric):
        self.scheduler.step(metric)

    def set_model_to_train_mode(self):
        self.model.train()

    def set_model_to_eval_mode(self):
        self.model.eval()

    def audio_stft(self, audio):
        # audio [B, S]
        # [B, S] -> [B, F, T, 2]
        spec = torch.stft(
            audio,
            self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            return_complex=False,
        )
        return spec

    def audio_istft(self, mask, spec):
        # mask [B, 2, F, T]
        # spec [B, F, T, 2]
        mask_mags = (mask[:, 0, :, :] ** 2 + mask[:, 1, :, :] ** 2) ** 0.5
        phase_real = mask[:, 0, :, :] / (mask_mags + EPS)
        phase_imag = mask[:, 1, :, :] / (mask_mags + EPS)
        mask_phase = torch.atan2(phase_imag, phase_real)
        mask_mags = torch.tanh(mask_mags)
        enh_mags = mask_mags * torch.sqrt(spec[:, :, :, 0] ** 2 + spec[:, :, :, 1] ** 2)
        enh_phase = torch.atan2(spec[:, :, :, 1], spec[:, :, :, 0]) + mask_phase
        spec_real = enh_mags * torch.cos(enh_phase)
        spec_imag = enh_mags * torch.sin(enh_phase)
        # [B, F, T]
        cspec = spec_real + 1j * spec_imag
        # [B, S]
        audio = torch.istft(
            cspec,
            self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            return_complex=False,
        )
        audio = torch.clamp(audio, min=-1.0, max=1.0)
        return audio

    def train_epoch(self, epoch):
        loss_total = 0.0
        for noisy, clean in tqdm(self.train_iter, desc="train"):
            # 確保張量使用 float32 而非 float64
            noisy = noisy.to(self.device, dtype=torch.float32)
            clean = clean.to(self.device, dtype=torch.float32)

            # [B, S] -> [B, F, T, 2]
            noisy_spec = self.audio_stft(noisy)

            self.optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                mask = self.model(noisy_spec)

            # [B, S]
            enh = self.audio_istft(mask, noisy_spec)

            loss = self.loss(enh, clean)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_total += loss.item()

        # # update learning rate
        avg_loss = loss_total / len(self.train_iter)
        self.update_scheduler(avg_loss)

        # logs
        self.writer.add_scalar("loss/train", avg_loss, epoch)
        self.writer.add_scalar("lr", self.optimizer.state_dict()["param_groups"][0]["lr"], epoch)
        
        # 在终端打印当前epoch的训练结果
        print(f"\n[Epoch {epoch}] 训练结果:")
        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  学习率: {self.optimizer.state_dict()['param_groups'][0]['lr']:.6f}")

    @torch.no_grad()
    def valid_epoch(self, epoch):
        noisy_list = []
        clean_list = []
        enh_list = []
        noisy_files = []

        loss_total = 0.0
        for noisy, clean, noisy_file in tqdm(self.valid_iter, desc="valid"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            # [B, S] -> [B, F, T, 2]
            noisy_spec = self.audio_stft(noisy)

            mask = self.model(noisy_spec)

            # [B, S]
            enh = self.audio_istft(mask, noisy_spec)

            # 确保三个张量长度一致
            min_len = min(noisy.shape[-1], clean.shape[-1], enh.shape[-1])
            noisy = noisy[..., :min_len]
            clean = clean[..., :min_len]
            enh = enh[..., :min_len]

            loss = self.loss(enh, clean)

            loss_total += loss.item()

            noisy = noisy.detach().squeeze(0).cpu().numpy()
            clean = clean.detach().squeeze(0).cpu().numpy()
            enh = enh.detach().squeeze(0).cpu().numpy()
            assert len(noisy) == len(clean) == len(enh)

            noisy_list = np.concatenate([noisy_list, noisy], axis=0) if len(noisy_list) else noisy
            clean_list = np.concatenate([clean_list, clean], axis=0) if len(clean_list) else clean
            enh_list = np.concatenate([enh_list, enh], axis=0) if len(enh_list) else enh
            noisy_files = np.concatenate([noisy_files, noisy_file], axis=0) if len(noisy_files) else noisy_file

        # visual audio
        for i in range(self.visual_samples):
            self.audio_visualization(noisy_list[i], clean_list[i], enh_list[i], os.path.basename(noisy_files[i]), epoch)

        # logs
        avg_loss = loss_total / len(self.valid_iter)
        self.writer.add_scalar("loss/valid", avg_loss, epoch)

        # visual metrics and get valid score
        metrics_score = self.metrics_visualization(
            enh_list, clean_list, epoch, n_folds=self.n_folds, n_jobs=self.n_jobs
        )

        # 在终端打印当前epoch的验证结果
        print(f"\n[Epoch {epoch}] 验证结果:")
        print(f"  平均损失: {avg_loss:.6f}")
        # 使用保存的当前指标值，而不是尝试从writer.scalar_dict中获取
        print(f"  STOI 指标: {self.current_stoi:.4f}")
        print(f"  WB_PESQ 指标: {self.current_wb_pesq:.4f}")
        print(f"  综合评分: {metrics_score:.4f}")
        
        if self.is_best_epoch(metrics_score):
            print(f"  【新的最佳模型】当前评分 {metrics_score:.4f} 优于之前最佳评分 {self.best_score - metrics_score + metrics_score:.4f}")
        
        return metrics_score

    def log_experiment_info(self):
        """记录实验信息到 experiments.log"""
        import datetime
        import json
        
        log_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "dataset_version": self.config["dataset"].get("version", "未指定"),
            "model_name": self.config["model"]["name"],
            "model_config": self.config["model"],
            "experiment_description": self.config.get("description", "未指定"),
        }
        
        log_path = os.path.join(self.base_path, "experiments.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
        print(f"实验信息已记录到 {log_path}")

    def __call__(self):
        # to device
        self.model.to(self.device)

        # init pre load model
        if self.pre_model_path:
            self.load_pre_model()

        # resume
        if self.resume:
            self.resume_checkpoint()

        # init logs
        self.init_logs()

        # loop train
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"{'=' * 20} {epoch} epoch start {'=' * 20}")

            self.set_model_to_train_mode()
            self.train_epoch(epoch)

            if self.save_checkpoint_interval != 0 and (epoch % self.save_checkpoint_interval == 0):
                self.save_checkpoint(epoch)

            # valid
            if epoch % self.valid_interval == 0 and epoch >= self.valid_start_epoch:
                print(f"Train has finished, Valid is in progress...")

                self.set_model_to_eval_mode()
                metric_score = self.valid_epoch(epoch)

                if self.is_best_epoch(metric_score):
                    self.save_checkpoint(epoch, is_best_epoch=True)

            print(f"{'=' * 20} {epoch} epoch end {'=' * 20}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="knowledge distillation trainer")
    parser.add_argument("-C", "--config", required=True, type=str, help="Config (*.toml).")
    args = parser.parse_args()

    # config device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # get config
    config = toml.load(args.config)

    # get dataset path
    dataset_path = os.path.join(os.getcwd(), "dataset_csv")

    # get dataloader args
    batch_size = config["dataloader"]["batch_size"]
    num_workers = 0 if device == "cpu" else config["dataloader"]["num_workers"]
    drop_last = config["dataloader"]["drop_last"]
    pin_memory = config["dataloader"]["pin_memory"]

    # get train_iter
    train_set = DNS_Dataset(dataset_path, config, mode="train")
    train_iter = DataLoader(
        train_set,
        batch_size=batch_size[0],
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    # get valid_iter
    valid_set = DNS_Dataset(dataset_path, config, mode="valid")
    valid_iter = DataLoader(
        valid_set,
        batch_size=batch_size[1],
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    # config model
    model = globals().get(config["model"]["name"])(
        n_fft=config["dataset"]["n_fft"],
        rnn_layers=config["model"]["rnn_layers"],
        rnn_units=config["model"]["rnn_units"],
        kernel_num=config["model"]["kernel_num"],
        kernel_size=config["model"]["kernel_size"],
    )

    # trainer
    trainer = BaseTrainer(config, model, train_iter, valid_iter, device)

    # train
    trainer()

    pass
