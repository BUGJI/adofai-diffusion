"""DenseChartDataset — 从 train/ 的 (ogg, adofai) 对构建训练样本。

每对:
  - 音频 -> 梅尔谱 (128, T)  [log1p]
  - adofai -> 稠密谱面 (3, T): onset热图 / dir / twirl热图 (chart_repr.adofai_to_dense)
  - onset 包络 = dense[0] (1, T), 作为扩散的局部条件
  - bpm (标量, 外部条件, 不进张量)
切成定长 CHUNK 帧 (需被 16 整除), 步长 stride 做数据增广。
"""
from __future__ import annotations
import os, glob, json
import numpy as np
import torch
import torch.utils.data as td
import librosa

from adofai_parse import load_adofai
from chart_repr import adofai_to_dense, SR, HOP

# 128 全链路：HOP=128 ≈5.805ms/帧，故 4096 帧≈23.8s（与原 512 网格 1024 帧同窗口）。
# 均须被 16 整除（VAE 4 级 stride-2 下采样 -> 潜空间 T/16）。
CHUNK = 4096
STRIDE = 3072         # 块间滑动步长(重叠增广)
N_MELS = 128


def _mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=HOP, n_mels=N_MELS)
    return np.log1p(mel).astype(np.float32)      # (128, T)


def _pairs(train_dir):
    # 支持【多目录】(list, 或 os.pathsep 分隔的字符串) —— 默认综合模型喂
    # melody+vocal 双目录全量(2026-12 用户指令); 单目录行为不变。
    if isinstance(train_dir, (list, tuple)):
        dirs = [str(d) for d in train_dir]
    else:
        dirs = [d for d in str(train_dir).split(os.pathsep) if d]
    pairs = []
    for train_dir_ in dirs:
        if not os.path.isdir(train_dir_):
            print(f"[dataset] 跳过不存在的目录: {train_dir_}")
            continue
        for name in sorted(os.listdir(train_dir_)):
            d = os.path.join(train_dir_, name)
            if not os.path.isdir(d):
                continue
            auds = [f for f in os.listdir(d)
                    if f.lower().endswith(('.ogg', '.mp3', '.wav'))]
            ads = [f for f in os.listdir(d) if f.lower().endswith('.adofai')]
            if not auds or not ads:
                continue
            # 优先级 ogg > mp3 > wav（与原 ogg/mp3 优先行为一致，且方括号安全）
            def _pri(f):
                return {'ogg': 0, 'mp3': 1, 'wav': 2}.get(f.lower().rsplit('.', 1)[-1], 3)
            auds.sort(key=_pri)
            pairs.append((os.path.join(d, auds[0]), os.path.join(d, sorted(ads)[0])))
    print(f"[dataset] 配对 {len(pairs)} 首 (来自 {len(dirs)} 个目录)")
    return pairs


class DenseChartDataset(td.Dataset):
    def __init__(self, train_dir, chunk=CHUNK, stride=STRIDE):
        self.chunk = chunk
        self.stride = stride
        self.items = []          # (mel_path_or_y, dense, bpm) 预存稠密+音频懒加载
        self.samples = []        # (pair_idx, start)
        for pi, (ogg, adf) in enumerate(_pairs(train_dir)):
            try:
                lvl = load_adofai(adf)
                if not isinstance(lvl, dict):
                    continue
                bpm = float((lvl.get("settings") or {}).get("bpm", 120.0) or 120.0)
                # 预先算 mel 与 dense 并缓存到内存(105~250 首, 可控)
                y, _ = librosa.load(ogg, sr=SR, mono=True)
                mel = _mel(y)
                T = mel.shape[1]
                dense = adofai_to_dense(lvl, T)
                if dense.sum() == 0:
                    continue
                if mel.shape[1] != dense.shape[1]:
                    T = min(mel.shape[1], dense.shape[1])
                    mel = mel[:, :T]; dense = dense[:, :T]
                self.items.append((mel, dense, bpm))
                for s in range(0, max(1, T - chunk) + 1, stride):
                    if s + chunk <= T:
                        self.samples.append((len(self.items) - 1, s))
            except Exception as e:
                # 多目录全量训练: 单首解码/解析失败只跳过该首, 不让整集报废
                print(f"[dataset] 跳过失败样本: {os.path.basename(ogg)} ({e})")
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pi, s = self.samples[idx]
        mel, dense, bpm = self.items[pi]
        mel_c = mel[:, s:s + self.chunk]
        dense_c = dense[:, s:s + self.chunk]
        onset_env = dense_c[0:1]                 # (1, chunk)
        return (torch.from_numpy(mel_c), torch.from_numpy(dense_c),
                torch.tensor(bpm, dtype=torch.float32),
                torch.from_numpy(onset_env))
