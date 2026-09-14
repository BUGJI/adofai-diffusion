"""train_onset.py — 训练踩点神经网络 OnsetNet

监督信号：train/ 里每首歌的 adofai 经 chart_repr.adofai_to_dense 产出的
C0 onset 热图（= 用户谱面里每个 tile 的真实时间点，落在 mel 帧网格上）。
模型学"这首歌哪里该踩点"，替代旧的信号检测器。

用法（在 venv 里，已含 CUDA torch）：
  venv/Scripts/python.exe app/training/train_onset.py
可加 --epochs 80 --chunk 512 --stride 256
"""
from __future__ import annotations
import os, sys, glob, json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as td
import threading
import queue
import random
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "app" / "training"))
os.environ.setdefault("MKL_THREADING_LAYER", "sequential")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # Demucs 权重已缓存, 离线加载跳过网络重试
torch.set_num_threads(1)

import librosa
from adofai_parse import load_adofai
from chart_repr import adofai_to_dense, SR, HOP
from onset_net import OnsetNet, normalize_mel
from demucs_mel import demucs_mel, HOP_MS_ONSET, HOP_ONSET, STEMS, release_sep, _cache_path
from beat_this_align import estimate_beats, build_beat_phase, BEAT_COND_CH
from device_util import get_safe_device
from paths import train_ckpt_dir

N_MELS = 128
# 老显卡(如 GTX 10 系) torch.cuda.is_available() 会假阳性，必须用一次真实内核探测
DEVICE = get_safe_device()


def _mel(y):
    # 仅作兜底/调试用（单通道）。正式训练走 demucs_mel（多通道 + hop128）。
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=HOP_ONSET, n_mels=N_MELS)
    return np.log1p(mel).astype(np.float32)      # (128, T) @ hop128


def _pairs(train_dir):
    # 用 os.listdir 而非 glob：方括号目录名安全，且认 wav/ogg/mp3（优先级 ogg>mp3>wav）
    # train_dir 支持【多目录】(list/多路径, 以 os.pathsep 分隔的字符串也可)：
    # 默认模型用 melody+vocal 双目录全量训练(见 2026-12 用户指令)；单目录行为不变。
    if isinstance(train_dir, (list, tuple)):
        dirs = [str(d) for d in train_dir]
    else:
        dirs = [d for d in str(train_dir).split(os.pathsep) if d]
    pairs = []
    seen = set()
    for td_ in dirs:
        if not os.path.isdir(td_):
            print(f"[pairs] 跳过不存在的目录: {td_}")
            continue
        for name in sorted(os.listdir(td_)):
            d = os.path.join(td_, name)
            if not os.path.isdir(d):
                continue
            # 同名歌目录在多目录间去重（按音频文件真名，避免双份标签）
            key = os.path.basename(os.path.normpath(d))
            if key in seen:
                continue
            og = sorted([os.path.join(d, f) for f in os.listdir(d)
                         if f.lower().endswith(('.ogg', '.mp3', '.wav'))])
            ad = sorted([os.path.join(d, f) for f in os.listdir(d)
                         if f.lower().endswith('.adofai')])
            if og and ad:
                seen.add(key)
                pairs.append((og[0], ad[0]))
    print(f"[pairs] 配对 {len(pairs)} 首 (来自 {len(dirs)} 个目录)")
    return pairs


def build_samples(train_dir, chunk=512, stride=256, beat_grid=False):
    """返回 (refs, targets, pos_weight)。

    beat_grid=True 时，每个样本 mel 为 9 通道 = 6 Demucs + 3 节拍相位条件
    (sin2πφ, cos2πφ, downbeat_gate)，与 onset 真值同 hop128 网格逐帧对齐。

    refs: 每个切块一条 (mel_cache_path, bg_cache_path_or_None, s, cl)
          训练时由 OnsetDataset 按需从 .npy 缓存 mmap 流式读取，
          不再把整集 mel 预加载进内存 —— 彻底避免 131 首全量预加载引发的 OOM。
    targets（onset 切片，每个仅 1×chunk，内存可忽略）仍常驻 RAM。
    """
    refs, targets = [], []
    np_sum = 0.0
    np_count = 0
    pairs = _pairs(train_dir)
    total = len(pairs)
    for idx, (ogg, adf) in enumerate(pairs):
        try:
            lvl = load_adofai(adf)
            if not isinstance(lvl, dict):
                print(f"[demucs] 跳过 {idx+1}/{total} (非合法谱面): {os.path.basename(ogg)}")
                continue
            # 6 通道 Demucs 分离 mel @ hop128（首跑分离并缓存，之后走缓存秒读）
            _cp = _cache_path(ogg)
            if _cp and os.path.exists(_cp):
                print(f"[demucs] 缓存命中 {idx+1}/{total}: {os.path.basename(ogg)}", flush=True)
            else:
                print(f"[demucs] 分离 {idx+1}/{total}: {os.path.basename(ogg)}", flush=True)
            mel = demucs_mel(ogg, device=DEVICE)   # (C=6, 128, T) @ hop128
            T = mel.shape[2]                        # 时间轴在最后一维
            # 真值 onset 热图也在 hop128 网格上，与 mel 逐帧对齐
            dense = adofai_to_dense(lvl, T, hop_ms=HOP_MS_ONSET)  # (3, T) @ hop128
            if dense.sum() == 0:
                continue
            onset = dense[0:1]                 # (1, T) 真值热图
            # —— Phase B：Beat This! 节拍相位条件（3 通道）与 mel 同网格，逐块拼接 ——
            bg_path = None
            if beat_grid:
                try:
                    _b, _db = estimate_beats(ogg, device="cpu")
                    bg = build_beat_phase(_b, _db, T)   # (3,128,T) @ hop128
                    bg_path = _cp + ".bg.npy"
                    np.save(bg_path, bg.astype(np.float32))
                except Exception as e:
                    print(f"[beat_this] 条件构建失败({e})，该曲退化纯 mel")
                    bg_path = None
            for s in range(0, max(1, T - chunk) + 1, stride):
                if s + chunk <= T:
                    # 只存引用 + 切片位置，mel 数据不进内存
                    refs.append((_cp, bg_path, s, chunk))
                    targets.append(onset[:, s:s + chunk])  # (1,chunk) 时间轴在最后一维
                    np_sum += float(onset[:, s:s + chunk].sum())
                    np_count += chunk
        except Exception as e:
            print(f"[warn] skip {os.path.basename(ogg)}: {e}")
            continue
    release_sep()   # 分离完成, 立即释放 Demucs 模型(占 ~1.2G 显存), 避免后续训练 OOM
    if np_count:
        frac_pos = np_sum / np_count
        pos_weight = min(100.0, max(1.0, (1.0 - frac_pos) / max(frac_pos, 1e-4)))
        print(f"[data] 切块数={len(refs)}  正类占比≈{frac_pos:.4f}  pos_weight={pos_weight:.1f}")
    else:
        pos_weight = 1.0
        print(f"[data] 切块数={len(refs)}  (无正类样本, pos_weight=1.0)")
    return refs, targets, pos_weight


class OnsetDataset(td.Dataset):
    def __init__(self, refs, targets):
        self.refs = refs
        self.targets = targets

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, i):
        mel_path, bg_path, s, cl = self.refs[i]
        # mmap 按需读取切片，整集 mel 不占内存（根治 OOM）
        mel = np.load(mel_path, mmap_mode="r")          # (6,128,T)
        mel_c = np.asarray(mel[:, :, s:s + cl])         # (6,128,cl) 物化该切片
        if bg_path is not None and os.path.exists(bg_path):
            bg = np.load(bg_path, mmap_mode="r")        # (3,128,T)
            bg_c = np.asarray(bg[:, :, s:s + cl])
            mel_c = np.concatenate([mel_c, bg_c], axis=0)  # (9,128,cl)
        tgt = self.targets[i]
        return normalize_mel(torch.from_numpy(mel_c.astype("float32"))), \
               torch.from_numpy(tgt.astype("float32"))


class PrefetchLoader:
    """单进程 + 线程池并行读 .npy 的预取 DataLoader 替代:

    - 用线程池并行从磁盘加载样本(并行 I/O, 吃满多核), 主进程用单一 CUDA 上下文训练
      -> GPU 不等料(利用率拉满), 且显存低/页面文件不炸(不 Spawn worker 进程,
      num_workers>0 会让每个 worker 各开一份 CUDA 上下文, 撑爆显存+页面文件 WinError 1455)。
    - 每个 epoch 重建一次以重新 shuffle。
    """

    def __init__(self, dataset, batch_size, prefetch=3, n_io_threads=16):
        self.dataset = dataset
        self.bs = batch_size
        self.n = len(dataset)
        self.nb = max(1, self.n // batch_size)
        self.q = queue.Queue(maxsize=prefetch)
        self.executor = ThreadPoolExecutor(max_workers=n_io_threads)
        self._order = list(range(self.n))
        random.shuffle(self._order)
        self._pos = 0
        self._produced = 0
        self._stop = False
        self._producer = threading.Thread(target=self._produce, daemon=True)
        self._producer.start()

    def _produce(self):
        while not self._stop and self._produced < self.nb:
            idxs = self._order[self._pos:self._pos + self.bs]
            self._pos += len(idxs)
            futures = [self.executor.submit(self.dataset.__getitem__, i) for i in idxs]
            try:
                batch = [f.result() for f in futures]
            except Exception as e:
                print(f"[prefetch] 批次加载失败: {e}", flush=True)
                self._stop = True
                break
            mels = torch.stack([b[0] for b in batch])
            tgts = torch.stack([b[1] for b in batch])
            self.q.put((mels, tgts))
            self._produced += 1
        self.q.put(None)  # 哨兵 -> 触发 StopIteration

    def __iter__(self):
        return self

    def __next__(self):
        item = self.q.get()
        if item is None:
            raise StopIteration
        return item

    def __len__(self):
        return self.nb

    def close(self):
        self._stop = True
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir",
                    default=os.environ.get("ADOFAI_TRAIN_DIR",
                        str(ROOT / "train_single" / "melody")),
                    help="训练数据目录（每首一个子文件夹: 音频+谱面）。"
                         "支持多目录: 重复传或用分号/冒号(os.pathsep)分隔 —— "
                         "默认模型喂 melody+vocal 双目录全量")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--out", default=str(Path(train_ckpt_dir()) / "onset_net.pt"))
    ap.add_argument("--beat_grid", action="store_true",
                    help="把 Beat This! 节拍相位条件(3通道)拼入 OnsetNet，in_channels=9（需重训）")
    ap.add_argument("--resume", default=None,
                    help="微调: 从已有 onset_net.pt 加载权重继续训练(保留其他歌能力); "
                         "不指定则从头训(会覆盖其他歌知识, 慎用)")
    ap.add_argument("--init", default=None,
                    help="warm-start: 从已有权重初始化(同源模型, 如 melody->vocal 迁移); "
                         "与 --resume 区别: 仅用于初始化, 不打印'继续微调'文案。建议配合余弦衰减使用")
    ap.add_argument("--min_lr", type=float, default=1e-5,
                    help="余弦衰减的最小学习率(默认 1e-5)")
    a = ap.parse_args()

    in_ch = len(STEMS) + (BEAT_COND_CH if a.beat_grid else 0)
    out = a.out
    if a.beat_grid:
        out = str(Path(train_ckpt_dir()) / "onset_net_beatgrid.pt")
    # best 权重跟随 --out 同目录、同基础名（此前写死便携 data/checkpoints/best_onset.pt：
    # Program Files 安装下无写权限静默失败，且并发训 melody/vocal 会互相覆盖）。
    best_path = str(Path(out).with_name(Path(out).stem + ".best.pt"))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    print(f"[cfg] device={DEVICE}  train_dir={a.train_dir}  beat_grid={a.beat_grid}")
    mels, targets, pos_weight = build_samples(a.train_dir, a.chunk, a.stride, beat_grid=a.beat_grid)
    if not mels:
        print("[err] 没有可用训练样本"); return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # 再清一次, 确保 Demucs 显存彻底释放
    ds = OnsetDataset(mels, targets)
    # 单进程 + 线程池并行读 .npy(多线程预取): 并行 I/O 吃满多核, 主进程单一 CUDA 上下文
    # -> 显存低/页面文件不炸, 且 GPU 不等料(利用率拉满)。不 Spawn worker 进程(num_workers>0
    # 会让每个 worker 各开 CUDA 上下文, 撑爆显存+页面文件 WinError 1455)。
    _io_threads = max(1, (os.cpu_count() or 8) - 4)   # 留 4 核给主进程/系统, 其余全用于并行 I/O
    print(f"[loader] 多线程预取: io_threads={_io_threads} batch={a.batch} prefetch=3 (单CUDA上下文, 不Spawn进程)")

    model = OnsetNet(n_mels=N_MELS, in_channels=in_ch).to(DEVICE)
    if a.resume and os.path.exists(a.resume):
        try:
            model.load_state_dict(torch.load(a.resume,  map_location=DEVICE))
            print(f"[resume] 已从 {a.resume} 加载权重, 继续微调(保留其他歌能力)")
        except Exception as e:
            print(f"[warn] --resume 加载失败({e}), 改为从头训")
    elif a.resume:
        print(f"[warn] --resume 指定但文件不存在: {a.resume}, 将从头训")
    elif a.init and os.path.exists(a.init):
        # warm-start: 仅用于初始化(同源模型迁移, 如 melody->vocal), 不保留微调语义
        try:
            model.load_state_dict(torch.load(a.init, map_location=DEVICE))
            print(f"[init] 已从 {a.init} 初始化权重(warm-start, 从头计 epoch)")
        except Exception as e:
            print(f"[warn] --init 加载失败({e}), 改为从头训")
    elif a.init:
        print(f"[warn] --init 指定但文件不存在: {a.init}, 将从头训")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] OnsetNet 参数={n_params:,}  in_channels={in_ch} "
          f"({('含节拍网格 beatgrid' if a.beat_grid else 'Demucs 6通道')})")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    # 余弦退火: 从 a.lr 平滑降到 a.min_lr, 后期不再因固定 lr 在极小点附近跑飞
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=a.epochs, eta_min=a.min_lr)

    import time as _time
    _t0 = _time.time()
    best_loss = float("inf")
    best_epoch = -1
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        dl = PrefetchLoader(ds, batch_size=a.batch, prefetch=3, n_io_threads=_io_threads)
        nb = len(dl)
        for bi, (mel, tgt) in enumerate(dl, 1):
            mel, tgt = mel.to(DEVICE), tgt.to(DEVICE)
            opt.zero_grad()
            logits = model(mel)                  # (B, chunk)
            loss = criterion(logits, tgt.squeeze(1))
            loss.backward()
            opt.step()
            tot += float(loss.item()) * mel.size(0)
            # 每 batch 实时进度(强制 flush, 避免被缓冲吞掉)
            print(f"  [epoch {ep:03d}/{a.epochs}] batch {bi}/{nb} loss={loss.item():.4f}", flush=True)
        dl.close()
        avg = tot / len(ds)
        _el = _time.time() - _t0
        cur_lr = scheduler.get_last_lr()[0]
        print(f"[epoch {ep:03d}/{a.epochs}] loss={avg:.4f}  lr={cur_lr:.2e}  已用 {_el/60:.1f}min", flush=True)
        scheduler.step()  # 每个 epoch 衰减一次

        # 仅按 train loss 最低保留 best 权重; 不覆盖最终输出, 二者独立
        if avg < best_loss:
            best_loss = avg
            best_epoch = ep
            torch.save(model.state_dict(), best_path)
            print(f"  -> [best] loss={best_loss:.4f} 已保存 {best_path}", flush=True)
        # 原策略: 每 20 epoch 及最后也落一份到 a.out(便于续训), 但 best 才是真优
        if ep % 20 == 0 or ep == a.epochs:
            torch.save(model.state_dict(), a.out)
            print(f"  -> 已保存(末轮/周期) {a.out}", flush=True)

    print(f"[done] 训练完成 | best_loss={best_loss:.4f} @ epoch {best_epoch} | 权重: "
          f"{a.out} (末轮) / best_onset.pt (最优)", flush=True)


if __name__ == "__main__":
    main()
