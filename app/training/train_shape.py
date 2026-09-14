"""
train_shape.py — 训练「摆形状」模型(ShapeModel) on traindata。

从人工谱面(traindata 里的 .ogg+.adofai 配对)学习每格左右转向：
  - 标签：angleData 逐格转向符号(shortest(angleData[i]-angleData[i-1]) 符号)
  - 特征：OnsetNet 概率 + Demucs 各 stem 能量 + BPM 节拍相位(见 shape_model.extract_tile_features)
训练/推理特征完全一致 -> 学到的"走向风格"可迁移到新歌的 OnsetNet 检测格上。

用法:
  venv/Scripts/python.exe app/training/train_shape.py \
      --data "<训练数据根: 每子文件夹一首 .adofai+音频>" \
      --epochs 80
  (--out 缺省写运行时 checkpoints/shape_model.pt, 一般无需指定)

注：本脚本会跑 OnsetNet + Demucs 处理全部训练歌，较耗时；按用户铁律，启动需用户明确授权。
"""
from __future__ import annotations
import os
import sys
import re
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 修正：此前 ROOT = dirname(dirname(__file__)) 实际算出 app/（注释却写 portable/），
# 导致 CKPT_DIR/FEAT_CACHE 指向 app/data/ 这个黑洞目录——GUI 启动训练不传 --out 时，
# 权重写进 app/data/checkpoints（没人读），而监控/推理都在看运行时目录或便携目录。
_APP = os.path.dirname(os.path.abspath(__file__))        # app/training/
ROOT = os.path.dirname(_APP)                              # app/
PORTABLE_ROOT = os.path.dirname(ROOT)                     # 便携/仓库根
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from timing_engine import compute_note_times
from chart_repr import HOP_MS, _to_angle_data
from onset_net import OnsetNet, predict_onset_frames
from demucs_mel import demucs_mel
from shape_model import ShapeModel, extract_tile_features
from paths import train_ckpt_dir, resolve_checkpoint, RUNTIME_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 输出/缓存统一走运行时目录（GUI 传 ADOFAI_DATA_DIR=RUNTIME_DIR；网页监控也数这里）。
CKPT_DIR = str(train_ckpt_dir())
FEAT_CACHE = str(RUNTIME_DIR / "shape_feat_cache")


def _load_json_tolerant(path):
    """容错加载 ADOFAI .adofai：兼容尾逗号/单引号/行注释等非标准 JSON。

    训练集很多谱面是第三方/编辑器导出的非标准 JSON（带尾逗号、单引号键等），
    标准 json.load 直接报错被跳过 -> 训练样本暴减。这里逐级容错兜底。
    """
    raw = open(path, encoding="utf-8-sig").read()
    # 1) 标准（strict=False 容忍 ADOFAI 编辑器导出的控制字符，救回非标准谱面）
    try:
        return json.loads(raw, strict=False)
    except Exception:
        pass
    # 2) 去尾逗号（JS / ADOFAI 编辑器常见）
    t = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(t)
    except Exception:
        pass
    # 3) 去 // 行注释
    t2 = re.sub(r"//[^\n]*", "", t)
    try:
        return json.loads(t2)
    except Exception:
        pass
    # 4) 单引号 -> 双引号（大多数字符串内无单引号冲突）
    t3 = t2.replace("'", '"')
    try:
        return json.loads(t3)
    except Exception:
        pass
    raise ValueError(f"无法解析 JSON: {path}")


def _find_pairs(data_dir):
    """递归扫描配对(2026-08-23 升级: 支持任意嵌套深度)。

    原版只扫 data_dir/子目录/(ogg+adofai) 一层 -> 三层结构(如 训练数据/iDQ/人声/歌名/)
    完全漏掉。现改用 os.walk, 凡某目录同时含音频(.ogg/.mp3/.wav) 与 .adofai 即记一对。
    """
    pairs = []
    if not os.path.isdir(data_dir):
        return pairs
    for root, dirs, files in os.walk(data_dir):
        low_files = {f.lower(): f for f in files}
        ogg = None
        for ext in (".ogg", ".mp3", ".wav"):
            for lf, f in low_files.items():
                if lf.endswith(ext):
                    ogg = os.path.join(root, f)
                    break
            if ogg:
                break
        ado = None
        for lf, f in low_files.items():
            if lf.endswith(".adofai"):
                ado = os.path.join(root, f)
                break
        if ogg and ado:
            pairs.append((ogg, ado))
    return pairs


def _load_labels_and_features(ogg, ado, onset_net):
    import json, hashlib
    # 特征缓存：demucs+onset 仅首跑计算，重训直接秒加载
    _fcp = None
    try:
        so = os.stat(ogg); sa = os.stat(ado)
        _fh = hashlib.sha1(
            f"{os.path.abspath(ogg)}|{so.st_size}|{so.st_mtime:.3f}|"
            f"{os.path.abspath(ado)}|{sa.st_size}|{sa.st_mtime:.3f}|shape13".encode()
        ).hexdigest()[:16]
        _fcp = os.path.join(FEAT_CACHE, _fh + ".npz")
        if _fcp and os.path.exists(_fcp):
            _d = np.load(_fcp)
            if _d["feats"].shape[1] == 12:   # 仅接受新 12 维特征缓存, 旧 8 维忽略重算
                return (_d["feats"].astype(np.float32), _d["labels"].astype(np.float32),
                        float(_d["bpm"]))
    except Exception:
        pass
    lvl = _load_json_tolerant(ado)
    ad = _to_angle_data(lvl)
    settings = lvl.get("settings") or {}
    actions = lvl.get("actions") or []
    bpm = float(settings.get("bpm", 120.0))
    if not ad:
        return None
    try:
        nt = compute_note_times(ad, settings, actions, add_offset=True)
    except Exception:
        return None
    if not nt or len(nt) < 4:
        return None
    times = [float(x[0]) if isinstance(x, (tuple, list)) else float(x) for x in nt]
    n = len(ad)
    # 逐格转向标签
    labels = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        d = ((ad[i] - ad[i - 1] + 180.0) % 360.0) - 180.0
        labels[i] = 1.0 if d >= 0 else -1.0
    labels[0] = labels[1] if n > 1 else 0.0
    tile_frames = [max(0, int(round(t / HOP_MS))) for t in times]
    # 几何上下文用真实 angleData 作为 angle_hint（与推理期 magnitudes 模拟对应）
    angle_hint = [float(x) % 360.0 for x in ad]
    # 由相邻 time 差估算每格拍数 magnitude（与推理期 magnitudes 同源：间隔->拍数）
    magnitudes = []
    for i in range(n - 1):
        seg_sec = max(0.0, times[i + 1] - times[i])
        beats = seg_sec * float(bpm) / 60.0
        magnitudes.append(beats)
    if len(magnitudes) < n:
        magnitudes.append(1.0)  # 末格补 1 拍占位，长度对齐
    # 特征：Demucs 分离 GPU 提速（崩了降级单通道）；
    # OnsetNet GPU 分块推理（predict_onset_frames 内 CHUNK 切分，超长歌不爆显存/不非法访问，
    # 8/17+8/23 踩坑根因=整首长歌塞 GPU，现已根治）。
    mel = demucs_mel(ogg, device="cuda")            # (6,128,T) GPU 分离
    if mel is None:
        return None
    T = mel.shape[2]
    onset_prob = np.zeros(T, dtype=np.float32)
    try:
        _, prob = predict_onset_frames(onset_net, mel, device="cuda")
        if prob is not None and len(prob) == T:
            onset_prob = prob.astype(np.float32)
        else:
            raise RuntimeError("OnsetNet 输出长度不匹配")
    except Exception as e:
        print(f"  [warn] OnsetNet 失败 {os.path.basename(ogg)} -> 回退 CPU: {e}", flush=True)
        try:
            onset_net = onset_net.to("cpu")
            _, prob = predict_onset_frames(onset_net, mel, device="cpu")
            if prob is not None and len(prob) == T:
                onset_prob = prob.astype(np.float32)
        except Exception as e2:
            print(f"  [warn] OnsetNet CPU 也失败 {os.path.basename(ogg)}: {e2}", flush=True)
    stem_energy = np.mean(np.abs(mel), axis=1).astype(np.float32)  # (6,T)
    # 2026-08-23 升级: 传入 angle_hint + magnitudes, 让几何上下文通道(8~11)有真值
    feats = extract_tile_features(tile_frames, onset_prob, stem_energy, bpm, HOP_MS,
                                  magnitudes=magnitudes, angle_hint=angle_hint)
    if feats.shape[0] != n:
        # 帧取整导致 1 格偏差，截断对齐
        m = min(feats.shape[0], n)
        feats = feats[:m]; labels = labels[:m]
    if feats.shape[0] < 4:
        return None
    feats = feats.astype(np.float32); labels = labels.astype(np.float32)
    if _fcp:
        try:
            os.makedirs(FEAT_CACHE, exist_ok=True)
            np.savez(_fcp, feats=feats, labels=labels, bpm=np.float32(bpm))
        except Exception:
            pass
    return feats, labels, bpm


class ShapeDataset(Dataset):
    def __init__(self, pairs, onset_net, limit=None):
        self.items = []
        total = len(pairs) if not limit else min(limit, len(pairs))
        for i, (ogg, ado) in enumerate(pairs):
            if limit and i >= limit:
                break
            print(f"[feat] {i+1}/{total}  {os.path.basename(ado)}", flush=True)
            try:
                r = _load_labels_and_features(ogg, ado, onset_net)
            except Exception as e:
                print(f"  [skip] {os.path.basename(ado)}: {e}", flush=True)
                continue
            if r is None:
                continue
            feats, labels, bpm = r
            self.items.append((feats, labels))
            print(f"  + done tiles={len(labels)} (累计样本 {len(self.items)})", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _collate(batch):
    feats, labels = zip(*batch)
    L = max(f.shape[0] for f in feats)
    F = feats[0].shape[1]
    x = np.zeros((len(batch), L, F), dtype=np.float32)
    y = np.zeros((len(batch), L), dtype=np.float32)
    mask = np.zeros((len(batch), L), dtype=np.float32)
    for i, (f, l) in enumerate(zip(feats, labels)):
        x[i, :f.shape[0]] = f
        y[i, :l.shape[0]] = l
        mask[i, :l.shape[0]] = 1.0
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


def main():
    ap = argparse.ArgumentParser()
    # 2026-08-23 升级: --data 可多次传入(多个训练根), 合并扫描。
    # 例: --data "<路径1>" --data "<路径2>"
    ap.add_argument("--data", action="append", default=None,
                    help="训练数据根目录(可多次); 缺省用默认 portable/train")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(CKPT_DIR, "shape_model.pt"))
    ap.add_argument("--limit", type=int, default=0, help="仅前 N 首(冒烟测试用)")
    args = ap.parse_args()

    # 默认根：便携根/train（此前写死开发机遗留相对路径 ../new_last/portable/train）
    data_roots = args.data if args.data else [os.path.join(PORTABLE_ROOT, "train")]
    print(f"[train_shape] device={DEVICE} data_roots={data_roots}")
    pairs = []
    for dr in data_roots:
        ps = _find_pairs(dr)
        print(f"[train_shape]   {dr} -> {len(ps)} 对")
        pairs.extend(ps)
    print(f"[train_shape] 合计 {len(pairs)} 对训练数据")
    if not pairs:
        print("无训练数据，退出"); return

    print("[train_shape] 加载 OnsetNet 权重...")
    onset_net = OnsetNet(in_channels=6)
    # 统一解析：运行时目录(用户训练)优先，回退便携内置出厂权重
    op = resolve_checkpoint("onset_net.pt")
    if op is not None:
        onset_net.load_state_dict(torch.load(str(op), map_location="cpu"))
        print(f"[train_shape] OnsetNet <- {op}")
    else:
        print("[train_shape] 警告: 未找到 onset_net.pt（运行时/内置均无），使用随机初始化")
    # 特征提取用 CPU：OnsetNet 在 GPU 上对个别长歌会触发 illegal memory access，
    # 一旦出错会毒死整个 CUDA 上下文，导致后续训练全崩。CPU 稳定不崩。
    onset_net = onset_net.to(DEVICE).eval()
    print(f"[train_shape] OnsetNet 已加载(device=cpu，特征提取稳定)")

    ds = ShapeDataset(pairs, onset_net, limit=args.limit or None)
    if len(ds) == 0:
        print("[train_shape] 无可用样本，退出"); return
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=_collate)

    # 训练模型放 GPU（特征提取已全在 CPU，上下文干净）；GPU 不可用则退 CPU
    try:
        model = ShapeModel().to(DEVICE)
        train_device = DEVICE
    except Exception as e:
        print(f"[train_shape] GPU 建模型失败，退 CPU: {e}")
        model = ShapeModel().to("cpu")
        train_device = "cpu"
    print(f"[train_shape] ShapeModel 训练设备: {train_device}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss(reduction="none")

    print(f"[train_shape] 开始训练 {args.epochs} 轮, 样本={len(ds)}")
    for ep in range(args.epochs):
        model.train()
        tot = 0.0; cnt = 0
        for x, y, mask in dl:
            x, y, mask = x.to(train_device), y.to(train_device), mask.to(train_device)
            logit = model(x).reshape(y.shape)
            tgt = (y + 1.0) / 2.0  # +1/-1 -> 1/0
            loss = (crit(logit, tgt) * mask).sum() / mask.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * mask.sum().item(); cnt += mask.sum().item()
        print(f"[train_shape] ep {ep+1}/{args.epochs}  loss={tot/max(cnt,1):.4f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"[train_shape] 已保存 -> {args.out}")


if __name__ == "__main__":
    main()
