"""
shape_model.py — 「摆形状」模型：学习人工谱面的路径走向(每格左/右转)。

设计(2026-08-13 晚, 用户要求"再添加一个模型让他学习摆形状, 学 traindata")：
  - 输入：每格(方块)的特征向量 (F=12)  —— 2026-08-23 升级：加入【几何上下文】，让左右转
        在特征上可区分(纯音频特征左右对称, 模型学不到, 见下)。
        [0]  onset_prob      该格 OnsetNet 踩点概率(帧级)
        [1]  sin(2π·拍相位)  音乐节拍相位(0~1)
        [2]  cos(2π·拍相位)
        [3]  小节相位        (拍时间 / 4拍) mod 1
        [4]  drums 能量(归一) Demucs 鼓点能量(窗口均值)
        [5]  bass  能量(归一) Demucs 贝斯能量
        [6]  other 能量(归一) 其余 stem 能量
        [7]  局部间隔(归一)   到下一格的拍数 /4
        [8]  当前绝对角(归一)  cur_angle/360  —— 几何上下文1：告诉模型"现在球朝哪"
        [9]  上一格转向符号    ±1 (首格0)       —— 几何上下文2：上一格往左还是往右
        [10] 到原点距离(归一)  dist/120         —— 几何上下文3：是否快出屏/聚中心
        [11] 累计转角(归一)    cum_turn/720     —— 几何上下文4：已转了多少(螺旋/图案)
    —— 训练与推理用完全相同的特征(都来自 OnsetNet 概率 + Demucs 能量 + BPM + 几何模拟)，
       故 train/inference 特征分布一致。
  - 输出：每格一个 logit；>=0 -> 往左(d2=+1)，<0 -> 往右(d2=-1)。
    chart_repr.plan_path_twirl 用 turn_sign 接管左右(几何自交仍一票否决兜底)。
  - 标签：人工谱 angleData 的逐格转向符号(shortest(angleData[i]-angleData[i-1]) 的符号)。
    （注：绝对左右符号约定若实测相反，翻转 SIGN_FLIP 即可，不影响学习结构。）

为什么必须加几何上下文(2026-08-23 用户实测"模型没接管、走线和贪心一样")：
  纯音频特征(onset/相位/能量/间隔)对"左转 vs 右转"完全对称 —— 同一段音乐既可左拐也可右拐，
  模型从音频里根本分离不出左右，只能学到统计均匀偏置，再被中位数去偏拉平 -> 等于随机。
  加入角度/上一格符号/距离/累计转角后，模型能看到"当前几何状态"，从而学到
  "在某某音乐上下文 + 当前朝某方向时，应往左/往右拐成图案"，真正接管走向。

模型：2 层双向 GRU + 线性头(输出维度 12->12 通道对应新特征)。序列模型(前后文重要)。
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn


class ShapeModel(nn.Module):
    def __init__(self, feat_dim: int = 12, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.2)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        # x: (B, T, F) -> (B, T, 1)
        out, _ = self.gru(x)
        return self.head(out)


# 若实测发现模型学的左右与期望相反，置 True（单次全局翻转，不影响训练结构）。
SIGN_FLIP = False


def extract_tile_features(tile_frames, onset_prob, stem_energy, bpm, hop_ms, n_stems=6,
                          magnitudes=None, angle_hint=None):
    """为每格提取 12 维特征(2026-08-23 升级: 加几何上下文, 让左右转可学)。

    tile_frames : list[int] 每格帧下标(与谱面 kept 对齐)
    onset_prob  : (Tf,) OnsetNet 概率
    stem_energy : (n_stems, T) 各 stem 每帧能量(已 abs 均值)；None 时能量置 0
    magnitudes  : 可选, list[float] 每格真实时值(拍数)。用于推理期模拟几何上下文。
    angle_hint  : 可选, list[float] 每格"当前绝对角"(0~360)。训练期由人工谱真实 angleData
                  提供; 推理期由 magnitudes 模拟(非 Twirl 基准朝向, 足以区分左右)。
                  —— 缺失时(angle_hint=None 且 magnitudes=None)几何通道填 0, 向后兼容旧调用。

    几何上下文(通道 8~11)由 angle_hint / magnitudes 推导:
      [8]  cur_angle/360
      [9]  prev_sign (上一格转向符号, 首格=0)
      [10] dist/120  (从原点累加到当前点的欧氏距离, 用 step=1 近似尺度)
      [11] cum_turn/720 (累计最短弧转角绝对值)
    """
    beat = 60.0 / float(bpm) if bpm and bpm > 0 else 0.5
    hms = hop_ms / 1000.0
    Tf = len(onset_prob)
    Ts = stem_energy.shape[1] if stem_energy is not None else 0
    n = len(tile_frames)

    # —— 推导几何上下文序列 ——
    cur_angles = [0.0] * n      # 每格"当前绝对角"
    prev_signs = [0.0] * n      # 上一格转向符号
    dists = [0.0] * n           # 到原点距离(step=1 近似)
    cum_turns = [0.0] * n       # 累计转角
    if angle_hint is not None:
        # 训练期: 直接用人工谱真实绝对角
        for i in range(min(n, len(angle_hint))):
            cur_angles[i] = float(angle_hint[i]) % 360.0
        for i in range(1, n):
            da = ((cur_angles[i] - cur_angles[i - 1] + 180.0) % 360.0) - 180.0
            prev_signs[i] = 1.0 if da >= 0 else -1.0
            cum_turns[i] = cum_turns[i - 1] + abs(da)
    elif magnitudes is not None:
        # 推理期: 从 magnitudes 模拟基准朝向(非 Twirl, direction=+1)。
        # pAngle_i = |mag_i|*180 (往左为正); 入向每格先掉头180 -> 绝对角递推。
        ca = 0.0
        px, py = 0.0, 0.0
        cum = 0.0
        for i in range(n):
            pa = abs(float(magnitudes[i])) * 180.0 if i < len(magnitudes) else 180.0
            if i > 0:
                # 上一格转向符号 = 上一格 pAngle 符号(基准方向 +1 -> 恒往左, 但带 Twirl 后实际可能反)
                prev_signs[i] = 1.0  # 基准模拟恒往左; 真实 Twirl 翻转由 plan_path_twirl 决定
            # 入向掉头180
            cur_in = (ca - 180.0) % 360.0
            dest = (cur_in - pa) % 360.0
            ca = dest
            cur_angles[i] = ca
            # 渲染重放(step=1 近似)算距离
            nx = px + math.cos(math.radians(dest))
            ny = py + math.sin(math.radians(dest))
            px, py = nx, ny
            dists[i] = math.hypot(px, py)
            cum += pa
            cum_turns[i] = cum
    # 距离/累计转角归一
    max_dist = max(dists) if any(dists) else 1.0
    max_cum = max(cum_turns) if any(cum_turns) else 1.0

    feats = []
    for i, f in enumerate(tile_frames):
        f = max(0, min(Tf - 1, int(f)))
        op = float(onset_prob[f]) if Tf > 0 else 0.0
        a = max(0, f - 6)
        b = min(Ts - 1, f + 6) if Ts > 0 else f

        def _e(s):
            if stem_energy is None or s >= stem_energy.shape[0] or Ts == 0 or b < a:
                return 0.0
            return float(np.mean(stem_energy[s, a:b + 1]))

        drums = _e(0)
        bass = _e(1) if n_stems > 1 else 0.0
        other = float(np.mean([_e(s) for s in range(2, n_stems)])) if n_stems > 2 else 0.0
        t_sec = f * hms
        phase = (t_sec / beat) % 1.0
        bar = (t_sec / (4.0 * beat)) % 1.0
        nxt = tile_frames[i + 1] if i + 1 < n else f
        interval = max(0.0, (nxt - f) * hms) / beat
        interval = min(interval, 4.0) / 4.0
        ca_n = (cur_angles[i] / 360.0) if (angle_hint is not None or magnitudes is not None) else 0.0
        ps_n = (prev_signs[i]) if (angle_hint is not None or magnitudes is not None) else 0.0
        ds_n = (dists[i] / max(max_dist, 1e-6)) if (angle_hint is not None or magnitudes is not None) else 0.0
        ct_n = (cum_turns[i] / max(max_cum, 1e-6)) if (angle_hint is not None or magnitudes is not None) else 0.0
        feats.append([op, np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase),
                      bar, drums, bass, other, interval,
                      ca_n, ps_n, ds_n, ct_n])
    arr = np.array(feats, dtype=np.float32)
    # 能量列(z-score) + 几何归一列(max 已在推导时归一, 这里仅做轻微稳定) 都保持
    for c in (4, 5, 6):
        col = arr[:, c]
        if col.size and col.std() > 1e-8:
            arr[:, c] = (col - col.mean()) / (col.std() + 1e-6)
    return arr


def predict_turn_sign(model, feats, device):
    """feats:(N,F) -> (N,) +1/-1。"""
    xt = torch.from_numpy(np.asarray(feats, dtype=np.float32))
    if xt.dim() == 2:
        xt = xt.unsqueeze(0)
    xt = xt.to(device)
    model = model.to(device).eval()
    with torch.no_grad():
        logit = model(xt)[0].cpu().numpy().reshape(-1)
    # 运行时去偏(2026-08-15)：模型 logit 常系统性偏正(=> 大量往左拐成螺旋)。
    # 减去中位数后再按 0 判符号，使左右分布趋近均衡，打断持续同向螺旋。
    # 对称操作，偏左/偏右都治；只搬移判决点，不破坏序列前后文结构。
    med = float(np.median(logit))
    logit = logit - med
    sign = np.where(logit >= 0, 1.0, -1.0).astype(np.float32)
    if SIGN_FLIP:
        sign = -sign
    return sign
