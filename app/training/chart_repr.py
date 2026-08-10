"""
chart_repr.py — ADOFAI 谱面 <-> 稠密张量 (C, T) 双向转换
========================================================
按大佬架构(两阶段流水线):
  - 角度由【转换器】按 "间隔×180/拍长" 确定性算出,模型不碰角度 →
    彻底绕开「绝对角度回归塌缩」+「276s 物理错位」两个坑。
  - 方向(左/右):左/右转在同一时刻音频完全相同,模型学不到方向信号,故由
    plan_directions【几何路径规划】指派(不自交 + 留屏内),模型预测仅作平局破冰。
    这正对应大佬"转换器把时间点转成角度"的做法——方向是确定性布局决策。
  - SetSpeed 不做(大佬未做);Twirl 已启用:由 plan_path_twirl 计时反推式植入
    (不破坏踩点;Twirl 仅做视觉镜像翻转),由 onset 重音 + 模型 C2 弱偏置驱动。

  通道布局 (C=3):
    C0 onset : 格子起点处高斯热图(峰=1,附近平滑衰减)
    C1 dir   : 转弯方向类别 0=左转(+), 1=右转(-);非 onset 帧为 0(解码时忽略)
    C2 twirl : Twirl 事件热图(驱动 Twirl 放置;生成时由 onset 重音 + 该通道弱偏置决定)

  全局 BPM 不作为通道,而是【外部条件】传入(训练取谱面自身 bpm,推理取 UI/默认 120)。

  adofai_to_dense  : level dict -> (3, T) float32(训练目标;dir 取自真实角度符号)
  dense_to_adofai  : (3, T) -> level dict(转换器按间隔算角度 + 路径规划指派方向)
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from timing_engine import compute_note_times
from adofai_parse import _to_angle_data   # pathData(str/list) 与 angleData 统一转 angleData

N_CH = 3
DIR_LEFT = 0
DIR_RIGHT = 1
SR = 22050
# ── 128 全链路版本 ──────────────────────────────────────────────
# hop_length=128 ≈ 5.805 ms/帧（原 512 的 1/4 时间分辨率），踩点/事件网格都跑 128，
# 整条链路（OnsetNet 踩点 + VAE/扩散加事件）同一网格，落谱无需换算。
HOP = 128
HOP_MS = 1000.0 * HOP / SR          # ≈ 5.805 ms
# onset / twirl 通道用「高斯热图」而非单帧脉冲：峰值=1、附近平滑衰减（σ 帧）。
# 这样 VAE 重建能学出清晰峰值，避免稀疏单帧脉冲被后验塌缩抹平成常数。
# 128 网格下 σ=1.5 帧 ≈ ±4.5 帧(±26ms) 的峰值包络，比 512 网格更锐利。
ONSET_SIGMA = 1.5


def _bump(center, T, sigma=ONSET_SIGMA):
    """返回 (lo, hi, weights)，用于在 center 处叠加一段高斯热图。"""
    lo = max(0, int(round(center - 3.0 * sigma)))
    hi = min(T, int(round(center + 3.0 * sigma)) + 1)
    d = np.arange(lo, hi, dtype=np.float32) - center
    w = np.exp(-(d ** 2) / (2.0 * sigma * sigma))
    return lo, hi, w


def _events_by_floor(actions):
    """返回 twirl_floors:set。"""
    twirl = set()
    for a in (actions or []):
        if not isinstance(a, dict):
            continue
        et = a.get("eventType")
        fl = a.get("floor")
        if et == "Twirl" and fl is not None:
            twirl.add(int(fl))
    return twirl


def adofai_to_dense(level, T, hop_ms=HOP_MS, global_bpm=120.0):
    """level(dict) -> (3, T) float32 稠密谱面。解析失败返回全零。

    通道: C0 onset 热图; C1 dir(0=左/1=右, 取自真实角度符号); C2 twirl 热图(阶段一=0)。
    """
    ad = _to_angle_data(level) or []
    settings = level.get("settings") or {}
    actions = level.get("actions") or []
    if not ad:
        return np.zeros((N_CH, T), np.float32)
    try:
        nt = compute_note_times(ad, settings, actions, add_offset=True)
    except Exception:
        return np.zeros((N_CH, T), np.float32)
    if not nt:
        return np.zeros((N_CH, T), np.float32)
    times = [float(x[0]) if isinstance(x, (tuple, list)) else float(x) for x in nt]
    twirl = _events_by_floor(actions)
    dense = np.zeros((N_CH, T), np.float32)
    n = len(ad)
    for i in range(n):
        t0 = times[i - 1] if i > 0 else 0.0
        t1 = times[i]
        f0 = int(round(t0 / hop_ms))
        f1 = int(round(t1 / hop_ms))
        f0 = max(0, min(T, f0))
        f1 = max(f0, min(T, f1))
        if f1 <= f0:
            continue
        ang = ad[i]
        if abs(ang - 999) < 1.0:          # 中旋：转角对计时为 0，表示成 0 转角
            ang = 0.0
        lo, hi, w = _bump(f0, T)
        dense[0, lo:hi] = np.maximum(dense[0, lo:hi], w)
        # 方向类别:用相邻角度差的真实符号(模360归到[-180,180])。
        #   delta<0 -> 右转(1); 否则左转(0)。这是真实转弯方向, 之前的 ang<0 判定对 0~360
        #   角度几乎恒为左转, 是错的。中旋(ang==0)或近0差按左转处理。
        prev = ad[i - 1] if i > 0 else 0.0
        if abs(prev - 999) < 1.0:
            prev = 0.0
        delta = ((ang - prev + 180.0) % 360.0) - 180.0
        dense[1, f0] = DIR_RIGHT if delta < -1e-6 else DIR_LEFT
        if i in twirl:
            dense[2, lo:hi] = np.maximum(dense[2, lo:hi], w)
    return dense


def _seg_intersect(p1, p2, p3, p4):
    """标准线段相交判定(跨立实验)。端点重合不算相交(相邻砖块共用顶点)。"""
    def _ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = _ccw(p3, p4, p1); d2 = _ccw(p3, p4, p2)
    d3 = _ccw(p1, p2, p3); d4 = _ccw(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _fmod(a, b):
    """数学取模, 结果落在 [0, b) (同 C# / timing_engine)。"""
    return a - b * math.floor(a / b)


def _shortest(a):
    """把角度归到 (-180, 180]，取最短弧表示。"""
    a = _fmod(a, 360.0)
    return a - 360.0 if a > 180.0 else a


def plan_path_twirl(magnitudes, twirl_desire=None, model_twirl=None, step=1.0):
    """计时反推式路径规划(含可选 Twirl)——这是大佬"第二个模型加事件"的可微替代。

    核心: ADOFAI 计时引擎里 Twirl 会翻转 direction, 而引擎按 p_angle=(dest_{i-1}-180-dest_i)*dir
    算每格时长(不取最短弧)。由此反推得【计时精确】的闭式递推:
        dest_i = (dest_{i-1} - 180 - m_i * d_i)  mod 360
    其中 m_i = 该格目标转角幅度(由节拍锁死: 间隔×bpm×3), d_i = Twirl 奇偶(+1/-1)。
    代入验证: p_angle_i = (dest_{i-1}-180-dest_i)*d_i = m_i*d_i^2 = m_i -> 每格时长恒等于目标,
    无论是否 Twirl, 踩点误差=0。视觉上 Twirl 仅把该格转向翻成镜像(原路径在屏内则镜像也在屏内)。

    逐格二选一(不 Twirl / Twirl), 评分:
      - 硬约束: 新线段与已有线段相交 -> 一票否决(不自交)
      - 软目标: 离屏幕中心越近越好(不出屏)
      - Twirl 偏好: twirl_desire[i] 高(音乐重音)时减分, 诱导在此翻身; 扎堆时惩罚
      - 模型 C2 微弱偏置(可选, 通道偏弱, 仅点缀)
    返回 (angleData, twirl_floors)  angleData=list[int], twirl_floors=1-based 格下标 list。
    """
    import math
    pos = (0.0, 0.0)
    heading = 90.0  # 朝上(与 ADOFAI 视觉一致)
    dest_prev = 0.0  # 上一格线段绝对方向
    d = 1.0         # Twirl 奇偶(+1=无翻转累积方向)
    segs = []
    angleData = []
    twirls = []
    last_tw = -10
    R_TARGET = 25.0
    for i, mag in enumerate(magnitudes):
        mag = float(mag)
        best = None
        for tw in (False, True):
            d2 = d if not tw else -d
            dest_i = _fmod(dest_prev - 180.0 - mag * d2, 360.0)
            vt = _shortest(dest_i - dest_prev)          # 视觉转向(用于几何)
            nh = heading + vt
            nx = pos[0] + step * math.cos(math.radians(nh))
            ny = pos[1] + step * math.sin(math.radians(nh))
            new_seg = (pos, (nx, ny))
            inter = False
            for s in segs[:-1]:                          # 跳过紧邻前一条(共用顶点)
                if _seg_intersect(s[0], s[1], new_seg[0], new_seg[1]):
                    inter = True
                    break
            dist = math.hypot(nx, ny)
            over = max(0.0, dist - R_TARGET)
            # Twirl 偏好
            td = 0.0
            if twirl_desire is not None and i < len(twirl_desire):
                td = float(twirl_desire[i])
            score = (1e9 if inter else 0.0) + over - td * 6.0
            if tw:
                score += 2.0                             # 基础成本: 无重音时不过度 Twirl
                if (i - last_tw) < 4:                    # 与上个 Twirl 至少隔 4 格
                    score += 50.0
            if model_twirl is not None and i < len(model_twirl) and tw:
                score -= float(model_twirl[i]) * 1.0     # 模型 C2 微弱偏置
            if best is None or score < best[0]:
                best = (score, tw, d2, dest_i, vt, nh, (nx, ny), new_seg)
        _, tw, d2, dest_i, vt, nh, new_pos, new_seg = best
        angleData.append(int(round(_shortest(dest_i))))
        if tw:
            # floor = 0-based tile index，与 compute_note_times(parsed[fl]) 一致
            # （本项目引擎把 floor 当 parsed 下标，训练/解析均用 0-based，故保持一致）
            twirls.append(i)
            last_tw = i
        pos = new_pos
        heading = nh
        dest_prev = dest_i
        # 关键：Twirl 翻转旋转方向，引擎第二遍会「累计」翻转后续所有 tile 的方向。
        # 规划器必须把 d 翻成 d2，否则后续 tile 的 dest 用错奇偶 -> 踩点漂移。
        d = d2
        segs.append(new_seg)
    return angleData, twirls


def plan_directions(magnitudes, model_dir=None, step=1.0):
    """向后兼容包装: 仅做无 Twirl 的计时反推规划(等价于 twirl_desire 全 0)。"""
    ang, _ = plan_path_twirl(magnitudes, twirl_desire=None, model_twirl=model_dir, step=step)
    return ang


def dense_to_adofai(dense, global_bpm=120.0, hop_ms=HOP_MS, song="generated.mp3",
                    onset_frames=None, twirl_desire=None):
    """(3, T) -> level dict（含 angleData/settings）。无法构成有效谱面返回 None。

    onset_frames: 可选。若提供（频谱 onset 检测器给出的帧下标），直接用作谱面格
        起点（绕过 VAE 后验塌缩把稀疏 onset 抹平的问题）。角度由转换器按
        「间隔×180/拍长」算出(确定、不塌缩、不错位);方向(左/右)由 plan_directions
        几何规划指派(左/右转音频相同, 模型学不到, 故用路径规划保证不自交、留屏内)。
    """
    if dense is None or dense.ndim != 2 or dense.shape[0] < 2:
        return None
    C, T = dense.shape
    onset = dense[0]

    if onset_frames is not None:
        # 外部 onset（频谱检测器）直接驱动：绕过 VAE 抹平造成的稀疏化
        frames = [int(f) for f in onset_frames if 0 <= int(f) < T]
        frames.sort()
    else:
        # 兜底：从 VAE 解码的 onset 通道做局部峰值检测（易因后验塌缩而稀疏）
        peak = float(np.max(onset))
        if peak < 1e-3:
            return None  # 完全无 onset 结构（可能后验塌缩）
        thr = max(0.05, 0.25 * peak)
        frames = []
        for f in range(T):
            pv = onset[f - 1] if f > 0 else -1.0
            nv = onset[f + 1] if f < T - 1 else -1.0
            if onset[f] >= pv and onset[f] > nv and onset[f] > thr:
                frames.append(f)
    # 合并距离<2帧的重复 onset，保留合法短格（如 90°@120bpm≈5 帧）。
    kept = []
    for f in frames:
        if kept and (f - kept[-1]) < 2:
            continue
        kept.append(f)
    if len(kept) < 2:
        return None

    # 方向(左/右)：左/右转音频完全相同，模型学不到，由几何路径规划指派。
    # Twirl：由「音乐重音 twirl_desire(onset 包络强度) + 模型 C2(弱偏置)」驱动，
    # 用计时反推式 plan_path_twirl 植入。无论是否 Twirl，每格时长恒等于目标转角
    # 幅度 -> 踩点误差=0（不破坏踩点；Twirl 仅做视觉镜像翻转）。
    twirl_desire_per_tile = None
    model_twirl_per_tile = None
    if twirl_desire is not None:
        td_len = len(twirl_desire)
        twirl_desire_per_tile = np.array(
            [float(twirl_desire[f]) if 0 <= f < td_len else 0.0 for f in kept],
            dtype=np.float32)
        if C >= 3:
            twirl_ch = dense[2]
            cl = twirl_ch.shape[0]
            model_twirl_per_tile = np.array(
                [float(twirl_ch[f]) if 0 <= f < cl else 0.0 for f in kept],
                dtype=np.float32)

    # 先按【转换器】算每格转角幅度(确定性, 不塌缩/不错位)
    magnitudes = []
    for k, f in enumerate(kept):
        if k < len(kept) - 1:
            df = kept[k + 1] - kept[k]
        else:
            df = (kept[k] - kept[k - 1]) if len(kept) > 1 else 1
        seg_sec = max(0.0, df) * hop_ms / 1000.0
        beats = seg_sec * float(global_bpm) / 60.0
        magnitude = beats * 180.0
        # 不再量化到 15° 倍数：直接取连续角度，最终 int(round) 取整度即可。
        # ADOFAI angleData 本就是「整数度」，不必是 15° 倍数（15° 只是 path token 的
        # 紧凑记号，游戏会转回度数）。保留 15° 量化是无谓的精度损失：每格时长误差从
        # ~20ms/格(15°≈41.7ms)降到 ≤1.4ms/格(1°≈2.78ms@120bpm)，踩点显著更准。
        if magnitude <= 0:
            magnitude = 2.0
        if magnitude >= 360:
            magnitude = 358.0
        magnitudes.append(magnitude)
    # 路径规划：方向 + 可选 Twirl（计时精确）
    angleData, twirls = plan_path_twirl(
        magnitudes,
        twirl_desire=twirl_desire_per_tile,
        model_twirl=model_twirl_per_tile,
    )
    # 把规划出的 Twirl 写成事件（0-based floor，与 compute_note_times 一致）
    actions = [{"floor": int(f), "eventType": "Twirl"} for f in twirls]

    settings = {
        "bpm": float(global_bpm), "pitch": 100, "offset": 0, "song": song,
        "songArtist": "", "songName": song, "difficulty": 1, "volume": 100,
        "audioOffset": 0, "timeScale": 1.0, "mirror": 0, "flip": 0,
    }
    # 反解 offset：令 tile 0（出生点，关卡时间=0）落在第一个 onset 帧时间。
    # 计时链条：第 k 个 segment 时长 = (onset[k+1]-onset[k])，故 tile k 的歌曲时间
    #   = offset + sum(seg 0..k-1) = offset + (onset[k]-onset[0])。
    # 要让 tile k 对齐到 onset[k]，只需 offset = onset[0] = kept[0]*hop_ms。
    # 旧公式写成 kept[0]*hop_ms - first_arrival，错误地把【tile 1】而非 tile 0 对齐到
    # 首个 onset，使整谱相对音乐恒定早一拍(~kept[1]-kept[0]≈一个 segment)，
    # 即"踩不到点"的实质根因；与 BPM 翻倍是并列的两大元凶。
    raw_offset = int(round(kept[0] * hop_ms))
    # 安全夹：极端 offset 会让整谱错位，夹到 ±3000ms。
    settings["offset"] = max(-3000, min(3000, raw_offset))

    return {
        "angleData": [int(round(a)) for a in angleData],
        "settings": settings,
        "actions": actions,   # 阶段二:允许输出 Twirl(计时精确,不破坏踩点)
        "decorations": [],
    }


def validate(level, timestamps, onset_idx=None, out_path=None):
    """用计时引擎校验产出谱面的到达时间（含 offset）。

    参数
    ----
    onset_idx : list[int] | None
        各 onset 对应的格子下标。为 None 时退化为旧式 1:1 对齐（仅适用于无拆分的情况）。

    返回 (max_time_error_ms, has_setspeed)。
    """
    has_ss = any(a.get('eventType') == 'SetSpeed' for a in level.get('actions', []))
    nt = compute_note_times(level['angleData'], level['settings'],
                            level['actions'], add_offset=True)
    if onset_idx is not None:
        errs = []
        for i, idx in enumerate(onset_idx):
            if 0 <= idx < len(nt):
                errs.append(abs(nt[idx][0] - timestamps[i] * 1000))
    else:
        m = min(len(timestamps), len(nt))
        errs = [abs(nt[i][0] - timestamps[i] * 1000) for i in range(m)]
    max_err = max(errs) if errs else 0.0
    if has_ss:
        note = ""
    elif max_err < 50.0:
        note = "（纯角度已精确踩点，无 SetSpeed）"
    else:
        note = "（无 SetSpeed：个别超长间隔超出单格转角上限，略漂移）"
    print(f"[validate] tiles={len(level['angleData'])} offset={level['settings']['offset']}ms "
          f"has_SetSpeed={has_ss} max_time_error={max_err:.2f}ms{note}")
    if out_path:
        print(f"[validate] file : {out_path}")
    return max_err, has_ss


if __name__ == "__main__":
    # 自测：构造一个小谱面，验证 (1) 幅度往返一致 (2) 方向规划产出非退化谱面
    import json
    lvl = {
        "angleData": [90, -90, 180, -90, 90, -180],
        "settings": {"bpm": 120, "pitch": 100, "offset": 0},
        "actions": [],
        "decorations": [],
    }
    T = 1024
    d = adofai_to_dense(lvl, T, global_bpm=120.0)
    print("dense shape:", d.shape, "onset count:", int(d[0].sum()),
          "dir unique:", np.unique(d[1]))
    onsets = np.where(d[0] > 0.5)[0]
    back = dense_to_adofai(d, global_bpm=120.0, onset_frames=onsets)
    ad = back["angleData"]
    mags = [abs(a) for a in ad]
    L = sum(1 for a in ad if a < 0)   # 左转(负)数
    R = sum(1 for a in ad if a > 0)   # 右转(正)数
    print("reconstructed angleData:", ad)
    print("magnitudes match input abs-values (±1°):",
          np.allclose(sorted(mags), sorted([abs(x) for x in lvl["angleData"]]),
                      atol=1.0))
    print("non-degenerate (both signs present):", L > 0 and R > 0,
          f"(L={L}, R={R})")
    print("offset:", back["settings"]["offset"])
    print("OK plan_directions round-trip")
