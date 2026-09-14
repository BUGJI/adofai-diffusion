"""
apply_vfx.py — 把 VFXNet 的帧级视觉特效预测，吸附到生成谱面的方块上，
注入为 ADOFAI 顶层 actions（与 Twirl 同格式：{"floor": <原生 0-based floor>, "eventType": ...}）。

 设计（用户拍板，2026-08-11：完全自由发挥；2026-08-29 重申拔掉重音门）
 ------------------------------------
   - 模型对每一时刻预测 20 类特效的激活概率 ev[e,fi]。
   - 自由发挥：不设「重音门」，每个方块都是候选，由模型对各事件的预测概率
     （越过 ABS_THR 存在性下限）自行决定放不放、放几个、落在哪格。
   - 注入「所有超过存在性下限 ABS_THR 的事件」—— 爱放几个放几个，
     不做 top-K 精选、不堆满上限、不限制总数、不卡出现位置。模型自由发挥。
   - 特效幅度由模型直接决定（逆变换 [-1,1]→量纲 + 非负下界），不随音乐强度缩放。
   - 已开放全部视觉事件（含 MoveTrack/AnimateTrack/SetFilter/SetFilterAdvanced 等），
     filterType 仅从受控白名单选，绝不自创。仅 MultiPlanet(结构事件) / PositionTrack(用户禁用) 不注入。
   - aux 构造与训练端 extract_vfx 完全一致：aux[0] 只在 Twirl 重音处给包络，
     aux[1] 局部间隔、aux[2] 转角、aux[3] 歌曲位置。

用法：
    from apply_vfx import apply_vfx
    level2 = apply_vfx(level, audio_path, intensity=0.5, device="cuda")
"""
from __future__ import annotations
import os

import numpy as np
import torch

from extract_vfx import VFX_EVENTS
from effects_schema import build_action, FILTER_TYPES
from chart_repr import compute_note_times, HOP_MS
from vfx_net import VFXNet, predict_vfx
from demucs_mel import demucs_mel
try:
    from device_util import get_safe_device
except Exception:
    def get_safe_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

N_FILTERS = len(FILTER_TYPES)  # 与训练端 effects_schema / train_vfx 同步（当前 120 种真实滤镜名）
# VFX 权重经 paths.resolve_checkpoint 统一解析：运行时目录(网页/GUI 训练产出)优先，
# 回退便携内置目录(出厂权重)。修复：旧版写死便携目录，网页训练出的 vfx_net.pt 永远读不到。
CKPT_NAME = "vfx_net.pt"

# 每帧绝对下限：模型对该事件把握低于此值就不注入（不塞入完全不信的特效）
ABS_THR = 0.30
# SetFilter/SetFilterAdvanced 专用更低的出现门槛：训练集外歌曲上模型把滤镜几乎只绑在
# 「重音/翻身」帧，非重音帧想加滤镜的概率常<0.30 而被丢弃，导致很多歌一个滤镜都没有
# （用户反馈「新歌没滤镜」）。放低到 0.15，让「有点想加」的帧也能出滤镜（提高出现率，
# 非人为幅度上限，符合自由发挥原则）。
SETFILTER_THR = 0.30  # 2026-08-15 用户反馈滤镜添加过频，由 0.15 提至 0.30 减弱次数（仍低于 ABS_THR，保证有歌可出滤镜）
# 非 Twirl 帧要成为重音，模型峰值必须达到此值（适中：让更多强拍能注入特效）
ACCENT_BAR = 0.55
# 滤镜种类采样温度：训练数据滤镜种类极不均衡（Fisheye/Grayscale 占 35%），
# filt 头 argmax 会垄断成 Grayscale。用温度 softmax + top-k 采样打散，让种类多样。
FILT_TEMP = 2.2
FILT_TOPK = 8
# 灰阶去重偏好（用户反馈"滤镜全是灰阶、没有别的"，2026-08-14）：
# 训练数据 Grayscale 等去色类占比高，仅温度采样仍偏它。在 filt 头 logit 上给纯灰阶减分，
# 强制彩色/鲜艳滤镜（Neon/Fisheye/Glitch/VHS…）更易出现。只改"选哪种滤镜"的偏好，
# 不动任何幅度/位置/数量（守 VFX 自由发挥铁律）。
_GRAY_NAMES = ["Grayscale", "CameraFilterPack_Color_GrayScale"]
_GRAY_IDXS = []
for _gn in _GRAY_NAMES:
    try:
        _GRAY_IDXS.append(FILTER_TYPES.index(_gn))
    except Exception:
        pass
GRAYSCALE_BIAS = 2.0

# —— 闪光(Flash)频率抑制（用户反馈生成的 Flash 太密、快闪刺眼，2026-08-14）——
# 只控制时间密度，绝不改 opacity/duration/plane（遵守视觉自由发挥铁律）。
# FLASH_MODE:
#   "off"    —— 完全不注入 Flash（彻底止住快闪）
#   "sparse" —— 默认：相邻 Flash 至少间隔 FLASH_GAP_SEC 秒，且整首不超过 FLASH_MAX_COUNT 个
FLASH_MODE = "sparse"
FLASH_GAP_SEC = 1.5
FLASH_MAX_COUNT = 24

# —— 绽放(Bloom)阈值下压（用户反馈生成的 Bloom 阈值太高、泛光只在最亮处出现、效果偏弱，2026-08-14）——
# 只调整 Bloom 的 threshold 字段（下压到更明显的区间），绝不改 intensity/color（遵守视觉自由发挥铁律）。
# 与 Flash 降频同层：属于「注入后的后处理干预」，不碰 effects_schema.build_action 的幅度映射。
# 模型生成阈值多在 15~35（cap=120 线性映射结果），×BLOOM_TH_SCALE 落到约 5~12，泛光明显且不糊屏。
BLOOM_TH_SCALE = 0.35
# 绽放(Bloom)强度下压（用户反馈生成的 Bloom 强度偏高、整体偏过曝，2026-08-14）：
# 仅对 Bloom 的 intensity 字段做乘性下压，不动 threshold/color（守 VFX 自由发挥铁律）。
# 与阈值下压同层：注入后的后处理干预，不碰 effects_schema.build_action 的幅度映射。
# 模型生成强度多在 40~170（cap=5000 有符号逆变换结果），×BLOOM_INT_SCALE 落到约 6~26，泛光更柔和。
# 用户反馈 0.5 仍偏高（2026-08-14 20:07）由 0.5 降至 0.3；现再反馈"还是有点高"（2026-08-15）降至 0.15。
BLOOM_INT_SCALE = 0.15

# —— 绽放(Bloom)出谱总衰减（2026-09 用户反馈"晃得人看不到轨道了"）——
# 阈值和强度在写进谱面前，额外再 ÷ BLOOM_OUT_DIV（两层下压叠加）。
# 叠加后实际系数：阈值 ×0.35÷6 = 0.0583、强度 ×0.15÷6 = 0.0250。
# 只改 threshold/intensity 两个字段，不动颜色/时长/数量/位置（守 VFX 自由发挥铁律）。
BLOOM_OUT_DIV = 6.0


def _bump(center, T, sigma=1.5):
    lo = max(0, int(round(center - 3.0 * sigma)))
    hi = min(T, int(round(center + 3.0 * sigma)) + 1)
    d = np.arange(lo, hi, dtype=np.float32) - center
    return lo, hi, np.exp(-(d ** 2) / (2.0 * sigma * sigma))


def _clip(x, lo, hi):
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, xf))


def _build_aux(level, T, times=None):
    """构造 4 通道节奏辅助 (4,T)，与训练时 extract_vfx 的 aux 格式严格对齐。

    aux[0] Twirl 重音高斯包络（仅翻身处）  aux[1] 局部间隔(秒)
    aux[2] 局部转角幅度(/180)              aux[3] 歌曲相对位置(0~1)

    times：外部传入的方块时间轴(ms)。apply_vfx 先做「时间轴回缩对齐」再传入，
    保证 aux 特征落帧与注入端读取 ev 的帧轴是同一条时间轴（2026-09 运镜丢帧修复：
    旧版 aux 用未回缩时间构造、注入却读回缩后的帧，两条帧轴错位随时间线性放大，
    模型在方块帧上的概率峰值几乎全部被错过 -> 运镜被大面积丢弃）。
    缺省时自行 compute_note_times，行为同旧版。
    """
    aux = np.zeros((4, T), np.float32)
    aux[3, :] = np.linspace(0.0, 1.0, T, dtype=np.float32)
    angle_data = level.get("angleData", []) or []
    actions = level.get("actions", []) or []
    if times is None:
        try:
            nt = compute_note_times(angle_data, level.get("settings", {}) or {},
                                     actions, add_offset=True)
            times = [float(x[0]) if isinstance(x, (tuple, list)) else float(x) for x in nt]
        except Exception:
            return aux
    else:
        times = [float(x[0]) if isinstance(x, (tuple, list)) else float(x) for x in times]
    if not times:
        return aux
    n = len(times)
    frames = [max(0, min(T - 1, int(round(t / HOP_MS)))) for t in times]
    ad = (list(angle_data) + [0] * max(0, n - len(angle_data)))[:n]

    # Twirl 翻身帧（真正重音）。floor 为 0-based tile 序号: Twirl@floor F 在
    # tile F 到达瞬间触发 = nt[F-1] = frames[F-1], 故 ti=fl-1 是【触发帧】索引
    # (时间索引, 与 floor 0/1-based 约定无关, 数值不变)。floor 0(触发=起点帧 0)
    # 会被 fl-1=-1 丢弃 —— 语料与生成端都无 floor 0 事件, 无实际影响。
    twirl_floors = set()
    n_ad = len(angle_data)
    for a in actions:
        if isinstance(a, dict) and a.get("eventType") == "Twirl":
            try:
                ti = int(round(float(a.get("floor")))) - 1
            except Exception:
                continue
            if 0 <= ti < n_ad:
                twirl_floors.add(ti)

    # aux[0] 只在 Twirl 重音处打高斯包络
    for i, f in enumerate(frames):
        if i in twirl_floors and 0 <= f < T:
            lo, hi, w = _bump(f, T)
            aux[0, lo:hi] = np.maximum(aux[0, lo:hi], w)
    # aux[2] 转角幅度（节奏上下文，所有 tile）
    for i, f in enumerate(frames):
        if 0 <= f < T:
            ang = abs(ad[i]) if abs(ad[i] - 999) > 1.0 else 0.0
            aux[2, f] = _clip(ang / 180.0, 0, 1)
    # aux[1] 局部间隔（按所在 segment 广播）
    for f in range(T):
        seg = 0
        for i in range(n - 1):
            if frames[i] <= f < frames[i + 1]:
                seg = i
                break
        else:
            seg = n - 2 if n >= 2 else 0
        seg = max(0, min(n - 2, seg))
        inter = (times[seg + 1] - times[seg]) / 1000.0 if n >= 2 else 0.5
        aux[1, f] = _clip(inter, 0.05, 4.0)
    return aux


def _make_action(name, pa_vec, intensity, mag=1.0, ease_idx=0, filt_idx=0, disable=False, seed_key=0):
    """委托 effects_schema.build_action —— 字段名按 ADOFAI-JS 官方定义，滤镜名受控、
    缓动由模型预测索引决定。seed_key（帧号）驱动未学习维度（如 MoveCamera 方向）的
    确定性伪随机，保证同一帧每次生成结果一致。"""
    return build_action(name, pa_vec, intensity, mag, ease_idx=ease_idx, filt_idx=filt_idx, disable=disable, seed_key=seed_key)


def _sanitize(o):
    """递归把 NaN/Inf 换成 0，避免写出非法 JSON 导致游戏加载失败。"""
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, bool):
        return o
    if isinstance(o, (int, float)):
        try:
            f = float(o)
        except Exception:
            return o
        return 0.0 if not np.isfinite(f) else o
    return o


def _ftime_of(a, nt):
    """取某动作（含 floor）对应的方块时间（秒）。
    floor 为原生 0-based: 事件在 tile F 到达瞬间触发 = nt[F-1], fl-1 是【触发时间】
    索引(时间索引, 与 floor 约定无关)。nt[j] = tile j+1 的到达时间（ms）。
    2026-09 修复: 旧版漏除 1000, 返回的是毫秒, 而 _filter_flash 拿它跟
    FLASH_GAP_SEC(秒) 比较 -> 最小间隔形同虚设(只靠数量上限兜底), 现统一回秒。"""
    try:
        fl = int(a.get("floor", 1)) - 1
    except Exception:
        return 0.0
    if 0 <= fl < len(nt):
        t = nt[fl]
        if isinstance(t, (list, tuple)):
            try:
                return float(t[0]) / 1000.0
            except Exception:
                return 0.0
        try:
            return float(t) / 1000.0
        except Exception:
            return 0.0
    return 0.0


def _filter_flash(actions, nt):
    """对所有 Flash 动作按时间降频 / 关闭。原地修改 actions。

    - FLASH_MODE=="off"    : 移除全部 Flash（彻底止住快闪）。
    - FLASH_MODE=="sparse" : 相邻保留的 Flash 至少间隔 FLASH_GAP_SEC 秒，
      且整首不超过 FLASH_MAX_COUNT 个；其余 Flash 丢弃。
    不改动任何保留 Flash 的 opacity/duration/plane（视觉自由发挥铁律）。
    """
    if FLASH_MODE == "off":
        keep = [a for a in actions if a.get("eventType") != "Flash"]
        removed = len(actions) - len(keep)
        actions[:] = keep
        if removed:
            print(f"[vfx] FLASH_MODE=off：已移除 {removed} 个 Flash（彻底关闭闪光）")
        return
    # sparse：按时间最小间隔 + 数量上限
    flash_items = [(i, a) for i, a in enumerate(actions) if a.get("eventType") == "Flash"]
    if not flash_items:
        return
    flash_items.sort(key=lambda x: _ftime_of(x[1], nt))   # 按时间升序
    last_keep = -1e9
    keep_idx = set()
    count = 0
    for i, a in flash_items:
        t = _ftime_of(a, nt)
        if (t - last_keep) >= FLASH_GAP_SEC and count < FLASH_MAX_COUNT:
            keep_idx.add(i)
            last_keep = t
            count += 1
    new_actions = [a for i, a in enumerate(actions)
                   if a.get("eventType") != "Flash" or i in keep_idx]
    removed = len(actions) - len(new_actions)
    actions[:] = new_actions
    if removed:
        print(f"[vfx] FLASH_MODE=sparse：Flash 降频，保留 {count} 个、移除 {removed} 个"
              f"（间隔≥{FLASH_GAP_SEC}s，上限{FLASH_MAX_COUNT}）")


def _lower_bloom_threshold(actions):
    """对全部 Bloom 动作的 threshold 做下压（仅改 threshold 字段，不动 intensity/color）。
    原地修改 actions。阈值是「亮度高于此值的像素才发光」：越低泛光范围越大、效果越明显。

    与 _filter_flash 同层：注入后的后处理干预，不动 build_action 的模型自由发挥映射（守 VFX 铁律）。
    """
    changed = 0
    _k = BLOOM_TH_SCALE / BLOOM_OUT_DIV
    for a in actions:
        if a.get("eventType") != "Bloom":
            continue
        try:
            th = float(a.get("threshold", 0.0))
        except Exception:
            continue
        new_th = max(0.0, th * _k)
        a["threshold"] = round(new_th, 3)
        changed += 1
    if changed:
        print(f"[vfx] Bloom 阈值下压：{changed} 个 Bloom 的阈值 ×{_k:.4f}"
              f"(={BLOOM_TH_SCALE}÷{BLOOM_OUT_DIV:g})（不动强度/颜色）")


def _lower_bloom_intensity(actions):
    """对全部 Bloom 动作的 intensity 做乘性下压（仅改 intensity 字段，不动 threshold/color）。
    原地修改 actions。强度越高泛光越亮、越易过曝刺眼。

    与 _lower_bloom_threshold 同层：注入后的后处理干预，不动 build_action 的模型自由发挥映射
    （守 VFX 铁律）。系数 BLOOM_INT_SCALE 可调（调小=更弱）。
    """
    changed = 0
    _k = BLOOM_INT_SCALE / BLOOM_OUT_DIV
    for a in actions:
        if a.get("eventType") != "Bloom":
            continue
        try:
            it = float(a.get("intensity", 0.0))
        except Exception:
            continue
        new_it = max(0.0, it * _k)
        a["intensity"] = round(new_it, 3)
        changed += 1
    if changed:
        print(f"[vfx] Bloom 强度下压：{changed} 个 Bloom 的 intensity ×{_k:.4f}"
              f"(={BLOOM_INT_SCALE}÷{BLOOM_OUT_DIV:g})（不动阈值/颜色）")


def apply_vfx(level, audio, vfx_ckpt=None, intensity=1.0, device=None):
    """给生成谱面 level 注入视觉特效动作，原地修改并返回 level。
    intensity 固定 1.0（幅度不再有滑块/地板压缩，由模型自由决定）；
    mag=1.0（不再随音乐强度缩放，纯模型控制）。"""
    if vfx_ckpt is None:
        from paths import resolve_checkpoint
        vfx_ckpt = resolve_checkpoint(CKPT_NAME)
    if vfx_ckpt is None:
        print(f"[vfx] 权重缺失（{CKPT_NAME}，已查找运行时目录与便携内置目录），跳过特效注入")
        return level
    if device is None:
        try:
            device = get_safe_device()
        except Exception:
            device = "cpu"

    # 1) 模型
    sd = torch.load(vfx_ckpt, map_location="cpu")
    model = VFXNet(n_events=len(VFX_EVENTS), n_filters=N_FILTERS, param_dim=3)
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()

    # 2) 输入准备：mel（音频）+ nt（方块时间轴，先回缩对齐再喂 aux）。
    #    顺序是关键：aux 必须用与注入读取 ev 完全相同的（回缩后）时间轴构造，
    #    否则两条帧轴错位随时间线性放大，模型峰值被大面积错过（运镜丢帧根因）。
    mel = demucs_mel(audio, device=device)        # (6,128,T)
    T = mel.shape[2]
    try:
        nt = compute_note_times(level.get("angleData", []), level.get("settings", {}),
                                level.get("actions", []), add_offset=True)
    except Exception:
        print("[vfx] compute_note_times 失败，无法定位方块，跳过")
        return level
    if not nt:
        return level

    # ── 时间轴对齐保护（防御「后半段复制几十次」）────────────────────
    # 根因：BPM 自动检测偏差 或 复用半长/错长 mel 缓存，使「方块时间轴」超出
    # 「音频 mel 时长」，后半段 fi 全部 clamp 到末帧 -> 特效在谱面后半段
    # 一模一样复制几十次（看着就是没特效）。正常对齐的谱 _scale≈1 不触发。
    # 处理：若方块最大时间超过 mel 时长，按全局比例线性回缩到 [0,(T-1)*HOP_MS]，
    # 整曲仍铺满、且后半段不再复制末帧。
    # 2026-09 运镜修复：回缩挪到 aux 构造【之前】，aux 与注入共用这条回缩后的
    # 时间轴。旧版 aux 用未回缩时间构造、注入读回缩后的帧 -> 帧轴错位 ->
    # 运镜等特效被大面积丢弃（98格谱仅14格过阈，实际模型想给95格）。
    try:
        _max_t = max(((t[0] if isinstance(t, (tuple, list)) else t) for t in nt))
        _t_cap = (T - 1) * HOP_MS
        if _max_t > _t_cap * 1.001:
            _scale = _t_cap / float(_max_t)
            nt = [((t[0] * _scale, t[1]) if isinstance(t, (tuple, list)) else t * _scale)
                  for t in nt]
            print(f"[vfx] 时间轴回缩×{_scale:.3f}（方块最大时间 {_max_t:.0f}ms 超 mel 时长 "
                  f"{_t_cap:.0f}ms），避免后半段复制")
    except Exception:
        pass

    # aux 由 level 反算方块时间构造，与训练端 extract_vfx 格式严格对齐；
    # 传入回缩后的 nt —— 与注入读取 ev 的帧轴同轴（运镜丢帧修复的核心）。
    aux = _build_aux(level, T, times=nt)                     # (4,T)

    # 3) 推理
    with torch.no_grad():
        ev, pa, fl, ez, dis = predict_vfx(model, mel, aux, device=device)  # (V,T),(V,3,T),(F,T),(V,E,T),(T,)

    I = float(np.clip(intensity, 0.0, 1.0))
    # 自由发挥（2026-08-29 用户重申「自由发挥，除了我之前单独限制过的」）：
    # 拔掉「重音门」——不再用 ACCENT_BAR / Twirl 位置卡住特效出现在哪格。
    # 每个方块都是候选，由模型对各事件的预测概率(越过 ABS_THR 存在性下限)自行决定，
    # 爱放几个放几个、落在哪格完全由模型定，不强制对齐到重音/翻身。
    angle_data = level.get("angleData", [])
    new_actions = []
    for i, tms in enumerate(nt):
        if i >= len(angle_data):
            break
        if isinstance(tms, (tuple, list)):
            tms = tms[0]
        fi = max(0, min(T - 1, int(round(tms / HOP_MS))))
        mag = 1.0
        # 候选：ev 超过存在性下限 ABS_THR 的事件全部注入（自由发挥，不限数量、不限位置）。
        cands = []
        for e in range(len(VFX_EVENTS)):
            name = VFX_EVENTS[e]
            s = float(ev[e, fi])
            # SetFilter/SetFilterAdvanced 用更低的出现门槛，其余事件用 ABS_THR
            thr = SETFILTER_THR if name in ("SetFilter", "SetFilterAdvanced") else ABS_THR
            if s < thr:
                continue
            cands.append((s, e, name))
        for s, e, name in cands:
            # 用户要求（2026-08-29）：SetFilterAdvanced 输出时去掉 "Advanced" 后缀，
            # 统一当作 SetFilter 处理——SetFilter 在 ADOFAI 里可正常识别，
            # Advanced 变体的额外字段（rotation/noise/transitionTime/speed）不需要。
            if name == "SetFilterAdvanced":
                name = "SetFilter"
            # 用户明确要求禁用（位置轨道难看，2026-08-14）：仅 PositionTrack。
            if name in ("PositionTrack",):
                continue
            # 模型预测该事件该用哪种缓动 / 哪种滤镜（自由，不再写死 InOutSine/Linear）
            ez_vec = ez[e, :, fi]                       # (E,)
            ease_idx = int(np.argmax(ez_vec))
            if name in ("SetFilter",):
                # 带温度 softmax + top-k 采样，避免 Grayscale 等高频类垄断
                _lg = fl[:, fi] / FILT_TEMP
                for _gi in _GRAY_IDXS:           # 压低纯灰阶，让别的滤镜出来
                    _lg[_gi] -= GRAYSCALE_BIAS
                _p = np.exp(_lg - _lg.max()); _p /= _p.sum()
                _top = np.argsort(_p)[::-1][:FILT_TOPK]
                _pt = _p[_top]; _pt /= _pt.sum()
                _rng = np.random.default_rng(int(fi) + 1)
                filt_idx = int(_top[_rng.choice(len(_top), p=_pt)])
                # 模型 disable_head 因训练数据 85% 为 False 而塌缩（输出恒≈0，max=0.045），
                # 无法可靠决定 disableOthers → 滤镜会无限堆叠成"灰泥"。
                # 推理端强制每帧滤镜独占（disableOthers=True），避免灰蒙蒙叠加。
                # 根治需重训时给 BCE 加 pos_weight≈5.7 抵消类别不平衡（见 train_vfx.py）。
                disable = True
            else:
                filt_idx = 0
                disable = False
            act = _make_action(name, pa[e, :, fi], intensity, mag,
                               ease_idx=ease_idx, filt_idx=filt_idx, disable=disable,
                               seed_key=fi)
            if act is None:
                continue
            act["floor"] = i + 1  # floor 0-based: 要在 nt[i](tile i+1 到达瞬间)触发 -> floor=i+1
            new_actions.append(act)

    level = _sanitize(level)
    level.setdefault("actions", []).extend(new_actions)
    # 闪光频率抑制：按方块时间对 Flash 降频 / 关闭（用户反馈快闪刺眼，2026-08-14）
    try:
        _filter_flash(level.setdefault("actions", []), nt)
    except Exception as e:
        print(f"[vfx] Flash 过滤跳过({e})")
    # 绽放阈值下压：让泛光更明显（用户反馈生成的 Bloom 阈值太高，2026-08-14）
    try:
        _lower_bloom_threshold(level.setdefault("actions", []))
    except Exception as e:
        print(f"[vfx] Bloom 阈值下压跳过({e})")
    # 绽放强度下压：让泛光更柔和不过曝（用户反馈生成的 Bloom 强度偏高，2026-08-14）
    try:
        _lower_bloom_intensity(level.setdefault("actions", []))
    except Exception as e:
        print(f"[vfx] Bloom 强度下压跳过({e})")
    print(f"[vfx] 注入 {len(new_actions)} 个视觉特效动作 "
          f"(intensity={I:.2f}, 自由发挥：数量/位置不限)")
    return level


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 3:
        print("usage: apply_vfx.py <input.adofai> <audio> [intensity]")
        sys.exit(1)
    lv = json.load(open(sys.argv[1], encoding="utf-8"))
    out = apply_vfx(lv, sys.argv[2],
                    intensity=float(sys.argv[3]) if len(sys.argv) > 3 else 0.5)
    dst = sys.argv[1] + ".vfx.json"
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("written", dst)
