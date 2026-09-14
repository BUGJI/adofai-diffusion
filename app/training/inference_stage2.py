"""inference_stage2.py — ADOFAI Diffusion 生成: 音频 -> OnsetNet(踩点) + VAE/扩散(加事件) -> .adofai

所有网格统一为 HOP=128 (≈5.805ms/帧):
  - 踩点 : Demucs 6 通道 mel @128 -> OnsetNet -> onset 帧(128 网格)
  - 扩散 : 单通道 full-mix mel @128 -> VAE/扩散 -> 稠密谱面(128 网格)
  - 落谱 : dense_to_adofai(onset_frames=踩点帧, hop_ms=5.805)
整条链路同一网格, 无需 512<->128 换算(这就是和 512 版本最大的干净之处)。
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)  # 避免后台/子进程多线程段错误
# 2026-11: cudnn 确定性 —— 同一音频多次生成必须得到同一份踩点/谱面。
# 此前 cudnn 非确定性卷积让临界 onset 逐次漂移(某钢琴曲「有时对、有时整体
# 晚一格」、某长谱首段跨次 +10ms 漂移的根源之一): 两次前向的微小区别
# 让贴着峰拾阈值的 onset 闪现/闪没, 中段一旦多/少一个, 之后 tile 与音的
# 对应就整体错位一格。确定化后同 wav 重跑逐帧一致, 便于复现与回归。
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "app" / "training"))
os.environ.setdefault("MKL_THREADING_LAYER", "sequential")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # Demucs 权重已缓存, 离线加载跳过网络重试

import librosa
from dataset import CHUNK, N_MELS, _mel
from vae import ChartVAE
from diffusion import DDPM
from chart_repr import dense_to_adofai, SR, HOP, HOP_MS
from onset_net import OnsetNet, predict_onset_frames
from demucs_mel import demucs_mel, HOP_ONSET, STEMS
from beat_this_align import (estimate_beats, estimate_bpm, build_beat_phase, BEAT_COND_CH)
from device_util import get_safe_device
from paths import resolve_checkpoint, train_ckpt_dir

# 权重查找走 paths.resolve_checkpoint 统一解析：
#   ADOFAI_DATA_DIR 显式指定 > 运行时目录(用户训练) > 便携内置目录(出厂权重)。
# 修复：旧版只读 env 指定的一处目录，GUI(=LOCALAPPDATA)/bat(=便携 data)两套入口
# 各读各的，另一套权重永远读不到。
CKPT = str(train_ckpt_dir())
# 老显卡(如 GTX 10 系) torch.cuda.is_available() 会假阳性，必须用一次真实内核探测
DEVICE = get_safe_device()


# 信号检测器(频谱通量峰值)已移除：踩点固定走 OnsetNet 模型，不再回退。


_ONSET_NETS = {}  # 按 beat_grid 分缓存：False->标准6通道, True->9通道(含节拍网格)


def _load_onset_net(beat_grid=False, onset_ckpt=None):
    """加载踩点神经网络。权重缺失/加载失败直接报错，不回退信号检测器。

    权重文件经 paths.resolve_checkpoint 统一解析（运行时目录优先 > 便携内置）。
    beat_grid=True 时加载含节拍相位条件的版本(onset_net_beatgrid.pt, in_channels=9)，
    该权重需重训得到（见 train_onset.py --beat_grid）；缺失即报错，不静默回退。
    onset_ckpt 指定时加载对应权重文件（如 onset_net_vocal.pt / onset_net_melody.pt），
    实现「分轨专门踩点模型」切换；缺省用标准 onset_net.pt。
    """
    if beat_grid:
        fname = "onset_net_beatgrid.pt"
        in_ch = len(STEMS) + BEAT_COND_CH
        tag = "节拍网格(beatgrid)"
    else:
        in_ch = len(STEMS)
        fname = onset_ckpt or "onset_net.pt"
        tag = f"指定({onset_ckpt})" if onset_ckpt else "标准"
    _key = ("bg" if beat_grid else "std") + ":" + (onset_ckpt or "default")
    if _key in _ONSET_NETS:
        return _ONSET_NETS[_key]
    p = resolve_checkpoint(fname)
    if p is None:
        raise RuntimeError(
            f"[infer] 未找到{tag}踩点模型权重 {fname}"
            f"（已查找：运行时目录 {CKPT} 与便携内置 data/checkpoints；"
            f"分轨模型需先训练、beatgrid 需 --beat_grid 重训）")
    try:
        m = OnsetNet(n_mels=N_MELS, in_channels=in_ch).to(DEVICE).eval()
        m.load_state_dict(torch.load(str(p), map_location=DEVICE))
    except Exception as e:
        raise RuntimeError(f"[infer] {tag}踩点模型加载失败({e})，无法生成")
    _ONSET_NETS[_key] = m
    print(f"[infer] 已加载{tag}踩点模型 {p.name} <- {p.parent} (模型踩点, in_channels={in_ch})")
    return _ONSET_NETS[_key]


def _detect_onsets(mel_onset, beat_grid=False, onset_ckpt=None):
    """踩点检测：固定用神经网络踩点。
    输入 mel_onset: (C,128,T128) @ hop128（beat_grid 时为 9 通道，否则 6 通道）。
    输出 (frames:list[int], prob:np.ndarray @128)。
    """
    net = _load_onset_net(beat_grid, onset_ckpt)
    frames, prob = predict_onset_frames(net, mel_onset, device=DEVICE)
    return frames, prob


# —— 静音门控(2026-10) ——
# OnsetNet 在前奏纯静音段会"幻觉"出 onset：峰值拾取的局部自适应阈值在静音段
# 是自指的（窗口内峰值就是幻影尖峰自身，0.15×自身 < 自身 必过），模型的训练
# 先验（人声/旋律常从 t=0 起声）又放大了这种触发。人声曲实测：
# 0-650ms 数字静音（mel 能量≈0），vocal 权重凭空触发 128/401/627ms 三个伪
# 踩点，变成开头三个幽灵格子把整首踩点推后 —— 即"开头还是多一个 floor"。
# 修复：用全混合波形的本地 RMS 做能量门控，(近)静音里的 onset 一律砍掉。
_SIL_ABS_RMS = 3e-4   # 绝对底：数字静音/解码底噪(~1e-4)之上、可听内容之下
_SIL_REL_RMS = 0.01   # 相对底：比"典型 onset 响度"(中位)低 40dB 视为不可听


# —— 前导静音裁剪(2026-10) ——
# OnsetNet 对"长数字静音后起声"的歌会漏掉第一个音头: 实测 800ms 静音 + 硬攻击,
# 模型在攻击处概率仅 0.001, 要等约 500ms 音乐上下文后才恢复触发; 同一段音频
# 从 t=0 起声则 25ms 即触发 —— 长静音把模型"毒化"了。静音段同时也是幻影踩点
# 的温床(见 _gate_silent_onsets)。修复: 检测前把前导(近)静音按帧裁掉, 谱面在
# 裁剪后的时间轴上构建, 裁掉时长补进 settings.offset —— 引擎真相公式
# (2026-12 实证): 歌曲时间 = 关卡时间 - 1拍 + offset, offset 增大 = 关卡时钟
# 相对音乐前移 = 游戏里恰好跳过前导静音, 首踩踩在首个可听音头上,
# 绝对踩点时刻与不裁剪时完全一致。
_SIL_TRIM_RMS = 1e-3    # 20ms 窗 RMS 越过此值(-60dB)视为有内容, 低于任何真实乐音


def _leading_silence_frames(y, sr, hop, win_ms=20.0, pre_ms=25.0, sustain=3):
    """返回应裁掉的前导静音帧数 n0(0=不裁)。

    越过阈值需连续 sustain 个 hop 步(≈32ms)的能量支持(排除单击爆音);
    裁剪点回退 pre_ms=25ms 保住音头前响(实测模型对 ≤50ms 前导静音可正常
    触发, 125ms 即漏); 音频不足 0.5s 或全静音返回 0(交下游按 onset 不足处理)。
    """
    y = np.asarray(y)
    total_f = y.size // hop
    keep_f = max(1, int(0.5 * sr / hop))
    if total_f <= keep_f:
        return 0
    half = max(1, int(win_ms * 1e-3 * sr) // 2)
    rms = np.empty(total_f, np.float64)
    for f in range(total_f):
        c = f * hop
        lo, hi = max(0, c - half), min(y.size, c + half)
        rms[f] = float(np.sqrt(np.mean(np.square(y[lo:hi])))) if hi > lo else 0.0
    cross = -1
    for f in range(total_f - sustain):
        if all(rms[f + k] > _SIL_TRIM_RMS for k in range(sustain)):
            cross = f
            break
    if cross < 0:
        return 0
    n0 = max(0, cross - int(pre_ms * 1e-3 * sr / hop))
    return min(n0, total_f - keep_f)


def _gate_silent_onsets(y, frames, prob, sr, hop, win_ms=35.0):
    """能量门控：砍掉落在(近)静音里的伪 onset。

    y: 全混合单通道波形(采样率 sr)；frames: onset 帧列表(@hop 采样网格)；
    prob: (T,) onset 概率。返回 (门控后 frames, prob)——被砍帧位及其 ±1 邻域
    概率一并清零，保证喂给 DDPM 的 onset 包络条件与最终踩点列表一致。
    阈值 = max(绝对底, 1% × 全部 onset 的中位本地 RMS)：
      数字静音(≈1e-4)必被砍；渐入段可听内容(如 0.0016 起)保留。
    整曲近静音(病态输入)时不砍，交由下游按"onset 不足"正常报错。
    """
    if not frames:
        return frames, prob
    y = np.asarray(y)
    if y.size == 0:
        return frames, prob
    half = max(1, int(win_ms * 1e-3 * sr))

    def _rms(f):
        c = int(f) * hop
        lo, hi = max(0, c - half), min(y.size, c + half)
        if hi <= lo:
            return 0.0
        return float(np.sqrt(np.mean(np.square(y[lo:hi]))))

    lv = np.array([_rms(f) for f in frames])
    ref = float(np.median(lv)) if lv.size else 0.0
    thr = max(_SIL_ABS_RMS, _SIL_REL_RMS * ref)
    keep = lv >= thr
    if not keep.any():
        return frames, prob
    kept = [int(f) for f, k in zip(frames, keep) if k]
    n_cut = len(frames) - len(kept)
    if n_cut:
        prob = np.array(prob, copy=True)
        for f, k in zip(frames, keep):
            if not k:
                for d in (-1, 0, 1):
                    i = int(f) + d
                    if 0 <= i < len(prob):
                        prob[i] = 0.0
        print(f"[infer] 静音门控: 砍掉 {n_cut} 个静音段伪 onset "
              f"(RMS 阈值 {thr:.5f}, 典型 onset RMS {ref:.5f}); "
              f"首个保留 onset={kept[0] * hop / sr * 1000:.0f}ms")
    return kept, prob


# —— 人声起声门控(2026-11) ——
# 人声曲人声模式修复后开头仍"多一个 floor": 前奏伴奏渐入段(≈830ms)响度
# 极低, Demucs 在低响度下分离塌陷, 伴奏能量约对半渗进 vocals stem(实测该处
# 人声/伴奏 RMS 比 0.84, 而真人声段为 2.8~6.4), vocal 权重据此在人声真正进歌
# (≈1033ms)之前触发伪 onset —— 全混合静音门控拦不住(伴奏本身可听)。
# 注意不能做全曲人声能量门控: 实测全曲 onset 的 vocals 能量分布连续
# (rel=0.25 会误杀 270/841 个弱唱/间奏真人声 onset, 边界裕度仅 1.01x),
# 故只裁"进歌之前"的前导段。
# 三修(2026-12, 方案A重训后回归): 重训后的 vocal/melody 权重在某人声曲
# 前奏软渐入段(伴奏渗漏, voc/acc≈0.83-0.94, sv≈0.1-2.4)也触发, 旧判据C
# (台阶跳变, 3x run_max)在 #0@644ms(sv0.15)之后 #1@708ms(sv0.98)即咬住
# —— 谱面首踩 708ms, 比 GT 人声谱首踩(1448ms)/人工核对的开唱(1242ms
# 强音节)早 500ms+。另一人声曲同病: 旧 A/B 在 #0@23ms(voc/acc=2.29
# 半渗漏)咬住, GT 人声谱首踩 139.4ms 恰是跳过它之后的 #1@174ms(比 7.96)。
# 实测渗漏段 voc/acc 上限 2.44(渗漏段 1039ms 弱起), 真进歌 2.94-7.96;
# 故三修后进歌 = 首个同时满足下述两条的 onset:
#   A) sv >= entry_rel*p75 且 sv >= entry_rel*p50(中位数地板, 整曲弱唱时
#      p75 阈值会跟着塌, 用中位数兜底);
#   B) sv >= dom_ratio*sa (人声压过伴奏渗漏; dom_ratio 由 1.5 提至 3.0,
#      依据上面的实测分布, 渗漏/真进歌在 2.4~2.9 之间有干净分界)。
# 删除: 判据C(台阶跳变 —— 两曲的全部误触发都由它单独引起); 判据D
# (melody 参照对齐 —— 重训后的 melody 权重同样在软渐入段触发, 参照自身
# 污染, 会把进歌点顺延到错误位置)。
# 扫描上限 entry_window=64 个 onset(约 2.7s): 病态输入(纯伴奏/整曲渗漏)
# 找不到满足 A+B 的点时【不裁】, 安全回退 —— 避免无限扫描把整段前奏
# 误删。判据E 的 melody 参照(_ref_frames)保留, 只用于拖音判据的确认。
_VOCAL_ENTRY_REL = 0.5
_VOCAL_DOM_RATIO = 3.0
_VOCAL_ENTRY_WINDOW = 64
# 伴奏静默豁免(2026-09-13): onset 处伴奏能量 sa 低于全曲 onset 处 sa 中位的此比例时,
# 判该处"伴奏几乎没响" → 不可能有伴奏渗漏进 vocals(渗漏的前提是伴奏在响), 这里的
# onset 只能来自人声自身(清唱/念白/无伴奏段), 于是【豁免判据 A 的人声绝对响度地板】,
# 只凭判据 B(人声压伴奏)放行。修某清唱前奏曲目: 轻柔清唱前奏被判据 A(地板被副歌抬高)
# 整段误杀 → 进歌点误推到 10.5s(用户"offset 一万多")。实测该曲前奏 sa≈0.06 vs 全曲
# onset 处 sa 中位 53.8(相差近千倍), 与渗漏段(sa≈67, ratio≈0.1)分界巨大, 阈值稳健。
_VOCAL_ACCOMP_QUIET_FRAC = 0.10


def _stem_frame_energy(lin, f):
    """onset 帧 f 处的 stem 能量(RMS 代理): 取 f±1 帧线性 mel 能量最大值开方。"""
    lo, hi = max(0, f - 1), min(lin.size, f + 2)
    if hi <= lo:
        return 0.0
    return float(np.sqrt(max(0.0, float(lin[lo:hi].max()))))


def _gate_vocal_lead_onsets(mel_voc, mel_acc, frames, prob, sr, hop,
                            entry_rel=_VOCAL_ENTRY_REL,
                            dom_ratio=_VOCAL_DOM_RATIO,
                            entry_window=_VOCAL_ENTRY_WINDOW,
                            min_onsets=8,
                            accomp_quiet_frac=_VOCAL_ACCOMP_QUIET_FRAC):
    """人声踩点: 砍掉人声真正进歌之前的伴奏渗漏伪 onset(只裁前导段)。

    mel_voc/mel_acc: vocals/accomp 通道的 log-mel (@hop 网格, 与 frames 同轴)。
    能量 = sum(expm1(logmel))(线性 mel 功率, log1p 的精确逆), 开方作 RMS 代理。
    进歌判定(三修): 首个同时满足 A(绝对响度: entry_rel×p75 且 entry_rel×p50)
    和 B(人声主导: sv >= dom_ratio×sa)的 onset; 前后共 entry_window 个 onset
    (约 2.7s)内找不到则【不裁】(安全回退, 见函数头注释)。少于 min_onsets
    个 onset 时统计不可靠, 同样不裁。
    返回 (裁剪后 frames, prob)。
    """
    frames = [int(f) for f in frames]
    if len(frames) < min_onsets:
        return frames, prob
    lin_v = np.expm1(np.asarray(mel_voc, dtype=np.float32)).sum(axis=0)
    lin_a = (np.expm1(np.asarray(mel_acc, dtype=np.float32)).sum(axis=0)
             if mel_acc is not None else None)
    sv = np.array([_stem_frame_energy(lin_v, f) for f in frames])
    sa = (np.array([_stem_frame_energy(lin_a, f) for f in frames])
          if lin_a is not None else None)
    p75 = float(np.percentile(sv, 75)) if sv.size else 0.0
    p50 = float(np.percentile(sv, 50)) if sv.size else 0.0
    # 伴奏静默地板: sa 中位的 accomp_quiet_frac 以下判"伴奏几乎没响"(渗漏不可能)。
    sa_med = float(np.median(sa)) if (sa is not None and sa.size) else 0.0
    quiet_floor = accomp_quiet_frac * sa_med
    entry = None
    lim = min(len(frames), entry_window)
    for i in range(lim):
        # 伴奏静默段: 渗漏物理上不可能(没有伴奏能量可渗), 该 onset 只能来自人声
        # 自身 → 豁免判据 A 的人声绝对响度地板(否则轻柔清唱前奏被副歌抬高的地板
        # 整段误杀), 仅凭判据 B(此时 ratio 上百必过)放行。
        accomp_quiet = sa is not None and sa[i] <= quiet_floor
        if not accomp_quiet:
            # 判据 A: 绝对响度达典型人声量级(双分位地板, 见函数头注释)
            if not (p75 > 0 and p50 > 0
                    and sv[i] >= entry_rel * p75 and sv[i] >= entry_rel * p50):
                continue
        # 判据 B: 人声明显压过伴奏渗漏(重训后渗漏段 voc/acc 实测上限 2.44,
        # 真进歌 2.94~7.96, dom_ratio=3.0 卡在干净分界上; 伴奏静默段 ratio 上百,
        # 等价于 sa 地板已满足)
        if sa is None or sa[i] <= 1e-9 or sv[i] >= dom_ratio * sa[i]:
            entry = i
            break
    if entry is None or entry == 0:
        return frames, prob
    prob = np.array(prob, copy=True)
    for f in frames[:entry]:
        for d in (-1, 0, 1):
            j = f + d
            if 0 <= j < len(prob):
                prob[j] = 0.0
    t_in = frames[entry] * hop / sr * 1000.0
    print(f"[infer] 人声起声门控: 人声进歌≈{t_in:.0f}ms(裁剪后时间轴), "
          f"砍掉 {entry} 个进歌前伪 onset; 首个保留 onset={t_in:.0f}ms")
    return frames[entry:], prob


# —— 判据 E(2026-11): 拖音/衰减再触发抑制(人声模式, 全曲) ——
# OnsetNet 还会把【长音的持续/衰减段】误判成新音节。人声曲实测: ~1954ms
# 的 onset 落在 1904ms 峰值(包络 1824)之后的衰减坡上(947→609, 35ms 内无任何
# 上升), 旋律参照模型在该处无起音 —— 是无字的"多一格", 唱名与格子从该点起
# 整体错位一格, 即用户反馈的「整体还是晚一个」。此类 onset 是否出现还会随
# cudnn 非确定性逐次漂移(临界概率), 时有时无。
# 判据: 过 onset 点的平滑人声包络若在降/平(env[f] < 1.05*env[f-6],
# 6 帧≈35ms 内无新起升 —— 真音节必然迎着音头上升), 且旋律参照 ±3 帧内不
# 确认(排除真连音/装饰音, 它们包络可以不升但旋律模型听得到), 则裁掉。
# 反之有新起升(真音头)或旋律确认者一律保留; 进歌点前的清理已由判据 A-D 负责。
def _gate_sustain_retrigger(mel_voc, frames, prob, sr, hop,
                            ref_frames=None, ref_tol=3, rise_win=6, rise_ratio=1.05):
    """裁掉人声"拖音/衰减再触发"伪 onset, 返回 (frames, prob)。"""
    frames = [int(f) for f in frames]
    if not frames:
        return frames, prob
    lin_v = np.expm1(np.asarray(mel_voc, dtype=np.float32)).sum(axis=0)
    _k = np.ones(4, dtype=np.float32) / 4.0
    env = np.convolve(lin_v, _k, mode="same")
    ref = [int(r) for r in (ref_frames or [])]
    keep = []
    for f in frames:
        if f >= rise_win:
            rise = float(env[f]) / max(float(env[f - rise_win]), 1e-6)
        else:
            rise = 2.0  # 曲首信息不足, 保留(进歌点已由判据 A-D 把关)
        mel_ok = any(abs(f - r) <= ref_tol for r in ref)
        if rise < rise_ratio and not mel_ok:
            continue  # 拖音/衰减处的再触发: 无新起升且旋律不确认
        keep.append(f)
    if len(keep) == len(frames):
        return frames, prob
    prob = np.array(prob, copy=True)
    _kept = set(keep)
    for f in frames:
        if f in _kept:
            continue
        for d in (-1, 0, 1):
            j = f + d
            if 0 <= j < len(prob):
                prob[j] = 0.0
    n = len(frames) - len(keep)
    t0 = keep[0] * hop / sr * 1000.0 if keep else -1.0
    print(f"[infer] 拖音门控(判据E): 裁掉 {n} 个拖音/衰减再触发伪 onset, "
          f"保留 {len(keep)} 个; 首个保留 onset={t0:.0f}ms(裁剪后时间轴)")
    return keep, prob


# —— 判据 F(2026-11): 连发簇合并(全模式) ——
# 症状(实测反馈): 匀速强节奏/长摩擦声等【同一事件】的位置, OnsetNet 在概率
# 平台上反复触发, 一个节奏点劈里啪啦虚踩一大片(实测单簇 2~15 击, 相邻
# 11~70ms), 打起来"糊"。
# 实测依据(6 曲症状歌, OnsetNet 概率曲线, 4000+ 对相邻踩点):
#   - 真音头之间概率必然塌谷: 间隔>=80ms 的谷深比 vr(=谷底/两峰较强者)中位
#     数 0.00~0.03; 即便是 45ms 的真 16 分连打(330bpm)vr p25 也
#     仅 0.25 —— 深谷=真连打;
#   - 同一事件的再触发骑在概率平台上: vr>=0.55(间隔 11~70ms 桶里占 60~100%);
#   - 某曲开头琶音(真音符, 人工核对确认"开头点采的准")间隔可达 116ms
#     且谷浅 —— 故谷深判据必须设【间隔上限】, >70ms 一律不合并, 保护真连打
#     与rubato段。
# 规则: 相邻踩点 gap<ABS_MS(机械双触发, 亚帧/近帧) 或 (gap<=LINK_MS 且
# vr>=VR_T)(平台再触发) 判为同一事件连成簇; 簇内仅保留概率最强的一击。
# 340bpm 的 16 分音符(44ms)深谷者不会被并(>35ms 且 vr 低); 独立音乐事件
# (>=80ms 或深谷)一律保留。合并后谱面由 _fill_missing_beats 在长空隙补
# 整拍, "本来一个节奏点"的位置恢复为节拍级踩点。
_BURST_ABS_MS = 35.0   # 绝对不应期: <35ms 必为同事件机械双触发(2~6 帧)
_BURST_LINK_MS = 70.0  # 谷深判据适用上限: 更远的一律独立(真连打/琶音保护)
_BURST_VR = 0.55       # 谷深比阈值: 两踩点间概率不低于强者 55% = 同一平台


def _merge_burst_onsets(frames, prob, min_keep=2, global_bpm=None, enabled=True):
    """判据 F: 合并同一事件的连发伪踩点(全模式), 簇内保留概率最强一击。

    enabled=False(界面「同音多采拦截」关): 直接原样返回, 不做任何合并,
    供 A/B 试听拦截前后的真实效果。

    frames: onset 帧列表(@hop 网格, 已过静音/人声/拖音门控); prob: (T,) 概率。
    返回 (合并后 frames, prob)——被并帧位及其 ±1 邻域概率清零, 保证喂给
    DDPM 的 onset 包络条件与最终踩点列表一致。病态输入(合并后不足
    min_keep 个)不合并, 安全回退。
    """
    frames = [int(f) for f in frames]
    if not enabled:
        print(f"[infer] 同音多采拦截=关: 跳过连发簇合并(判据F), 保留全部 {len(frames)} 个踩点")
        return frames, prob
    if len(frames) < min_keep:
        return frames, prob
    p = np.asarray(prob)

    def _vr(a, b):
        seg = p[a + 1:b]
        if seg.size == 0:
            return 1.0
        mx = max(float(p[a]), float(p[b]))
        if mx <= 1e-9:
            return 1.0
        return float(seg.min()) / mx

    keep, i, n_cut, biggest = [], 0, 0, (0, -1.0)
    while i < len(frames):
        j = i
        while j + 1 < len(frames):
            gap = (frames[j + 1] - frames[j]) * HOP_MS
            if gap < _BURST_ABS_MS or (
                    gap <= _BURST_LINK_MS and _vr(frames[j], frames[j + 1]) >= _BURST_VR):
                j += 1
            else:
                break
        if j > i:
            mem = frames[i:j + 1]
            ps = [float(p[f]) for f in mem]
            best = mem[int(np.argmax(ps))]
            keep.append(best)
            n_cut += len(mem) - 1
            if len(mem) > biggest[0]:
                biggest = (len(mem), best * HOP_MS)
        else:
            keep.append(frames[i])
        i = j + 1
    if n_cut == 0 or len(keep) < min_keep:
        return frames, prob
    prob = np.array(prob, copy=True)
    _kept = set(keep)
    for f in frames:
        if f in _kept:
            continue
        for d in (-1, 0, 1):
            k = f + d
            if 0 <= k < len(prob):
                prob[k] = 0.0
    print(f"[infer] 连发簇合并(判据F): 裁掉 {n_cut} 个同事件再触发伪踩点, "
          f"合并 {len(frames) - len(keep)} 簇(最大簇 {biggest[0]} 击@"
          f"{biggest[1] / 1000:.1f}s), 保留 {len(keep)} 个")
    return keep, prob


@torch.no_grad()
def generate(audio, out_path, base_bpm=120.0, steps=50, guidance=2.5,
             onset_track="all", vfx=False, intensity=0.5,
             auto_bpm=False, beat_grid=False, onset_ckpt=None,
             seed=None, dual_block=True):
    # —— 确定性种子(2026-11): 同歌同设置 -> 逐位一致的谱面 ——
    # 此前谱面摆形全靠 DDPM 无种子采样(torch.randn), 每次跑都是新的随机
    # 抖动; 叠加 cudnn 非确定性卷积, 就是某钢琴狂奏曲「有时正确有时晚
    # 一个」偶发症状的根源之一。现在在 generate() 入口统一播种:
    #   - seed=None(默认, web/CLI 均此路径) -> 由音频内容指纹派生(前 64s 采样
    #     均值+时长, 不随文件路径/缓存变动), 同一首歌永远同一个种子;
    #   - seed 显式给定 -> 完全可复现实验。
    # cudnn 确定性标志已设(见模块顶部), 两者配合后整条推理链不再有随机源。
    if seed is None:
        y, _ = librosa.load(audio, sr=SR, mono=True)
        _fp = y[: int(64 * SR)].astype(np.float32)
        _h = float(np.asarray(_fp.reshape(-1, 4).mean(axis=1)).sum()) * 1000.0
        seed = int(abs(_h * 1e6) % (2 ** 31 - 1)) or 1
        del _fp
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31 - 1))
    import random as _random
    _random.seed(seed)
    print(f"[infer] 确定性种子 seed={seed}(DDPM 采样/摆形完全可复现)")
    y, _ = librosa.load(audio, sr=SR, mono=True)
    # —— 前导静音裁剪(2026-10): 防 OnsetNet 被长静音"毒化"漏掉首个音头 ——
    _n0 = _leading_silence_frames(y, SR, HOP)
    _trim_ms = _n0 * HOP_MS
    if _n0 > 0:
        y = y[_n0 * HOP:]
        print(f"[infer] 前导静音裁剪 {_trim_ms:.0f}ms(谱面建于裁剪后时间轴, "
              f"时长补回 offset=游戏跳过该段静音)")
    mel = _mel(y)                       # (128, T) 单通道 —— 扩散条件 mc 用(@128)
    T = mel.shape[1]
    # 踩点专用：Demucs 6 通道分离 mel @ hop128，模型踩点（与扩散同网格，无换算）
    mel_onset = demucs_mel(audio, device=DEVICE)   # (6, 128, T128)
    if _n0 > 0:
        # 帧级切片 = 裁掉前导静音(Demucs 只跑一次; mel 帧中心对齐采样 f*hop,
        # 切片后与对裁剪波形重算的 mel 一致)
        mel_onset = np.ascontiguousarray(mel_onset[:, :, _n0:])
    # —— 人声占比标定(2026-09-12, 11 首实测) ——
    # 分母只用 4 条原子轨(drums+bass+other+vocals, 不含 full/accomp 复制品):
    # 有人唱 4.8%~14.6%(6.95/5.82/4.77/14.60/12.93),
    # 纯音乐全部 ≤0.6%(最高 0.56/其余≈0),
    # ~10 倍干净间隔 → 阈值 2%。(旧版 6ch 分母 ≥1%: full/accomp 摊薄比例, 采样上也能分,
    # 但原子分母不依赖复制品权重, 标定值直接可比。)
    _lin = np.expm1(mel_onset.astype(np.float32))
    _atom_idx = [STEMS.index(s) for s in ("drums", "bass", "other", "vocals")]
    _voc_e = float((_lin[STEMS.index("vocals")] ** 2).sum())
    _atom_e = float((_lin[_atom_idx] ** 2).sum())
    voc_frac = (_voc_e / _atom_e) if _atom_e > 0 else 0.0
    # —— 选音轨采音：把选中轨复制填充全部 6 通道喂 OnsetNet(维度不变, 免重训) ——
    # 音轨名归一化：网页下拉可能传 "vocal"（单数），统一映射到 STEMS 里的 "vocals"。
    _TRACK_ALIAS = {"vocal": "vocals", "voice": "vocals", "voc": "vocals",
                    "mel": "melody", "melody": "melody"}
    if onset_track and onset_track != "all":
        onset_track = _TRACK_ALIAS.get(onset_track.lower(), onset_track)
        if onset_track in STEMS:
            if onset_track == "vocals" and voc_frac < 0.02:
                # 纯音乐硬选人声轨 = 给模型喂空通道 → 谱面塌掉。要求: 没人唱的
                # 伴奏要踩背景, 这里自动回退全混合(和自动聚焦同一条 2% 判定线)。
                print(f"[infer] 采音音轨=vocals, 但 vocals 能量占比 "
                      f"{voc_frac * 100:.2f}% < 2%, 判定纯音乐 → 回退全混合踩背景伴奏")
            else:
                single = mel_onset[STEMS.index(onset_track)]  # (128, T) 选中轨
                # 关键：把选中轨复制填充到全部 6 通道，而非置零其他通道。
                # OnsetNet 训练时 6 通道始终同时有内容，若置零会触发"静音"分布导致踩点塌成 0；
                # 复制填充保持"多通道均非零"的分布，模型只从该轨的 onset 特征踩点。
                mel_onset = np.stack([single] * len(STEMS), axis=0)
                print(f"[infer] 采音音轨={onset_track}（已复制填充至全部 6 通道供 OnsetNet 踩点）")
        else:
            print(f"[infer] 未知音轨 '{onset_track}'，回退全部混合踩点")
    # —— 人声模式(2026-09-13 修正): 自动聚焦【已废除】, 模型始终吃真 Demucs 6 通道 ——
    # 2026-09-12 曾把人声模式的 mel 换成"vocals 单通道复制×6"以躲开伴奏渗漏伪踩点。
    # 但训练侧(train_onset.build_samples)喂的永远是 demucs_mel 真 6 通道, 复制填充
    # 是模型从未见过的分布 → 输出糊成 46ms 宽平台、峰位随机漂(实测同一 vocal
    # 权重: 真6通道 离音头 23.2ms/踩空 16.7%/间隔乱度 35 种; 假6通道 34.8ms/31.5%/
    # 102 种 —— "采的乱七八糟像喝了酒"的根因就是它, 不是网格不是 bpm)。
    # 渗漏防护交还给下面的人声起声门控(判据 A+B, 本来就基于 vocals/accomp 通道
    # 能量, 设计上就是配合真 6 通道输入的)。auto 模式采得好, 正因为它一直吃真通道。
    if (onset_ckpt and "vocal" in Path(str(onset_ckpt)).stem.lower()
            and (not onset_track or onset_track == "all")):
        if voc_frac >= 0.02:
            print(f"[infer] 人声模式: 模型吃真 6 通道(与训练分布一致), vocals 占比 "
                  f"{voc_frac * 100:.1f}%, 伴奏渗漏防护走起声/拖音门控")
        else:
            print(f"[infer] 人声模式: vocals 原子能量占比 {voc_frac * 100:.2f}% < 2%, "
                  f"判定纯音乐, 全混合踩背景伴奏")
    # —— SOTA 节拍对齐 (Beat This!) ——
    # 一次推理拿到 beats/downbeats，Phase A 用于 BPM、Phase B 用于节拍相位条件。
    _bt_beats, _bt_down = None, None
    if auto_bpm or beat_grid:
        try:
            _bt_beats, _bt_down = estimate_beats(audio, device="cpu")
        except Exception as e:
            print(f"[beat_this] 推理失败({e})，跳过节拍对齐")
        # 前导静音裁剪后 beats 也要平移到裁剪后时间轴(丢掉裁剪点之前的)
        if _bt_beats is not None and _n0 > 0:
            _t0 = _n0 * HOP / float(SR)
            _bt_beats = [float(b) - _t0 for b in _bt_beats if float(b) >= _t0]
            if _bt_down is not None:
                _bt_down = [float(b) - _t0 for b in _bt_down if float(b) >= _t0]
    # Phase A：用 Beat This! 估计真实 BPM，替换写死的 base_bpm（无需重训，立即生效）。
    if auto_bpm:
        if _bt_beats is not None and len(_bt_beats) >= 2:
            est = estimate_bpm(_bt_beats)
            print(f"[beat_this] 估计 BPM={est}（原 base_bpm={base_bpm}），已采用")
            base_bpm = est
        else:
            print("[beat_this] 未检出足够 beats，沿用 base_bpm")
    # Phase B：把节拍相位条件(3通道)拼到 mel 后 -> 9 通道，喂 beatgrid 模型（需重训）。
    if beat_grid:
        if _bt_beats is not None:
            bg = build_beat_phase(_bt_beats, _bt_down, mel_onset.shape[2])  # (3,128,T)
            mel_onset = np.concatenate([mel_onset, bg], axis=0)            # (9,128,T)
            print(f"[beat_this] 已拼节拍相位条件 -> in_channels={mel_onset.shape[0]}")
        else:
            print("[beat_this] 无 beats 可构建节拍条件，回退纯 mel")
    onsets, onset_prob = _detect_onsets(mel_onset, beat_grid=beat_grid, onset_ckpt=onset_ckpt)
    # 静音门控(2026-10)：砍掉前奏/间奏静音段里 OnsetNet 的幻影踩点，
    # 否则它们变成开头幽灵格子把整首踩点推后（见 _gate_silent_onsets 注释）。
    onsets, onset_prob = _gate_silent_onsets(y, onsets, onset_prob, SR, HOP)
    # —— 人声起声门控(2026-11): vocal 专模型会咬住前奏伴奏渐入的渗漏伪踩点
    # (低响度下 Demucs 分离塌陷, 伴奏对半渗进 vocals stem), 全混合门控拦不住。
    # 只在 vocal 权重 + 采音轨为 all/vocals 时启用; 旋律/标准模式不受影响。
    # (2026-09-12) 输入已聚焦人声(自动聚焦/显式人声轨)时【跳过起声门控】: 纯人声
    # 输入没有伴奏渗漏可砍, 门控只会误杀"伴奏安静段落里的轻人声"(某曲前奏喊话
    # 被整段砍掉 → offset 一万多)。拖音判据 E 与输入无关, 照常执行。
    # (2026-09-12 二修) 纯音乐(vocals 原子占比<2%)时【整套跳过】: 判据 E 的
    # rise=env[f]/env[f-6] 在空人声通道上恒 <1.05, 又少有旋律参照可确认, 会把
    # 背景踩点整片误杀 —— 恰好违反"没人唱的伴奏要踩背景"的要求。 ——
    if (onset_ckpt and "vocal" in Path(str(onset_ckpt)).stem.lower()
            and onset_track in ("all", "vocals")):
        if voc_frac < 0.02:
            print(f"[infer] vocals 原子占比 {voc_frac * 100:.2f}% < 2%, 纯音乐 → "
                  f"跳过整套人声门控(起声+拖音), 保护背景伴奏踩点")
        else:
            try:
                _ref_frames = None
                if resolve_checkpoint("onset_net_melody.pt") is not None:
                    try:
                        # 判据 E 参照: melody 权重的起音帧(同样过静音门控), 只用于
                        # 拖音判据的确认, 不参与进歌点(其自身也会咬住软渐入渗漏,
                        # 三修后进歌点改由 A+B 双判据独立判定, 见 _gate_vocal_lead_onsets)。
                        _rf, _rp = _detect_onsets(mel_onset, beat_grid=False,
                                                  onset_ckpt="onset_net_melody.pt")
                        _rf, _ = _gate_silent_onsets(y, _rf, _rp, SR, HOP)
                        _ref_frames = _rf
                    except Exception as e:
                        print(f"[infer] 旋律参照踩点失败({e})，拖音判据仅用包络")
                if onset_track == "vocals":
                    # 用户显式选人声轨 → 输入是真·人声单轨, 无伴奏渗漏可砍,
                    # 起声门控只会误杀前奏轻人声, 跳过; 拖音判据 E 照常。
                    # (auto 人声模式 track=all 走不到这里, 门控正常运行扛渗漏)
                    print("[infer] 采音音轨=vocals(纯人声输入), 跳过起声门控(防误杀安静前奏人声)")
                else:
                    onsets, onset_prob = _gate_vocal_lead_onsets(
                        mel_onset[STEMS.index("vocals")],
                        mel_onset[STEMS.index("accomp")],
                        onsets, onset_prob, SR, HOP)
                # 判据 E: 拖音/衰减再触发抑制(全曲, 人声模式) —— 修「整体晚一个」。
                onsets, onset_prob = _gate_sustain_retrigger(
                    mel_onset[STEMS.index("vocals")], onsets, onset_prob, SR, HOP,
                    ref_frames=_ref_frames)
            except Exception as e:
                print(f"[infer] 人声起声门控失败({e})，跳过")
    # 判据 F: 连发簇合并(全模式, 最后一道门) —— 修「一个节奏点被劈里啪啦
    # 踩一大片」。注意必须放在所有门控之后: 此时 onset 列表已经是最终踩点
    # 候选, 簇内保留概率最强一击; prob 同步清零保证 DDPM 包络条件一致。
    try:
        onsets, onset_prob = _merge_burst_onsets(onsets, onset_prob, global_bpm=float(base_bpm),
                                                 enabled=bool(dual_block))
    except Exception as e:
        print(f"[infer] 连发簇合并失败({e})，跳过")
    print(f"[infer] audio {Path(audio).name} T={T} frames, 模型(Demucs-OnsetNet) 踩点 onsets={len(onsets)}")

    vae_p = resolve_checkpoint("vae.pt")
    if vae_p is None:
        print(f"[infer] 缺 vae.pt（已查找 {CKPT} 与便携内置 data/checkpoints），无法生成")
        return None
    ddpm_p = resolve_checkpoint("ddpm.pt")
    if ddpm_p is None:
        print(f"[infer] 缺 ddpm.pt（已查找 {CKPT} 与便携内置 data/checkpoints），无法生成")
        return None
    vae = ChartVAE().to(DEVICE).eval()
    vae.load_state_dict(torch.load(str(vae_p), map_location=DEVICE))
    ddpm = DDPM().to(DEVICE).eval()
    ddpm.load_state_dict(torch.load(str(ddpm_p), map_location=DEVICE))

    # 逐块处理（每块 CHUNK 帧 @128）。
    # 修复 2026-08-28: 末尾不足 CHUNK 的最后一块曾被 `break` 直接丢弃,
    # 导致整首尾奏被截断 T%CHUNK 帧(最长可达 23.8s; 实测曾丢 ~11.6s)。
    # 现改为把最后一块边缘补零到 CHUNK(4096, 已被16整除, VAE/扩散须定长)后照常跑,
    # 再把补零部分裁掉, 保证整曲不丢帧。
    dense_chunks = []
    for s in range(0, T, CHUNK):
        seg_len = min(CHUNK, T - s)
        seg = mel[:, s:s + seg_len]
        pad = CHUNK - seg_len
        if pad > 0:
            # 边缘复制(末列重复)补到定长, 避免硬零跳变; 扩散条件该段 onset=0
            seg = np.concatenate([seg, np.repeat(seg[:, -1:], pad, axis=1)], axis=1)
        mc = torch.from_numpy(seg)[None].to(DEVICE)
        # onset 包络 = 该块内音头条件（模型概率平滑）
        oe_np = np.zeros((1, 1, CHUNK), np.float32)
        for f in range(s, min(s + CHUNK, len(onset_prob))):
            oe_np[0, 0, f - s] = float(onset_prob[f])
        oe = torch.from_numpy(oe_np).to(DEVICE)
        z = ddpm.sample(mc, oe, steps=steps, guidance=guidance, device=DEVICE)  # (1,16,chunk/16)
        rec = vae.decode(z)               # (1,3,chunk)
        rec = rec[0].cpu().numpy()
        rec[0] = 1.0 / (1.0 + np.exp(-rec[0]))     # sigmoid onset
        rec[2] = 1.0 / (1.0 + np.exp(-rec[2]))     # sigmoid twirl
        dense_chunks.append(rec[:, :seg_len])      # 裁掉补零部分, 只留真实长度
    if not dense_chunks:
        print("[infer] 音频过短, 无法生成"); return None
    dense = np.concatenate(dense_chunks, axis=1)    # (3, T')
    # 仅保留与原始 onset 帧一致的范围
    Tf = dense.shape[1]
    onset_frames = [f for f in onsets if f < Tf]

    # 摆形状模型(可选): 权重存在则加载, 让模型接管左右转向; 无权重退回几何贪心(当前行为)
    shape_model = None
    shape_ckpt = resolve_checkpoint("shape_model.pt")
    if shape_ckpt is not None:
        try:
            from shape_model import ShapeModel
            sd = torch.load(str(shape_ckpt), map_location=DEVICE)
            shape_model = ShapeModel()
            shape_model.load_state_dict(sd, strict=True)
            shape_model = shape_model.to(DEVICE).eval()
            print(f"[shape] 已加载摆形状模型 {shape_ckpt}")
        except Exception as e:
            print(f"[shape] 加载失败, 退回几何贪心: {e}")
            shape_model = None
    # 各 stem 能量必须来自 Demucs 6 通道分离 mel（(6,128,T128)），与 onset_prob 同网格。
    # 之前误用单通道全混合 mel 的 np.mean(axis=1) -> (128,) 一维，导致
    # extract_tile_features 里 stem_energy.shape[1] 索引越界、模型接管失败退回几何贪心。
    stem_energy = np.mean(np.abs(mel_onset), axis=1).astype(np.float32)  # (6, T128)
    level = dense_to_adofai(dense, global_bpm=float(base_bpm), hop_ms=HOP_MS,
                            song=Path(audio).name, onset_frames=onset_frames,
                            twirl_desire=dense[2],
                            shape_model=shape_model, onset_prob=onset_prob,
                            stem_energy=stem_energy, device=DEVICE,
                            merge_dup=bool(dual_block))
    if level is None:
        print("[infer] dense_to_adofai 返回 None (onset 不足)")
        return None
    # —— 前导静音裁剪的时长补回 offset(引擎真相公式 2026-12 实证:
    # 歌曲时间 = 关卡时间 - 1拍 + offset, offset 增大即游戏跳过前导静音)
    # —— 绝对踩点时刻与不裁剪时完全一致。
    if _n0 > 0:
        _s = level.setdefault("settings", {})
        _s["offset"] = int(round(float(_s.get("offset", 0)) + _trim_ms))
        print(f"[infer] offset += {_trim_ms:.0f}ms(前导静音裁剪补回)")

    # —— 第三阶段：注入视觉特效（VFXNet 帧级预测吸附到方块）——
    n_vfx = 0
    if vfx:
        try:
            from apply_vfx import apply_vfx as _apply_vfx
            before = len(level.get("actions", []))
            level = _apply_vfx(level, audio, intensity=float(intensity), device=DEVICE)
            n_vfx = len(level.get("actions", [])) - before
            print(f"[infer] VFX 注入完成：新增 {n_vfx} 个视觉特效动作")
        except Exception as e:
            print(f"[infer] VFX 注入失败（已跳过，谱面仍可用）：{e}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(level, ensure_ascii=False, indent=1), encoding="utf-8-sig")
    n_twirl = sum(1 for a in level.get("actions", [])
                  if isinstance(a, dict) and a.get("eventType") == "Twirl")
    print(f"[infer] wrote {out_path} | tiles={len(level['angleData'])} "
          f"| twirl={n_twirl} | vfx={n_vfx} | bpm={base_bpm}")
    return level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bpm", type=float, default=120.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--track", default="all",
                    help="采音音轨: all/drums/bass/other/vocals/accomp（选某轨则只盯该轨踩点）")
    ap.add_argument("--onset_ckpt", default=None,
                    help="指定踩点模型权重文件名（如 onset_net_vocal.pt / onset_net_melody.pt），"
                         "实现分轨专门踩点；缺省用 onset_net.pt")
    ap.add_argument("--vfx", action="store_true",
                    help="生成时加入视觉特效（VFXNet 帧级预测吸附到方块）")
    ap.add_argument("--intensity", type=float, default=0.5,
                    help="视觉特效强度 0..1（越大越密、幅度越狠）")
    ap.add_argument("--auto_bpm", action="store_true",
                    help="用 Beat This! 估计真实 BPM 替换 --bpm（默认写死值）")
    ap.add_argument("--beat_grid", action="store_true",
                    help="把节拍相位条件拼入 OnsetNet（需先 --beat_grid 重训 onset_net_beatgrid.pt）")
    ap.add_argument("--no_dual_block", action="store_true",
                    help="关闭同音多采拦截(判据F连发合并+落谱<2帧合并), 保留全部近距踩点供试听")
    a = ap.parse_args()
    generate(a.audio, a.out, a.bpm, a.steps, a.guidance, a.track,
             vfx=a.vfx, intensity=a.intensity, auto_bpm=a.auto_bpm,
             beat_grid=a.beat_grid, onset_ckpt=a.onset_ckpt,
             dual_block=not a.no_dual_block)

if __name__ == "__main__":
    main()
