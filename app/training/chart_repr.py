"""
chart_repr.py — ADOFAI 谱面 <-> 稠密张量 (C, T) 双向转换
========================================================
采用两阶段流水线架构:
  - 角度由【转换器】按 "间隔×180/拍长" 确定性算出,模型不碰角度 →
    彻底绕开「绝对角度回归塌缩」+「276s 物理错位」两个坑。
  - 方向(左/右):左/右转在同一时刻音频完全相同,模型学不到方向信号,故由
    plan_directions【几何路径规划】指派(不自交 + 留屏内),模型预测仅作平局破冰。
    这正对应"转换器把时间点转成角度"的做法——方向是确定性布局决策。
  - SetSpeed 不做;Twirl 已启用:由 plan_path_twirl 计时反推式植入
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
# ── ADOFAI Diffusion（hop=128 高分辨率网格）─────────────────────
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
    """返回 twirl:set —— Twirl 所在的内部 0-based tile 下标。

    原生 .adofai 的 Twirl floor 本身就是 0-based tile 序号(见 timing_engine
    事件循环注释的 ADOFAI-JS / AutoCat 源码证据): floor F = 离开 tile F 的弧,
    故直接 ti = fl(floor<0 的非法值丢弃)。消费方(adofai_to_dense 的
    `if i in twirl`)按 0-based tile/弧下标比较, i 即弧下标 = tile 下标。
    """
    twirl = set()
    for a in (actions or []):
        if not isinstance(a, dict):
            continue
        et = a.get("eventType")
        fl = a.get("floor")
        if et == "Twirl" and fl is not None:
            try:
                ti = int(round(float(fl)))
            except (TypeError, ValueError):
                continue
            if ti >= 0:
                twirl.add(ti)
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
    twirl = _events_by_floor(actions)   # 0-based tile/弧下标(原生 floor 即 0-based)
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


def _cumulative_snap(angles, grid=15.0):
    """累积传导式吸附(2026-08-23 定稿方案)。

    与"每格独立吸到最近 grid 倍数"不同: 每吸附一格, 修正量会传导给后续
    所有格(后续有效角 -= 修正量), 使整条轨迹绝对朝向保持连续, 仅抹平
    局部抖动, 不产生累积漂移。

    参数:
        angles: 已 round 到 1° 的整数角度列表
        grid:   目标吸附步进(默认 15°, 即 ADOFAI 合法角度步进)
    返回:
        吸附后的整数角度列表(仍落在 grid 倍数上)
    """
    out = []
    drift = 0.0
    for a in angles:
        eff = a + drift                 # 当前格有效角(含历史漂移)
        t = round(eff / grid) * grid    # 吸附到最近 grid 倍数
        delta = t - eff                 # 本格修正量
        drift += delta                  # 传导: 后续格全部偏移 -delta
        # 吸附结果落在 grid 倍数, 再 round 到 1° 保证整数
        out.append(int(round(t)))
    return out


def _cumulative_round_diff(angles):
    """角度取整: 累积和取整再差分(主修 2026-11-14, 长谱漂移根因)。

    逐角度独立 round 会让 ±0.5°/格的量化误差沿上千块砖随机游走 ——
    实测一张 1484 块、bpm=200 的长谱累计 ~30ms 相位漂移, 呈"越走越快、
    后半谱面比音乐提前一格"。改为对累积角度取整再差分:
      cum_k = Σ_{i<=k} a_i (浮点),  out_k = round(cum_k) - round(cum_{k-1})
    任意前缀的整数累积和都是浮点累积和的最优整数近似, 累积误差被压在
    ±0.5°(单砖时间差 ≤ 半个 HOP_MS), 任何后段位置的到达时间不再漂。
    """
    acc = 0.0
    prev = 0
    out = []
    for a in angles:
        acc += float(a)
        c = int(round(acc))
        out.append(c - prev)
        prev = c
    return out



def plan_path_twirl(magnitudes, twirl_desire=None, model_twirl=None, turn_sign=None, step=1.0):
    """踩点驱动反推式路径规划(含 Twirl 镜像)——重写于 2026-08-22。

    核心机制(对照 SharpFAI LevelUtils.cs + 真实谱面数据验证):
      ADOFAI 计时引擎: 每格 pAngle = (curAngle - destAngle) * direction (mod 360, 不取最短弧),
      该格时长 = pAngle/180 * 拍长。Twirl 是「全局 direction 永久开关」——遇到 Twirl 后,
      后续所有格的 direction 乘 -1, 即把整条后续轨道镜像到另一侧走。

      本项目 OnsetNet 已给出每个 tile 的【真实时值(拍数)】= magnitudes[i]。由此反推:
        pAngle_i = magnitudes[i] * 180          (正值=往左转, 负值=往右转)
        dest_i   = curAngle - pAngle_i * direction   (mod 360, 绝对角)
      代入引擎公式验证: pAngle_real = (curAngle - dest_i)*direction = pAngle_i -> 每格时长
      恒等于 OnsetNet 给出的真实时值(**踩点零误差, 不依赖任何近似**)。

      Twirl 决策(不再白送、不再每格糊翻):
        Twirl 翻转 direction -> 后续所有格的「往左/往右」整体镜像。这一步才轮到
        模型/音乐决定: 当 turn_sign 想让轨道换边、或 twirl_desire(重音)/model_twirl(DDPM C2)
        强烈驱动时, 才在此处翻身。基础成本 TW_BASE 保证「无驱动纯换边」不划算。

      渲染重放(与 SharpFAI CreateFloors 一致): 每格球朝 angleData[i] 绝对方向走 step。
      方向序列 direction 由累积 Twirl 决定, 全局生效。

    参数:
      magnitudes   : list[float], 每格真实时值(拍数, 可任意 0.25/0.5/0.75/1.0/1.5...)
      twirl_desire : 可选, 每格 Twirl 重音驱动(来自 dense[2]/onset 重音), >=0
      model_twirl  : 可选, 每格 DDPM C2 通道 Twirl 概率(强驱动)
      turn_sign    : 可选, 每格模型走线方向偏置(<0=往右, >=0=往左); 提供即「模型接管」
      step         : 渲染步长(像素/单位), 仅影响几何铺开尺度, 不影响计时
    返回 (angleData, twirl_floors):
      angleData      : list[float], 每格绝对行星角(0~360, 任意 15° 步进)
      twirl_floors   : list[int], 0-based tile 下标(翻转处); tile0 永不 Twirl
    """
    import math
    # —— 评分权重 ——
    # 模型接管(有 turn_sign): 放开几何硬约束, 让模型主导「往左/往右」; Twirl 由驱动决定。
    shape_mode = turn_sign is not None
    if shape_mode:
        INTER_PENALTY = 0.0      # 允许自交(真实谱面本就交叉)
        OVER_W = 0.0             # 出屏惩罚移除(球可敞开铺满)
        # 模型方向偏置: 偏置每格往左/往右选择(接管走向)。2026-08-23 提升 4->10:
        # 之前权重过低, 被 twirl_desire(18)/model_twirl(12) 完全压制, 模型方向偏好几乎
        # 不影响几何 -> 走线和贪心(默认往左)几乎一致、聚成一团。提到 10 后模型左右偏置
        # 在非翻转格真正生效, 走出与贪心不同的形状。仍低于翻转驱动, 不抢 Twirl 决策。
        SHAPE_W = 10.0
        # 铺开激励(2026-09): 奖励让轨迹远离当前所在位置的候选(径向增益), 抑制
        # 「形状缩成一团」。step=4 下一格外向格最多 +SPREAD_W*4≈8 分, 与 SHAPE_W 同量级、
        # 低于重音驱动(18/12), 只在左右候选间挑更铺开的那个, 不抢 Twirl/重音决策。
        SPREAD_W = 2.0
        # Twirl 解耦: 翻转 direction 有基础成本, 只在重音/模型强驱动时才划算。
        TW_BASE = 8.0            # 翻转全局方向的基础成本(不再任何情况白送)
        TW_GAP_PENALTY = 30.0    # 最小间隔: 连续两格都翻 -> 严惩
        TW_DESIRE_W = 18.0       # 重音驱动(必须 > TW_BASE: 有重音才划算翻)
        TW_MODEL_W = 12.0        # DDPM C2 通道强驱动(同样 > TW_BASE)
        step = 4.0               # 铺开尺度(像素)
    else:
        INTER_PENALTY = 1e9      # 几何贪心: 自交一票否决
        OVER_W = 0.0
        SHAPE_W = 0.0
        TW_BASE = 8.0
        TW_GAP_PENALTY = 30.0
        TW_DESIRE_W = 18.0
        TW_MODEL_W = 12.0
        step = 3.0

    # —— 预计算每格转角幅度(恒正)与「直线格」标记 ——
    # 直线格: p_angle_base 为 180 的奇数倍(=1拍/3拍...)。此时 Twirl 翻转 direction 后
    # 该格 dest 数学重合(0 与 360 等价), 翻转对【当前格】几何无效; 其价值仅在【未来拐点格】
    # 显现。故直线格上的翻转不能靠当前格加分, 必须靠前瞻未来拐点驱动才划算(见前瞻逻辑)。
    pa_list = [abs(float(m)) * 180.0 for m in magnitudes]
    straight_list = [abs((pa % 360.0) - 180.0) < 0.5 for pa in pa_list]

    # —— 前瞻窗口: 翻转 direction 后, 未来 LOOK 格整体更贴合驱动/模型时才划算 ——
    LOOK = 6                      # 前瞻格数
    LOOK_W = 0.15                 # 前瞻驱动累计权重(相对当格, 低折扣防噪声累积狂翻)
    FLIP_MIN_CUR = 0.10           # 翻转当格自身必须有驱动(重音/C2)下限, 禁纯靠前瞻翻当前格

    pos = (0.0, 0.0)
    cur_angle = 0.0             # 上一格 destAngle (引擎 curAngle)
    direction = 1.0             # 全局 direction(由累积 Twirl 决定)
    segs = []
    angleData = []
    twirls = []
    last_tw = -10
    for i, mag in enumerate(magnitudes):
        mag = float(mag)
        # —— 转角幅度恒为正(0~360) ——
        # ADOFAI 引擎 pAngle = Fmod((cur_angle-dest)*direction, 360) 不做最短弧归一,
        # 故 pAngle 必须恒为正(=|mag*180|)。「往左/往右」不由 pAngle 符号决定, 而由
        # 累积 direction(=Twirl  flips)决定 —— 这正是模型 turn_sign 要控制的: 想往另一边
        # 拐, 就在合适处放 Twirl 翻转 direction。
        p_angle_base = pa_list[i]
        # 模型方向偏置(2026-09 修正语义): turn_sign[i] 是「本格几何转向」(左/右),
        # 而非「是否翻转」。要真正右转需 dir_cand=-1, dir_cand = direction*(flip?-1:1)
        #   => 想 右 转: flip = (direction > 0)
        #   => 想 左 转: flip = (direction < 0)
        #   合并: desired_flip = (turn_sign[i] < 0) == (direction > 0)
        # 旧版直接用 desired_flip = turn_sign[i] < 0, 只在 direction=+1 时碰巧正确;
        # 一旦发生过 Twirl(direction=-1), 模型说「右」代码却奖励翻转到 dir_cand=+1(左),
        # 偏好整体反转 -> 学到的形状被打成噪声 -> 左右均衡随机游走 -> 轨迹缩成一团。
        # 这就是「ShapeModel 画出来的形状缩在一起」的主根因。
        desired_flip = False
        if turn_sign is not None and i < len(turn_sign):
            desired_flip = (float(turn_sign[i]) < 0.0) == (direction > 0.0)
        # 径向增益基准(铺开激励用): 当前点到原点距离(候选评估期间 pos 不变)
        dist_cur = math.hypot(pos[0], pos[1])
        # —— Twirl 候选: 翻或不翻 direction ——
        # 不翻: 用当前 direction; 翻: direction 取反(后续全局镜像)。
        # 两候选的 dest 不同(因为 direction 不同 -> pAngle 镜像), 几何真正不同,
        # 模型/驱动才能在其中做选择。pAngle 恒正 -> 两候选计时都精确(零误差)。
        best = None
        # tile0 是出生格, 不允许放 Twirl -> 翻转候选赢了也无法兑现(direction 不翻、
        # 引擎也看不到事件), 却会让 dest 按 dir_cand=-1 反推 -> 首格 p_angle=360-p_base,
        # 计时凭空差一倍。故 i=0 只评估不翻候选(2026-09 修复)。
        for flip in ((False, True) if i > 0 else (False,)):
            dir_cand = direction * (-1.0 if flip else 1.0)
            # 引擎入向基准: curAngle = Fmod(curAngle - 180, 360) (每格先掉头180)
            cur_in = _fmod(cur_angle - 180.0, 360.0)
            # 反推绝对角: dest = cur_in - pAngle * dir_cand (mod 360)
            # 代入引擎验证: pAngle_real = Fmod((cur_in - dest)*dir_cand,360) = pAngle_base
            # (因 dir_cand^2=1) -> 每格时长 = pAngle_base/180*拍长 = |mag|*拍长 = 踩点零误差。
            dest_i = _fmod(cur_in - p_angle_base * dir_cand, 360.0)
            # 渲染重放(绝对角): 球朝 dest_i 方向走 step
            nx = pos[0] + step * math.cos(math.radians(dest_i))
            ny = pos[1] + step * math.sin(math.radians(dest_i))
            new_seg = (pos, (nx, ny))
            # 自交检查(跳过紧邻前一条共用顶点)
            inter = False
            for s in segs[:-1]:
                if _seg_intersect(s[0], s[1], new_seg[0], new_seg[1]):
                    inter = True
                    break
            dist = math.hypot(nx, ny)
            over = max(0.0, dist - 120.0)
            # Twirl 偏好(重音 + 模型 C2 驱动) —— 当格。
            # 关键修复(2026-09): td/mt 奖励只应记在【翻转候选】上 —— 重音想要的
            # 是"此处翻身", 不是"此处随便走哪边"。旧版把 -td*TW_DESIRE_W 记在
            # 翻/不翻两个候选上, 逐格比较时恒相消 -> 当格重音对翻转决策零贡献
            # (只剩约束 C 的放行门槛), Twirl 只能靠前瞻窗口累积或模型走向分歧才
            # 翻得动, 系统性晚于重音格 —— 正是"twirl 晚一格"这条实测反馈的推手之一。
            # 现仅翻转候选享受重音/C2 奖励, 兑现文档
            # "TW_DESIRE_W 必须 > TW_BASE: 有重音才划算翻"的设计意图。
            td = 0.0
            if twirl_desire is not None and i < len(twirl_desire):
                td = float(twirl_desire[i])
            mt = 0.0
            if model_twirl is not None and i < len(model_twirl):
                mt = float(model_twirl[i])
            score = (INTER_PENALTY if inter else 0.0) + over * OVER_W
            if flip:
                score -= td * TW_DESIRE_W
                score -= mt * TW_MODEL_W
            # 铺开激励(shape 模式): 径向增益为正减分(奖励外向), 为负加分(抑制回缩)。
            if shape_mode:
                score -= SPREAD_W * (dist - dist_cur)
            # 模型方向偏好: 本候选 flip 与模型期望的翻转一致 -> 减分(模型主导走向)
            if shape_mode and desired_flip == flip:
                score -= SHAPE_W
            if flip:
                score += TW_BASE
                # 最小间隔: 连续两格都翻 -> 严惩。但 shape 模式下【方向追踪式】翻转
                # (模型 sign 与当前 direction 不符, 翻转是为兑现模型走向) 是几何必需:
                # 交替符号(锯齿)本就需要每格一 Twirl(真实谱面锯齿段正是如此),
                # 间隔惩罚只应管【重音/噪声驱动的标点式】翻转 (2026-09 修正)。
                if (i - last_tw) < 2 and not (shape_mode and desired_flip == flip):
                    score += TW_GAP_PENALTY
                # —— 前瞻: 翻转后未来 LOOK 格若也强烈驱动/模型想翻, 整段受益(累计减分) ——
                # 关键修复(2026-08-23): 直线格(p=180)翻转后当前格 dest 重合, 单格无收益;
                # Twirl 的真正价值在后续拐点格(0.5/0.75拍)才显现。故翻转是否划算,
                # 须看未来窗口内驱动/模型是否同样指向翻转。前瞻累计收益压过 TW_BASE 才翻。
                look_bonus = 0.0
                for k in range(1, LOOK + 1):
                    j = i + k
                    if j >= len(magnitudes):
                        break
                    ltd = 0.0
                    if twirl_desire is not None and j < len(twirl_desire):
                        ltd = float(twirl_desire[j])
                    lmt = 0.0
                    if model_twirl is not None and j < len(model_twirl):
                        lmt = float(model_twirl[j])
                    # 未来格若也在强驱动区 -> 翻转后整段贴合, 累计减分
                    look_bonus -= (ltd * TW_DESIRE_W + lmt * TW_MODEL_W) * LOOK_W
                    # 模型模式: 本候选翻转后 direction=-direction, 未来 j 若无再翻则
                    # dir_cand_j=-direction; 其「想右转」被满足 <=> (sign_j<0)==(direction>0)
                    # (与当格 desired_flip 同式, 2026-09 修正: 旧版用 ==flip 在
                    #  direction=-1 时同样整体反转, 已随主 bug 一并修正)
                    if shape_mode and turn_sign is not None and j < len(turn_sign):
                        if (float(turn_sign[j]) < 0.0) == (direction > 0.0):
                            look_bonus -= SHAPE_W * LOOK_W
                score += look_bonus
                has_future_kink = any(not straight_list[j]
                                      for j in range(i + 1, min(len(magnitudes), i + 1 + LOOK)))
                # 约束(B): 【直线格】上翻转后当前格 dest 重合、几何无即时变化, 未来 LOOK 格
                # 内必须至少有一个「非直线拐点」, 否则翻转纯亏 -> 重罚禁止。
                # (2026-09 修正: 当格自身是拐点时, 翻转立刻改变本格走向(左<->右),
                #  有即时几何价值, 不受本约束 —— 旧版一刀切误伤拐点格翻转。)
                if straight_list[i] and not has_future_kink:
                    score += 1e9
                # 约束(C): 翻转当格自身要有驱动(重音/C2); shape 模式下「模型方向分歧」
                # 本身就是合法驱动(模型接管的题中之义), 豁免之 —— 否则非重音格永远
                # 翻不了 direction, 全图被迫单向拐(全左回环), 也是聚团根因之一
                # (2026-09 修正: 旧版一刀切禁翻, 模型在无重音处完全失声)。
                if (td + mt) < FLIP_MIN_CUR and not (shape_mode and desired_flip == flip):
                    score += 1e9
            if best is None or score < best[0]:
                best = (score, flip, dest_i, (nx, ny), new_seg)
        _, flip, dest_i, new_pos, new_seg = best
        # angleData 存最短弧表示(-180~180), 引擎内部会再 mod; 用 _shortest 保证数值稳定,
        # 避免负角被 _fmod 映射成 >180 的大角导致后续 cur_in 累积偏差。
        angleData.append(_shortest(dest_i))
        if flip and i > 0:                  # tile0 永不 Twirl(出生格)
            direction *= -1.0
            twirls.append(i)
            last_tw = i
        pos = new_pos
        segs.append(new_seg)
        cur_angle = dest_i
    return angleData, twirls


def plan_freeform_path(n, turn_sign, turn_base=12.0, target_times=None,
                       global_bpm=120.0, hop_ms=HOP_MS, pitch=1.0,
                       spiral_decay=0.0015):
    """豪放自由走线（方案 B）—— 纯 angleData 大回环，不依赖 pathData。

    为什么需要它：
      原 plan_path_twirl 把每格转角锁死成 magnitude=beats*180（时间决定角度），
      纯 angleData 下轨迹必为「90°/180° 折叠方波」。而新版本 ADOFAI 只用 angleData
      算位置、pathData 无效，所以必须让 angleData 自身形成大回环。

    做法：
      1. 每格转 turn_base*sign 的【小角度】连续转弯 -> 轨迹是自由大回环/花瓣
         （不再是急折返）；sign 由 ShapeModel 决定（左/右），模型真正主导走向。
      2. 计时陷阱：timing_engine 的 p_angle=(cur_angle-dest)*dir 会把小角度差分
         算成近整圈(338°)导致时间暴涨。解决：在【转向符号变化处】放 Twirl 翻转
         dir，使 p_angle 恒=|turn|（小角度）。
      3. 每格再用 SetSpeed(Multiplier) 把时间校准回 target_times（踩点） ->
         形状与时间彻底解耦。

    返回 (angleData, twirls, setspeeds)
      angleData  : 连续绝对方向(0~360) list[float]，引擎会 mod360，无妨
      twirls     : 计时用 Twirl 的 0-based tile 索引列表（符号变化处；发射时需 +1）
      setspeeds  : 每格 SetSpeed 事件列表（校准时间到 onset 间隔；floor 已按原生 1-based 写）

    ⚠️ 已停用：硬性约束「生成的谱子杜绝 SetSpeed」——生成一律走 plan_path_twirl，
    本函数仅作存档参考，勿在推理链路调用。
    """
    angleData = []
    twirls = []
    setspeeds = []
    # 核心：ADOFAI 计时引擎每格 cur_angle 先减 180 度(出生偏移)，故【几何转向】与
    # 【计时 p_angle】天然差 180 度。要让渲染是大回环，angleData 差分须为【小角度】
    # (+-turn -> 连续转弯)；计时 p_angle 因此=180+-turn(半圈附近)。配合 dir=sign，
    # 可令 p_angle 恒=180-|turn|，再用 SetSpeed 把时间校准到 target_times。
    # cur_dir 从 0 度起步(而非 180 度)，使首格 p_angle 也=180-|turn| 对齐后续。
    cur_dir = 0.0
    eff_sign = 1.0   # 初始 dir=+1 => 期望 sign=+1（dir 必须=sign）
    for i in range(n):
        sign = 1.0
        if turn_sign is not None and i < len(turn_sign):
            sign = -1.0 if float(turn_sign[i]) < 0 else 1.0
        decay = max(0.4, 1.0 - i * spiral_decay)   # 螺旋发散：曲率渐减 -> 圈逐渐变大铺开
        turn = turn_base * sign * decay
        cur_dir += turn          # 渲染：小角度连续转弯（螺旋发散）-> 大回环铺开
        angleData.append(_fmod(cur_dir, 360.0))
        # 计时 Twirl：本格 sign 与「当前有效 sign」不符时翻转 dir，使 dir=sign
        # （p_angle 恒=180-|turn| 半圈附近，时间合理且可校准）。
        # floor 0（第一格）ADOFAI 禁止放 Twirl：跳过首格
        if i > 0 and sign != eff_sign:
            twirls.append(i)
            eff_sign = sign
        # SetSpeed：把该格时间校准到 target_times[i]（踩点）。
        # 必须用 speedType='Bpm'（绝对覆盖），不能用 'Multiplier'（跨格累积相乘爆炸）。
        if target_times is not None and i < len(target_times):
            Ti = max(1e-3, float(target_times[i]))
            # p_angle 恒=180-|turn|：校准 bpm 使该格时间=Ti
            p_angle = 180.0 - abs(turn)
            bpm_eff = p_angle / 180.0 * 60000.0 / Ti
            setspeeds.append({
                "floor": int(i) + 1, "eventType": "SetSpeed",   # 原生 1-based floor
                "speedType": "Bpm", "beatsPerMinute": float(bpm_eff / pitch),
            })
    return angleData, twirls, setspeeds


def plan_directions(magnitudes, model_dir=None, step=1.0):
    """向后兼容包装: 仅做无 Twirl 的计时反推规划(等价于 twirl_desire 全 0)。"""
    ang, _ = plan_path_twirl(magnitudes, twirl_desire=None, model_twirl=model_dir, step=step)
    return ang


def _grid_snap_keep(kept, hop_ms, step_sec=None, max_fill_units=8):
    """把检测到的音头帧吸附到规则节拍网格并补齐漏拍，根除时序漂移/跳踩。

    网格步长优先用「音乐真实拍长」step_sec（由 BPM 给出，60/bpm）；未提供退化为中位数。
    关键修复(2026-08-13 晚): 在 [首个音头, 末个音头] 跨度内生成【完整节拍网格】(每拍一点)，
    凡是附近(≤0.5拍)没有 OnsetNet 音头的网格点一律补入 -> OnsetNet 漏检的拍 100% 被补上,
    彻底消除"跳踩"。长空隙(>max_fill_units 拍, 视为静音)不补, 避免灌水。
    保留每一个真实 onset(绝不删除), 仅去 <2 帧极近重复。
    """
    if len(kept) < 2:
        return kept
    kept = sorted(int(f) for f in kept)
    if step_sec is None or step_sec <= 1e-4:
        gaps = np.diff(kept).astype(np.float64) * (hop_ms / 1000.0)
        step_sec = float(np.median(gaps)) if len(gaps) else float((kept[1] - kept[0]) * hop_ms / 1000.0)
    if step_sec <= 1e-4:
        return kept
    hms = hop_ms / 1000.0
    t0 = kept[0] * hms
    t_end = kept[-1] * hms
    step = step_sec
    half = step * 0.5
    max_dist = max_fill_units * step
    # 完整节拍网格(覆盖整段已演奏区间)
    grid = []
    k = 0
    while True:
        t = t0 + k * step
        if t > t_end + half:
            break
        grid.append(t)
        k += 1
    if not grid:
        return kept
    onset_t = [f * hms for f in kept]
    out = []
    gi = 0
    n = len(grid)
    for ot in onset_t:
        # 插入本 onset 之前、且离任何 onset 都远(>half)的网格点 = 被漏检的拍。
        # 关键修正(2026-08-16)：填满整段网格, 不再因空隙>max_fill_units 而跳过
        # -> 根治「中间整段(如 2:00~2:10)被吞、后面直接拼接」的问题。真静音仍不灌水
        # (网格只覆盖 [首个 onset, 末个 onset] 跨度, 之前/之后皆不补)。
        while gi < n and grid[gi] < ot - half:
            gp = grid[gi]
            out.append(gp)
            gi += 1
        # 跳过与本 onset 重合(±half 内)的网格点, 避免重复
        while gi < n and grid[gi] <= ot + half:
            gi += 1
        out.append(ot)
    # 尾部网格点: 距最后一个 onset ≤max_dist 才补(避免尾奏静音灌水)
    while gi < n:
        gp = grid[gi]
        if (gp - onset_t[-1]) <= max_dist:
            out.append(gp)
        gi += 1
    out.sort()
    res = []
    for t in out:
        f = int(round(t / hms))
        if res and (f - res[-1]) < 2:
            continue
        res.append(f)
    return res


def _fill_missing_beats(kept, hop_ms, beat_sec, max_gap_beats=1.5):
    """保守漏拍填充: 保留原始 onset 间隔(精确踩点), 只在明显漏检处(间隔>max_gap_beats 拍)
    补整拍。不吸附、不拆细分(0.5/0.75 拍间隔不会触发填充, 保持原样)。"""
    if len(kept) < 2 or beat_sec <= 1e-4:
        return kept
    hms = hop_ms / 1000.0
    max_gap_frames = max_gap_beats * beat_sec / hms
    out = [kept[0]]
    for k in range(1, len(kept)):
        gap = kept[k] - kept[k - 1]
        if gap > max_gap_frames:
            # 在中间补整拍(不超过 gap 本身)
            n_fill = int(gap / (beat_sec / hms)) - 1
            for j in range(1, n_fill + 1):
                fp = kept[k - 1] + j * int(round(beat_sec / hms))
                if fp < kept[k] and (not out or fp - out[-1] >= 2):
                    out.append(fp)
        if kept[k] - out[-1] >= 2:
            out.append(kept[k])
        else:
            out[-1] = kept[k]   # 极近重复, 替换
    return out


def dense_to_adofai(dense, global_bpm=120.0, hop_ms=HOP_MS, song="generated.mp3",
                    onset_frames=None, twirl_desire=None,
                    turn_sign=None, shape_model=None, onset_prob=None,
                    stem_energy=None, device=None, merge_dup=True):
    """(3, T) -> level dict（含 angleData/settings）。无法构成有效谱面返回 None。

    onset_frames: 可选。若提供（频谱 onset 检测器给出的帧下标），直接用作谱面格
        起点（绕过 VAE 后验塌缩把稀疏 onset 抹平的问题）。每格【真实时值(拍数)】由
        onset 间隔算出, 传给 plan_path_twirl 反推绝对角(踩点零误差); 方向(左/右)与
        Twirl 由模型/音乐驱动(见 plan_path_twirl 文档)。

    计时不变式(2026-10): tile k+1 恰好踩在第 k 个 onset 上 —— 首个 onset 由 tile 1
        踩中(count-in 起手旋转), 前奏由 offset 等待吸收, 末格踩最后一个 onset。
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
    # merge_dup=False（同音多采拦截关）: 只去同帧精确重复——0 拍 tile 引擎无法表达,
    # 属格式硬约束不是拦截; 1 帧(5.8ms)以上的近距重复原样保留, 供试听。
    kept = []
    _min_gap = 2 if merge_dup else 1
    for f in frames:
        if kept and (f - kept[-1]) < _min_gap:
            continue
        kept.append(f)
    if len(kept) < 2:
        return None
    # 真实节拍步长（秒）：优先用 BPM 真拍(60/bpm)，彻底消除"帧取整把检测中位数偏置到
    # 整数帧"导致的逐格累积漂移（例：200bpm 真拍=51.677帧，round 偏成 52 帧=301.9ms，
    # 每拍 +1.9ms、整曲漂移 ~380ms）。用 BPM 真拍则每格恒=60/bpm，零漂移。
    med_sec = 60.0 / float(global_bpm) if global_bpm and global_bpm > 0 else float(hop_ms / 1000.0)
    if med_sec <= 1e-4:
        med_sec = float(hop_ms / 1000.0)
    # —— 保守漏拍填充（不吸附、不拆细分）——
    # 旧版 _grid_snap_keep 把 onset 吸附到整拍/半拍网格 -> 0.75 拍被压成 1 拍、0.5 拍被
    # 拆错 -> 踩点错位 700ms+。现改为: 保留原始 onset 间隔(精确匹配踩点), 仅在「间隔 >
    # 1.5 拍」的明显漏检处补整拍(半拍粒度不会触发, 不破坏 0.5/0.75 切分)。
    kept = _fill_missing_beats(kept, hop_ms, beat_sec=med_sec)
    if len(kept) < 2:
        return None

    # —— 计时结构(2026-10 重构): 首拍必须被踩中 ——
    # 目标不变式: 引擎 nt[k] = offset + (m_0+...+m_k)*beat_ms ≡ kept[k]*hop_ms,
    # 即 tile k+1 恰好踩在第 k 个 onset 上:
    #   m_0 = count-in 起手旋转: 关卡时钟起点(歌曲时间 offset)到首个 onset 的拍数
    #        (经典 1 拍起手; 首个 onset 本身不足 1 拍时用其全部时长, 保证 offset>=0);
    #   m_k = kept[k]-kept[k-1] (k>=1): 引向第 k 个 onset 的弧。
    # 由此第一个 onset 由 tile 1 踩中、最后一个 onset 由末格踩中(不再有尾部幻影拍)。
    # (2026-08-13 的"用实际 onset 间隔算转角"原则不变: 每格时长仍=真实间隔, 踩点零误差;
    #  长间隔已由 _fill_missing_beats 预先补整拍, 单格恒 <2 拍=360°, 引擎可精确表达。)
    # (2026-12 还原: 15° 量化层已按要求移除 —— 每格时长恢复为【原始 onset 间隔
    #  的精确拍数】, 不再吸附到 1/12 拍倍数; 谱面角度恢复 1° 整数自由(见输出侧
    #  grid=1.0)。引擎单弧上限 360°(2 拍)的封顶保留: count-in 起手弧超过 2 拍的
    #  部分引擎 Fmod 会静默砍整圈, 多余等待并入 raw_offset(出生点后移, 原生合法),
    #  首 tap 时刻不受影响。)
    beat_ms = med_sec * 1000.0
    first_ms = kept[0] * hop_ms
    # m_0 = 起手拍数: 整拍 count-in(经典 1 拍起手); 首 onset 不足 1 拍时用其全部
    # 时长(offset=0, 首 tap 即首个 onset); 超过 2 拍的部分封顶并入 offset(见上)。
    m0_beats = first_ms / beat_ms if beat_ms > 1e-6 else float(kept[0])
    if m0_beats >= 1.0:
        m0_beats = min(float(int(m0_beats)), 2.0)
    raw_offset = max(0.0, first_ms - m0_beats * beat_ms)
    # 拍数 magnitudes: m_0 + 各弧真实间隔拍数(2026-08-13 原则: 间隔=真实时值,
    # 计时不变式由此严格成立 —— 引擎按 m*180 还原时长, 踩点零误差)。
    magnitudes = [float(m0_beats)]
    for k in range(1, len(kept)):
        df = kept[k] - kept[k - 1]
        seg_sec = max(0.0, df) * hop_ms / 1000.0
        magnitudes.append(seg_sec / med_sec)

    # 方向(左/右)：左/右转音频完全相同，模型学不到，由几何路径规划指派。
    # Twirl：由「音乐重音 twirl_desire(onset 包络强度) + 模型 C2(弱偏置)」驱动，
    # 用计时反推式 plan_path_twirl 植入。无论是否 Twirl，每格时长恒等于目标转角
    # 幅度 -> 踩点误差=0（不破坏踩点；Twirl 仅做视觉镜像翻转）。
    twirl_desire_per_tile = None
    model_twirl_per_tile = None
    if twirl_desire is not None:
        td_len = len(twirl_desire)
        # twirl_desire 是按「帧号」索引的数组(与 dense[2] 同维, 长度=时间帧 T),
        # 而 kept 存放的是 onset 帧号; 故第 k 个 onset 的驱动应取 twirl_desire[kept[k]]
        # (帧号索引), 而非 twirl_desire[k] (序号)。旧版用序号 k 导致驱动错位 -> Twirl 几乎为 0
        # (2026-08-23 修复: 此处与 inference 传 dense[2] 的帧号语义对齐)。
        # 槽位对齐(2026-10 重构): 重音 k 的 Twirl 必须挂在【踩中该重音的 tile k+1】上
        # (触发时刻 nt[k] = kept[k]*hop_ms, 与重构前的绝对触发时刻完全一致), 即驱动放
        # magnitudes 索引 k+1 = 离开 tile k+1 的弧。索引 0 是 count-in 出生弧(无重音,
        # 且 plan_path_twirl 本就禁止 tile0 翻转)置 0; 末位重音的弧不在规划范围, 弃用。
        twirl_desire_per_tile = np.array(
            [0.0] + [float(twirl_desire[kept[k]]) if 0 <= kept[k] < td_len else 0.0
                     for k in range(len(kept) - 1)],
            dtype=np.float32)
        if C >= 3:
            twirl_ch = dense[2]
            cl = twirl_ch.shape[0]
            model_twirl_per_tile = np.array(
                [0.0] + [float(twirl_ch[kept[k]]) if 0 <= kept[k] < cl else 0.0
                         for k in range(len(kept) - 1)],
                dtype=np.float32)

    # —— 摆形状模型：用模型决定每格左右(取代纯几何贪心) ——
    # turn_sign 优先用外部传入；否则若给了模型+onset 概率，则在此推断。
    # 2026-08-23 升级: 传入 magnitudes, 使几何上下文通道(8~11)在推理期也有真值,
    # 模型才能基于"当前几何状态"决定往左/往右, 真正接管走向(而非和贪心一样)。
    turn_sign_arg = turn_sign
    if turn_sign_arg is None and shape_model is not None and onset_prob is not None:
        try:
            from shape_model import extract_tile_features, predict_turn_sign
            _se = stem_energy if stem_energy is not None else np.zeros((6, len(onset_prob)), np.float32)
            _feats = extract_tile_features(kept, onset_prob, _se, float(global_bpm), hop_ms,
                                           magnitudes=magnitudes)
            if _feats.shape[0] >= 2 and _feats.shape[1] == 12:
                turn_sign_arg = predict_turn_sign(shape_model, _feats,
                                                  device if device else "cpu")
            else:
                print(f"[shape] 特征维度异常({_feats.shape}), 退回几何贪心")
                turn_sign_arg = None
        except Exception as e:
            print(f"[shape] 转向推断失败, 退回几何贪心: {e}")
            turn_sign_arg = None

    # —— 走线：踩点驱动反推框架(2026-08-22 重写) ——
    # magnitudes 传【每格真实时值(拍数)】, plan_path_twirl 反推绝对角 + Twirl 镜像。
    # 有 turn_sign(模型接管)时模型决定每格往左/往右; 否则几何贪心(默认往左)。
    if turn_sign_arg is not None:
        angleData, shape_twirls = plan_path_twirl(
            magnitudes,
            twirl_desire=twirl_desire_per_tile,
            model_twirl=model_twirl_per_tile,
            turn_sign=turn_sign_arg,
        )
        # 原生 .adofai floor 即 0-based tile 序号(ADOFAI-JS/AutoCat 源码证实,
        # 见 timing_engine 事件循环注释): plan_path_twirl 返回 0-based tile 索引
        # i, 直接发射 floor=i, 引擎/游戏读回 parsed[i]=离开 tile i 的弧, 恰好兑现规划。
        actions = [{"floor": int(f), "eventType": "Twirl"} for f in shape_twirls if int(f) > 0]
    else:
        angleData, twirls = plan_path_twirl(
            magnitudes,
            twirl_desire=twirl_desire_per_tile,
            model_twirl=model_twirl_per_tile,
            turn_sign=None,
        )
        # 把规划出的 Twirl 写成事件（plan_path_twirl 返回 0-based tile 索引 i,
        # 原生 floor 即 0-based -> 直接发射 floor=i)。tile0 永不放 Twirl(见
        # plan_path_twirl 的 `flip and i>0`)，此处再防御性过滤。
        actions = [{"floor": int(f), "eventType": "Twirl"} for f in twirls if int(f) > 0]

    settings = {
        "bpm": float(global_bpm), "pitch": 100, "offset": 0, "song": song,
        "songArtist": "", "songName": song, "difficulty": 1, "volume": 100,
        "audioOffset": 0, "timeScale": 1.0, "mirror": 0, "flip": 0,
    }
    # —— offset = 前奏等待(见上方 magnitudes 块推导): 关卡时钟(首格 count-in 旋转起点)
    # 落在歌曲时间 raw_offset, 音乐从 0 播起; 加上 m_0=count-in 后 tile 1 恰踩 kept[0]。
    # 旧版 offset=kept[0]*hop_ms(把首个 onset 当出生点) + magnitudes 从 g_0 起 —— 两者
    # 合成"全谱踩点推后一格", 已随 2026-10 重构一并废除; ±3000 旧安全夹同废(真实语料
    # offset 最大 42.4s, 长等待原生合法, 旧夹反而把长前奏谱整体夹错位)。
    # —— 真相公式(2026-12, v2 点击测试 4/4 实证) ——
    # 引擎语义: 歌曲时间 = 关卡时间 - 1拍 + offset, 即音乐在关卡时间「1拍-offset」
    # 处开播。
    # —— 起手角恒 0(硬性约束: 每首歌起手 angleData 必须是 0, 根据偏移调整) ——
    # 做法: 全谱角度同减 R=angleData[0] (整体旋转坐标系, 起手格归 0=出生朝向)。
    # 引擎计时只看相邻角差 (cur_angle-dest): 同减常数后差值不变 -> 第 2 格起所有
    # 弧的耗时/踩点分毫不动; 唯一变化是第一格(起手弧): 引擎 cur_angle 进弧后恒
    # 180°, 原 dest_0=180*(1-m_0) 给出 p_angle=m_0*180(=m_0 拍), 现 dest_0=0 ->
    # p_angle 恒 180(=1 拍)。m_0=1 的经典整拍起手角度本来就是 0, 行为完全不变;
    # m_0<1 (首 onset 不足一拍, 如 0.288 拍原起手角 128.1°) 起手弧并到 1 拍;
    # m_0>1 时多余等待并入 offset, 引擎原生合法。
    # 时间补偿推导(旋转后): nt_song[k] = (1+m_1+...+m_k)*beat - 1拍 + offset
    #   = (m_1+...+m_k)*beat + offset ≡ kept[k]*hop_ms = first_ms + (m_1+...+m_k)*beat
    #   -> offset = first_ms(=kept[0]*hop_ms, 首 onset 毫秒), 对任意 m_0 严格成立。
    # (旧公式 offset=raw_offset+1拍 与 first_ms 恰差 (1-m_0)拍, 即起手弧时长变化量。)
    if angleData:
        _R = float(angleData[0])
        if abs(_R) > 1e-9:
            angleData = [_shortest(float(a) - _R) for a in angleData]
    settings["offset"] = int(round(first_ms))

    # 角度输出 = 采样原值(硬性约束: 取消"精度是 1 的自动吸附",
    # 采出来是多少就是多少)。plan_path_twirl 的 angleData 本就是从每格真实
    # 时值(拍数)精确反推的浮点绝对角, 每格 pAngle=|mag|*180 严格成立, 踩点
    # 零量化误差 —— 不取整 = 无量化误差, 引擎计时与规划完全一致。
    # 历史后处理在此一并停用(两个函数保留定义, 供存档参考):
    #   _cumulative_round_diff: 累积和取整再差分, 当年修"逐角度独立 round
    #     的量化误差随机游走"(越走越快) —— 根因正是取整本身;
    #     现在完全不取整, 量化误差为零, 该 bug 无从发生。
    #   _cumulative_snap(grid=1.0): 1° 吸附, 整数输入下本就是 no-op, 一并移除。
    # (15° 吸附 2026-08-27 已取消; 1° 取整今日取消 —— 自此角度=采样原值。)
    return {
        # 原始浮点角度直出: 值域同 plan_path_twirl 输出(最短弧表示 -180~180,
        # 引擎读入自会 mod360, 见 plan_path_twirl 内 _shortest 注释)。
        "angleData": [float(a) for a in angleData],
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
    # 自测：(1) 往返非退化 (2) 计时不变式: 每个检测 onset 恰被一踩(nt[k]≈kept[k]帧)
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
    # 提取 bump 中心(局部极大, 阈值 0.9 避开高斯肩部: 肩部±1帧=0.80 也会过 0.5 阈,
    # 且肩部间隔恰为 2 帧逃过 <2 合并 -> 双倍伪 onset); frame0 是出生格标记(非踩点)。
    onsets = [int(f) for f in range(1, T - 1)
              if d[0][f] > 0.9 and d[0][f] >= d[0][f - 1] and d[0][f] > d[0][f + 1]]
    # 复刻 dense_to_adofai 内部管线(合并 <2 帧重复 + 漏拍填充), 得到对齐基准 kept
    kept = []
    for f in onsets:
        if kept and (f - kept[-1]) < 2:
            continue
        kept.append(f)
    kept = _fill_missing_beats(kept, HOP_MS, beat_sec=60.0 / 120.0)
    print("onset centers:", onsets, "-> kept(合并+补拍):", kept)
    back = dense_to_adofai(d, global_bpm=120.0, onset_frames=onsets)
    ad = back["angleData"]
    L = sum(1 for a in ad if a < 0)   # 左转(负)数
    R = sum(1 for a in ad if a > 0)   # 右转(正)数
    print("reconstructed angleData:", ad)
    print("non-degenerate (both signs present):", L > 0 and R > 0,
          f"(L={L}, R={R})")
    print("offset:", back["settings"]["offset"])
    # 不变式1: 每 onset 一格(首拍不被出生格吞掉)
    print("tile count == kept count:", len(ad) == len(kept),
          f"(adN={len(ad)}, kept={len(kept)})")
    # 不变式2: 引擎回算, 第 k 个 tap(nt[k]) 恰落在第 k 个 onset 上
    # (旧版 nt[k]≈kept[k+1]: 首拍踩空+尾部幻影拍 -> 踩点整体推后一格)
    nt = compute_note_times(ad, back["settings"], back["actions"], add_offset=True)
    errs = [abs(nt[k][0] - kept[k] * HOP_MS) for k in range(len(kept))]
    max_err = max(errs) if errs else 0.0
    first_err = errs[0] if errs else 0.0
    print(f"per-onset tap max_err={max_err:.2f}ms, first-tap err={first_err:.2f}ms",
          "OK" if max_err < 40.0 else "FAIL")
    # 不变式3: 起手角恒 0 (硬性约束: 每首歌第一个 angleData 必须是 0)
    print("first angle == 0:", ad[0] == 0.0, f"(ad[0]={ad[0]!r})")
    print("OK dense_to_adofai round-trip")
