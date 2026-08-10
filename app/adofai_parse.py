"""
宽容解析 ADOFAI .adofai 文件
===========================
.adofai 名义是 JSON, 但实际数据里常见:
  - UTF-8 BOM
  - 尾随逗号 (trailing comma)
  - 字段缺失 (settings/actions/angleData/pathData 任意缺)
  - 只用 pathData 不用 angleData
这里尽量宽容: json5 解析 + 标准 json 回退 + 去尾逗号兜底 + 异常兜底。
单个文件解析失败返回 None, 由调用方跳过, 不中断批量训练。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    import json5
except ImportError:
    json5 = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timing_engine import compute_note_times

# pathData 字符 -> 角度 (抄自 pyadofai_main, 完整表, 零依赖)
PATH_TO_ANGLE = {
    'R': 0, 'p': 15, 'J': 30, 'E': 45, 'T': 60, 'o': 75, 'U': 90, 'q': 105,
    'G': 120, 'Q': 135, 'H': 150, 'W': 165, 'L': 180, 'x': 195, 'N': 210,
    'Z': 225, 'F': 240, 'V': 255, 'D': 270, 'Y': 285, 'B': 300, 'C': 315,
    'M': 330, 'A': 345, '!': 999,
}


def _clean_json(text: str) -> str:
    """反复应用一组宽容变换, 修掉 ADOFAI 编辑器常见的非标准 JSON 结构。

    变换(顺序无关, 迭代到稳定):
      - 折叠多余逗号:  ", ," / ",,," -> ","
      - 去尾随逗号:    "x": 1, } / ] -> "x": 1 } / ]
      - 去括号间逗号:  }, ] -> } ]
      - 补缺失逗号(同级元素间漏逗号):
            } { -> }, {      ] { -> ], {
            } [ -> }, [      ] [ -> ], [
      注: 合法 JSON 中 '}' 与 '{' / '[' 之间必已有逗号, 故上述补逗号不会误伤合法文件。
    """
    cur = text
    for _ in range(12):
        nxt = cur
        nxt = re.sub(r",\s*,+", ",", nxt)                       # 折叠多余逗号
        nxt = re.sub(r",(\s*[}\]])", r"\1", nxt)                # 去尾随逗号
        nxt = re.sub(r"([}\]])\s*,(\s*[}\]])", r"\1\2", nxt)    # 去括号间逗号
        nxt = re.sub(r"\}\s*\{", "}, {", nxt)                  # 补缺失逗号
        nxt = re.sub(r"\]\s*\{", "], {", nxt)
        nxt = re.sub(r"\}\s*\[", "}, [", nxt)
        nxt = re.sub(r"\]\s*\[", "], [", nxt)
        nxt = re.sub(r"\}\s*\"", "}, \"", nxt)                 # } 后直接跟 "键" (缺逗号)
        nxt = re.sub(r"\]\s*\"", "], \"", nxt)                 # ] 后直接跟 "键" (缺逗号)
        if nxt == cur:
            break
        cur = nxt
    return cur


def loads_adofai(text: str):
    """从字符串宽容解析, 失败返回 None。

    处理 ADOFAI 编辑器常见的非标准 JSON:
      - UTF-8 BOM (调用方已用 utf-8-sig 读, 这里再兜底一次)
      - 双/多余逗号 ("difficulty": 1, ,  <- 某些编辑器会多打一个逗号)
      - 尾随逗号 ("bpm": 86, } / ] )
      - 括号间多余逗号 (}, ])
      - 缺失逗号 (对象/数组元素之间漏逗号: } { / ] [ 等)
      - 字符串内裸控制字符 (极端个案, 最后手段替换成空格)
    优先 json.loads, 其次 json5(若有), 不行再走兜底清洗链 + 控制字符兜底。
    """
    if not text or not text.strip():
        return None
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    candidates = [text, _clean_json(text)]
    # 最后手段: 字符串内裸控制字符替换成空格(仅在前述都失败时使用)
    ctrl = re.sub(r"[\x00-\x1f]", " ", text)
    if ctrl != text:
        candidates.append(_clean_json(ctrl))
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            pass
        if json5 is not None:
            try:
                return json5.loads(c)
            except Exception:
                pass
    return None


def load_adofai(path: str):
    try:
        raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return None
    return loads_adofai(raw)


def _to_angle_data(adofai):
    """优先 angleData; 只有 pathData 时按字符映射转; 都没有返回 None。"""
    if not isinstance(adofai, dict):
        return None
    angle = adofai.get("angleData")
    if angle:
        return angle
    path = adofai.get("pathData")
    if isinstance(path, str) and path.strip():
        try:
            return [PATH_TO_ANGLE.get(ch, 0) for ch in path]
        except Exception:
            return None
    if isinstance(path, list):
        return path
    return None


def extract_targets(adofai, T: int, hop_ms: float):
    """从 adofai dict 提取 (onset_frames, interval) 或 None。"""
    if not isinstance(adofai, dict):
        return None
    settings = adofai.get("settings") or {}
    if not isinstance(settings, dict):
        settings = {}
    actions = adofai.get("actions") or []
    if not isinstance(actions, list):
        actions = []
    angle = _to_angle_data(adofai)
    if angle is None:
        return None
    try:
        raw = compute_note_times(angle, settings, actions, add_offset=True)
    except Exception:
        return None
    if not raw:
        return None
    # compute_note_times 返回 (时间ms, 标志) 元组列表
    times = [float(x[0]) if isinstance(x, (tuple, list)) else float(x) for x in raw]
    onset = [0.0] * T
    interval = [0.0] * T
    for i, t in enumerate(times):
        f = int(round(float(t) / hop_ms))
        if 0 <= f < T:
            onset[f] = 1.0
            nxt = times[i + 1] if i + 1 < len(times) else (float(t) + hop_ms)
            interval[f] = max(0.05, (float(nxt) - float(t)) / 1000.0)
    return onset, interval


def validate_adofai(path: str) -> bool:
    """快速判断一个 .adofai 是否可解析并能提取到音符时间。"""
    ad = load_adofai(path)
    if not isinstance(ad, dict):
        return False
    angle = _to_angle_data(ad)
    if angle is None:
        return False
    settings = ad.get("settings") or {}
    actions = ad.get("actions") or []
    try:
        times = compute_note_times(
            angle,
            settings if isinstance(settings, dict) else {},
            actions if isinstance(actions, list) else [],
            add_offset=True)
    except Exception:
        return False
    return bool(times)
