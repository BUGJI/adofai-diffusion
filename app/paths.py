"""paths.py — 集中管理 data/ 目录、权重名与训练目录（跨平台唯一真相来源）。

所有脚本（web_server / training/* / 测试）统一从这里取路径，
避免各文件散落的 'data/checkpoints' 魔法字符串与 "+ \"/\"" 拼接。
路径与名称也可被环境变量覆盖，便于容器挂载与部署定制。
"""
import os
import sys
from pathlib import Path

# 项目根：本文件在 app/ 下 -> 上一级
APP_DIR = Path(__file__).resolve().parent          # app/
ROOT = APP_DIR.parent                               # 项目根（portable / 仓库根）

# data 根：可用环境变量 ADOFAI_DATA_DIR 覆盖（容器里可指向挂载卷）
DATA_DIR = Path(os.environ.get("ADOFAI_DATA_DIR", str(ROOT / "data")))

CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
PREVIEW_DIR = DATA_DIR / "preview"
TRAIN_LOG = DATA_DIR / "train.log"
DEMUCS_CACHE_DIR = DATA_DIR / "demucs_cache"

# 便携内置权重目录：永远指向 ROOT/data/checkpoints，不随 ADOFAI_DATA_DIR 变。
# 这是「出厂权重」的兜底来源——即便 GUI 把 ADOFAI_DATA_DIR 指到运行时目录，
# resolve_checkpoint 的最后一级也要能回退到这里（修复：此前用 CHECKPOINTS_DIR
# 作回退档，env 设置时它会跟着变，回退等于失效）。
PORTABLE_CHECKPOINTS_DIR = ROOT / "data" / "checkpoints"

# 权重文件名（训练/推理/评测共用的唯一真相）
CKPT_ONSET = "onset_net.pt"
CKPT_ONSET_MELODY = "onset_net_melody.pt"
CKPT_ONSET_VOCAL = "onset_net_vocal.pt"
CKPT_VAE = "vae.pt"
CKPT_DDPM = "ddpm.pt"
CKPT_VFX = "vfx_net.pt"
CKPT_SHAPE = "shape_model.pt"

# 训练数据根：可用环境变量 ADOFAI_TRAIN_DIR 覆盖
TRAIN_DIR_DEFAULT = str(ROOT / "train")

# ── 运行时数据目录（与 web_server / gui_main 保持一致）───────────────────
# 便携目录（ROOT/data）只放出厂权重，只读；用户训练产出 / 缓存 / 预览 / 日志
# 一律写到运行时目录（%LOCALAPPDATA%\ADOFAI_Diffusion），装到 Program Files
# 等无写权限位置时也能正常工作。
RUNTIME_DIR = Path(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")) / "ADOFAI_Diffusion"
RUNTIME_CHECKPOINTS_DIR = RUNTIME_DIR / "checkpoints"
RUNTIME_SHAPE_FEAT_CACHE = RUNTIME_DIR / "shape_feat_cache"


def train_ckpt_dir():
    """训练输出 checkpoints 目录：显式 ADOFAI_DATA_DIR > 运行时目录。

    便携 data/checkpoints 只放出厂权重，不应被训练覆盖（真正「内置 vs 已训练」分层）。
    """
    env = os.environ.get("ADOFAI_DATA_DIR")
    if env:
        return Path(env) / "checkpoints"
    return RUNTIME_CHECKPOINTS_DIR


def resolve_checkpoint(filename):
    """统一权重解析（推理/状态展示的唯一真相）：按优先级找第一个「存在且非空」的副本。

    顺序：
      1) ADOFAI_DATA_DIR 显式指定（容器挂载 / 显式覆盖）
      2) 运行时目录（用户训练出的权重）
      3) 便携内置目录（随包发布的出厂权重，不随 env 变）

    返回 Path；找不到或 <1KB（视为损坏）返回 None。
    """
    dirs = []
    env = os.environ.get("ADOFAI_DATA_DIR")
    if env:
        dirs.append(Path(env) / "checkpoints")
    dirs.append(RUNTIME_CHECKPOINTS_DIR)
    dirs.append(PORTABLE_CHECKPOINTS_DIR)
    seen = set()
    for d in dirs:
        d = Path(d)
        key = os.path.normcase(str(d))
        if key in seen:
            continue
        seen.add(key)
        p = d / filename
        try:
            if p.is_file() and p.stat().st_size >= 1024:
                return p
        except OSError:
            continue
    return None


def ensure_dirs():
    """确保 data 相关目录存在（幂等）。"""
    for d in (DATA_DIR, CHECKPOINTS_DIR, PREVIEW_DIR, DEMUCS_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def venv_python(portable_root, platform=None):
    """返回 venv 解释器路径；Windows 在 Scripts/，POSIX 在 bin/。

    portable_root: 项目根（str 或 Path）。
    platform: 显式指定平台（测试用）；缺省取当前平台。
    """
    platform = platform or sys.platform
    if platform == "win32":
        return os.path.join(str(portable_root), "venv", "Scripts", "python.exe")
    return os.path.join(str(portable_root), "venv", "bin", "python")
