"""separate_all.py — 一次分离全部音轨并导出可播放 wav，供网页「第二步」列出试听。

用法: python separate_all.py --audio <wav> --outdir <dir> --prefix <uid>
输出到 stdout 的 JSON（唯一输出）:
  {"ok": true, "stems": [{"name","label","file","duration_s"}, ...]}
  {"ok": false, "error": "..."}

必须在 venv 里跑（需要 torch + demucs）。分离只跑一次（htdemucs），
再合成 full(原) / accomp(原-人声)。

分离实现收拢到 separation.separate_stems（与 preview_track 共用一份，
替代此前散落的「加载 -> 44100 立体声 -> apply_model -> 重采样对齐」逻辑，避免行为漂移）。
"""
from __future__ import annotations
import os, sys, json, argparse

ROOT = os.path.dirname(os.path.abspath(__file__))          # app/training/
APP = os.path.dirname(ROOT)                                # app/
for p in (APP, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from separation import separate_stems, write_wav, SR
from device_util import get_safe_device

# 第二步要列出的「分解音轨」（不含原音频；原音频由前端用上传文件直接播放）
STEMS_OUT = [
    ("drums", "鼓点 drums"),
    ("bass", "贝斯 bass"),
    ("other", "旋律 / 钢琴 other"),
    ("vocals", "人声 vocals"),
    ("accomp", "伴奏（去人声）accomp"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", required=True)
    a = ap.parse_args()
    try:
        # 优先 GPU；若显卡架构不被打包的 PyTorch 支持
        # （CUDA error: no kernel image is available for the device），自动退回 CPU。
        # 退回后分离变慢但可用，避免老显卡用户直接报错。
        device = get_safe_device()
        try:
            stems = separate_stems(a.audio, device=device)
        except Exception as e:
            if device != "cuda":
                raise
            sys.stderr.write(f"[separate] GPU 分离失败({e})，退回 CPU\n")
            sys.stderr.flush()
            stems = separate_stems(a.audio, device="cpu")

        out = []
        for key, label in STEMS_OUT:
            y = stems[key]
            fname = f"{a.prefix}_{key}.wav"
            fpath = os.path.join(a.outdir, fname)
            write_wav(fpath, y)
            out.append({
                "name": key, "label": label, "file": fname,
                "duration_s": round(len(y) / SR, 2),
            })
        print(json.dumps({"ok": True, "stems": out}))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))


if __name__ == "__main__":
    main()
