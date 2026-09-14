"""preview_track.py — 分离单条音轨并导出可播放 wav，供网页「试听」与「选单轨生成」。

用法: python preview_track.py --audio <wav> --track <name> --out <out.wav>
输出到 stdout 的 JSON（唯一输出）:
  {"ok": true,  "track": "vocals", "duration_s": 123.45}
  {"ok": false, "error": "..."}

必须跑在 venv 里（需要 torch + demucs）。
契约对应 web_server.py 的 _run_stem_separate / preview_track()：
  --track all/full  -> 直接转写原音频（无需分离）
  --track vocals 等  -> Demucs 分离后取该轨（22050Hz 单声道 wav）

（背景：本脚本在 separation.py 重构时被误删未重建，导致网页端
「试听单轨」与「选单轨生成」两功能直接失败——2026-09 修复补回。
实现收拢到 separation.py 公共实现，避免与 demucs_mel/separate_all 三处漂移。）
"""
from __future__ import annotations
import os, sys, json, argparse

ROOT = os.path.dirname(os.path.abspath(__file__))          # app/training/
APP = os.path.dirname(ROOT)                                # app/
for p in (APP, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import librosa

from separation import separate_stems, write_wav, SR
from device_util import get_safe_device

# 网页/推理端可能传来的音轨别名 -> Demucs stem 名
_TRACK_ALIAS = {
    "vocal": "vocals", "voice": "vocals", "voc": "vocals",
    "mel": "other", "melody": "other",
    "inst": "accomp", "instrumental": "accomp",
    "mix": "full", "original": "full",
}
_VALID = {"drums", "bass", "other", "vocals", "full", "accomp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--track", default="all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        track = (a.track or "all").strip().lower()
        track = _TRACK_ALIAS.get(track, track)

        # all/full：无需分离，直接把原音频转成 22050 单声道 wav
        if track in ("all", "full", ""):
            y, _ = librosa.load(a.audio, sr=SR, mono=True)
            write_wav(a.out, y)
            print(json.dumps({"ok": True, "track": "all",
                              "duration_s": round(len(y) / SR, 2)}))
            return

        if track not in _VALID:
            # 未知轨名不炸链路：退回原音频全混合（与 inference 端的宽容回退一致）
            y, _ = librosa.load(a.audio, sr=SR, mono=True)
            write_wav(a.out, y)
            print(json.dumps({"ok": True, "track": "all", "fallback": True,
                              "duration_s": round(len(y) / SR, 2)}))
            return

        device = get_safe_device()
        try:
            stems = separate_stems(a.audio, device=device)
        except Exception as e:
            if device != "cuda":
                raise
            # 老显卡 no-kernel-image / 显存不足：退回 CPU 再试一次（慢但能用）
            sys.stderr.write(f"[preview] GPU 分离失败({e})，退回 CPU\n")
            sys.stderr.flush()
            stems = separate_stems(a.audio, device="cpu")

        y = stems[track]
        write_wav(a.out, y)
        print(json.dumps({"ok": True, "track": track,
                          "duration_s": round(len(y) / SR, 2)}))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))


if __name__ == "__main__":
    main()
