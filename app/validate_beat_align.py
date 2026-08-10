"""
validate_beat_align.py — 干净的"踩点"校验（修正 off-by-one）

正确语义（来自 timing_engine.compute_note_times）：
  nt[i][0] = 关卡时间下，到达 tile (i+1) 的累计时间。
  tile 0（出生点）在关卡时间 0，对应歌曲时间 = settings['offset']。
  故：tile k 的歌曲时间 = offset + (nt[k-1][0] 当 k>=1) ，tile 0 = offset。
  第 k 个 onset 帧 = onset_frames[k]。
  => 正确比对：tile k 的歌曲时间 应 ≈ onset_frames[k] * HOP_MS。

旧推理管线的校验把 nt[i]（到 tile i+1）误比 onset_frames[i]（onset i），
差一拍，制造出 ~341ms 的"漂移"假象。本脚本做两件事：
  A) 用【正确】比对（tile k <-> onset k）算最大误差；
  B) 用【旧】比对（nt[i] <-> onset i）复现旧脚本的数字，证明那是测试索引错位。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from onset_detector import detect_onsets
from chart_repr import HOP_MS
from timing_engine import compute_note_times


def main():
    if len(sys.argv) < 3:
        print("usage: validate_beat_align.py <audio.ogg> <chart.adofai> [bpm]")
        return
    audio = sys.argv[1]
    chart = sys.argv[2]

    level = json.load(open(chart, encoding="utf-8"))
    ad = level["angleData"]
    settings = level["settings"]
    actions = level.get("actions", [])
    offset = int(settings.get("offset", 0))
    bpm = float(settings.get("bpm", 120))

    # 复现推理期的 onset 检测（与大佬方案推理期完全一致）
    det_times, _, _ = detect_onsets(audio, wait=4, avg_win=24, k=1.0)
    T_full = int(round(det_times[-1] * 1000.0 / HOP_MS)) + 64 if det_times else 0
    onset_frames = sorted({int(round(t * 1000.0 / HOP_MS)) for t in det_times})
    print(f"[info] bpm={bpm} offset={offset} tiles={len(ad)} "
          f"detector_onsets={len(onset_frames)}")

    # 真实引擎逐格到达时间（含 offset）
    nt = compute_note_times(ad, settings, actions, add_offset=True)
    # nt[i][0] = 到 tile (i+1) 的歌曲时间；tile 0 在歌曲时间 = offset。
    # 构造 tile_k_songtime[k] = tile k 的歌曲时间
    tile_songtime = [offset] + [nt[i][0] for i in range(len(ad))]

    # ---- A) 正确比对：tile k <-> onset k ----
    errs_correct = []
    for k in range(min(len(onset_frames), len(tile_songtime))):
        target = onset_frames[k] * HOP_MS
        errs_correct.append(abs(tile_songtime[k] - target))
    max_correct = max(errs_correct) if errs_correct else 0.0
    mean_correct = sum(errs_correct) / len(errs_correct) if errs_correct else 0.0

    # ---- B) 旧比对（复现旧推理管线 off-by-one）----
    errs_old = []
    for i, f in enumerate(onset_frames):
        if i < len(nt):
            errs_old.append(abs(nt[i][0] - f * HOP_MS))
    max_old = max(errs_old) if errs_old else 0.0

    print()
    print("="*60)
    print("A) 正确比对 (tile k 的到达时间 vs 第 k 个 onset):")
    print(f"   max_tile_error = {max_correct:.3f} ms   mean = {mean_correct:.3f} ms")
    print("   => 若 ≈0，说明谱面真正卡在音乐节拍上")
    print("-"*60)
    print("B) 旧比对 (复现旧推理管线: nt[i] vs onset i):")
    print(f"   max_tile_error = {max_old:.3f} ms")
    print("   => 若较大，证明那是测试索引 off-by-one，非谱面 bug")
    print("="*60)

    # 逐格前 12 格打印正确比对，肉眼核对
    print("\n   k | song_ms | onset_ms | err_ms")
    for k in range(min(12, len(onset_frames), len(tile_songtime))):
        print(f"  {k:2d} | {tile_songtime[k]:8.2f} | {onset_frames[k]*HOP_MS:8.2f} | "
              f"{abs(tile_songtime[k]-onset_frames[k]*HOP_MS):7.2f}")

    verdict = "PASS (卡点)" if max_correct < 20 else "CHECK"
    print(f"\n结论: {verdict}  | 正确比对 max={max_correct:.2f}ms  旧比对 max={max_old:.2f}ms")


if __name__ == "__main__":
    main()
