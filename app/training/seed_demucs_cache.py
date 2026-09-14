r"""seed_demucs_cache.py — 为 vfx_all 合并目录（junction）预映射 Demucs 缓存键。

train_vfx.py 的 Dataset 走 junction 路径（vfx_all\<song>\<audio>）调用 demucs_mel()，
缓存键 = sha1(abspath|size|mtime|C6H128)。此前踩点/扩散训练用的是真实路径
（melody\<song>\ / vocal\<song>\），键不同 -> train_vfx 会对 264 首全部重新分离。

junction 指向同一物理目录，文件 size/mtime 完全一致，只有 abspath 不同。
本脚本对每首歌：
  真实键  = sha1("<real_dir>\<song>\<audio>|size|mtime|C6H128")
  合并键  = sha1("<vfx_all>\<song>\<audio>|size|mtime|C6H128")
若缓存目录里存在 真实键.npy 且缺 合并键.npy，则建硬链接（同卷，零磁盘开销）。

对 portable 与 runtime 两个缓存目录分别做（train_vfx 用哪个取决于环境，
两个都铺好最稳）。只补缺失的，绝不覆盖已有的。
"""
import os
import sys
import hashlib
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])   # 项目根(app/training/ 上两级), 不再硬编码盘符
VFX_ALL = os.path.join(ROOT, "train_single", "vfx_all")
KINDS = ["melody", "vocal"]
CACHE_DIRS = [
    os.path.join(ROOT, "data", "demucs_cache"),
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "ADOFAI_Diffusion", "demucs_cache"),
]
AUDIO_EXTS = (".ogg", ".mp3", ".wav")


def audio_of(folder):
    """与 extract_vfx._audio_of 完全一致的选音频规则：排序后第一个。"""
    fs = [f for f in os.listdir(folder)
          if f.lower().endswith(AUDIO_EXTS)]
    if fs:
        return sorted(fs)[0]
    return None


def key_of(abspath, size, mtime):
    h = hashlib.sha1(
        f"{abspath}|{size}|{mtime:.3f}|C6H128".encode()
    ).hexdigest()[:16]
    return h


def main():
    songs = sorted(os.listdir(VFX_ALL))
    print(f"[seed] vfx_all 歌曲目录: {len(songs)}")
    pairs = []  # (real_dir, song, audio, size, mtime)
    for song in songs:
        jdir = os.path.join(VFX_ALL, song)
        if not os.path.isdir(jdir):
            continue
        audio = audio_of(jdir)
        if not audio:
            print(f"  [skip] {song}: 无音频")
            continue
        # 找到真实父目录（melody 或 vocal）
        real_dir = None
        for kind in KINDS:
            rd = os.path.join(ROOT, "train_single", kind, song)
            if os.path.isdir(rd):
                real_dir = rd
                break
        if real_dir is None:
            print(f"  [skip] {song}: 找不到真实目录")
            continue
        ap = os.path.join(jdir, audio)
        st = os.stat(ap)
        pairs.append((real_dir, song, audio, st.st_size, st.st_mtime))

    print(f"[seed] 可配对歌曲: {len(pairs)}")
    total_linked = 0
    for cache_dir in CACHE_DIRS:
        if not os.path.isdir(cache_dir):
            print(f"[seed] 缓存目录不存在，跳过: {cache_dir}")
            continue
        n_link = n_have = n_miss = 0
        for real_dir, song, audio, size, mtime in pairs:
            real_key = key_of(os.path.join(real_dir, audio), size, mtime)
            junc_key = key_of(os.path.join(VFX_ALL, song, audio), size, mtime)
            src = os.path.join(cache_dir, real_key + ".npy")
            dst = os.path.join(cache_dir, junc_key + ".npy")
            if not os.path.exists(src):
                n_miss += 1
                continue
            if os.path.exists(dst):
                n_have += 1
                continue
            try:
                os.link(src, dst)
                n_link += 1
            except OSError as e:
                # 硬链接失败（跨卷/权限）退化为复制
                try:
                    import shutil
                    shutil.copy2(src, dst)
                    n_link += 1
                except Exception as e2:
                    print(f"  [err] {song}: {e2}")
        print(f"[seed] {cache_dir}")
        print(f"      新建硬链接 {n_link}，已有 {n_have}，缺源 {n_miss}")
        total_linked += n_link
    print(f"[seed] 完成，共新建 {total_linked} 个缓存映射")


if __name__ == "__main__":
    main()
