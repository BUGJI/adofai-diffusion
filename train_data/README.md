# train_data/ — 训练数据目录（不入库）

本目录用于存放你自己的训练数据，默认是空的。
Docker / compose 会把这里挂载到容器内 `/app/train_data`；
本地训练用环境变量 `ADOFAI_TRAIN_DIR` 指到任意路径也可以。

## 数据格式（train_onset / stage2 通用）

每一首歌 = 一个子文件夹，里面放**同名**的音频和谱面：

```
train_data/
├── MySongA/
│   ├── MySongA.mp3        # 或 .ogg / .wav（优先级 ogg > mp3 > wav）
│   └── MySongA.adofai     # ADOFAI 官方谱面文件
└── MySongB/
    ├── MySongB.ogg
    └── MySongB.adofai
```

要求：
- 子目录名可以随意，但音频与 `.adofai` 文件名主名最好一致（取各自排序第一个）。
- 一首歌只要音频+谱面齐全就会被自动收集，无需清单文件。
- `--train_dir` 支持多目录：用系统路径分隔符（Windows 是 `;`，Linux 是 `:`）
  分隔，例如 melody 一套、vocal 一套分开训练再合起来。

## 训练命令示例

```bash
# 拍点模型（onset）
python app/training/train_onset.py --train_dir ./train_data --epochs 80

# 谱面扩散主模型（VAE + DDPM 一起练）
python app/training/train_stage2.py          # 读 ADOFAI_TRAIN_DIR，默认 ./train
```

## 版权提醒

仓库不附带任何音乐和谱面。请只使用你有权使用的音频做训练。
