# ADOFAI Diffusion

**用扩散模型自动生成《A Dance of Fire and Ice》(ADOFAI) 节奏谱面的本地工具。**

输入一首歌（mp3 / ogg / wav / flac / m4a / aac），输出可以直接进游戏游玩的 `.adofai` 谱面文件。
整条流水线（音源分离、踩点检测、扩散生成、摆形状、视觉特效）都在本地跑，不上传任何数据。

- 界面：原生 GUI（Windows / WebView2）+ 网页端（跨平台，Docker 可直接部署）
- 模型：Demucs(htdemucs) + Beat This! + OnsetNet(CNN+BiGRU) + ChartVAE + DDPM + ShapeModel + VFXNet
- 权重：7 个 `.pt`（约 45 MB）随仓库发布，克隆即用

---

## 一、快速开始

### 1) Windows 源码直跑

```bat
git clone https://github.com/<你的用户名>/adofai-diffusion.git
cd adofai-diffusion
py -3.13 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe app\web_server.py --port 8420
```

浏览器打开 `http://127.0.0.1:8420`。或者直接双击 `run旧版网页端.bat`（它会找同目录的 `venv\Scripts\python.exe`）。

> venv 的 `pyvenv.cfg` 用相对路径找 base 解释器，**必须从项目根目录启动**，
> 否则报 "did not find executable"（GUI 已自动 chdir，手动跑注意一下）。

### 2) 原生 GUI（仅 Windows）

```bat
venv\Scripts\python.exe gui\gui_main.py
```

需要 `comtypes`（在 `requirements.txt` 里，标了 `sys_platform == 'win32'`）和系统自带的 WebView2 Runtime（Win11 已内置）。
GUI 只做窗口壳，生成/训练逻辑全部复用 `app/web_server.py`，改 `gui/`、`app/` 源码即时生效。

### 3) Docker（Linux，CPU 推理）

```bash
git clone https://github.com/<你的用户名>/adofai-diffusion.git
cd adofai-diffusion
docker compose up -d --build
# 或手动：
#   docker build -t adofai-diffusion .
#   docker run -d -p 8420:8420 -v ./data:/app/data -v ./train_data:/app/train_data adofai-diffusion
```

打开 `http://localhost:8420`。

要点：

- 镜像只装依赖 + 源码 + `beat_this` 权重（8 MB）；`data/`、`train_data/` 用 volume 挂载，
  容器里对应 `ADOFAI_DATA_DIR=/app/data`、`ADOFAI_TRAIN_DIR=/app/train_data`。
- 训练出来的权重写进挂载卷，重启不丢。
- 有 NVIDIA GPU 时：把 `requirements.txt` 里的 `torch/torchaudio` 换成 cu 索引版本，
  并在 compose 里加 `deploy.resources.reservations.devices`（`Dockerfile` 顶部有注释）。
- `ffmpeg` / `libsndfile1` 已装在镜像里，不用额外准备。

---

## 二、界面开关说明

| 开关 | 含义 |
|---|---|
| 视觉特效 (VFX) | 默认关闭。开启后注入 MoveTrack/Flash/SetFilter… 等事件 |
| 同音多采拦截 | 默认开启。合并过密的同音 onset（35 ms 内成簇、70 ms 内串联），避免同一音被判成多踩 |
| 分轨模式 | 用 melody / vocal 专用 OnsetNet，并只喂对应声源的 mel |

---

## 三、训练

数据格式：**每首歌一个子目录，音频与谱面同名配对**：

```
train_data/
└── 歌名A/
    ├── 歌名A.ogg
    └── 歌名A.adofai
```

```bash
# 踩点模型
python app/training/train_onset.py --train_dir train_data
# 风格层（VAE + DDPM），数据目录用 env 指定
ADOFAI_TRAIN_DIR=train_data python app/training/train_stage2.py
# 摆形状模型
python app/training/train_shape.py --data train_data
# 特效模型
python app/training/train_vfx.py --data train_data --cache train_data/.vfx_cache
```

产物默认进运行时目录 `checkpoints/`，网页端和 GUI 的训练页签有实时日志监控。
GPU 显存不够时，`train_stage2.py` 支持用 `VAE_EPOCHS` / `DDPM_EPOCHS` 控制轮数。

---

## 四、模型清单（`data/checkpoints/`）

| 权重文件 | 模型 | 作用 | 必需性 |
|---|---|---|---|
| `onset_net.pt` | OnsetNet (CNN+GRU) | 每帧 onset 概率（踩点） | **核心** |
| `vae.pt` + `ddpm.pt` | ChartVAE + DDPM | 风格层（事件热图生成） | **核心** |
| `onset_net_melody.pt` | OnsetNet 变体 | 分轨踩点（旋律轨） | 可选，缺则回退标准模型 |
| `onset_net_vocal.pt` | OnsetNet 变体 | 分轨踩点（人声轨） | 可选，缺则回退标准模型 |
| `shape_model.pt` | ShapeModel (GRU) | 每格左/右转（摆形状） | 可选，缺则几何贪心 |
| `vfx_net.pt` | VFXNet (多任务头) | 视觉特效预测 | 可选，缺则跳过注入 |

---

## 五、权重与路径解析（`app/paths.py`）

`resolve_checkpoint(filename)` 是唯一的加载入口，优先级：

```
1) ADOFAI_DATA_DIR           显式指定（容器挂载 / 覆盖）
2) 运行时目录  %LOCALAPPDATA%\ADOFAI_Diffusion\checkpoints   ← 你自己训练的权重
   （Linux/macOS 下取 ~/，容器里 compose 已把 LOCALAPPDATA 指到 /app/data）
3) 便携内置    data/checkpoints                               ← 出厂权重（不随 env 变）
```

配套规则：

- **训练产物**一律写运行时目录（env 优先），便携 `data/` 视为只读 —— 相关脚本：
  `train_onset.py` / `train_stage2.py` / `train_vfx.py` / `train_shape.py`。
- **推理加载**（`inference_stage2.py` / `apply_vfx.py`）、**状态展示**
  （`web_server._model_status_payload`、`gui_main.send_model_status`）都走同一个 resolver。
- 语义是「内置 vs 已训练」分层：你训练出来的权重优先生效；删掉运行时目录即恢复出厂。
  前端模型状态徽章会标注来源（内置 / 已训练）。

容器/多实例常用环境变量：

| 变量 | 读取处 | 作用 |
|---|---|---|
| `ADOFAI_DATA_DIR` | `app/paths.py` | 权重与 demucs 缓存根目录 |
| `ADOFAI_TRAIN_DIR` | `app/training/train_stage2.py` 等 | 训练数据目录 |
| `LOCALAPPDATA` | `app/paths.py`、`web_server.py` | 运行时目录（日志 / 预览图 / 训练产物） |
| `TORCH_HOME` | 启动时注入子进程 | beat_this 权重缓存位置 |
| `HF_HOME` | `app/separation.py` | HuggingFace 缓存 |
| `ADOFAI_GUI_OPAQUE` | `gui/gui_main.py` | 关闭窗口透明效果 |

---

## 六、目录结构

```
adofai-diffusion/
├── app/                        # 全部核心源码
│   ├── web_server.py           # ★ 入口：零依赖 http.server 后端（生成/分离/训练 API）
│   ├── paths.py                # ★ 路径与权重解析的唯一真相
│   ├── onset_detector.py       # 频谱通量 onset 检测（预览图）
│   ├── adofai_parse.py         # 宽容解析 .adofai（BOM / 尾逗号 / pathData）
│   ├── timing_engine.py        # ADOFAI 计时引擎忠实移植（SharpFAI GetNoteTimes）
│   ├── effects_schema.py       # 视觉特效字段唯一真相源（转录自 ADOFAI-JS）
│   ├── webui/                  # 网页版前端（index.html / vfx_train.html / shape_monitor.html）
│   └── training/               # 模型定义 / 训练 / 推理
│       ├── inference_stage2.py # ★ 生成主链路（web_server 以子进程调用）
│       ├── chart_repr.py       # 稠密谱面 ↔ .adofai 落谱 + 路径规划
│       ├── apply_vfx.py        # VFX 注入
│       └── train_*.py          # onset / stage2(vae+ddpm) / shape / vfx
├── gui/                        # 原生 GUI 壳（Windows：Win32 Mica 窗口 + WebView2）
├── data/checkpoints/           # 出厂权重（7 个 .pt）
├── torch_hub/                  # beat_this 的 ONSET 权重（免首跑联网）
├── train_data/                 # ← 训练数据放这里（见第三节）
├── requirements.txt            # GPU/CPU 通用（torch 默认 cu128 轮子）
├── requirements-cpu.txt        # 纯 CPU 环境（配合 pytorch cpu 索引）
├── Dockerfile / docker-compose.yml
└── run旧版网页端.bat           # Windows 网页端启动器
```

---

## 七、生成流水线（`app/training/inference_stage2.py`）

全链路统一 **hop=128 网格（≈5.805 ms/帧 @22050 Hz）**，踩点 / 扩散 / 落谱共用同一网格，无换算：

```
音频 → ffmpeg/librosa 解码
  ├─ Demucs(htdemucs) 分离 6 通道 mel：drums/bass/other/vocals/full/accomp
  ├─ [可选] Beat This! (ISMIR 2024) 估 BPM / 节拍相位条件
  ├─ OnsetNet(CNN+BiGRU) 踩点 → onset 帧（分轨模式：选中轨复制填充 6 通道）
  ├─ full-mix mel + onset 包络 → DDPM(CFG 2.5, 25~50 步, 分块 4096 帧)
  │     → VAE 潜空间(16ch, T/16) 采样 → 解码回 (3,T) 稠密谱面
  ├─ dense_to_adofai 落谱：
  │     每格时值(拍数) = onset 间隔 → plan_path_twirl 反推绝对角
  │     （pAngle = 拍数×180，踩点零误差；Twirl 由重音 + 模型 C2 打分决策）
  │     方向(左/右) 由 ShapeModel(12 维特征含几何上下文) 或几何贪心决定
  └─ [可选] apply_vfx：VFXNet 帧级预测吸附到方块，注入 19 类视觉事件
→ .adofai（回填 songName/songFilename、轨道淡入淡出、offset = 首个 onset）
```

---

## 八、关键设计决策（踩坑沉淀）

- **角度由「转换器」确定性反推**（pAngle = 拍数 × 180），模型不碰角度 → 绕开绝对角度回归塌缩；
  方向（左/右）音频上不对称、学不好，交给几何规划 / ShapeModel 决定。
- **每格时长恒等于 onset 间隔** → 踩点零误差；`offset` = 首 onset 时刻（tile0 对齐）。
- **不做网格吸附**：把 onset 强行对齐到节拍网格会听感发糊，实测劣化，已永久废弃。
- `VAE β=1e-4 + free_bits=0.5` 防后验塌缩；DDPM UNet 必须有时间步嵌入（否则中段 NaN）。
- `device_util.get_safe_device()`：老显卡（GTX 10 系）`torch.cuda.is_available()` 会假阳性，
  必须真跑一次内核再信；失败自动回退 CPU。
- **不生成 `SetSpeed`**：变速事件与「每格 = onset 间隔」的计时不变量冲突，输出前一律剔除。
- `timing_engine.py` 逐行对应 SharpFAI：floor 是 1-based（事件索引要 `fl-1`）、
  999 = mid-spin（deltaTime=0）、Twirl/Pause/Hold/MultiPlanet/FreeRoam 全部进计时。
  盘上 `.adofai` 原生事件 floor 为 1-based（floor 1 ↔ `angleData[0]`），内部索引一律 0-based。
- `effects_schema.py` 是特效字段唯一真相源（转录自 ADOFAI-JS），杜绝非法字段被游戏忽略。
- Windows 11 24H2 移除了 `wmic` → 杀训练进程改用 PowerShell CIM，Linux 下走 `pkill`。

---

## 九、已知边界

- GUI 壳依赖 Win32/WebView2，**只有 Windows 能用**；跨平台请用网页端或 Docker。
- CPU 推理可用但慢（一首 3 分钟的歌，分离 + 扩散约数分钟量级）。
- 生成的谱面质量强依赖训练语料；开箱权重是在有限曲目上训练的，换曲风建议自行微调。
- `.adofai` 的 `angleData[0]` 必须是 0（出生格朝向），落谱器已保证，二次编辑时别改。

---

## 十、许可与致谢

- 本项目源码采用 **Apache-2.0** 许可（见 `LICENSE`）。
- 依赖与转录的上游项目见 `NOTICE`，各自许可仍归上游。
- 特别提示：**Demucs 预训练权重**与其上游数据受音乐版权约束，本仓库仅按上游条款引用；
  用分离结果训练出的模型权重如需再分发，请自行确认合规。
- 本项目与 ADOFAI 官方无隶属关系。游戏本体版权归 ADOFAI 团队所有。
