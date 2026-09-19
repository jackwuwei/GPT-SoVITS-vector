# Vector 音色克隆：GPT-SoVITS Fine-tune 完整记录

把 Vector 机器人的英文音色克隆出来，用于 wire-pod 中文（及多语言）TTS。Apple Silicon (M-series) 本地训练，最终生成的中文语音具有 Vector 的童声/机器人音色特征。

> **实验日期**：2026-05-04
> **总耗时**：约 90 分钟（其中训练约 50 分钟，调试约 40 分钟）
> **结果**：✅ 成功，e30 SoVITS + e50 GPT 输出听感"像 Vector"
>
> **目录说明**：实验时本仓库嵌套在 `wire-pod/gpt-sovits/` 下，现已独立为 `~/workspace/gpt-sovits`。文中 `gpt-sovits/...` 指本仓库根目录，`samples/...` 指本仓库下的 `samples/`；Genie-TTS 零样本实验在 `experiments/genie-tts/`。

---

## 1. 背景

wire-pod 自带的 edge-tts 用真人女声念中文，与 Vector 原生音色（Acapela 合成的高音童声）差距大。Vector 自身的 TTS 不支持中文且不开源。需要在保留 Vector 音色特征的前提下让它说中文。

### 走过的弯路

| 方案 | 结果 | 原因 |
|---|---|---|
| edge-tts + DSP（pitch shift、ring mod、telephone bandpass）| ❌ | 神经 TTS + 滤镜永远是"加滤镜的真人"，缺机器人质感 |
| eSpeak-NG 共振峰合成 | ❌ | 机器人感对了但中文发音生硬 |
| Genie-TTS 零样本（带 ONNX 推理）| ❌ | 它的 reference audio 只迁移情绪/语调、不迁移音色 |
| GPT-SoVITS 上游零样本 | ❌ | 7.8 秒英文参考对 Vector 这种"非自然"音色不够 |
| **GPT-SoVITS Fine-tune** | ✅ | 本文档记录的方案 |

---

## 2. 数据准备

### 2.1 录音

让 Vector 通过 wire-pod SDK 念预设的英文短句，旁边用手机录音。两段录音合并：

- `samples/vector.m4a`（48 秒）—— Rainbow Passage 前 4 句
- `samples/vector2.m4a`（228 秒）—— Rainbow Passage 续段 5 句 + Harvard Sentences 5 句 + Vector 原生口头禅 6 句

驱动脚本（让 Vector 念 16 句）：`samples/say_record_script.sh`，调用 wire-pod 的 SDK API：

```
GET  http://escapepod.local/api-sdk/assume_behavior_control?serial=<S>&priority=high
GET  http://escapepod.local/api-sdk/say_text?serial=<S>&text=<TEXT>
GET  http://escapepod.local/api-sdk/release_behavior_control?serial=<S>
```

> wire-pod 默认监听 80 端口（不是 8080）。SDK API 在 `chipper/pkg/wirepod/sdkapp/server.go`。

**注意点**：

- Vector 的 `SayText` 长句会**阻塞**直到念完，使 curl 超时 10 秒 —— 实际 16 句录下来 228 秒（远长于估算的 78 秒），但语音内容完整
- 录音设置：手机 m4a, 48kHz stereo, 134kbps AAC

### 2.2 切片 + 转录

48s 录音里有效语音 ~22s，新录的 228s 里有效语音 ~50s。**总计 70 秒、20 句**，刚到 GPT-SoVITS 的 1 分钟下限。

切片用 `ffmpeg silencedetect`（阈值 d=1.5s/-35dB），脚本和片段时间在 `samples/segments.txt`。20 个 wav 文件输出到 `samples/dataset/`：

| 段 | 内容 | 时长 |
|---|---|---|
| orig_01..04 | Rainbow Passage 1-4（来自 vector.m4a）| 4-8s 各 |
| A1..A5 | Rainbow Passage 5-9 | 4-6s 各 |
| B1..B5 | Harvard Sentences 5 句 | 2-3s 各 |
| C1..C6 | "Hello, my name is Vector"、"Sure, why not" 等 6 句 | 1.5-3s 各 |

> 切片时 C 组短句容易被静音检测器切碎，C6 出现 part1/part2 拆分（part1 是噪声，丢掉）。需要听一遍人工确认对齐。

### 2.3 训练用 list 文件

GPT-SoVITS 期待的格式：`<wav_path>|<spk>|<lang>|<text>`，路径用绝对路径避免 cwd 歧义：

```
samples/dataset/orig_01.wav|vector|en|When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.
...
samples/dataset/C6.wav|vector|en|Goodbye!
```

写入 `gpt-sovits/dataset/vector.list`。

---

## 3. 环境

### 3.1 Python venv

```bash
cd ~/workspace
git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS.git gpt-sovits
cd gpt-sovits
python3.11 -m venv .venv
. .venv/bin/activate

# 国内必须切镜像（PyPI 直连超时拖死整个安装）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

pip install -r requirements.txt -r extra-req.txt
pip install torchcodec  # torchaudio 2.11 需要它做 audio I/O
```

### 3.2 模型下载（HF 镜像必备）

```bash
export HF_ENDPOINT=https://hf-mirror.com

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='lj1995/GPT-SoVITS',
    local_dir='GPT_SoVITS/pretrained_models',
    allow_patterns=[
        'chinese-hubert-base/*',
        'chinese-roberta-wwm-ext-large/*',
        'gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt',
        'gsv-v2final-pretrained/s2G2333k.pth',
        'gsv-v2final-pretrained/s2D2333k.pth',
    ],
)
"
```

约 1.06 GB，HF 镜像 + xethub CDN 速度 60-100 MB/s。

### 3.3 NLTK 资源（英文 G2P 用）

```bash
python -c "import nltk; [nltk.download(p) for p in ('averaged_perceptron_tagger_eng','averaged_perceptron_tagger','cmudict','punkt','punkt_tab')]"
```

### 3.4 fastText 语言检测模型缓存目录

```bash
mkdir -p GPT_SoVITS/pretrained_models/fast_langdetect
```

不预创会在第一次"中英混合"推理时报 `FileNotFoundError`。模型本身（lid.176.bin, 125MB）会自动下载到这。

### 3.5 MPS 验证

```python
import torch
assert torch.backends.mps.is_available()  # M-series Mac 必须 True
```

---

## 4. 预处理（3 stage）

`gpt-sovits/preprocess_vector.sh`：

| Stage | 脚本 | 输出 |
|---|---|---|
| 1a | `prepare_datasets/1-get-text.py` | `logs/vector/2-name2text.txt`（音素序列）+ `logs/vector/3-bert/`（中文用，英文为空）|
| 1b | `prepare_datasets/2-get-hubert-wav32k.py` | `logs/vector/4-cnhubert/*.pt`（HuBERT 特征）+ `logs/vector/5-wav32k/*.wav`（重采样到 32k）|
| 1c | `prepare_datasets/3-get-semantic.py` | `logs/vector/6-name2semantic.tsv`（语义 token）|

关键环境变量：

```bash
export PYTHONPATH="$PWD:$PWD/GPT_SoVITS:$PYTHONPATH"   # text 包要在 GPT_SoVITS 下
export PYTORCH_ENABLE_MPS_FALLBACK=1                   # MPS 不支持的算子 fallback CPU
export inp_text=dataset/vector.list
export inp_wav_dir=""                                   # list 用绝对路径就留空
export exp_name=vector
export opt_dir=logs/vector
export i_part=0 all_parts=1 _CUDA_VISIBLE_DEVICES=0
export is_half=False                                    # MPS fp16 不稳，强制 fp32
export version=v2
```

3 stage 总耗时：~5-10 分钟。

---

## 5. 训练

### 5.1 配置

`train_vector.py` 读 `configs/s2.json` + `configs/s1longer-v2.yaml`，patch 后写到 `TEMP/tmp_s{1,2}.{json,yaml}`：

| 参数 | 值 | 说明 |
|---|---|---|
| `batch_size` | 4 | 20 句小数据集 + MPS 内存 |
| `fp16_run` / `precision` | `False` / `"32"` | MPS 不要 fp16 |
| `epochs` (SoVITS) | 30 | 最终选 e30 |
| `epochs` (GPT) | 50 | 最终选 e50 |
| `save_every_epoch` | 5 | 每 5 个 epoch 一份 ckpt |
| `if_save_latest` | True | 自动 resume 关键 |
| `if_save_every_weights` | True | 推理用的轻量权重保存到 `SoVITS_weights_v2/` & `GPT_weights_v2/` |
| `lora_rank` | 32 | SoVITS 默认 |
| `data.num_workers` (s1) | 0 | macOS fork DataLoader 死锁见下面 |

### 5.2 命令

```bash
. .venv/bin/activate
python train_vector.py
```

`train_vector.py` 串联跑：

1. `python -s GPT_SoVITS/s2_train.py --config TEMP/tmp_s2.json`
2. `python -s GPT_SoVITS/s1_train.py --config_file TEMP/tmp_s1.yaml`

### 5.3 时间消耗（M-series MPS）

| 阶段 | 耗时 |
|---|---|
| SoVITS e1-e15 | 18 分钟（~2.85s/batch × 26 batch × 15 epoch）|
| SoVITS e16-e30（续训）| 18 分钟 |
| GPT e1-e20 | **6 分钟**（GPT 比 SoVITS 快很多）|
| GPT e21-e50（续训）| 7 分钟 |
| **合计** | **49 分钟** |

> 注：SoVITS 在 MPS 上跑（GPU），CPU 占用 0% 是正常的。GPT 部分在 CPU 上多线程跑（因为 Lightning + MPS 兼容性问题，部分算子 fallback），CPU 占用 100-200%。

---

## 6. 踩坑记录

### 6.1 DataLoader 死锁（macOS + MPS + num_workers > 0）

**症状**：训练卡在 `epoch 1, batch 7/26`，CPU=0%、内存不增长，`sample` 显示主线程在 `select_poll_poll` 等子进程数据。

**原因**：macOS 的 fork 多进程 DataLoader 与 MPS 张量分配冲突。

**修法**：
- `GPT_SoVITS/s2_train.py:119-127` 把 hardcoded `num_workers=5` 改成 MPS 时强制 0：

```python
_macos_workers = 0 if (os.environ.get("MACOS_SAFE_DATALOADER") or torch.backends.mps.is_available()) else 5
train_loader = DataLoader(
    train_dataset,
    num_workers=_macos_workers,
    shuffle=False,
    pin_memory=False if _macos_workers == 0 else True,
    collate_fn=collate_fn,
    batch_sampler=train_sampler,
    persistent_workers=(_macos_workers > 0),
    prefetch_factor=3 if _macos_workers > 0 else None,
)
```

- `GPT_SoVITS/AR/data/data_module.py:48-69` 同样的逻辑。
- `train_vector.py` patch s1 config 时加 `cfg["data"]["num_workers"] = 0`。

### 6.2 `prefetch_factor` 与 `num_workers=0` 冲突

**症状**：DataLoader 报 `ValueError: prefetch_factor option could only be specified in multiprocessing.`

**原因**：上面修了 num_workers 但 `prefetch_factor=16` 和 `persistent_workers=True` 在 num_workers=0 时不合法。

**修法**：见 6.1，已经把这两个改成条件 `(nw > 0)`。

### 6.3 PyTorch 2.6+ `weights_only=True` 默认值

**症状**：续训时 Lightning load_checkpoint 报 `WeightsUnpickler error: Unsupported global: GLOBAL pathlib.PosixPath was not an allowed global by default`。

**原因**：PT 2.6 默认 `torch.load(weights_only=True)`，Lightning ckpt 里有 `pathlib.PosixPath` 不在 safe globals。

**修法**：`GPT_SoVITS/s1_train.py` 顶部加：

```python
import pathlib
torch.serialization.add_safe_globals([pathlib.PosixPath])
```

### 6.4 Vector SayText 长句阻塞 + curl 超时

**症状**：录音脚本中 A 组（每句 ~7 秒）每次 curl 都超时 10 秒。

**影响**：录音里 A 组每句之间多了 ~10 秒静音。**对训练无影响**（自动切片会跳过空白段），但录音整体时长翻倍。

### 6.5 fastText `Cache directory not found`

**症状**：第一次推理 `中英混合` 模式时挂掉。

**修法**：手动 `mkdir GPT_SoVITS/pretrained_models/fast_langdetect`。

### 6.6 NLTK 资源缺失

**症状**：`Resource 'averaged_perceptron_tagger_eng' not found`。

**修法**：跑一次 `nltk.download('averaged_perceptron_tagger_eng')`。

### 6.7 PyPI 直连卡死

**症状**：`pip install genie-tts` 跑了 14 分钟无任何输出，CPU 0%、网络无活动。

**修法**：清华镜像（见 §3.1）。

---

## 7. 推理

### 7.1 命令

```bash
. .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH

python GPT_SoVITS/inference_cli.py \
  --gpt_model    GPT_weights_v2/vector-e50.ckpt \
  --sovits_model SoVITS_weights_v2/vector_e30_s780.pth \
  --ref_audio    samples/vector_ref_best.wav \
  --ref_text     ref_text.txt \
  --ref_language 英文 \
  --target_text  target_text.txt \
  --target_language 中英混合 \
  --output_path  output
```

`ref_text.txt` 内容是 `vector_ref_best.wav` 对应的英文转录；`target_text.txt` 是想合成的中文（含英文专有名词时用 `中英混合` 语种）。

### 7.2 性能

| 指标 | 值 |
|---|---|
| 输出采样率 | 32 kHz mono |
| 推理耗时（CPU + 缓存）| 5-6 秒 |
| 输出长度（5 秒中文）| 4-6 秒 |
| 模型加载时间（首次） | ~3-5 秒 |

接 wire-pod 时输出 32k 要重采样到 16k（沿用 edge-tts 现有的 `downsample` 流水线，把 24k→16k 改成 32k→16k）。

### 7.3 结果对比

| 输出 | 说明 |
|---|---|
| `samples/vector_ref_best.wav` | Vector 原声参考（8.2 秒，Rainbow Passage 第 3 句）|
| `gpt-sovits/output/output_finetuned.wav` | e15 SoVITS + e20 GPT —— "部分像，需要再训" |
| `gpt-sovits/output/output_e30e50.wav` | **e30 SoVITS + e50 GPT —— 听感"像 Vector"** ✅ |

---

## 8. 关键产出物

```
samples/
├── vector.m4a                       # 原始录音 1（48s）
├── vector2.m4a                      # 原始录音 2（228s）
├── vector_24k_mono.wav              # 录音 1 转换
├── vector2_24k_mono.wav             # 录音 2 转换
├── vector_ref_best.wav              # 8.2s 最干净段（推理用 reference）
├── say_record_script.sh             # 让 Vector 念 16 句的脚本
├── segments.txt                     # 录音 2 的切片时间表
└── dataset/                         # 20 个切好的训练 wav
    ├── orig_01..04.wav
    ├── A1..A5.wav
    ├── B1..B5.wav
    └── C1..C6.wav

gpt-sovits/
├── dataset/vector.list              # 20 行 transcript（训练用）
├── preprocess_vector.sh             # 3 stage 预处理 driver
├── train_vector.py                  # SoVITS + GPT 训练 driver
├── train_gpt_only.py                # 仅续训 GPT
├── ref_text.txt                     # 推理参考文本
├── target_text.txt                  # 推理目标文本
├── logs/vector/                     # 预处理产物（特征、token）
├── SoVITS_weights_v2/
│   ├── vector_e5_s130.pth
│   ├── ...
│   └── vector_e30_s780.pth          # ⭐ 当前最佳
├── GPT_weights_v2/
│   ├── vector-e5.ckpt
│   ├── ...
│   └── vector-e50.ckpt              # ⭐ 当前最佳
└── output/
    ├── output_finetuned.wav         # e15/e20 测试
    └── output_e30e50.wav            # e30/e50 测试 ✅
```

---

## 9. 接下来要做

1. **包成 HTTP 服务**：参考 `gpt-sovits/api_v2.py`，起一个常驻 FastAPI，提供 `POST /tts` 接口（输入 text + lang，返回 wav 字节）
2. **接进 wire-pod**：在 `chipper/pkg/wirepod/ttr/` 加 `sovits.go`，仿照 `edgetts.go` 的结构调 HTTP，并把现有的 24k→16k 改成 32k→16k
3. **`apiConfig.json` 加 TTS provider**：`knowledge.tts_provider` 增加 `"sovits"` 分支（同 `"edge-tts"` / `"openai"`）
4. **可选优化**：
   - 用 `inference_webui_fast.py` 路径替代 cli（更快的流式推理）
   - 把 e30/e50 模型转 ONNX，喂回 Genie-TTS 这套 ONNX 推理流水线，绕开 PyTorch 依赖
   - 录更多 Vector 中文语音（如果以后 Vector 固件支持）做 fine-tune V2，去掉跨语言这一步
