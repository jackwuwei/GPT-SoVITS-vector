# Vector 音色克隆：GPT-SoVITS Fine-tune 完整记录

把 Vector 机器人的英文音色克隆出来，用于 wire-pod 中文（及多语言）TTS。Apple Silicon (M-series) 本地训练，最终生成的中文语音具有 Vector 的童声/机器人音色特征。

> **当前模型**：v2 数据集，SoVITS e30 + GPT e30（`vectorv2_e30_s780.pth` / `vectorv2-e30.ckpt`），由 epoch sweep 逐个试听选出
> **部署**：`serve.py`（FastAPI `POST /tts`，返回 32kHz mono WAV），通过 `docker-compose.cpu.yaml` 跑在 CPU 主机上；wire-pod 侧用 `tts_provider=gpt-sovits` 接入
>
> **目录说明**：文中路径均相对本仓库根目录。Genie-TTS 零样本实验在 `experiments/genie-tts/`。

---

## 1. 背景

wire-pod 自带的 edge-tts 用真人女声念中文，与 Vector 原生音色（Acapela 合成的高音童声）差距大。Vector 自身的 TTS 不支持中文且不开源。需要在保留 Vector 音色特征的前提下让它说中文。

### 走过的弯路

| 方案 | 结果 | 原因 |
|---|---|---|
| edge-tts + DSP（pitch shift、ring mod、telephone bandpass）| ❌ | 神经 TTS + 滤镜永远是"加滤镜的真人"，缺机器人质感 |
| eSpeak-NG 共振峰合成 | ❌ | 机器人感对了但中文发音生硬 |
| Genie-TTS 零样本（带 ONNX 推理）| ❌ | 它的 reference audio 只迁移情绪/语调、不迁移音色 |
| GPT-SoVITS 上游零样本 | ❌ | 几秒英文参考对 Vector 这种"非自然"音色不够 |
| GPT-SoVITS Fine-tune，v1 数据（麦克风录 Vector 喇叭）| ⚠️ | 手机录音带房间噪声和喇叭失真，且合成参数用错（100/100/100），听感像普通男童声。数据已从仓库删除 |
| **GPT-SoVITS Fine-tune，v2 数据（直接合成的干声）** | ✅ | 本文档记录的方案 |

---

## 2. 数据准备（v2）

数据集在 `samples/vector_sovits_dataset_v2/`，详细说明见该目录下的 `README.md`。

### 2.1 来源

不再用麦克风录机器人，而是用 Vector 同款的 **Acapela BABILE 引擎 + Bendnn 声库**直接合成干声，合成参数与 Vector 真机 `tts_config.json` 一致：**speed=80, pitch=100, shape=130**。这样得到的是没有喇叭味、没有环境噪声、但 timbre 与真机一致的训练数据。

> 如果想要"喇叭味"也一致，思路是让模型先学干净的 timbre，推理后再叠真机喇叭的 IR/EQ，而不是用录音训练。

### 2.2 规格

| 项 | 值 |
|---|---|
| 片段数 | 20 段（`clips/vector_001.wav` .. `vector_020.wav`）|
| 总时长 | 115 秒（每段 5.4-9.7 秒）|
| 格式 | 32 kHz mono PCM 16-bit |
| 峰值 | -3 dBFS |
| 文本 | `phrases.tsv`（英文日常句 + 绕口令式覆盖音素的句子）|
| 全部拼接试听 | `PREVIEW_ALL.wav` |

### 2.3 训练用 list 文件

GPT-SoVITS 期待的格式：`<wav_path>|<spk>|<lang>|<text>`。`dataset/vectorv2.list` 里的路径相对仓库根目录（`preprocess_vector.sh` 会先 `cd` 到仓库根）：

```
samples/vector_sovits_dataset_v2/clips/vector_001.wav|vector|en|Hi there. My name is Vector. It's very nice to meet you today.
...
samples/vector_sovits_dataset_v2/clips/vector_020.wav|vector|en|...
```

`samples/vector_sovits_dataset_v2/vector.list` 是同一份清单、路径相对数据集目录，方便把数据集单独拷走（如 Colab）。

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

`preprocess_vector.sh`（默认 `EXP_NAME=vectorv2`、`LIST=dataset/vectorv2.list`）：

| Stage | 脚本 | 输出 |
|---|---|---|
| 1a | `prepare_datasets/1-get-text.py` | `logs/vectorv2/2-name2text.txt`（音素序列）+ `logs/vectorv2/3-bert/`（中文用，英文为空）|
| 1b | `prepare_datasets/2-get-hubert-wav32k.py` | `logs/vectorv2/4-cnhubert/*.pt`（HuBERT 特征）+ `logs/vectorv2/5-wav32k/*.wav`（重采样到 32k）|
| 1c | `prepare_datasets/3-get-semantic.py` | `logs/vectorv2/6-name2semantic.tsv`（语义 token）|

关键环境变量：

```bash
export PYTHONPATH="$PWD:$PWD/GPT_SoVITS:$PYTHONPATH"   # text 包要在 GPT_SoVITS 下
export PYTORCH_ENABLE_MPS_FALLBACK=1                   # MPS 不支持的算子 fallback CPU
export inp_text=dataset/vectorv2.list
export inp_wav_dir=""                                   # 留空：按 list 里的路径（相对仓库根）原样打开
export exp_name=vectorv2
export opt_dir=logs/vectorv2
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
| `epochs` (GPT) | 50 | 最终选 e30 |
| `save_every_epoch` | 5 | 每 5 个 epoch 一份 ckpt |
| `if_save_latest` | True | 自动 resume 关键 |
| `if_save_every_weights` | True | 推理用的轻量权重保存到 `SoVITS_weights_v2/` & `GPT_weights_v2/` |
| `lora_rank` | 32 | SoVITS 默认 |
| `data.num_workers` (s1) | 0 | macOS fork DataLoader 死锁见下面 |

### 5.2 命令

```bash
. .venv/bin/activate
./preprocess_vector.sh
python train_vector.py          # EXP 默认 vectorv2
```

`train_vector.py` 串联跑：

1. `python -s GPT_SoVITS/s2_train.py --config TEMP/tmp_s2.json`
2. `python -s GPT_SoVITS/s1_train.py --config_file TEMP/tmp_s1.yaml`

### 5.3 时间消耗（M-series MPS）

同规模数据（20 段、每 epoch 26 batch）的实测：

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

### 6.4 fastText `Cache directory not found`

**症状**：第一次推理 `中英混合` 模式时挂掉。

**修法**：手动 `mkdir GPT_SoVITS/pretrained_models/fast_langdetect`。

### 6.5 NLTK 资源缺失

**症状**：`Resource 'averaged_perceptron_tagger_eng' not found`。

**修法**：跑一次 `nltk.download('averaged_perceptron_tagger_eng')`。

### 6.6 PyPI 直连卡死

**症状**：`pip install genie-tts` 跑了 14 分钟无任何输出，CPU 0%、网络无活动。

**修法**：清华镜像（见 §3.1）。

---

## 7. 推理

### 7.1 命令

```bash
. .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH

python GPT_SoVITS/inference_cli.py \
  --gpt_model    GPT_weights_v2/vectorv2-e30.ckpt \
  --sovits_model SoVITS_weights_v2/vectorv2_e30_s780.pth \
  --ref_audio    samples/vector_sovits_dataset_v2/clips/vector_001.wav \
  --ref_text     ref_text.txt \
  --ref_language 英文 \
  --target_text  target_text.txt \
  --target_language 中英混合 \
  --output_path  output
```

`ref_text.txt` 是 `vector_001.wav` 的英文转录；`target_text.txt` 是想合成的中文（含英文专有名词时用 `中英混合` 语种）。

常驻服务用 `serve.py`，默认参考音频同为 `vector_001.wav`（容器内路径 `/samples/vector_001.wav`，由 `docker-compose.cpu.yaml` 挂载 `samples/vector_sovits_dataset_v2/clips/`）。

### 7.2 性能

| 指标 | 值 |
|---|---|
| 输出采样率 | 32 kHz mono |
| 推理耗时（CPU + 缓存）| 5-6 秒 |
| 输出长度（5 秒中文）| 4-6 秒 |
| 模型加载时间（首次） | ~3-5 秒 |

接 wire-pod 时 32k 重采样到 16k（wire-pod `ttr/sovits.go` 已处理）。

### 7.3 选 epoch

`synth_sweep.py` 用固定 seed 把各个 SoVITS × GPT epoch 组合各合成一遍，输出到 `output/sweep/`，逐个试听后选定 **SoVITS e30 + GPT e30**。`synth_compare.py` 用来对比指定的几组权重。

---

## 8. 关键产出物

```
samples/vector_sovits_dataset_v2/
├── README.md                        # 数据集说明
├── clips/vector_001..020.wav        # 20 段训练音频
├── phrases.tsv                      # 文本
├── vector.list                      # 清单（相对数据集目录）
└── PREVIEW_ALL.wav                  # 拼接试听

dataset/vectorv2.list                # 清单（相对仓库根，训练用）
preprocess_vector.sh                 # 3 stage 预处理 driver
train_vector.py                      # SoVITS + GPT 训练 driver
train_gpt_only.py                    # 仅续训 GPT
synth_sweep.py / synth_compare.py    # epoch 选择
serve.py                             # FastAPI /tts 服务
Dockerfile.cpu / docker-compose.cpu.yaml
ref_text.txt / target_text.txt       # 推理参考文本 / 目标文本

# 以下不入 git（本地生成）
logs/vectorv2/                       # 预处理产物（特征、token）
SoVITS_weights_v2/vectorv2_e30_s780.pth   # ⭐ 当前使用
GPT_weights_v2/vectorv2-e30.ckpt          # ⭐ 当前使用
```

---

## 9. 后续

已完成：`serve.py` HTTP 服务、CPU Docker 部署、wire-pod `gpt-sovits` TTS provider。

可选方向：
- ONNX / OpenVINO 推理（`export_vector_onnx.py`、`run_onnx.py`、`run_ov.py`）已试过，在目标 CPU 上不比 PyTorch 快，FP16/INT8 量化有可闻的音质损失，暂不采用
- 推理后叠加 Vector 喇叭的 IR/EQ，还原真机听感
