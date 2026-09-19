# Vector 音色 GPT-SoVITS 训练数据集 v2

**v2 更新**：使用 Vector 真机的 TTS 参数（speed=80, pitch=100, shape=130），听感与真机一致。

## 规格
- **采样率**: 32000 Hz mono PCM 16-bit
- **总时长**: 115 秒（1.92 分钟）
- **片段**: 20 段，每段 5.4-9.7 秒
- **峰值**: -3 dBFS
- **来源**: Acapela BABILE + Bendnn 声库
- **参数**: speed=80, pitch=100, shape=130（与 Vector 真机 tts_config.json 一致）

## 用法

### A. 零样本推理
GPT-SoVITS WebUI → 1C-Inference：
- 参考音频: `clips/vector_001.wav`
- 参考文本: `Hi there. My name is Vector. It's very nice to meet you today.`
- 参考语言: English
- 目标语言: 中文 / 中英混合
- 目标文本: 你想 Vector 说的中文

### B. 微调 fine-tune
1. 把整个 zip 解压到 Colab `/content/`
2. 修改 `vector.list` 里相对路径成绝对路径（或在 Colab 重做）：
   ```python
   import os
   DATA_DIR = "/content/dataset_v2"  # 解压后的路径
   with open(f"{DATA_DIR}/vector.list") as fin, open("/content/GPT-SoVITS/vector.list", "w") as fout:
       for line in fin:
           p, spk, lan, txt = line.strip().split("|")
           fout.write(f"{DATA_DIR}/{p}|{spk}|{lan}|{txt}\n")
   ```
3. GPT-SoVITS WebUI 1A → One-Click Formatting → 1B 训练 → 1C 推理

## 跟 v1 的区别
v1 用错了参数（100/100/100）+ 没做 audio layer pitch shift，听感像普通男童声。
v2 用对了基础参数，听感符合 Vector 真机干声（没有喇叭味，但 timbre 对了）。

如果你想要"喇叭味"完全一致，最佳做法是让模型先学正确的 timbre（用本数据集），推理后再叠真机喇叭的 IR/EQ。
