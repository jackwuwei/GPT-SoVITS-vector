# vector-tts

用 **Anki Vector 机器人音色**合成语音（中文、英文、中英混合）的 HTTP 服务。基于 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) v2，用 Vector 的录音做了微调。源码：[jackwuwei/GPT-SoVITS-vector](https://github.com/jackwuwei/GPT-SoVITS-vector)

- 模型都打包在镜像里（预训练模型和微调后的 Vector 权重），拉下来就能用，不用挂载
- 只用 CPU，不需要显卡
- 支持 `linux/amd64` 和 `linux/arm64`
- 配合 [jackwuwei/wire-pod-chinese](https://hub.docker.com/r/jackwuwei/wire-pod-chinese)，Vector 可以用自己的声音说中文

## 快速开始

```bash
docker run -d --name vector-tts -p 8020:8020 --restart unless-stopped jackwuwei/vector-tts:latest

curl -X POST http://localhost:8020/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，我是 Vector，很高兴认识你。"}' \
  -o out.wav
```

启动时会加载模型并预热一次，要等几十秒，`GET /healthz` 返回 `"warm": true` 后就可以用了。

## 接口

**`POST /tts`**：返回 32kHz 单声道 16 位 WAV。

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `text` | 必填 | 要合成的文本 |
| `lang` | `中英混合` | 也可以用 `中文`、`英文` |
| `seed` | `42` | 随机种子。输入和种子都相同时，输出也相同 |
| `ref_audio_path` / `ref_text` / `ref_lang` | 内置的 Vector 参考音频 | 换参考音频时使用，路径指的是容器内的路径 |
| `gain_db` / `target_peak_dbfs` | `0` | 音量增益和峰值归一化，默认关闭 |

响应头里有 `X-RTF`、`X-Audio-Duration`、`X-Inference-Wall`，可以用来看合成速度。

**`GET /healthz`**：健康检查。

## 性能

纯 CPU 推理。实时率（RTF，合成耗时 ÷ 音频时长）大约为：Apple M 系列 0.3，Intel Pentium Gold 8505 约 2。低功耗机器上达不到实时，适合短句。

可以用 `OMP_NUM_THREADS` / `MKL_NUM_THREADS` 设置线程数，一般设成 CPU 的线程数。

## 接入 wire-pod

在 wire-pod 配置页把 TTS provider 选为 `gpt-sovits`，服务地址填 `http://<本机 IP>:8020/tts`。

---

**English**: an HTTP TTS server that speaks in Anki Vector's voice (Chinese, English, or mixed), built on a GPT-SoVITS v2 fine-tune. CPU only, models baked in, amd64 + arm64. `POST /tts` with `{"text": "..."}` returns a 32 kHz mono WAV.
