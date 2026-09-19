"""End-to-end RTF benchmark for the fine-tuned Vector model.

Loads the model once, then runs the same inference path as inference_cli.py
across several target texts of different lengths. Reports per-call wall time,
output audio duration, and RTF (real-time factor).
"""

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "GPT_SoVITS"))
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

import soundfile as sf
from GPT_SoVITS.inference_webui import (
    change_gpt_weights,
    change_sovits_weights,
    get_tts_wav,
)

GPT_PATH = os.environ.get("GPT_PATH", "GPT_weights_v2/vector-e50.ckpt")
SOVITS_PATH = os.environ.get("SOVITS_PATH", "SoVITS_weights_v2/vector_e30_s780.pth")
# Inside the docker image samples are bind-mounted at /samples; on the host Mac
# they live under samples/ in this repo. Override via REF_AUDIO env var if elsewhere.
_default_ref = "/samples/vector_ref_best.wav" if os.path.exists("/samples") else \
    "/Users/wuwei/workspace/gpt-sovits/samples/vector_ref_best.wav"
REF_AUDIO = os.environ.get("REF_AUDIO", _default_ref)
REF_TEXT = "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon."

# Mix of short / medium / long, with embedded English (to exercise mixed-lang path)
PROMPTS = [
    ("短", "你好。"),
    ("中-1", "你好，我是 Vector。"),
    ("中-2", "今天天气怎么样？"),
    ("长-1", "你好，我是 Vector，今天天气很好，要不要一起出去玩？"),
    ("长-2", "我可以陪你聊天，识别你的脸，帮你记日程，还能跟你玩拳头游戏。"),
]


def main():
    print("Loading checkpoints…")
    t0 = time.time()
    change_gpt_weights(gpt_path=GPT_PATH)
    change_sovits_weights(sovits_path=SOVITS_PATH)
    print(f"  load: {time.time() - t0:.1f}s\n")

    print(f"{'tag':<6} {'words':>5} {'audio_s':>8} {'wall_s':>8} {'RTF':>6}  text")
    print("-" * 80)

    rtfs = []
    for tag, text in PROMPTS:
        t0 = time.time()
        result = list(get_tts_wav(
            ref_wav_path=REF_AUDIO,
            prompt_text=REF_TEXT,
            prompt_language="英文",
            text=text,
            text_language="中英混合",
            top_p=1, temperature=1,
        ))
        wall = time.time() - t0
        sr, audio = result[-1]
        audio_s = len(audio) / sr
        rtf = wall / audio_s
        rtfs.append((tag, audio_s, wall, rtf))
        print(f"{tag:<6} {len(text):>5} {audio_s:>8.2f} {wall:>8.2f} {rtf:>6.2f}  {text}")
        sf.write(f"output/bench_{tag}.wav", audio, sr)

    print("-" * 80)
    avg_audio = sum(r[1] for r in rtfs) / len(rtfs)
    avg_wall = sum(r[2] for r in rtfs) / len(rtfs)
    avg_rtf = sum(r[3] for r in rtfs) / len(rtfs)
    print(f"{'平均':<6} {'-':>5} {avg_audio:>8.2f} {avg_wall:>8.2f} {avg_rtf:>6.2f}")
    print(f"\nRTF < 1.0 = 实时；RTF > 1.0 = 比实时慢 RTF 倍")


if __name__ == "__main__":
    main()
