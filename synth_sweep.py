"""Epoch sweep for the vectorv2 fine-tune: two 1-D sweeps isolating SoVITS
(timbre/vocoder) vs GPT (prosody/articulation), so the best checkpoint combo
can be picked by ear.

  SoVITS sweep: GPT fixed @ e50, SoVITS in {e10..e30}
  GPT    sweep: SoVITS fixed @ e30, GPT in {e10..e40}

Fixed seed so differences come from weights, not sampling jitter.
Outputs into output/sweep/.
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
import numpy as np
_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

import soundfile as sf
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()
OUT = REPO / "output" / "sweep"
OUT.mkdir(parents=True, exist_ok=True)

REF_AUDIO = str(REPO / "samples" / "vector_sovits_dataset_v2" / "clips" / "vector_001.wav")
REF_TEXT = "Hi there. My name is Vector. It's very nice to meet you today."
REF_LANG = "英文"
TEXT = "你好，我是 Vector。很高兴见到你！"
LANG = "中英混合"
SEED = 42

SOV = {e: f"SoVITS_weights_v2/vectorv2_e{e[0]}_s{e[1]}.pth" for e in
       [("5", "130"), ("10", "260"), ("15", "390"), ("20", "520"), ("25", "650"), ("30", "780")]}
GPT = {e: f"GPT_weights_v2/vectorv2-e{e}.ckpt" for e in
       ["5", "10", "15", "20", "25", "30", "35", "40", "45", "50"]}


def set_sovits(path):
    try:
        next(change_sovits_weights(sovits_path=path))
    except (StopIteration, UnboundLocalError, Exception):
        pass


def synth(tag):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()
    result = list(get_tts_wav(
        ref_wav_path=REF_AUDIO, prompt_text=REF_TEXT, prompt_language=i18n(REF_LANG),
        text=TEXT, text_language=i18n(LANG), top_p=1, temperature=1,
    ))
    sr, audio = result[-1]
    out = OUT / f"{tag}.wav"
    sf.write(out, audio, sr, subtype="PCM_16")
    print(f"  {tag}: {len(audio)/sr:.1f}s in {time.time()-t0:.1f}s -> {out.name}")


def main():
    # --- SoVITS sweep, GPT @ e50 ---
    print("=== SoVITS sweep (GPT @ e50) ===")
    change_gpt_weights(gpt_path=GPT["50"])
    for ep in ["10", "15", "20", "25", "30"]:
        key = next(k for k in SOV if k[0] == ep)
        set_sovits(SOV[key])
        synth(f"sov{ep}_gpt50")

    # --- GPT sweep, SoVITS @ e30 ---
    print("=== GPT sweep (SoVITS @ e30) ===")
    set_sovits(SOV[("30", "780")])
    for ep in ["10", "20", "30", "40"]:
        change_gpt_weights(gpt_path=GPT[ep])
        synth(f"sov30_gpt{ep}")

    print("\nDone. sov30_gpt50 already exists as output/compare/vectorv2_mixed.wav")


if __name__ == "__main__":
    main()
