"""Run inside the NAS vector-tts container (/app). Epoch sweep:
SoVITS {e20,e25,e30} x GPT {e30,e40,e50} = 9 combos, one sentence each,
fixed seed + production sampling (top_p=1, temp=1), OpenVINO off, gain off.
Outputs to /app/output/nas_sweep/ (host: <compose dir>/output/nas_sweep).
"""
import os, sys, time
from pathlib import Path

REPO = Path("/app")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "GPT_SoVITS"))

import torch, numpy as np
_o = torch.load
torch.load = lambda *a, **k: _o(*a, **{**k, "weights_only": False})

import soundfile as sf
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()
OUT = Path("/app/output/nas_sweep"); OUT.mkdir(parents=True, exist_ok=True)
REF = "/samples/vector_001.wav"
RT = "Hi there. My name is Vector. It's very nice to meet you today."
TEXT = "你好，我是 Vector。很高兴见到你！"

SOV = {"20": "SoVITS_weights_v2/vectorv2_e20_s520.pth",
       "25": "SoVITS_weights_v2/vectorv2_e25_s650.pth",
       "30": "SoVITS_weights_v2/vectorv2_e30_s780.pth"}
GPT = {"30": "GPT_weights_v2/vectorv2-e30.ckpt",
       "40": "GPT_weights_v2/vectorv2-e40.ckpt",
       "50": "GPT_weights_v2/vectorv2-e50.ckpt"}


def set_sov(p):
    try:
        next(change_sovits_weights(sovits_path=p))
    except Exception:
        pass


for g in ["30", "40", "50"]:
    change_gpt_weights(gpt_path=GPT[g])
    for s in ["20", "25", "30"]:
        set_sov(SOV[s])
        torch.manual_seed(42); np.random.seed(42)
        r = list(get_tts_wav(ref_wav_path=REF, prompt_text=RT, prompt_language=i18n("英文"),
                             text=TEXT, text_language=i18n("中英混合"), top_p=1, temperature=1))
        sr, a = r[-1]
        sf.write(OUT / f"sweep_sov{s}_gpt{g}.wav", a, sr, subtype="PCM_16")
        print("RENDER", f"sov{s}_gpt{g}", f"{len(a)/sr:.1f}s", flush=True)
print("DONE")
