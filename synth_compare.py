"""One-off A/B synth: render the same sentences with the v1 vs vectorv2
fine-tunes so the user can compare the new dataset's voice.

Usage:
  .venv/bin/python synth_compare.py
Outputs WAVs into output/compare/.
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
_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

import soundfile as sf
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()
OUT = REPO / "output" / "compare"
OUT.mkdir(parents=True, exist_ok=True)

# v2 dataset reference clip + its transcript (English ref → cross-lingual synth)
REF_AUDIO = str(REPO / "samples" / "vector_sovits_dataset_v2" / "clips" / "vector_001.wav")
REF_TEXT = "Hi there. My name is Vector. It's very nice to meet you today."
REF_LANG = "英文"

SENTENCES = [
    ("mixed", "你好，我是 Vector。很高兴见到你！", "中英混合"),
    ("zh", "今天天气真不错，要不要出去走走？", "中文"),
    ("mixed2", "我的电量有点低了，能帮我找一下 charger 吗？", "中英混合"),
]

MODELS = {
    "vectorv2": (
        "SoVITS_weights_v2/vectorv2_e30_s780.pth",
        "GPT_weights_v2/vectorv2-e50.ckpt",
    ),
}


def load(sovits_path, gpt_path):
    try:
        next(change_sovits_weights(sovits_path=sovits_path))
    except (StopIteration, UnboundLocalError, Exception):
        pass
    change_gpt_weights(gpt_path=gpt_path)


def main():
    if not Path(REF_AUDIO).exists():
        print(f"ref audio missing: {REF_AUDIO}")
        sys.exit(1)
    for tag, (sov, gpt) in MODELS.items():
        if not (REPO / sov).exists() or not (REPO / gpt).exists():
            print(f"[{tag}] checkpoints missing, skipping ({sov} / {gpt})")
            continue
        print(f"\n=== loading {tag}: {sov} + {gpt} ===")
        load(sov, gpt)
        for name, text, lang in SENTENCES:
            t0 = time.time()
            result = list(get_tts_wav(
                ref_wav_path=REF_AUDIO,
                prompt_text=REF_TEXT,
                prompt_language=i18n(REF_LANG),
                text=text,
                text_language=i18n(lang),
                top_p=1, temperature=1,
            ))
            sr, audio = result[-1]
            out = OUT / f"{tag}_{name}.wav"
            sf.write(out, audio, sr, subtype="PCM_16")
            print(f"  [{tag}] {name}: {len(audio)/sr:.1f}s audio in {time.time()-t0:.1f}s -> {out.name}")


if __name__ == "__main__":
    main()
