"""Zero-shot clone smoke test using Genie-TTS + Vector reference audio.

Loads the predefined Chinese character "Feibi" and overrides its reference audio
with our 7.8s Vector clip so we can hear how much of Vector's timbre transfers
through Genie-TTS's reference-audio path. Output is written to test_out.wav.
"""

import os
import sys
import time

REF_AUDIO = "/Users/wuwei/workspace/gpt-sovits/samples/vector_ref_best.wav"
REF_TEXT = (
    "These take the shape of a long round arch, with its path high above, "
    "and its two ends apparently beyond the horizon."
)
OUT_PATH = "/Users/wuwei/workspace/gpt-sovits/experiments/genie-tts/test_out.wav"
TARGET_TEXT = "你好，我是 Vector，今天天气很好，要不要一起出去玩？"
CHARACTER = "Feibi"
LANG = "zh"
ONNX_MODEL_DIR = (
    "/Users/wuwei/workspace/gpt-sovits/experiments/genie-tts/models/CharacterModels/v2ProPlus/feibi/tts_models"
)

print("[1/4] importing genie_tts (first import auto-downloads ~391MB on first run)...")
t0 = time.time()
import genie_tts as genie  # noqa: E402
print(f"      imported in {time.time() - t0:.1f}s")

print(f"[2/4] loading character '{CHARACTER}' (lang={LANG})...")
t0 = time.time()
genie.load_character(character_name=CHARACTER, onnx_model_dir=ONNX_MODEL_DIR, language=LANG)
print(f"      loaded in {time.time() - t0:.1f}s")

print(f"[3/4] setting reference audio: {REF_AUDIO}")
genie.set_reference_audio(
    character_name=CHARACTER,
    audio_path=REF_AUDIO,
    audio_text=REF_TEXT,
    language="en",
)

print(f"[4/4] synthesizing: {TARGET_TEXT!r}")
t0 = time.time()
genie.tts(
    character_name=CHARACTER,
    text=TARGET_TEXT,
    play=False,
    save_path=OUT_PATH,
)
print(f"      synthesized in {time.time() - t0:.1f}s, wrote {OUT_PATH}")

if os.path.exists(OUT_PATH):
    print(f"      file size: {os.path.getsize(OUT_PATH)} bytes")
    sys.exit(0)
print("ERROR: output file not created")
sys.exit(1)
