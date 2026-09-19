"""Sanity check: synthesize using Feibi's bundled Chinese reference.

If this produces a normal-length wav, the pipeline works and the earlier
0.08s output must come from the cross-lingual reference (English audio +
Chinese target). If this also fails, something deeper is wrong.
"""

import os
import time

MODEL_DIR = "/Users/wuwei/workspace/gpt-sovits/experiments/genie-tts/models/CharacterModels/v2ProPlus/feibi"
REF_AUDIO = f"{MODEL_DIR}/prompt_wav/zh_vo_Main_Linaxita_2_1_10_26.wav"
REF_TEXT = "在此之前，请您务必继续享受旅居拉古那的时光。"
OUT_PATH = "/Users/wuwei/workspace/gpt-sovits/experiments/genie-tts/test_baseline_out.wav"
TARGET_TEXT = "你好，我是 Vector，今天天气很好，要不要一起出去玩？"

import genie_tts as genie

genie.load_character(character_name="Feibi", onnx_model_dir=f"{MODEL_DIR}/tts_models", language="zh")
genie.set_reference_audio(character_name="Feibi", audio_path=REF_AUDIO, audio_text=REF_TEXT)
t0 = time.time()
genie.tts(character_name="Feibi", text=TARGET_TEXT, play=False, save_path=OUT_PATH)
print(f"baseline synth in {time.time() - t0:.1f}s, size={os.path.getsize(OUT_PATH)} bytes")
