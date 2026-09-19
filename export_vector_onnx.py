"""Export the fine-tuned Vector model to ONNX.

The repo's onnx_export.export() builds dummy inputs and runs a debug forward,
but the actual ONNX export call (gpt_sovits.export) is commented out. This
wrapper replicates the dummy-input setup, then calls the export().
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

# PT 2.6 default weights_only=True breaks loading our trusted checkpoints
# (utils.HParams, pathlib.PosixPath, etc.). Restore the old behaviour.
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

# torch 2.11's torch.onnx.export defaults to dynamo=True which fails on
# data-dependent control flow in core_vq.py. Force legacy TorchScript exporter.
import torch.onnx as _onnx
_orig_onnx_export = _onnx.export
_onnx.export = lambda *a, **kw: _orig_onnx_export(*a, **{"dynamo": False, **kw})

import torchaudio

from GPT_SoVITS.onnx_export import GptSoVits, SSLModel, T2SModel, VitsModel
from text import cleaned_text_to_sequence


SOVITS_PATH = "SoVITS_weights_v2/vectorv2_e30_s780.pth"
GPT_PATH = "GPT_weights_v2/vectorv2-e30.ckpt"
PROJECT = "vector"
VERSION = "v2"


def main():
    out_dir = REPO / "onnx" / PROJECT
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading SoVITS:", SOVITS_PATH)
    vits = VitsModel(SOVITS_PATH)
    print("[2/4] loading GPT:", GPT_PATH)
    gpt = T2SModel(GPT_PATH, vits)
    gpt_sovits = GptSoVits(vits, gpt)
    ssl = SSLModel()

    # Phoneme dummies (Mandarin pinyin sequences) — used for trace shape only.
    ref_seq = torch.LongTensor([cleaned_text_to_sequence(
        ["n", "i2", "h", "ao3", ",", "w", "o3", "sh", "i4", "b", "ai2", "y", "e4"],
        version=VERSION)])
    text_seq = torch.LongTensor([cleaned_text_to_sequence(
        ["w", "o3", "sh", "i4", "b", "ai2", "y", "e4"] * 3,
        version=VERSION)])
    ref_bert = torch.randn((ref_seq.shape[1], 1024)).float()
    text_bert = torch.randn((text_seq.shape[1], 1024)).float()
    ref_audio = torch.randn((1, 48000 * 5)).float()
    ref_audio_16k = torchaudio.functional.resample(ref_audio, 48000, 16000).float()
    ref_audio_sr = torchaudio.functional.resample(ref_audio, 48000, vits.hps.data.sampling_rate).float()
    ssl_content = ssl(ref_audio_16k).float()

    print("[3/4] exporting GPT (Text2Semantic) ONNX...")
    t0 = time.time()
    gpt.export(ref_seq, text_seq, ref_bert, text_bert, ssl_content, PROJECT)
    print(f"      done in {time.time() - t0:.1f}s")

    print("[4/4] exporting SoVITS (Vits) ONNX...")
    t0 = time.time()
    gpt_sovits.export(ref_seq, text_seq, ref_bert, text_bert, ref_audio_sr, ssl_content, PROJECT)
    print(f"      done in {time.time() - t0:.1f}s")

    # Move per-onnx_export convention output (from cwd) to onnx/<project>/
    print("\nGenerated files:")
    for p in sorted(REPO.glob(f"onnx/{PROJECT}/*")):
        print(f"  {p.name}: {p.stat().st_size / 1024 / 1024:.1f} MB")
    for p in sorted(REPO.glob(f"{PROJECT}/*")):
        print(f"  ./{PROJECT}/{p.name}: {p.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
