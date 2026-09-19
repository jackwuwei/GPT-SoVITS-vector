"""Microbenchmark: ONNX Runtime vs PyTorch forward of the four GPT-SoVITS modules.

Real end-to-end speed depends on the autoregressive sampling loop (stage_decoder
runs 50-100 times per utterance) plus the vocoder (vits, runs once). We measure
single-call latency of each module on the same dummy inputs, then estimate
end-to-end per-utterance time from a typical iteration count.
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

import torchaudio
import onnxruntime as ort
import numpy as np

from GPT_SoVITS.onnx_export import GptSoVits, SSLModel, T2SModel, VitsModel
from text import cleaned_text_to_sequence

ONNX_DIR = REPO / "onnx" / "vector"
SOVITS_PATH = "SoVITS_weights_v2/vectorv2_e30_s780.pth"
GPT_PATH = "GPT_weights_v2/vectorv2-e30.ckpt"
VERSION = "v2"
WARMUP = 2
ITERS = 5
TYPICAL_AR_STEPS = 80  # rough avg autoregressive iterations per ~5s output


def make_inputs():
    ref_seq = torch.LongTensor([cleaned_text_to_sequence(
        ["n", "i2", "h", "ao3", ",", "w", "o3", "sh", "i4", "b", "ai2", "y", "e4"], version=VERSION)])
    text_seq = torch.LongTensor([cleaned_text_to_sequence(
        ["w", "o3", "sh", "i4", "b", "ai2", "y", "e4"] * 3, version=VERSION)])
    ref_bert = torch.randn((ref_seq.shape[1], 1024)).float()
    text_bert = torch.randn((text_seq.shape[1], 1024)).float()
    ref_audio = torch.randn((1, 48000 * 5)).float()
    return ref_seq, text_seq, ref_bert, text_bert, ref_audio


def time_call(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    t0 = time.time()
    for _ in range(iters):
        fn()
    return (time.time() - t0) / iters * 1000  # ms


def main():
    ref_seq, text_seq, ref_bert, text_bert, ref_audio = make_inputs()

    print("Loading PyTorch models (this loads everything once for both backends)…")
    t0 = time.time()
    vits = VitsModel(SOVITS_PATH)
    gpt = T2SModel(GPT_PATH, vits)
    ssl = SSLModel()
    print(f"  PT load: {time.time() - t0:.1f}s")

    ref_audio_16k = torchaudio.functional.resample(ref_audio, 48000, 16000).float()
    ref_audio_sr = torchaudio.functional.resample(ref_audio, 48000, vits.hps.data.sampling_rate).float()
    with torch.no_grad():
        ssl_content = ssl(ref_audio_16k).float()

    print("\nLoading ONNX sessions…")
    t0 = time.time()
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = os.cpu_count() // 2 or 1
    encoder_sess = ort.InferenceSession(str(ONNX_DIR / "vector_t2s_encoder.onnx"), sess_options)
    fsdec_sess = ort.InferenceSession(str(ONNX_DIR / "vector_t2s_fsdec.onnx"), sess_options)
    sdec_sess = ort.InferenceSession(str(ONNX_DIR / "vector_t2s_sdec.onnx"), sess_options)
    vits_sess = ort.InferenceSession(str(ONNX_DIR / "vector_vits.onnx"), sess_options)
    print(f"  ONNX load: {time.time() - t0:.1f}s")

    # === Encoder ===
    print("\n=== T2S Encoder (called once per utterance) ===")
    pt_encoder = lambda: gpt.onnx_encoder(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
    pt_ms = time_call(lambda: torch.no_grad().__enter__() or pt_encoder())
    enc_inputs = {
        "ref_seq": ref_seq.numpy(), "text_seq": text_seq.numpy(),
        "ref_bert": ref_bert.numpy(), "text_bert": text_bert.numpy(),
        "ssl_content": ssl_content.numpy(),
    }
    onnx_ms = time_call(lambda: encoder_sess.run(None, enc_inputs))
    print(f"  PyTorch:  {pt_ms:7.1f} ms")
    print(f"  ONNX RT:  {onnx_ms:7.1f} ms   (speedup {pt_ms / onnx_ms:.2f}×)")
    enc_pt = pt_ms; enc_onnx = onnx_ms

    # === First-stage decoder ===
    print("\n=== T2S First-stage Decoder (called once per utterance) ===")
    with torch.no_grad():
        x, prompts = gpt.onnx_encoder(ref_seq, text_seq, ref_bert, text_bert, ssl_content)
    pt_ms = time_call(lambda: torch.no_grad().__enter__() or gpt.first_stage_decoder(x, prompts))
    fsd_inputs = {"x": x.numpy(), "prompts": prompts.numpy()}
    onnx_ms = time_call(lambda: fsdec_sess.run(None, fsd_inputs))
    print(f"  PyTorch:  {pt_ms:7.1f} ms")
    print(f"  ONNX RT:  {onnx_ms:7.1f} ms   (speedup {pt_ms / onnx_ms:.2f}×)")
    fsd_pt = pt_ms; fsd_onnx = onnx_ms

    # === Stage decoder (the hot path: runs ~80x per utterance) ===
    print("\n=== T2S Stage Decoder (runs ~80× per utterance — the hot path) ===")
    with torch.no_grad():
        y, k, v, y_emb, x_example = gpt.first_stage_decoder(x, prompts)
    pt_ms = time_call(lambda: torch.no_grad().__enter__() or gpt.stage_decoder(y, k, v, y_emb, x_example))
    sdec_inputs = {
        "iy": y.numpy(), "ik": k.numpy(), "iv": v.numpy(),
        "iy_emb": y_emb.numpy(), "ix_example": x_example.numpy(),
    }
    onnx_ms = time_call(lambda: sdec_sess.run(None, sdec_inputs))
    print(f"  PyTorch:  {pt_ms:7.1f} ms")
    print(f"  ONNX RT:  {onnx_ms:7.1f} ms   (speedup {pt_ms / onnx_ms:.2f}×)")
    sdec_pt = pt_ms; sdec_onnx = onnx_ms

    # === Vits vocoder (called once per utterance) ===
    print("\n=== Vits / SoVITS vocoder (called once per utterance) ===")
    # Real semantic tokens shape unknown — synthesize a plausible (1, T) tensor.
    pred_semantic = torch.randint(0, 1024, (1, 200)).long()
    text_seq_v = text_seq
    ref_audio_input = ref_audio_sr
    pt_ms = time_call(lambda: torch.no_grad().__enter__() or vits(text_seq_v, pred_semantic, ref_audio_input))
    vits_inputs = {
        "text_seq": text_seq_v.numpy(),
        "pred_semantic": pred_semantic.numpy(),
        "ref_audio": ref_audio_input.numpy(),
    }
    try:
        onnx_ms = time_call(lambda: vits_sess.run(None, vits_inputs))
        print(f"  PyTorch:  {pt_ms:7.1f} ms")
        print(f"  ONNX RT:  {onnx_ms:7.1f} ms   (speedup {pt_ms / onnx_ms:.2f}×)")
        vits_pt = pt_ms; vits_onnx = onnx_ms
    except Exception as e:
        print(f"  PyTorch:  {pt_ms:7.1f} ms")
        print(f"  ONNX RT:  failed ({e})")
        vits_pt = pt_ms; vits_onnx = float('nan')

    # === End-to-end estimate ===
    print(f"\n=== Estimated per-utterance time (TYPICAL_AR_STEPS = {TYPICAL_AR_STEPS}) ===")
    pt_total = enc_pt + fsd_pt + sdec_pt * TYPICAL_AR_STEPS + (vits_pt if not np.isnan(vits_pt) else 0)
    onnx_total = enc_onnx + fsd_onnx + sdec_onnx * TYPICAL_AR_STEPS + (vits_onnx if not np.isnan(vits_onnx) else 0)
    print(f"  PyTorch total:  {pt_total / 1000:6.2f} s")
    print(f"  ONNX RT total:  {onnx_total / 1000:6.2f} s")
    print(f"  speedup:        {pt_total / onnx_total:.2f}×")
    print(f"  (stage_decoder dominates: PT {sdec_pt * TYPICAL_AR_STEPS / 1000:.2f}s vs ONNX {sdec_onnx * TYPICAL_AR_STEPS / 1000:.2f}s)")


if __name__ == "__main__":
    main()
