"""Microbenchmark: per-module latency of fp32 vs INT8 ONNX.

Runs each of encoder / fsdec / sdec / vits with dummy inputs of realistic
shape and reports per-call latency. Stage-decoder runs ~80x per utterance
so its delta dominates end-to-end RTF.

This script does NOT validate output quality — INT8 t2s sampling can diverge
silently. Run an end-to-end audio test separately before trusting the numbers.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parent
FP32_DIR = REPO / "onnx" / "vector"
INT8_DIR = REPO / "onnx" / "vector_int8"

WARMUP = 2
ITERS = 5
TYPICAL_AR_STEPS = 80

# Realistic shapes from the existing bench script. Phoneme IDs in pinyin
# vocab — just need the embedding lookup to succeed. We use small ints.
REF_LEN = 13
TEXT_LEN = 24
SSL_T = 250  # ~5s of 16kHz hubert frames
SSL_DIM = 768  # chinese-hubert-base hidden size


def make_inputs():
    rng = np.random.default_rng(42)
    ref_seq = rng.integers(low=1, high=200, size=(1, REF_LEN), dtype=np.int64)
    text_seq = rng.integers(low=1, high=200, size=(1, TEXT_LEN), dtype=np.int64)
    ref_bert = rng.standard_normal((REF_LEN, 1024)).astype(np.float32)
    text_bert = rng.standard_normal((TEXT_LEN, 1024)).astype(np.float32)
    ssl_content = rng.standard_normal((1, SSL_DIM, SSL_T)).astype(np.float32)
    return {
        "ref_seq": ref_seq, "text_seq": text_seq,
        "ref_bert": ref_bert, "text_bert": text_bert,
        "ssl_content": ssl_content,
    }


def make_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def time_run(sess, feed, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, feed)
    return (time.perf_counter() - t0) / iters * 1000  # ms


def bench(label, root, threads, enc_inputs, vits_root=None):
    """Bench t2s trio from `root`. VITS comes from `vits_root` if given (mixed
    mode: int8 t2s + fp32 vits is the actual deployment shape, since vits
    can't be dynamic-quantized profitably)."""
    vits_root = vits_root or root
    print(f"\n=== {label} (t2s={root.name}, vits={vits_root.name}) — threads={threads} ===")
    enc = make_session(root / "vector_t2s_encoder.onnx", threads)
    fsd = make_session(root / "vector_t2s_fsdec.onnx", threads)
    sdec = make_session(root / "vector_t2s_sdec.onnx", threads)
    vits_path = vits_root / "vector_vits.onnx"
    if not vits_path.exists():
        vits_path = FP32_DIR / "vector_vits.onnx"
    vits = make_session(vits_path, threads)

    # Encoder
    enc_ms = time_run(enc, enc_inputs)
    # Pull encoder output to feed fsdec
    x, prompts = enc.run(None, enc_inputs)
    fsd_inputs = {"x": x, "prompts": prompts}

    fsd_ms = time_run(fsd, fsd_inputs)

    # First-stage outputs feed stage decoder. Names mirror the export script.
    fsd_out = fsd.run(None, fsd_inputs)
    sdec_input_names = [i.name for i in sdec.get_inputs()]
    if len(sdec_input_names) != len(fsd_out):
        print(f"  WARN: fsd outputs={len(fsd_out)} but sdec inputs={len(sdec_input_names)}; using positional zip")
    sdec_inputs = dict(zip(sdec_input_names, fsd_out[:len(sdec_input_names)]))
    sdec_ms = time_run(sdec, sdec_inputs)

    # Vits — synthesize plausible inputs
    rng = np.random.default_rng(7)
    pred_semantic = rng.integers(0, 1024, size=(1, 1, 200), dtype=np.int64)
    # Reference audio: vits input is at SoVITS sampling rate (32k for v2)
    ref_audio_v = rng.standard_normal((1, 32000 * 5)).astype(np.float32)
    vits_inputs = {
        "text_seq": enc_inputs["text_seq"],
        "pred_semantic": pred_semantic,
        "ref_audio": ref_audio_v,
    }
    try:
        vits_ms = time_run(vits, vits_inputs)
    except Exception as e:
        print(f"  vits failed: {e}")
        vits_ms = float("nan")

    print(f"  encoder:  {enc_ms:7.1f} ms  (1x per utterance)")
    print(f"  fsdec:    {fsd_ms:7.1f} ms  (1x per utterance)")
    print(f"  sdec:     {sdec_ms:7.1f} ms  (~{TYPICAL_AR_STEPS}x per utterance — HOT PATH)")
    print(f"  vits:     {vits_ms:7.1f} ms  (1x per utterance)")
    total_s = (enc_ms + fsd_ms + sdec_ms * TYPICAL_AR_STEPS + (0 if np.isnan(vits_ms) else vits_ms)) / 1000
    print(f"  est. end-to-end (AR={TYPICAL_AR_STEPS}): {total_s:.2f}s")
    return enc_ms, fsd_ms, sdec_ms, vits_ms, total_s


def main():
    threads_arg = int(sys.argv[1]) if len(sys.argv) > 1 else max((os.cpu_count() or 2) // 2, 1)
    enc_inputs = make_inputs()

    fp = bench("fp32", FP32_DIR, threads_arg, enc_inputs)
    if INT8_DIR.exists():
        # Mixed deployment: int8 t2s + fp32 vits. Hybrid path is what we'd ship.
        q = bench("int8 t2s + fp32 vits", INT8_DIR, threads_arg, enc_inputs, vits_root=FP32_DIR)
        print("\n=== speedup (fp32 / int8-hybrid) ===")
        labels = ["encoder", "fsdec", "sdec", "vits", "end-to-end"]
        for label, a, b in zip(labels, fp, q):
            if a and b and not (np.isnan(a) or np.isnan(b)):
                print(f"  {label:11s} {a/b:.2f}x")
    else:
        print(f"\nINT8 dir missing: {INT8_DIR} (run quantize_int8.py first)")


if __name__ == "__main__":
    main()
