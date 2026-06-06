"""Microbench: sdec single-call latency, OpenVINO Runtime vs ONNX Runtime.

Notes on OV: the upstream ONNX export contains Squeeze ops with dynamic rank
which OV CPU plugin can't handle. We work around by `model.reshape(...)` with
concrete shapes derived from a real encoder→fsdec→sdec chain run via ORT first.

Usage (inside container):
  python /app/output/_bench/bench_ov_vs_ort.py [iters]
"""

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import openvino as ov
from openvino import PartialShape

ONNX_DIR = Path("/app/output/_bench/onnx/vector")
WARMUP = 3
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
THREADS = 6

# Realistic frontend shapes
REF_LEN = 76
TEXT_LEN = 15
SSL_T = 424
SSL_DIM = 768

rng = np.random.default_rng(42)
ref_seq = rng.integers(low=1, high=200, size=(1, REF_LEN), dtype=np.int64)
text_seq = rng.integers(low=1, high=200, size=(1, TEXT_LEN), dtype=np.int64)
ref_bert = rng.standard_normal((REF_LEN, 1024)).astype(np.float32)
text_bert = rng.standard_normal((TEXT_LEN, 1024)).astype(np.float32)
ssl_content = rng.standard_normal((1, SSL_DIM, SSL_T)).astype(np.float32)


def make_ort(name):
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(ONNX_DIR / name),
        sess_options=so,
        providers=["CPUExecutionProvider"],
        disabled_optimizers=["SimplifiedLayerNormFusion"],
    )


def chain_to_sdec_inputs():
    enc = make_ort("vector_t2s_encoder.onnx")
    fsd = make_ort("vector_t2s_fsdec.onnx")
    enc_feed = {
        "ref_seq": ref_seq, "text_seq": text_seq,
        "ref_bert": ref_bert, "text_bert": text_bert,
        "ssl_content": ssl_content,
    }
    x, prompts = enc.run(None, enc_feed)
    print(f"      encoder out: x={x.shape} prompts={prompts.shape}")
    fsd_outs = fsd.run(None, {"x": x, "prompts": prompts})
    y, k, v, y_emb, x_example = fsd_outs
    print(f"      fsdec out:   y={y.shape} k={k.shape} v={v.shape} y_emb={y_emb.shape} x_example={x_example.shape}")
    return {"iy": y, "ik": k, "iv": v, "iy_emb": y_emb, "ix_example": x_example}


print(f"sdec single-call bench, warmup={WARMUP} iters={ITERS}, threads={THREADS}")

# Step 1: derive realistic sdec shapes via ORT chain
print("\n[setup] deriving sdec input shapes via ORT encoder->fsdec ...")
sdec_feed = chain_to_sdec_inputs()
shapes = {k: list(v.shape) for k, v in sdec_feed.items()}
print(f"      sdec shapes: {shapes}")

# === ORT (CPU EP) bench ===
print("\n[ORT-CPU] sdec ...")
t0 = time.perf_counter()
ort_sdec = make_ort("vector_t2s_sdec.onnx")
print(f"      load: {time.perf_counter()-t0:.1f}s")
for _ in range(WARMUP):
    ort_sdec.run(None, sdec_feed)
t0 = time.perf_counter()
for _ in range(ITERS):
    ort_sdec.run(None, sdec_feed)
ort_cpu_ms = (time.perf_counter() - t0) / ITERS * 1000
print(f"      sdec/call: {ort_cpu_ms:.1f} ms")

# === ORT (OpenVINO EP) bench ===
print("\n[ORT-OV] sdec ...")
t0 = time.perf_counter()
try:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # let OV do its own
    ort_ov_sdec = ort.InferenceSession(
        str(ONNX_DIR / "vector_t2s_sdec.onnx"),
        sess_options=so,
        providers=[("OpenVINOExecutionProvider", {"device_type": "CPU", "num_of_threads": THREADS}),
                   "CPUExecutionProvider"],
    )
    print(f"      load: {time.perf_counter()-t0:.1f}s, providers={ort_ov_sdec.get_providers()}")
    for _ in range(WARMUP):
        ort_ov_sdec.run(None, sdec_feed)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        ort_ov_sdec.run(None, sdec_feed)
    ort_ov_ms = (time.perf_counter() - t0) / ITERS * 1000
    print(f"      sdec/call: {ort_ov_ms:.1f} ms")
except Exception as e:
    print(f"      FAILED: {str(e)[:300]}")
    ort_ov_ms = float("nan")

# === OpenVINO bench ===
print("\n[OV] sdec ...")
t0 = time.perf_counter()
core = ov.Core()
print(f"      devices: {core.available_devices}")
ov_model = core.read_model(str(ONNX_DIR / "vector_t2s_sdec.onnx"))
# Reshape to concrete input shapes so dynamic-rank Squeeze can be folded.
input_shapes = {inp.any_name: PartialShape(shapes[inp.any_name]) for inp in ov_model.inputs}
print(f"      reshaping to: {input_shapes}")
ov_model.reshape(input_shapes)
ov_compiled = core.compile_model(
    ov_model,
    "CPU",
    config={"INFERENCE_NUM_THREADS": THREADS, "PERFORMANCE_HINT": "LATENCY"},
)
print(f"      load+compile: {time.perf_counter()-t0:.1f}s")
ov_req = ov_compiled.create_infer_request()
for _ in range(WARMUP):
    ov_req.infer(sdec_feed)
t0 = time.perf_counter()
for _ in range(ITERS):
    ov_req.infer(sdec_feed)
ov_ms = (time.perf_counter() - t0) / ITERS * 1000
print(f"      sdec/call: {ov_ms:.1f} ms")

print("\n=== summary ===")
print(f"  ORT-CPU       : {ort_cpu_ms:7.1f} ms")
print(f"  ORT-OV (EP)   : {ort_ov_ms:7.1f} ms  (speedup vs ORT-CPU: {ort_cpu_ms/ort_ov_ms:.2f}x)" if not np.isnan(ort_ov_ms) else "  ORT-OV (EP)   : FAILED")
print(f"  OV native     : {ov_ms:7.1f} ms  (speedup vs ORT-CPU: {ort_cpu_ms/ov_ms:.2f}x)")
print()
print(f"Estimated AR loop (50 steps):")
print(f"  ORT-CPU      {ort_cpu_ms*50/1000:5.2f}s")
if not np.isnan(ort_ov_ms):
    print(f"  ORT-OV (EP)  {ort_ov_ms*50/1000:5.2f}s")
print(f"  OV native    {ov_ms*50/1000:5.2f}s")
