"""Convert the Vector ONNX exports from fp32 to fp16.

Uses onnxruntime.transformers.float16.convert_float_to_float16 with
keep_io_types=True so the public I/O of each module stays fp32 — run_onnx.py
needs no changes.

Caveats on x86 without AVX-512_FP16 / AMX (Pentium 8505):
- Weights stored fp16 in RAM (~2× smaller), but ORT inserts Cast nodes that
  upcast to fp32 right before each MatMul. So inference compute is fp32 with
  per-op cast overhead — expect equal or slightly slower wall time, NOT a
  speedup.
- VITS contains many Conv ops; ORT's CPU EP fp16 Conv coverage is partial.
  We aggressively block Conv-related ops there to avoid runtime errors.

Usage:
    python convert_fp16.py                # all four files
    python convert_fp16.py vits           # subset
"""

import sys
from pathlib import Path

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

REPO = Path(__file__).resolve().parent
SRC = REPO / "onnx" / "vector"
DST = REPO / "onnx" / "vector_fp16"

NAMES = {
    "encoder": "vector_t2s_encoder.onnx",
    "fsdec": "vector_t2s_fsdec.onnx",
    "sdec": "vector_t2s_sdec.onnx",
    "vits": "vector_vits.onnx",
}

# Ops where ORT's CPU EP fp16 implementation is missing or buggy. Keeping
# them fp32 forces ORT to insert Cast nodes around them (cheap) instead of
# erroring out at session creation.
COMMON_BLOCK = ["LayerNormalization", "InstanceNormalization", "GroupNormalization"]
VITS_EXTRA_BLOCK = ["Conv", "ConvTranspose", "BatchNormalization", "GridSample"]


def convert_one(name: str):
    if name not in NAMES:
        print(f"  unknown target: {name} (valid: {list(NAMES)})")
        return
    f = NAMES[name]
    src = SRC / f
    if not src.exists():
        print(f"  skip {f}: source missing")
        return
    block_list = list(COMMON_BLOCK)
    if name == "vits":
        block_list += VITS_EXTRA_BLOCK

    print(f"converting {f} (block_list={block_list}) ...")
    model = onnx.load(str(src))
    fp16_model = convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=block_list,
    )
    dst = DST / f
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(fp16_model, str(dst))
    size_in = src.stat().st_size / 1024 / 1024
    size_out = dst.stat().st_size / 1024 / 1024
    print(f"  {size_in:.1f} MB -> {size_out:.1f} MB ({size_in/size_out:.2f}x)")


def main():
    targets = sys.argv[1:] or list(NAMES.keys())
    for t in targets:
        convert_one(t)


if __name__ == "__main__":
    main()
