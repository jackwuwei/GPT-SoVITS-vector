"""Dynamic INT8 quantization of the Vector ONNX exports.

Default targets are the t2s trio (encoder/fsdec/sdec) because they are
MatMul-dominated. VITS is conv-dominated (HiFiGAN vocoder); dynamic quant
inserts Dequantize nodes around every Conv with no offsetting speedup —
benchmarked at 5.8x SLOWER on Mac. Quantize VITS only via static quant with
calibration data, which this script does not do.

Caveat: stage decoder bakes top-p sampling into its graph. INT8 perturbs
pre-softmax logits enough that token selection may diverge from fp32 —
audio quality must be A/B tested, not assumed.

Usage:
    python quantize_int8.py                  # quantize encoder/fsdec/sdec (default)
    python quantize_int8.py sdec             # subset
    python quantize_int8.py all              # include vits (NOT recommended)
"""

import sys
from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType

REPO = Path(__file__).resolve().parent
SRC = REPO / "onnx" / "vector"
DST = REPO / "onnx" / "vector_int8"

NAMES = {
    "encoder": "vector_t2s_encoder.onnx",
    "fsdec": "vector_t2s_fsdec.onnx",
    "sdec": "vector_t2s_sdec.onnx",
    "vits": "vector_vits.onnx",
}


DEFAULT_TARGETS = ["encoder", "fsdec", "sdec"]


def main():
    args = sys.argv[1:]
    if not args:
        requested = DEFAULT_TARGETS
    elif args == ["all"]:
        requested = list(NAMES.keys())
    else:
        requested = args
    DST.mkdir(parents=True, exist_ok=True)

    total_in = total_out = 0
    for key in requested:
        if key not in NAMES:
            print(f"  unknown target: {key} (valid: {list(NAMES)})")
            continue
        f = NAMES[key]
        src = SRC / f
        dst = DST / f
        if not src.exists():
            print(f"  skip {f}: source missing at {src}")
            continue
        size_in = src.stat().st_size / 1024 / 1024
        print(f"quantizing {f} ({size_in:.1f} MB) ...")
        quantize_dynamic(
            model_input=str(src),
            model_output=str(dst),
            weight_type=QuantType.QInt8,
        )
        size_out = dst.stat().st_size / 1024 / 1024
        ratio = size_in / size_out if size_out > 0 else float("inf")
        print(f"  -> {dst.relative_to(REPO)}: {size_out:.1f} MB ({ratio:.2f}x smaller)")
        total_in += size_in
        total_out += size_out

    if total_out:
        print(f"\nTotal: {total_in:.1f} MB -> {total_out:.1f} MB ({total_in / total_out:.2f}x)")


if __name__ == "__main__":
    main()
