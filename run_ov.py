"""End-to-end TTS via OpenVINO Runtime (CPU device).

Mirrors run_onnx.py but uses openvino.Core to compile each ONNX module.
Handles dynamic shapes via model.reshape(PartialShape) with -1 for growing
dims so the AR loop doesn't trigger per-step recompile.

Usage (inside vector-tts container on NAS):
  python /app/output/_bench/run_ov.py \
      --t2s-dir /app/output/_bench/onnx/vector \
      --vits-dir /app/output/_bench/onnx/vector \
      --text "..." --ref-audio /samples/vector_ref_best.wav \
      --out /app/output/_bench/ov.wav
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent  # /app
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "GPT_SoVITS"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

import numpy as np
import soundfile as sf
import librosa
import openvino as ov
from openvino import PartialShape

DEFAULT_REF_AUDIO = "/samples/vector_ref_best.wav"
DEFAULT_REF_TEXT = (
    "These take the shape of a long round arch, with its path high above, "
    "and its two ends apparently beyond the horizon."
)
DEFAULT_REF_LANG = "en"
DEFAULT_GPT_PATH = "GPT_weights_v2/vector-e50.ckpt"
DEFAULT_SOVITS_PATH = "SoVITS_weights_v2/vector_e30_s780.pth"
EOS_TOKEN = 1024
MAX_AR_STEPS = 1500


def init_upstream(sovits_path, gpt_path):
    from GPT_SoVITS.inference_webui import change_sovits_weights, change_gpt_weights
    try:
        next(change_sovits_weights(sovits_path=sovits_path))
    except Exception:
        pass
    change_gpt_weights(gpt_path=gpt_path)
    from GPT_SoVITS import inference_webui as iw
    return iw


def prep_ref(iw, ref_wav_path):
    wav16k, _ = librosa.load(ref_wav_path, sr=16000)
    wav16k = torch.from_numpy(np.concatenate([wav16k, np.zeros(int(16000 * 0.3), dtype=np.float32)])).float()
    with torch.no_grad():
        ssl = iw.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2).float()
        codes = iw.vq_model.extract_latent(ssl)
        prompt_semantic = codes[0, 0].unsqueeze(0)
    return ssl.cpu().numpy().astype(np.float32), prompt_semantic.cpu().numpy().astype(np.int64)


def get_text(iw, text, lang):
    from GPT_SoVITS.inference_webui import get_phones_and_bert
    phones, bert, _ = get_phones_and_bert(text, lang, "v2")
    phones_np = np.asarray(phones, dtype=np.int64)[None, :]
    bert_np = bert.detach().cpu().numpy().astype(np.float32)
    if bert_np.shape[0] == 1024:
        bert_np = bert_np.T
    return phones_np, bert_np


def compile_with_dynamic(core, onnx_path, dynamic_dims, threads):
    """Read ONNX, mark `dynamic_dims` (mapping input_name -> list of dim indices
    that should be dynamic) and compile for CPU."""
    model = core.read_model(str(onnx_path))
    if dynamic_dims:
        partial = {}
        for inp in model.inputs:
            name = inp.any_name
            shp = list(inp.partial_shape)
            dyn = dynamic_dims.get(name, [])
            new = [-1 if i in dyn else (s.get_length() if not s.is_dynamic else -1)
                   for i, s in enumerate(shp)]
            partial[name] = PartialShape(new)
        model.reshape(partial)
    cfg = {"INFERENCE_NUM_THREADS": str(threads), "PERFORMANCE_HINT": "LATENCY"}
    return core.compile_model(model, "CPU", config=cfg)


def ov_run(model, feed):
    req = model.create_infer_request()
    res = req.infer(feed)
    return [res[o] for o in model.outputs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t2s-dir", required=True)
    p.add_argument("--vits-dir", default=None)
    p.add_argument("--text", required=True)
    p.add_argument("--lang", default="all_zh")
    p.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    p.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    p.add_argument("--ref-lang", default=DEFAULT_REF_LANG)
    p.add_argument("--out", required=True)
    p.add_argument("--gpt", default=DEFAULT_GPT_PATH)
    p.add_argument("--sovits", default=DEFAULT_SOVITS_PATH)
    p.add_argument("--threads", type=int, default=max((os.cpu_count() or 2), 1))
    p.add_argument("--max-steps", type=int, default=MAX_AR_STEPS)
    args = p.parse_args()

    t2s_dir = Path(args.t2s_dir)
    vits_dir = Path(args.vits_dir) if args.vits_dir else t2s_dir
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] init upstream PyTorch frontend ...")
    t = time.time()
    iw = init_upstream(args.sovits, args.gpt)
    print(f"      done in {time.time()-t:.1f}s")

    print(f"[2/5] preparing ref + text ...")
    t = time.time()
    ssl_content, _ = prep_ref(iw, args.ref_audio)
    ref_seq, ref_bert = get_text(iw, args.ref_text, args.ref_lang)
    text_seq, text_bert = get_text(iw, args.text, args.lang)
    sr = int(iw.hps.data.sampling_rate)
    ref_audio_v, _ = librosa.load(args.ref_audio, sr=sr)
    ref_audio_v = ref_audio_v.astype(np.float32)[None, :]
    print(f"      ref_seq={ref_seq.shape} text_seq={text_seq.shape} done in {time.time()-t:.1f}s")

    print(f"[3/5] OV compiling 4 models (threads={args.threads}) ...")
    t = time.time()
    core = ov.Core()
    print(f"      OV devices: {core.available_devices}")
    # All inputs for these modules have variable lengths (audio, text, kv).
    # Mark every non-1 dim as dynamic so OV doesn't recompile per call.
    enc_dyn = {"ref_seq": [1], "text_seq": [1], "ref_bert": [0], "text_bert": [0], "ssl_content": [2]}
    fsd_dyn = {"x": [1], "prompts": [1]}
    sdec_dyn = {"iy": [1], "ik": [1], "iv": [1], "iy_emb": [1], "ix_example": [1]}
    vits_dyn = {"text_seq": [1], "pred_semantic": [2], "ref_audio": [1]}

    enc_m = compile_with_dynamic(core, t2s_dir / "vector_t2s_encoder.onnx", enc_dyn, args.threads)
    fsd_m = compile_with_dynamic(core, t2s_dir / "vector_t2s_fsdec.onnx", fsd_dyn, args.threads)
    sdec_m = compile_with_dynamic(core, t2s_dir / "vector_t2s_sdec.onnx", sdec_dyn, args.threads)
    vits_m = compile_with_dynamic(core, vits_dir / "vector_vits.onnx", vits_dyn, args.threads)
    print(f"      compile done in {time.time()-t:.1f}s")

    print(f"[4/5] running pipeline ...")
    pipe_t0 = time.perf_counter()

    t0 = time.perf_counter()
    x, prompts = ov_run(enc_m, {
        "ref_seq": ref_seq, "text_seq": text_seq,
        "ref_bert": ref_bert, "text_bert": text_bert,
        "ssl_content": ssl_content,
    })
    t_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    y, k, v, y_emb, x_example = ov_run(fsd_m, {"x": x, "prompts": prompts})
    t_fsd = time.perf_counter() - t0

    t0 = time.perf_counter()
    sdec_req = sdec_m.create_infer_request()
    output_names = [o.any_name for o in sdec_m.outputs]
    samples_idx = output_names.index("samples")
    y_idx = output_names.index("y")
    k_idx = output_names.index("k")
    v_idx = output_names.index("v")
    yemb_idx = output_names.index("y_emb")

    new_token_count = 1
    sample_log = [int(y[0, -1])]
    eos_seen = False
    for step in range(args.max_steps):
        feed = {"iy": y, "ik": k, "iv": v, "iy_emb": y_emb, "ix_example": x_example}
        res = sdec_req.infer(feed)
        outs = [res[o] for o in sdec_m.outputs]
        y = outs[y_idx]
        k = outs[k_idx]
        v = outs[v_idx]
        y_emb = outs[yemb_idx]
        new_tok = int(outs[samples_idx][0, 0])
        sample_log.append(new_tok)
        new_token_count += 1
        if new_tok == EOS_TOKEN or int(y[0, -1]) == EOS_TOKEN:
            eos_seen = True
            break
    t_sdec = time.perf_counter() - t0

    y[0, -1] = 0
    pred_semantic = y[:, -new_token_count:][None, :, :].astype(np.int64)

    t0 = time.perf_counter()
    audio = ov_run(vits_m, {
        "text_seq": text_seq, "pred_semantic": pred_semantic, "ref_audio": ref_audio_v,
    })[0]
    t_vits = time.perf_counter() - t0
    wall = time.perf_counter() - pipe_t0

    audio_s = len(audio) / sr
    print(f"      AR steps: {new_token_count}  EOS: {eos_seen}")
    print(f"      enc {t_enc*1000:.0f}ms  fsd {t_fsd*1000:.0f}ms  "
          f"sdec_total {t_sdec*1000:.0f}ms ({t_sdec*1000/max(new_token_count,1):.1f}ms/step)  "
          f"vits {t_vits*1000:.0f}ms")
    print(f"      e2e wall: {wall:.2f}s  audio: {audio_s:.2f}s  RTF: {wall/audio_s:.2f}")

    print(f"[5/5] writing {out_path}")
    sf.write(out_path, audio, sr, subtype="PCM_16")
    tok_path = out_path.with_suffix(".tokens.txt")
    tok_path.write_text(",".join(str(t) for t in sample_log) + "\n")


if __name__ == "__main__":
    main()
