"""End-to-end TTS via ONNX Runtime — fp32 or int8-hybrid.

Borrows upstream GPT-SoVITS's text frontend (get_phones_and_bert handles
中英混合) and ssl/vq prompt extraction, then drives the four ONNX modules
manually. AR loop adapted from genie_tts.Core.Inference.t2s_cpu and from
upstream GPT_SoVITS/onnx_export.py.

Usage:
    python run_onnx.py --t2s-dir onnx/vector --vits-dir onnx/vector \
        --text "你好，我是 Vector。" --out output/fp32.wav
    python run_onnx.py --t2s-dir onnx/vector_int8 --vits-dir onnx/vector \
        --text "你好，我是 Vector。" --out output/int8.wav

Both runs use the same seed so semantic-token divergence is purely from quant.
"""

import argparse
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

import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort

DEFAULT_REF_AUDIO = str(REPO.parent / "samples" / "vector_ref_best.wav")
DEFAULT_REF_TEXT = (
    "These take the shape of a long round arch, with its path high above, "
    "and its two ends apparently beyond the horizon."
)
DEFAULT_REF_LANG = "en"  # internal code, not 英文
DEFAULT_GPT_PATH = "GPT_weights_v2/vector-e50.ckpt"
DEFAULT_SOVITS_PATH = "SoVITS_weights_v2/vector_e30_s780.pth"
EOS_TOKEN = 1024
MAX_AR_STEPS = 1500


def make_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # SimplifiedLayerNormFusion crashes on fp16-converted t2s graphs because
    # the conversion renames Cast nodes the fusion expects to find. Safe to
    # disable for fp32 too (it's a minor opt).
    return ort.InferenceSession(
        str(path),
        sess_options=so,
        providers=["CPUExecutionProvider"],
        disabled_optimizers=["SimplifiedLayerNormFusion"],
    )


def init_upstream(sovits_path: str, gpt_path: str):
    """Load upstream globals (hps, ssl_model, vq_model, t2s_model). We only
    need ssl_model + vq_model.extract_latent for prompt prep, plus
    get_phones_and_bert for text. The PyTorch t2s/vits aren't used at runtime."""
    from GPT_SoVITS.inference_webui import change_sovits_weights, change_gpt_weights
    try:
        next(change_sovits_weights(sovits_path=sovits_path))
    except (StopIteration, UnboundLocalError, Exception):
        pass
    change_gpt_weights(gpt_path=gpt_path)
    from GPT_SoVITS import inference_webui as iw
    return iw


def prep_ref(iw, ref_wav_path: str):
    """Run upstream ssl_model + vq_model.extract_latent to produce the
    inputs that our encoder ONNX expects. Returns (ssl_content_np, prompts_np)
    where ssl_content has shape (1, 768, T) and prompts is (1, P) int64."""
    device = "cpu"
    wav16k, _ = librosa.load(ref_wav_path, sr=16000)
    if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
        print(f"warn: ref audio is {wav16k.shape[0]/16000:.1f}s — upstream wants 3-10s")
    # Pad with 0.3s silence (matches get_tts_wav)
    zero = np.zeros(int(iw.hps.data.sampling_rate * 0.3), dtype=np.float32)
    wav16k = torch.from_numpy(np.concatenate([wav16k, np.zeros(int(16000 * 0.3), dtype=np.float32)]))
    wav16k = wav16k.float().to(device)
    with torch.no_grad():
        ssl = iw.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2).float()
        codes = iw.vq_model.extract_latent(ssl)  # (1, K, T_codes) but we use first codebook
        prompt_semantic = codes[0, 0].unsqueeze(0)  # (1, P)
    return ssl.cpu().numpy().astype(np.float32), prompt_semantic.cpu().numpy().astype(np.int64)


def get_text(iw, text: str, lang: str):
    """Return (phones_np[1,T] int64, bert_np[T,1024] float32) via upstream."""
    from GPT_SoVITS.inference_webui import get_phones_and_bert
    phones, bert, norm_text = get_phones_and_bert(text, lang, "v2")
    phones_np = np.asarray(phones, dtype=np.int64)[None, :]
    bert_np = bert.detach().cpu().numpy().T.astype(np.float32) if bert.ndim == 2 and bert.shape[0] == 1024 else bert.detach().cpu().numpy().astype(np.float32)
    # encoder expects ref_bert/text_bert as (T, 1024)
    if bert_np.shape[0] == 1024:
        bert_np = bert_np.T
    return phones_np, bert_np


def load_ref_audio_for_vits(ref_wav_path: str, sr: int):
    audio, _ = librosa.load(ref_wav_path, sr=sr)
    return audio.astype(np.float32)[None, :]  # (1, T)


def run_pipeline(sessions, ref_seq, text_seq, ref_bert, text_bert, ssl_content,
                 ref_audio_for_vits, max_steps=MAX_AR_STEPS):
    enc, fsd, sdec, vits = sessions["enc"], sessions["fsd"], sessions["sdec"], sessions["vits"]

    t0 = time.perf_counter()
    x, prompts = enc.run(None, {
        "ref_seq": ref_seq, "text_seq": text_seq,
        "ref_bert": ref_bert, "text_bert": text_bert,
        "ssl_content": ssl_content,
    })
    t_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    y, k, v, y_emb, x_example = fsd.run(None, {"x": x, "prompts": prompts})
    t_fsd = time.perf_counter() - t0

    t0 = time.perf_counter()
    sdec_input_names = [i.name for i in sdec.get_inputs()]
    sdec_output_names = [o.name for o in sdec.get_outputs()]
    y_idx = sdec_output_names.index("y")
    k_idx = sdec_output_names.index("k")
    v_idx = sdec_output_names.index("v")
    yemb_idx = sdec_output_names.index("y_emb")
    samples_idx = sdec_output_names.index("samples")

    new_token_count = 1  # fsdec already produced one
    eos_seen = False
    sample_log = [int(y[0, -1])]  # fsdec's emitted token
    for step in range(max_steps):
        feed = {
            sdec_input_names[0]: y,
            sdec_input_names[1]: k,
            sdec_input_names[2]: v,
            sdec_input_names[3]: y_emb,
            sdec_input_names[4]: x_example,
        }
        outs = sdec.run(None, feed)
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

    # Strip EOS, expand to (1, 1, T) for vits
    y[0, -1] = 0
    pred_semantic = y[:, -new_token_count:][None, :, :].astype(np.int64)

    t0 = time.perf_counter()
    audio = vits.run(None, {
        "text_seq": text_seq,
        "pred_semantic": pred_semantic,
        "ref_audio": ref_audio_for_vits,
    })[0]
    t_vits = time.perf_counter() - t0

    return {
        "audio": audio,
        "tokens": sample_log,
        "ar_steps": new_token_count,
        "eos_seen": eos_seen,
        "timing": {"enc": t_enc, "fsd": t_fsd, "sdec": t_sdec, "vits": t_vits},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t2s-dir", required=True, help="dir with vector_t2s_{encoder,fsdec,sdec}.onnx")
    p.add_argument("--vits-dir", default=None, help="dir with vector_vits.onnx (default: same as --t2s-dir)")
    p.add_argument("--text", required=True)
    p.add_argument("--lang", default="all_zh", help="upstream lang code: all_zh, en, all_en, zh, ja, all_ja, auto, auto_yue, all_yue")
    p.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    p.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    p.add_argument("--ref-lang", default=DEFAULT_REF_LANG)
    p.add_argument("--out", required=True)
    p.add_argument("--gpt", default=DEFAULT_GPT_PATH)
    p.add_argument("--sovits", default=DEFAULT_SOVITS_PATH)
    p.add_argument("--threads", type=int, default=max((os.cpu_count() or 2) // 2, 1))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=MAX_AR_STEPS)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t2s_dir = Path(args.t2s_dir)
    vits_dir = Path(args.vits_dir) if args.vits_dir else t2s_dir
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] init upstream models for text/ssl frontend ...")
    t = time.time()
    iw = init_upstream(args.sovits, args.gpt)
    print(f"      done in {time.time() - t:.1f}s")

    print(f"[2/5] preparing reference (ssl_content + prompt_semantic) ...")
    t = time.time()
    ssl_content, prompts_for_text = prep_ref(iw, args.ref_audio)
    ref_seq, ref_bert = get_text(iw, args.ref_text, args.ref_lang)
    text_seq, text_bert = get_text(iw, args.text, args.lang)
    sr = int(iw.hps.data.sampling_rate)
    ref_audio_v = load_ref_audio_for_vits(args.ref_audio, sr)
    print(f"      ref_seq={ref_seq.shape} text_seq={text_seq.shape} ssl={ssl_content.shape}  done in {time.time() - t:.1f}s")

    print(f"[3/5] loading ONNX sessions  (t2s={t2s_dir.name}, vits={vits_dir.name}) threads={args.threads} ...")
    t = time.time()
    sessions = {
        "enc": make_session(t2s_dir / "vector_t2s_encoder.onnx", args.threads),
        "fsd": make_session(t2s_dir / "vector_t2s_fsdec.onnx", args.threads),
        "sdec": make_session(t2s_dir / "vector_t2s_sdec.onnx", args.threads),
        "vits": make_session(vits_dir / "vector_vits.onnx", args.threads),
    }
    print(f"      done in {time.time() - t:.1f}s")

    print(f"[4/5] running pipeline ...")
    t_full = time.perf_counter()
    res = run_pipeline(
        sessions, ref_seq, text_seq, ref_bert, text_bert, ssl_content, ref_audio_v,
        max_steps=args.max_steps,
    )
    wall = time.perf_counter() - t_full

    audio = res["audio"]
    audio_s = len(audio) / sr
    tim = res["timing"]
    print(f"      AR steps: {res['ar_steps']}  EOS: {res['eos_seen']}")
    print(f"      enc {tim['enc']*1000:.0f}ms  fsd {tim['fsd']*1000:.0f}ms  "
          f"sdec_total {tim['sdec']*1000:.0f}ms ({tim['sdec']*1000/max(res['ar_steps'],1):.1f}ms/step)  "
          f"vits {tim['vits']*1000:.0f}ms")
    print(f"      e2e wall: {wall:.2f}s  audio: {audio_s:.2f}s  RTF: {wall/audio_s:.2f}")

    print(f"[5/5] writing {out_path}")
    sf.write(out_path, audio, sr, subtype="PCM_16")

    # Save tokens to a sidecar for cross-run divergence comparison
    tok_path = out_path.with_suffix(".tokens.txt")
    tok_path.write_text(",".join(str(t) for t in res["tokens"]) + "\n")
    print(f"      tokens -> {tok_path}")


if __name__ == "__main__":
    main()
