"""FastAPI TTS server for the fine-tuned Vector model.

Endpoint:
  POST /tts
    body JSON: {"text": str, "lang": "zh"|"en"|"中英混合"|... , optionally:
                 "ref_audio_path": str, "ref_text": str, "ref_lang": str}
    returns: audio/wav (32kHz mono PCM)

Models are loaded once on startup.  A single global lock serializes inference
since GPT-SoVITS keeps mutable global state (loaded weights) across calls.
"""

import hashlib
import io
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

# PT 2.6 default weights_only=True breaks our trusted ckpts.
_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})

# inference_webui.py loads a default GPT/SoVITS weight at IMPORT time (it reads
# weight.json or scans the weights dirs). If weight.json still points at a
# deleted v1 checkpoint, the import crashes. Pin the lowercase gpt_path/
# sovits_path env vars it reads BEFORE the import so it loads vectorv2 directly.
os.environ.setdefault("gpt_path", os.environ.get("GPT_PATH", "GPT_weights_v2/vectorv2-e30.ckpt"))
os.environ.setdefault("sovits_path", os.environ.get("SOVITS_PATH", "SoVITS_weights_v2/vectorv2_e30_s780.pth"))

# Imported after the patch.
from GPT_SoVITS.inference_webui import (  # noqa: E402
    change_gpt_weights,
    change_sovits_weights,
    get_tts_wav,
)
from tools.i18n.i18n import I18nAuto  # noqa: E402

_i18n = I18nAuto()


def _norm_lang(label: str) -> str:
    """Map Chinese-display labels (英文/中文/中英混合) to internal codes via i18n.
    Pass-through for already-translated values."""
    try:
        return _i18n(label)
    except Exception:
        return label


def _hash_tensor(t: torch.Tensor) -> str:
    return hashlib.md5(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def install_ref_caches(iw, *, phb_cap: int = 256) -> dict[str, OrderedDict]:
    """Monkey-patch upstream get_tts_wav's expensive ref-side helpers to memoize.

    Cached:
      - ssl_model.model(wav)               keyed by wav bytes
      - vq_model.extract_latent(ssl)       keyed by ssl bytes
      - get_phones_and_bert(text, lang, …) keyed by args (LRU bounded)
      - get_spepc(hps, path, dtype, …)     keyed by args
    NOT cached: t2s_model.infer_panel, vq_model.decode (depend on target text).

    Reference inputs are tiny (~1-2 entries) so unbounded except phones/bert
    which is per-target-text and capped at phb_cap entries.
    """
    caches = {
        "ssl": OrderedDict(),
        "extract_latent": OrderedDict(),
        "phb": OrderedDict(),
        "spepc": OrderedDict(),
    }

    real_ssl = iw.ssl_model.model

    class _CachedHFSSL(torch.nn.Module):
        """nn.Module wrapper around the HF cnhubert model that memoizes forward
        by content hash of the input wav. Must inherit nn.Module because
        upstream's ssl_model is a Module and Module.__setattr__ rejects
        non-Module assignments to child slots."""

        def __init__(self, real):
            super().__init__()
            self.real = real  # registered as child module

        def forward(self, x, *args, **kwargs):
            key = _hash_tensor(x) if isinstance(x, torch.Tensor) else None
            if key is not None and key in caches["ssl"]:
                return caches["ssl"][key]
            res = self.real(x, *args, **kwargs)
            if key is not None:
                caches["ssl"][key] = res
            return res

        def __getattr__(self, name):
            # nn.Module.__getattr__ resolves _parameters/_buffers/_modules; fall
            # through to wrapped object for anything else (config, etc.).
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.real, name)

    iw.ssl_model.model = _CachedHFSSL(real_ssl)

    real_extract = iw.vq_model.extract_latent

    def _cached_extract(ssl_content):
        key = _hash_tensor(ssl_content) if isinstance(ssl_content, torch.Tensor) else None
        if key is not None and key in caches["extract_latent"]:
            return caches["extract_latent"][key]
        res = real_extract(ssl_content)
        if key is not None:
            caches["extract_latent"][key] = res
        return res

    iw.vq_model.extract_latent = _cached_extract

    real_phb = iw.get_phones_and_bert

    def _cached_phb(text, language, version, final=False):
        key = (text, language, version, bool(final))
        if key in caches["phb"]:
            caches["phb"].move_to_end(key)
            return caches["phb"][key]
        res = real_phb(text, language, version, final)
        caches["phb"][key] = res
        if len(caches["phb"]) > phb_cap:
            caches["phb"].popitem(last=False)
        return res

    iw.get_phones_and_bert = _cached_phb

    real_spepc = iw.get_spepc

    def _cached_spepc(hps, filename, dtype, device, is_v2pro=False):
        key = (filename, int(hps.data.sampling_rate), str(dtype), bool(is_v2pro))
        if key in caches["spepc"]:
            return caches["spepc"][key]
        res = real_spepc(hps, filename, dtype, device, is_v2pro)
        caches["spepc"][key] = res
        return res

    iw.get_spepc = _cached_spepc

    return caches


def boost_volume(audio: np.ndarray, target_peak_dbfs: float, gain_db: float) -> np.ndarray:
    """Optional level adjustment. Both controls default to disabled
    (target_peak_dbfs >= 0 means no peak-normalize; gain_db == 0 means no gain),
    in which case the audio passes through unchanged — the vectorv2 model is
    already loud enough at source (-3dBFS dataset), no artificial boost needed.
    When enabled: peak-normalize to target_peak_dbfs, then apply gain_db with
    tanh soft-clip on overflow."""
    if target_peak_dbfs >= 0 and gain_db == 0:
        return audio
    if audio.dtype != np.int16:
        audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    peak = int(np.abs(audio).max())
    if peak == 0:
        return audio
    target = 32767.0 * (10.0 ** (target_peak_dbfs / 20.0))
    scale = (target / peak) * (10.0 ** (gain_db / 20.0))
    f = audio.astype(np.float32) * scale
    if np.abs(f).max() > 32767:
        # Soft saturate: linear up to ~70% then tanh-shaped knee
        f = np.tanh(f / 32767.0) * 32767.0
    return f.astype(np.int16)

GPT_PATH = os.environ.get("GPT_PATH", "GPT_weights_v2/vectorv2-e30.ckpt")
SOVITS_PATH = os.environ.get("SOVITS_PATH", "SoVITS_weights_v2/vectorv2_e30_s780.pth")
DEFAULT_REF_AUDIO = os.environ.get("REF_AUDIO", "/samples/vector_001.wav")
DEFAULT_REF_TEXT = os.environ.get(
    "REF_TEXT",
    "Hi there. My name is Vector. It's very nice to meet you today.",
)
DEFAULT_REF_LANG = os.environ.get("REF_LANG", "英文")
USE_OPENVINO = os.environ.get("USE_OPENVINO", "0") not in ("0", "false", "False", "")

app = FastAPI(title="vector-tts", version="0.1")
_lock = threading.Lock()
_warm = {"done": False}
_caches: dict[str, OrderedDict] = {}


class TTSReq(BaseModel):
    text: str
    lang: str = "中英混合"
    ref_audio_path: str | None = None
    ref_text: str | None = None
    ref_lang: str | None = None
    # Volume controls — DISABLED by default (vectorv2 dataset is -3dBFS source,
    # loud enough without artificial boost). Set gain_db>0 and/or
    # target_peak_dbfs<0 per-request to re-enable. See boost_volume().
    target_peak_dbfs: float = 0.0
    gain_db: float = 0.0
    # Fixed seed makes output reproducible per input AND identical across
    # machines: with top_p=1/temperature=1 (no nucleus truncation) the AR
    # sampling follows the same token path on ARM (Mac) and x86 (NAS) — verified
    # MFCC-identical. Keep top_p=1; do NOT use conservative top_p<1, which
    # diverges cross-platform at the nucleus boundary.
    seed: int = 42


@app.on_event("startup")
def _startup():
    print(f"[startup] loading SoVITS: {SOVITS_PATH}")
    # change_sovits_weights is a generator that yields gradio UI updates. The
    # second yield references locals only defined when prompt_language is set —
    # upstream itself wraps the call in try/except (inference_webui.py:404).
    # The model-loading side effects (incl. setting `hps`) complete before the
    # broken yield, so this is safe.
    try:
        next(change_sovits_weights(sovits_path=SOVITS_PATH))
    except (StopIteration, UnboundLocalError, Exception):
        pass
    print(f"[startup] loading GPT:    {GPT_PATH}")
    change_gpt_weights(gpt_path=GPT_PATH)

    if USE_OPENVINO:
        try:
            import openvino  # noqa: F401  (registers torch.compile backend)
            from GPT_SoVITS import inference_webui as iw

            t0 = time.time()
            print("[startup] torch.compile backend=openvino on vq_model …")
            # Only compile vq_model (SoVITS+vocoder, deterministic, the heavy
            # path). t2s_model uses top-p sampling which is sensitive to the
            # numerical jitter introduced by compile.
            try:
                iw.vq_model = torch.compile(iw.vq_model, backend="openvino", mode="reduce-overhead")
            except Exception as e:
                print(f"  vq_model compile failed (skipping): {e}")
            print(f"[startup] torch.compile attempts done in {time.time() - t0:.1f}s "
                  "(actual graph compile happens on first /tts call)")
        except ImportError:
            print("[startup] openvino not installed — skipping torch.compile")

    # Ref-side caches save ~1s/call BUT change RNG consumption before t2s
    # sampling (cached ssl/phb skip ops that would otherwise advance the RNG),
    # so a cached run produces a DIFFERENT seeded take than a non-cached render.
    # Disabled by default so production output is bit-identical to the
    # reference renders (sweep / Mac). Re-enable with USE_REF_CACHE=1 only if
    # you don't care about matching a specific reference take.
    if os.environ.get("USE_REF_CACHE", "0") not in ("0", "false", "False", ""):
        print("[startup] installing ref-side caches (ssl/extract_latent/phb/spepc) …")
        from GPT_SoVITS import inference_webui as iw
        _caches.update(install_ref_caches(iw))

    if Path(DEFAULT_REF_AUDIO).exists():
        print(f"[startup] warm-up call (ref={DEFAULT_REF_AUDIO})…")
        try:
            t0 = time.time()
            torch.manual_seed(42)
            np.random.seed(42)
            list(get_tts_wav(
                ref_wav_path=DEFAULT_REF_AUDIO,
                prompt_text=DEFAULT_REF_TEXT,
                prompt_language=_norm_lang(DEFAULT_REF_LANG),
                text="你好。", text_language=_norm_lang("中文"),
                top_p=1, temperature=1,
            ))
            print(f"[startup] warm-up done in {time.time() - t0:.1f}s")
            _warm["done"] = True
        except Exception as e:
            print(f"[startup] warm-up failed (non-fatal): {e}")
    else:
        print(f"[startup] ref audio missing: {DEFAULT_REF_AUDIO} (warm-up skipped)")


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "warm": _warm["done"],
        "cache_sizes": {k: len(v) for k, v in _caches.items()},
    }


@app.post("/tts")
def tts(req: TTSReq):
    ref_audio = req.ref_audio_path or DEFAULT_REF_AUDIO
    ref_text = req.ref_text or DEFAULT_REF_TEXT
    ref_lang = req.ref_lang or DEFAULT_REF_LANG

    if not Path(ref_audio).exists():
        raise HTTPException(400, f"reference audio not found: {ref_audio}")

    with _lock:
        t0 = time.time()
        torch.manual_seed(req.seed)
        np.random.seed(req.seed)
        try:
            result = list(get_tts_wav(
                ref_wav_path=ref_audio,
                prompt_text=ref_text,
                prompt_language=_norm_lang(ref_lang),
                text=req.text,
                text_language=_norm_lang(req.lang),
                top_p=1, temperature=1,
            ))
        except Exception as e:
            raise HTTPException(500, f"synth failed: {e}")
        sr, audio = result[-1]
        wall = time.time() - t0

    audio = boost_volume(audio, req.target_peak_dbfs, req.gain_db)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    audio_s = len(audio) / sr
    new_peak = int(np.abs(audio).max())
    new_rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    headers = {
        "X-Sample-Rate": str(sr),
        "X-Audio-Duration": f"{audio_s:.3f}",
        "X-Inference-Wall": f"{wall:.3f}",
        "X-RTF": f"{wall / audio_s:.3f}",
        "X-Peak-dBFS": f"{20*np.log10(new_peak/32767) if new_peak else -120:.1f}",
        "X-RMS-dBFS": f"{20*np.log10(new_rms/32767) if new_rms else -120:.1f}",
    }
    return Response(content=buf.getvalue(), media_type="audio/wav", headers=headers)


@app.get("/")
def index():
    return JSONResponse({
        "service": "vector-tts",
        "endpoints": {"GET /healthz": "liveness", "POST /tts": "synth"},
        "models": {"sovits": SOVITS_PATH, "gpt": GPT_PATH},
        "ref_audio": DEFAULT_REF_AUDIO,
    })
