"""Download everything serve.py needs at runtime into /app, for the self-contained image.

- Pretrained frontend models (cnhubert, roberta) from the upstream lj1995/GPT-SoVITS repo.
- G2PW (Chinese polyphone) model from XXXXRT/GPT-SoVITS-Pretrained.
- fast-langdetect's lid.176.bin (used for 中英混合 segmentation).
- The fine-tuned Vector GPT + SoVITS checkpoints from the GitHub release at $VECTOR_WEIGHTS_URL.

Honors HF_ENDPOINT, so the same script works with hf-mirror.com in China.
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

APP = Path("/app")
PRETRAINED = APP / "GPT_SoVITS" / "pretrained_models"

WEIGHTS_URL = os.environ["VECTOR_WEIGHTS_URL"].rstrip("/")
GPT_FILE = os.environ.get("VECTOR_GPT_FILE", "vectorv2-e30.ckpt")
SOVITS_FILE = os.environ.get("VECTOR_SOVITS_FILE", "vectorv2_e30_s780.pth")
LID_URL = os.environ.get(
    "LID_URL", "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
)


def fetch(repo, filename, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo, filename)
    target = dest_dir / Path(filename).name
    shutil.copy(path, target)
    print(f"{repo}/{filename} -> {target} ({target.stat().st_size >> 20} MB)")


for name in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
    fetch("lj1995/GPT-SoVITS", f"chinese-hubert-base/{name}", PRETRAINED / "chinese-hubert-base")
for name in ("config.json", "tokenizer.json", "pytorch_model.bin"):
    fetch(
        "lj1995/GPT-SoVITS",
        f"chinese-roberta-wwm-ext-large/{name}",
        PRETRAINED / "chinese-roberta-wwm-ext-large",
    )

g2pw_zip = hf_hub_download("XXXXRT/GPT-SoVITS-Pretrained", "G2PWModel.zip")
with zipfile.ZipFile(g2pw_zip) as z:
    z.extractall(APP / "GPT_SoVITS" / "text")
print("G2PWModel ->", APP / "GPT_SoVITS" / "text" / "G2PWModel")


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)
    print(f"{url} -> {target} ({target.stat().st_size >> 20} MB)")


download(LID_URL, PRETRAINED / "fast_langdetect" / "lid.176.bin")
download(f"{WEIGHTS_URL}/{GPT_FILE}", APP / "GPT_weights_v2" / GPT_FILE)
download(f"{WEIGHTS_URL}/{SOVITS_FILE}", APP / "SoVITS_weights_v2" / SOVITS_FILE)

# hf_hub_download caches a second copy; drop it so it doesn't bloat the layer.
shutil.rmtree(Path.home() / ".cache" / "huggingface", ignore_errors=True)
