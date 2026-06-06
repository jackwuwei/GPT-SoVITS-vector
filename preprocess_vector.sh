#!/usr/bin/env bash
# GPT-SoVITS v2 preprocessing pipeline for the Vector dataset, MPS-friendly.

set -e
cd "$(dirname "$0")"

source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/GPT_SoVITS:$PYTHONPATH"
export PYTORCH_ENABLE_MPS_FALLBACK=1   # let MPS silently fall back to CPU for unsupported ops

EXP_NAME="${EXP_NAME:-vector}"
LIST="${LIST:-dataset/vector.list}"
OPT_DIR="logs/${EXP_NAME}"
BERT_DIR="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
HUBERT_DIR="GPT_SoVITS/pretrained_models/chinese-hubert-base"
PRETRAINED_S2G="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
S2_CONFIG="GPT_SoVITS/configs/s2.json"

mkdir -p "$OPT_DIR"

common() {
    export inp_text="$LIST"
    export inp_wav_dir=""
    export exp_name="$EXP_NAME"
    export opt_dir="$OPT_DIR"
    export i_part="0"
    export all_parts="1"
    export _CUDA_VISIBLE_DEVICES="0"
    export is_half="False"
    export version="v2"
}

echo "=== Stage 1a: BERT feature extraction ==="
common
export bert_pretrained_dir="$BERT_DIR"
python -s GPT_SoVITS/prepare_datasets/1-get-text.py
# merge per-part output to single file (1 part here, but matches webui logic)
mv "${OPT_DIR}/2-name2text-0.txt" "${OPT_DIR}/2-name2text.txt"

echo "=== Stage 1b: HuBERT (cnhubert) feature + 32k wav resample ==="
common
export cnhubert_base_dir="$HUBERT_DIR"
export sv_path="GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
python -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py

echo "=== Stage 1c: Semantic token extraction ==="
common
export pretrained_s2G="$PRETRAINED_S2G"
export s2config_path="$S2_CONFIG"
python -s GPT_SoVITS/prepare_datasets/3-get-semantic.py
# merge per-part tsv
{
    echo -e "item_name\tsemantic_audio"
    cat "${OPT_DIR}/6-name2semantic-0.tsv"
} > "${OPT_DIR}/6-name2semantic.tsv"
rm "${OPT_DIR}/6-name2semantic-0.tsv"

echo "=== preprocess complete ==="
ls -la "$OPT_DIR"
