"""GPT-SoVITS v2 fine-tune driver for the Vector dataset.

Patches s2.json (SoVITS) and s1longer-v2.yaml (GPT) to run on Apple Silicon
MPS (fp32, small batch) and invokes the official training scripts in sequence.

Outputs:
  SoVITS_weights_v2/<EXP>_e*_s*.pth      <- SoVITS fine-tune checkpoints
  GPT_weights_v2/<EXP>-e*.ckpt           <- GPT fine-tune checkpoints

EXP defaults to "vectorv2" (dataset/vectorv2.list, preprocessed by
preprocess_vector.sh into logs/vectorv2/).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
EXP = os.environ.get("EXP", "vectorv2")
EXP_DIR = REPO / "logs" / EXP
TMP = REPO / "TEMP"
TMP.mkdir(exist_ok=True)

S2_BASE = REPO / "GPT_SoVITS/configs/s2.json"
S1_BASE = REPO / "GPT_SoVITS/configs/s1longer-v2.yaml"

PRETRAINED_S2G = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
PRETRAINED_S2D = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"
PRETRAINED_S1 = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"

# Conservative settings for 20-utterance dataset on MPS
SOVITS_EPOCHS = 30
GPT_EPOCHS = 50
BATCH_SIZE = 4  # small batch, fp32, fits Apple Silicon unified memory


def patch_s2_config() -> Path:
    with open(S2_BASE) as f:
        cfg = json.load(f)

    cfg["train"]["fp16_run"] = False
    cfg["train"]["batch_size"] = BATCH_SIZE
    cfg["train"]["epochs"] = SOVITS_EPOCHS
    cfg["train"]["text_low_lr_rate"] = 0.4
    cfg["train"]["pretrained_s2G"] = PRETRAINED_S2G
    cfg["train"]["pretrained_s2D"] = PRETRAINED_S2D
    cfg["train"]["if_save_latest"] = True
    cfg["train"]["if_save_every_weights"] = True
    cfg["train"]["save_every_epoch"] = 5
    cfg["train"]["gpu_numbers"] = "0"
    cfg["train"]["grad_ckpt"] = False
    cfg["train"]["lora_rank"] = 32
    cfg["model"]["version"] = "v2"
    cfg["data"]["exp_dir"] = str(EXP_DIR)
    cfg["s2_ckpt_dir"] = str(EXP_DIR)
    cfg["save_weight_dir"] = "SoVITS_weights_v2"
    cfg["name"] = EXP
    cfg["version"] = "v2"

    out = TMP / "tmp_s2.json"
    out.write_text(json.dumps(cfg))
    return out


def patch_s1_config() -> Path:
    with open(S1_BASE) as f:
        cfg = yaml.safe_load(f)

    cfg["train"]["precision"] = "32"
    cfg["train"]["batch_size"] = BATCH_SIZE
    cfg["train"]["epochs"] = GPT_EPOCHS
    cfg["train"]["save_every_n_epoch"] = 5
    cfg["train"]["if_save_every_weights"] = True
    cfg["train"]["if_save_latest"] = True
    cfg["train"]["if_dpo"] = False
    cfg["train"]["half_weights_save_dir"] = "GPT_weights_v2"
    cfg["train"]["exp_name"] = EXP
    cfg["pretrained_s1"] = PRETRAINED_S1
    cfg["train_semantic_path"] = str(EXP_DIR / "6-name2semantic.tsv")
    cfg["train_phoneme_path"] = str(EXP_DIR / "2-name2text.txt")
    cfg["output_dir"] = str(EXP_DIR / "logs_s1_v2")
    cfg["data"]["num_workers"] = 0  # macOS fork-based DataLoader deadlocks on MPS

    out = TMP / "tmp_s1.yaml"
    out.write_text(yaml.dump(cfg, default_flow_style=False))
    return out


def run(stage: str, cmd: list):
    print(f"\n=== {stage} starting at {time.strftime('%H:%M:%S')} ===")
    print(" ".join(cmd))
    sys.stdout.flush()
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO)
    print(f"=== {stage} finished in {time.time() - t0:.0f}s, exit={proc.returncode} ===")
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def main():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["_CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["hz"] = "25hz"
    pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{REPO}:{REPO / 'GPT_SoVITS'}:{pp}"

    # Pre-create output dirs
    (REPO / "SoVITS_weights_v2").mkdir(exist_ok=True)
    (REPO / "GPT_weights_v2").mkdir(exist_ok=True)
    (EXP_DIR / "logs_s2_v2").mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "logs_s1_v2").mkdir(parents=True, exist_ok=True)

    s2_cfg = patch_s2_config()
    s1_cfg = patch_s1_config()

    py = str(REPO / ".venv/bin/python")

    run("Stage 2: SoVITS training", [py, "-s", "GPT_SoVITS/s2_train.py", "--config", str(s2_cfg)])
    run("Stage 3: GPT training", [py, "-s", "GPT_SoVITS/s1_train.py", "--config_file", str(s1_cfg)])

    print("\n=== ALL TRAINING DONE ===")
    print("SoVITS checkpoints:")
    for p in sorted((REPO / "SoVITS_weights_v2").glob(f"{EXP}*")):
        print(f"  {p}")
    print("GPT checkpoints:")
    for p in sorted((REPO / "GPT_weights_v2").glob(f"{EXP}*")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
