"""Re-run only the GPT (s1) training stage after data_module.py was patched."""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from train_vector import patch_s1_config

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["_CUDA_VISIBLE_DEVICES"] = "0"
os.environ["hz"] = "25hz"
pp = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{REPO}:{REPO / 'GPT_SoVITS'}:{pp}"

s1_cfg = patch_s1_config()
print(f"=== Stage 3: GPT training (re-run) starting at {time.strftime('%H:%M:%S')} ===")
t0 = time.time()
proc = subprocess.run(
    [str(REPO / ".venv/bin/python"), "-s", "GPT_SoVITS/s1_train.py", "--config_file", str(s1_cfg)],
    cwd=REPO,
)
print(f"=== Stage 3: GPT training finished in {time.time() - t0:.0f}s, exit={proc.returncode} ===")
sys.exit(proc.returncode)
