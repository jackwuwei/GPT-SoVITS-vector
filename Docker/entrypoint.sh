#!/usr/bin/env bash
# Modes:
#   serve   — FastAPI HTTP server on :8020 (default; what wire-pod talks to)
#   bench   — RTF microbench (one-shot)
#   shell   — bash for debugging
set -e

case "${1:-serve}" in
  serve)
    echo "=== vector-tts FastAPI on 0.0.0.0:8020 ==="
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-?}  MKL_NUM_THREADS=${MKL_NUM_THREADS:-?}"
    cd /app
    exec python -m uvicorn serve:app --host 0.0.0.0 --port 8020
    ;;
  bench)
    echo "=== CPU info ==="
    grep -E '^model name|^cpu MHz|^cpu cores' /proc/cpuinfo | head -8
    echo
    echo "=== Mem ==="
    free -h | head -3
    echo
    cd /app
    exec python /app/bench_rtf.py
    ;;
  shell)
    exec /bin/bash
    ;;
  *)
    exec "$@"
    ;;
esac
