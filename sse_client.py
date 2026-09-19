#!/usr/bin/env python3
"""SSE client for the vector-tts /tts-stream endpoint.

Drives `curl -sN` as a subprocess so we get reliable streaming (Python's
stdlib urlopen buffers SSE in some configurations).

Usage:
  ./sse_client.py "你好，我是 Vector，今天天气很好，要不要一起出去玩？"
  ./sse_client.py --no-play "..."     # parse + time only, don't play
  ./sse_client.py --host 127.0.0.1 --port 8020 "..."
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time

DEFAULT_TEXT = "你好，我是 Vector，今天天气很好，要不要一起出去玩？"


def stream(host: str, port: int, text: str, lang: str, play: bool) -> None:
    url = f"http://{host}:{port}/tts-stream"
    body = json.dumps({"text": text, "lang": lang}, ensure_ascii=False)

    print(f"POST {url}\n  text={text!r}\n  lang={lang!r}")
    t_request = time.time()
    first_byte = None

    proc = subprocess.Popen(
        ["curl", "-sN", "--no-buffer", "-X", "POST", url,
         "-H", "Content-Type: application/json",
         "--data-binary", body],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None

    chunks_played = 0
    buf = b""
    with tempfile.TemporaryDirectory() as tmp:
        while True:
            piece = proc.stdout.read1(8192)
            if not piece:
                break
            buf += piece
            while b"\n\n" in buf:
                evt_raw, buf = buf.split(b"\n\n", 1)
                line = evt_raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].lstrip()
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError as e:
                    print(f"  [parse fail: {e}] {payload[:80]}")
                    continue
                arrival = time.time() - t_request
                if first_byte is None:
                    first_byte = arrival
                if evt.get("done"):
                    print(f"\n=== DONE ===")
                    print(f"  chunks:        {evt.get('n_chunks')}")
                    print(f"  total_audio_s: {evt.get('total_audio_s')}")
                    print(f"  total_wall_s:  {evt.get('total_wall_s')}")
                    print(f"  total_rtf:     {evt.get('total_rtf')}")
                    print(f"  first_byte:    {first_byte:.3f}s")
                elif "error" in evt:
                    print(f"  [{arrival:6.2f}s] idx={evt['idx']} ERROR: {evt['error']}")
                else:
                    print(f"  [{arrival:6.2f}s] idx={evt['idx']:>2} "
                          f"audio={evt['audio_s']:5.2f}s wall={evt['wall_s']:5.2f}s "
                          f"rtf={evt['rtf']:5.2f}  '{evt['sentence']}'")
                    if play:
                        wav_path = os.path.join(tmp, f"chunk_{chunks_played:03d}.wav")
                        with open(wav_path, "wb") as f:
                            f.write(base64.b64decode(evt["audio_b64"]))
                        # afplay blocks → chunks play in arrival order, which is
                        # exactly the streaming UX we want.
                        subprocess.run(["afplay", wav_path], check=False)
                        chunks_played += 1

    rc = proc.wait()
    if rc != 0:
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        print(f"curl exit {rc}: {err}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="?", default=DEFAULT_TEXT)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8020)
    p.add_argument("--lang", default="中英混合")
    p.add_argument("--no-play", action="store_true", help="parse + time only, don't afplay")
    args = p.parse_args()
    stream(args.host, args.port, args.text, args.lang, play=not args.no_play)


if __name__ == "__main__":
    main()
