#!/usr/bin/env bash
# Drives Vector through 16 utterances via wire-pod SDK while user records.
# Pacing: ~0.42s/word + 2s tail buffer keeps Vector finished before the next line.

set -u
HOST="http://escapepod.local"
SERIAL="00408959"

say() {
  local delay=$1
  local text=$2
  printf '[%s] [%4.1fs] %s\n' "$(date +%H:%M:%S)" "$delay" "$text"
  curl -sS --max-time 10 \
    -G "${HOST}/api-sdk/say_text" \
    --data-urlencode "serial=${SERIAL}" \
    --data-urlencode "text=${text}" >/dev/null
  sleep "$delay"
}

# 3-second pre-roll so user has clean head silence to trim later
echo "=== pre-roll 3s of silence ==="
sleep 3

echo "=== A: Rainbow Passage continuation ==="
say 6.5  "Throughout the centuries people have explained the rainbow in various ways."
say 5.5  "Some have accepted it as a miracle without physical explanation."
say 7.5  "To the Hebrews it was a token that there would be no more universal floods."
say 7.5  "The Greeks used to imagine that it was a sign from the gods to foretell war."
say 9.0  "The Norsemen considered the rainbow as a bridge over which the gods passed from earth to the sky."

echo "=== B: Harvard Sentences ==="
say 4.0  "The birch canoe slid on the smooth planks."
say 4.0  "Glue the sheet to the dark blue background."
say 4.0  "It's easy to tell the depth of a well."
say 4.0  "These days a chicken leg is a rare dish."
say 3.5  "Rice is often served in round bowls."

echo "=== C: Vector native phrases ==="
say 3.0  "Hello, my name is Vector."
say 3.0  "I'm a small home robot."
say 2.5  "Sure, why not."
say 3.5  "I don't know what you mean."
say 3.0  "Have a nice day."
say 2.5  "Goodbye!"

echo "=== 2s tail silence ==="
sleep 2

echo "=== releasing behavior control ==="
curl -sS --max-time 10 -G "${HOST}/api-sdk/release_behavior_control" \
  --data-urlencode "serial=${SERIAL}" >/dev/null
echo "DONE"
