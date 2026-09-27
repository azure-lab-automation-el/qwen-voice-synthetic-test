#!/usr/bin/env bash
# Public console displays only this script invocation and a generic exit status.
set -euo pipefail
umask 077
# Install the public encryption binary before capturing sensitive work.
# If package installation fails, it prints no diagnostic to public logs.
if ! (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq age) >/dev/null 2>&1; then
  printf 'Unable to prepare private run. No synthesis attempted.\n' >&2
  exit 1
fi
LOG_FILE="$(mktemp)"
OUT_DIR=mysterious-output
KEY_FILE="$(mktemp)"
AGE_RECIPIENT='age1wa58xw6szfc34ghfrs4vqp0yx3sqjxwmsf7xa9m9yucvwyzxzdesj2u2r5'
finish() {
  status=$?
  trap - EXIT
  set +e
  age -r "$AGE_RECIPIENT" -o moon-run.log.age "$LOG_FILE" >/dev/null 2>&1 || status=1
  if [ -d "$OUT_DIR" ]; then
    tar -czf moon-output.tar.gz "$OUT_DIR" >/dev/null 2>&1 || status=1
    if [ -s moon-output.tar.gz ]; then
      age -r "$AGE_RECIPIENT" -o moon-output.tar.gz.age moon-output.tar.gz >/dev/null 2>&1 || status=1
    fi
  fi
  rm -f "$LOG_FILE" "$KEY_FILE" moon-output.tar.gz
  if [ "$status" -eq 0 ]; then
    printf 'Moon synthesis completed. Encrypted logs and output uploaded by the next step.\n'
  else
    printf 'Moon synthesis failed (exit %s). Inspect the encrypted log artifact.\n' "$status"
  fi
  exit "$status"
}
trap finish EXIT
{
  test -n "${MOON_AGE_KEY:-}"
  printf '%s\n' "$MOON_AGE_KEY" > "$KEY_FILE"
  mkdir -p mysterious-references "$OUT_DIR"
  age -d -i "$KEY_FILE" -o mysterious-references/moon.wav 1-moon.wav.age
  echo '48ce34759c7265f0fab5bceab0f896b789a160471785b30a34cd2c2b5883e5ea  mysterious-references/moon.wav' | sha256sum -c -
  printf '%s\n' 'There is a small secret in this lesson. Wait for the last page, and I will tell you what it is.' > mysterious-references/moon.txt
  test -x /opt/qwentts/build/qwen-tts
  test -s /opt/models/talker.gguf
  test -s /opt/models/codec.gguf
  python - <<'PY_G2P'
from renikud_onnx import G2P
from pathlib import Path
text='יֵשׁ לִי סוֹד קָטָן... וַאֲנִי אֲגַלֶּה לְךָ אוֹתוֹ רַק בְּסוֹף הַשִּׁיעוּר.'
g2p=G2P(model_path='/opt/models/renikud/model.onnx',niqqud='use',datastore=None)
ipa=g2p.phonemize(text,speaker=2,target_speaker=2)
Path('mysterious-output/prompt.ipa').write_text(ipa,encoding='utf-8')
PY_G2P
  python generate.py mysterious-hebrew
} > "$LOG_FILE" 2>&1
