#!/usr/bin/env python3
"""Extract Miku-only (Character2dId 21) spoken lines from PJSK card assets.

Two problems this fixes, both caught by the user rather than by my first pass:

1. **Not all voices are Miku.** A card belongs to Miku, but its episode is a
   conversation: Ichika, Saki, Honami, Kanade, … all speak in it. Filtering by
   card/character 21 gets you ~2.8% Miku; the rest is other cast. The speaker is
   named per-voice by `Voices[].Character2dId`, so filter on that, not on the card.

2. **The text is Japanese.** The transcript lives in `Body` on the same TalkData
   record as the voice, so we can recover it exactly instead of paying for ASR —
   and it is the *original* text, which is what the katakana frontend wants anyway
   (Japanese-orthography input through OpenJTalk).

The card `.asset` files are plain JSON (not Unity bundles), served at
  character/member/res021_no<NNN>/<card>_miku<nn>.asset
and each holds a `TalkData` list of {TalkCharacters, WindowDisplayName, Body, Voices[]}.

Output:
  <out>/*.wav          only Miku-spoken clips, RIFF-validated
  <out>/manifest.list  "relpath|text" — GPT-SoVITS-ready, one line per clip

Usage: python3 scripts/extract_miku_lines.py [--out data/miku21] [--speaker 21]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

BUCKET = "https://storage.sekai.best/sekai-jp-assets/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
UA = "MikuTTS/1.0"  # CDN 403s urllib's default agent


def _open(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60)


def list_keys(prefix: str, suffix: str = ".asset") -> list[str]:
    keys: list[str] = []
    token = ""
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(_open(BUCKET + "?" + urllib.parse.urlencode(q)).read())
        keys += [c.findtext(NS + "Key") for c in root.findall(NS + "Contents")]
        if (root.findtext(NS + "IsTruncated") or "").lower() != "true":
            break
        token = root.findtext(NS + "NextContinuationToken") or ""
        if not token:
            break
    return [k for k in keys if k.endswith(suffix)]


def fetch_json(key: str) -> dict | None:
    try:
        return json.loads(_open(BUCKET + urllib.parse.quote(key)).read())
    except Exception as e:  # noqa: BLE001
        print(f"  asset read failed {key}: {e}", file=sys.stderr)
        return None


def fetch_wav(voice_id: str, card_dir: str, dest: str) -> int:
    """Download one voice; verify RIFF so error bodies never land on disk."""
    if os.path.exists(dest) and os.path.getsize(dest) > 44:
        return 0
    url = BUCKET + urllib.parse.quote(f"sound/card_scenario/voice/{card_dir}/{voice_id}.wav")
    data = _open(url).read()
    if not data.startswith(b"RIFF"):
        raise ValueError(f"not RIFF ({len(data)} B): {card_dir}/{voice_id}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    return len(data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/miku21")
    ap.add_argument("--speaker", type=int, default=21, help="Character2dId to keep (21 = Miku)")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("listing card assets …", flush=True)
    assets = list_keys("character/member/res021")
    print(f"{len(assets)} card assets")

    # A VoiceId is reused across cards and a card can ship several variant dirs
    # (021001_miku01 / 021001_miku02). The asset basename is NOT reliable for the
    # voice directory — resolving against the real listing is. Build vid -> [dirs].
    print("listing voice files …", flush=True)
    voice_files = list_keys("sound/card_scenario/voice/021", ".wav")
    vid_to_dirs: dict[str, list[str]] = {}
    for k in voice_files:
        parts = k.split("voice/", 1)[1].split("/")
        if len(parts) != 2:
            continue
        d, fn = parts
        vid_to_dirs.setdefault(fn.rsplit(".", 1)[0], []).append(d)
    print(f"{len(voice_files)} voice files in {len({d for ds in vid_to_dirs.values() for d in ds})} card dirs")

    jobs: list[tuple[str, str, str]] = []   # (voice_id, card_dir, text)
    seen: set[str] = set()
    missing = 0
    for key in assets:
        d = fetch_json(key)
        if not d:
            continue
        for td in d.get("TalkData") or []:
            body = (td.get("Body") or "").strip()
            for v in td.get("Voices") or []:
                # THE filter that was missing: speaker id, not card owner
                if v.get("Character2dId") != args.speaker:
                    continue
                vid = v.get("VoiceId")
                if not vid or not body or vid in seen:
                    continue
                dirs = vid_to_dirs.get(vid)
                if not dirs:
                    missing += 1
                    continue
                seen.add(vid)
                jobs.append((vid, dirs[0], body))

    print(f"{len(jobs)} Miku-spoken lines (speaker {args.speaker})")
    if missing:
        print(f"  {missing} voice ids absent from the bucket", file=sys.stderr)

    ok = skipped = failed = 0
    total = 0
    lines: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for vid, card_dir, text in jobs:
            dest = os.path.join(args.out, f"{vid}.wav")
            futs[ex.submit(fetch_wav, vid, card_dir, dest)] = (vid, dest, text)
        for i, fut in enumerate(as_completed(futs), 1):
            vid, dest, text = futs[fut]
            try:
                n = fut.result()
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  FAIL {vid}: {e}", file=sys.stderr)
                continue
            if n == 0:
                skipped += 1
            else:
                ok += 1
                total += n
            lines.append(f"{os.path.relpath(dest, args.out)}|{text.replace(chr(10), ' ')}")
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} …", flush=True)

    manifest = os.path.join(args.out, "manifest.list")
    with open(manifest, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\ndone: {ok} downloaded, {skipped} present, {failed} failed")
    print(f"bytes: {total/1e6:.1f} MB -> {args.out}")
    print(f"manifest: {manifest} ({len(lines)} lines)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
