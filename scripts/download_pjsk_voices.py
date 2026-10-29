#!/usr/bin/env python3
"""Download all Hatsune Miku (id 21) PJSK card voices directly from the asset bucket.

FindUrVoicesPJSK reconstructs card-voice URLs from cardEpisodes/scenario ids, but the
current sekai-jp-assets layout no longer matches: it requests
  sound/card_scenario/voice/<scenario_id>/<VoiceId>.wav
while the bucket actually serves
  sound/card_scenario/voice/021<card>_miku<nn>/partvoice_<n>_021_<unit>.wav
and its card `.asset` URLs (/character/member/<assetbundleName>/…) 404 too. The downloader
also writes S3 XML error bodies to disk as ".wav" (it ignores HTTP status), which is how
1,424/1,530 files came out as NoSuchKey XML.

So: enumerate the bucket prefix (it is publicly listable) and fetch what actually exists.
Stdlib only — no venv needed.

Usage:  python3 scripts/download_pjsK_voices.py [--out data/raw/miku_cards]
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

BUCKET = "https://storage.sekai.best/sekai-jp-assets/"
PREFIX = "sound/card_scenario/voice/021"  # all Miku card dirs (021001_miku01, …)
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
# The CDN in front of the bucket rejects urllib's default agent with 403,
# so every request must carry a User-Agent explicitly.
UA = "FindUrVoicesPJSK/v1"


def _open(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=60)


def list_keys(prefix: str, suffix: str = ".wav") -> list[str]:
    """Page through the public bucket listing and return every matching key."""
    keys: list[str] = []
    token = ""
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        url = BUCKET + "?" + urllib.parse.urlencode(q)
        with _open(url) as r:
            root = ET.fromstring(r.read())
        for c in root.findall(f"{NS}Contents"):
            k = c.findtext(f"{NS}Key") or ""
            if k.endswith(suffix):
                keys.append(k)
        truncated = (root.findtext(f"{NS}IsTruncated") or "").lower() == "true"
        token = root.findtext(f"{NS}NextContinuationToken") or ""
        if not truncated or not token:
            break
    return keys


def fetch(key: str, out_root: str) -> tuple[str, int]:
    rel = key.split("voice/", 1)[1]          # e.g. 021001_miku01/partvoice_01_021_piapro.wav
    dest = os.path.join(out_root, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 44:
        return dest, 0                         # already have it
    url = BUCKET + urllib.parse.quote(key)
    with _open(url) as r:
        data = r.read()
    if not data.startswith(b"RIFF"):
        raise ValueError(f"not a RIFF/WAV payload ({len(data)} B): {key}")
    with open(dest, "wb") as f:
        f.write(data)
    return dest, len(data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/miku_cards")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    print(f"listing {PREFIX} …", flush=True)
    keys = list_keys(PREFIX)
    if not keys:
        print("no keys found — bucket layout may have changed", file=sys.stderr)
        return 1
    dirs = sorted({k.split("voice/", 1)[1].split("/")[0] for k in keys})
    print(f"{len(keys)} wav keys across {len(dirs)} card dirs")

    ok = skipped = failed = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, k, args.out): k for k in keys}
        for i, fut in enumerate(as_completed(futs), 1):
            k = futs[fut]
            try:
                dest, n = fut.result()
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  FAIL {k}: {e}", file=sys.stderr)
                continue
            if n == 0:
                skipped += 1
            else:
                ok += 1
                total_bytes += n
            if i % 200 == 0:
                print(f"  {i}/{len(keys)} …", flush=True)

    print(f"\ndone: {ok} downloaded, {skipped} already present, {failed} failed")
    print(f"bytes written: {total_bytes / 1e6:.1f} MB -> {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
