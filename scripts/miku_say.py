#!/usr/bin/env python3
"""Zero-shot Miku TTS with the katakana-English phonetic-bash frontend.

No fine-tuning: GPT-SoVITS clones the timbre from a short reference clip. This is the
cheapest test of whether the §C aesthetic works at all (plan §D says zero-shot first).

The core trick: English is transliterated to katakana, which is then fed to the
Japanese frontend with text_lang='ja'. OpenJTalk then produces genuine Japanese
phonemes, so the model stays in its training distribution while the result sounds
like Japanese-accented English ("the" -> ざ, "world" -> わーるど).

Run from vendor/GPT-SoVITS so the repo's modules are importable:
  cd vendor/GPT-SoVITS
  PYTHONPATH=.:./GPT_SoVITS ../../.venv/bin/python ../../scripts/miku_say.py --out ../../out
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

# --- CPU config (measured in milestone 1: 6 threads + fp32 is fastest on AVX2) ---
torch.set_num_threads(6)
torch.set_num_interop_threads(1)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PM = os.path.join("GPT_SoVITS", "pretrained_models")

PROFILES = {
    "v2ProPlus": (
        f"{PM}/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        f"{PM}/v2Pro/s2Gv2ProPlus.pth",
    ),
    "v2Pro": (
        f"{PM}/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        f"{PM}/v2Pro/s2Gv2Pro.pth",
    ),
    "v4": (
        f"{PM}/s1v3.ckpt",
        f"{PM}/gsv-v4-pretrained/s2Gv4.pth",
    ),
}

# Katakana-English substitution table. e2k handles the general case (see
# transliterate()), but these are hand-tuned: function words and the ones whose
# textbook transliteration reads "too correct" and loses the aesthetic.
OVERRIDES = {
    "the": "ざ", "a": "あ", "an": "あん", "you": "ゆー", "your": "ゆあ",
    "to": "とぅ", "and": "あんど", "of": "おぶ", "is": "いず", "are": "あー",
    "was": "わず", "were": "わー", "be": "びー", "I": "あい", "me": "みー",
    "my": "まい", "we": "うぃー", "he": "ひー", "she": "しー", "it": "いっと",
    "this": "でぃす", "that": "ざっと", "there": "ぜあ", "here": "ひあ",
    "what": "わっと", "world": "わーるど", "girl": "がーる", "computer": "こんぴゅーたー",
    "love": "らぶ", "music": "みゅーじっく", "stage": "すてーじ", "dream": "どりーむ",
    "sing": "しんぐ", "song": "そんぐ", "voice": "ぼいす", "heart": "はーと",
}

_e2k = None

# Rule-based English -> katakana approximation, used when e2k is unavailable or
# returns nothing. Not linguistically rigorous, but the whole point of the
# aesthetic is that it is approximate: we want plausible Japanese phonemes.
_ONSETS = [
    ("sch", "しゅ"), ("tch", "っち"), ("ch", "ち"), ("sh", "し"),
    ("th", "ざ"), ("ph", "ふ"), ("wh", "うぃ"), ("ck", "っく"),
    ("qu", "く"), ("ng", "んぐ"), ("kn", "ん"), ("wr", "ら"),
    ("b", "ぶ"), ("c", "く"), ("d", "ど"), ("f", "ふ"), ("g", "ぐ"),
    ("h", "は"), ("j", "じ"), ("k", "く"), ("l", "る"), ("m", "む"),
    ("n", "ん"), ("p", "ぷ"), ("r", "ら"), ("s", "す"), ("t", "と"),
    ("v", "ぶ"), ("w", "わ"), ("x", "くす"), ("y", "や"), ("z", "ず"),
]
_VOWELS = {
    "a": "あ", "e": "え", "i": "い", "o": "お", "u": "う",
    "ai": "あい", "ay": "えい", "ea": "いー", "ee": "いー", "ie": "いー",
    "oa": "おー", "oo": "うー", "ou": "あう", "ow": "おう", "oy": "おい",
    "au": "おー", "ei": "えい", "ey": "えい", "ue": "うー", "ui": "うい",
}


def _rule_kana(word: str) -> str:
    out, i, n = [], 0, len(word)
    while i < n:
        ch = word[i]
        # vowel-ish runs first
        two = word[i:i + 2]
        if two in _VOWELS:
            out.append(_VOWELS[two]); i += 2; continue
        if ch in _VOWELS:
            out.append(_VOWELS[ch]); i += 1; continue
        if ch == "r" and i + 1 < n and word[i + 1] not in "aeiou":
            out.append("ー"); i += 1; continue          # r-coloured: car -> かー
        if ch == "c" and i + 1 < n and word[i + 1] in "ei":
            out.append("す"); i += 1; continue
        if ch == "g" and i + 1 < n and word[i + 1] in "ei":
            out.append("じ"); i += 1; continue
        for pat, kana in _ONSETS:
            if word.startswith(pat, i):
                out.append(kana); i += len(pat)
                # a consonant must be followed by a vowel in Japanese;
                # inserted う keeps it pronounceable without distorting much
                if i < n and word[i] not in "aeiouy":
                    out.append("う")
                break
        else:
            i += 1
    return "".join(out)


def _to_kana(word: str) -> str | None:
    """Try e2k C2K (character-level, takes plain words), then the rule table.

    NOTE: e2k exposes several classes and the obvious-looking ones do not take
    plain English. ``P2K`` expects ARPAbet phonemes (its in_table is AA0/AA1/...),
    so calling it on a word silently returns ''. ``C2K`` is the character-level
    model that maps a written word, which is what we want.
    """
    global _e2k
    if _e2k is None:
        try:
            from e2k import C2K
            _e2k = C2K()
        except Exception as e:  # noqa: BLE001
            print(f"warning: e2k unavailable ({e}); using rule table only", file=sys.stderr)
            _e2k = False
    if _e2k:
        try:
            r = _e2k(word)
            if isinstance(r, str) and r.strip():
                return r
        except Exception:
            pass
    r = _rule_kana(word)
    return r or None


def transliterate(text: str) -> str:
    """English -> katakana, using overrides first then e2k/rules for the rest."""
    out = []
    for raw in text.replace("\n", " ").split(" "):
        word = raw.strip()
        if not word:
            continue
        key = word.lower().strip(".,!?;:\"'")
        tail = ""
        if key and word.lower().startswith(key):
            tail = word[len(key):]
        if key in OVERRIDES:
            out.append(OVERRIDES[key] + tail)
            continue
        kana = _to_kana(key)
        out.append((kana or key) + tail)
    return " ".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", nargs="*", help="English lines to synthesize")
    ap.add_argument("--ref", default=os.path.join(ROOT, "data/ref/miku_prompt.wav"))
    ap.add_argument("--ref-text", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "out"))
    ap.add_argument("--version", default="v2ProPlus", choices=sorted(PROFILES))
    ap.add_argument("--truncate", action="store_true")
    args = ap.parse_args()

    lines = args.text or [
        "Hello, I am Hatsune Miku.",
        "Today I want to sing about the world.",
        "The voice of the future is here.",
    ]

    os.chdir(os.path.join(ROOT, "vendor", "GPT-SoVITS"))
    sys.path.insert(0, ".")
    sys.path.insert(0, "GPT_SoVITS")

    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    t2s, vits = PROFILES[args.version]
    for p in (t2s, vits, args.ref):
        if not os.path.exists(p):
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    print(f"loading {args.version} on CPU …", flush=True)
    t0 = time.time()
    config = TTS_Config({"device": "cpu", "is_half": False,
                         "t2s_weights_path": t2s, "vits_weights_path": vits,
                         "version": args.version})
    pipe = TTS(config)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    audio_total = proc_total = 0.0

    for i, line in enumerate(lines, 1):
        kata = transliterate(line)
        print(f"\n[{i}] EN : {line}")
        print(f"    KATA: {kata}")

        req = {
            "text": kata, "text_lang": "ja",
            "ref_audio_path": args.ref, "prompt_text": args.ref_text,
            "prompt_lang": "ja",
            "top_k": 15, "top_p": 1.0, "temperature": 1.0,
            "text_split_method": "cut5", "batch_size": 1,
            "speed_factor": 1.0, "return_fragment": False,
            "seed": 42, "parallel_infer": False, "repetition_penalty": 1.35,
        }

        t0 = time.time()
        audio = sr = None
        for sr, audio in pipe.run(req):
            pass                      # not streaming: keep the final full buffer
        dt = time.time() - t0
        if audio is None:
            print("    -> no audio returned", file=sys.stderr)
            continue

        # GPT-SoVITS yields int16 in this build. Writing that as float32 clamps
        # everything to +/-1 and produces a square wave, so normalise explicitly.
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32) / 32768.0
        elif np.abs(audio).max() > 1.5:
            audio = audio.astype(np.float32) / 32768.0
        peak = float(np.abs(audio).max())
        if peak > 1.0:
            audio = audio / peak

        dur = len(audio) / sr
        audio_total += dur
        proc_total += dt

        path = os.path.join(args.out, f"zeroshot_{i}.wav")
        sf.write(path, audio.astype(np.float32), sr)
        print(f"    -> {path}  {dur:.2f}s audio in {dt:.2f}s (RTF {dt/dur:.2f}) peak {peak:.2f}")

    if audio_total:
        print(f"\ntotal: {audio_total:.1f}s audio, {proc_total:.1f}s CPU "
              f"(overall RTF {proc_total/audio_total:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
