#!/usr/bin/env python3
"""Miku speech synthesis: English and Japanese, on CPU.

Two English routes, deliberately different:

  phonemes  English -> CMUdict/ARPAbet -> text_lang='en'
            The target aesthetic. English's own segments survive (/DH/, /3:/,
            final /ld/ clusters) and the *model's* Japanese training supplies the
            accent. This is the Vocaloid condition: consonants land, vowels bend.

  loanword  English -> katakana -> text_lang='ja'
            Loanword transcription: world -> わーるど. Sounds like a Japanese
            speaker reading English. Kept because it is a genuinely different
            character, and because it is the same path Japanese input takes.

And one Japanese route:

  japanese  Japanese text (kanji/kana) -> text_lang='ja'
            The katakana pipeline is just this with the katakana supplied by the
            transliterator instead of by you.

Vowel character (phonemes route only):
  natural   keep the frontend's stress marks. Schwa stays schwa. Most English.
  full      promote unstressed vowels to their full form (AH0 -> AH1). Miku's
            English voicebank over-articulates; this is that.
  mash      promote to full form AND swap in Japanese-shaped vowels where English
            has a distinctive one. Harshest, most obviously "not a native".

Run from vendor/GPT-SoVITS so the repo's modules import:
  cd vendor/GPT-SoVITS
  PYTHONPATH=.:./GPT_SoVITS ../../.venv/bin/python ../../scripts/miku_say.py \
      --mode phonemes --vowels full --out ../../out/ab
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

# --- CPU config (measured: 6 threads + fp32 is fastest on AVX2) ---
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
    "v4": (f"{PM}/s1v3.ckpt", f"{PM}/gsv-v4-pretrained/s2Gv4.pth"),
}

# ---------------------------------------------------------------------------
# katakana route (loanword + Japanese input)
# ---------------------------------------------------------------------------

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

_c2k = None


def _to_kana(word: str) -> str | None:
    """e2k C2K (character-level, takes written words), else a rule table.

    NOTE: e2k also exposes P2K, which looks like the word->katakana model but
    takes ARPAbet phonemes (in_table is AA0/AA1/...), so P2K()(word) silently
    returns ''. C2K is the one that accepts written English.
    """
    global _c2k
    if _c2k is None:
        try:
            from e2k import C2K
            _c2k = C2K()
        except Exception as e:  # noqa: BLE001
            print(f"warning: e2k unavailable ({e}); rule table only", file=sys.stderr)
            _c2k = False
    if _c2k:
        try:
            r = _c2k(word)
            if isinstance(r, str) and r.strip():
                return r
        except Exception:
            pass
    return None


def transliterate(text: str) -> str:
    """English -> katakana. Used by the loanword route; Japanese input bypasses it."""
    out = []
    for raw in text.replace("\n", " ").split(" "):
        word = raw.strip()
        if not word:
            continue
        key = word.lower().strip(".,!?;:\"'")
        tail = word[len(key):] if key and word.lower().startswith(key) else ""
        if key in OVERRIDES:
            out.append(OVERRIDES[key] + tail)
            continue
        out.append((_to_kana(key) or key) + tail)
    return " ".join(out)


# ---------------------------------------------------------------------------
# phoneme route (the target aesthetic)
# ---------------------------------------------------------------------------

VOWELS = ("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER",
          "EY", "IH", "IY", "OW", "OY", "UH", "UW")

# "mash": substitute Japanese-shaped vowels for the English ones that most
# announce themselves as foreign. Deliberately approximate -- the point is that
# it reads as a Japanese-trained voice reaching for English, not as an accent
# coach doing a perfect job.
MASH = {
    "AE": "AA",   # cat   -> more open
    "AH": "AA",   # comma -> a, not schwa
    "IH": "IY",   # sit   -> seat
    "UH": "UW",   # book  -> boohk
    "ER": "AA",   # her   -> aah (loses the r-colour)
    "AO": "AA",   # thought -> aah
}
# Any target here must exist in text.symbols2 for every stress digit, otherwise
# cleaned_text_to_sequence raises KeyError (learned the hard way: 'OU1' is not a
# phoneme, 'OW1' is).


def _split_stress(ph: str):
    """'ER1' -> ('ER', '1'); 'DH' -> ('DH', '')."""
    if ph and ph[-1].isdigit():
        return ph[:-1], ph[-1]
    return ph, ""


def restyle_vowels(ph: list[str], style: str) -> list[str]:
    """Reshape vowel stress/identity. Consonants are never touched."""
    if style == "natural":
        return ph
    out = []
    for p in ph:
        base, stress = _split_stress(p)
        if base not in VOWELS:
            out.append(p)
            continue
        if style == "full":
            # promote reduced vowels so nothing collapses to schwa
            out.append(base + ("1" if stress in ("0", "") else stress))
        elif style == "mash":
            b = MASH.get(base, base)
            out.append(b + ("1" if stress in ("0", "") else stress))
        else:
            raise ValueError(f"unknown vowel style: {style}")
    return out


def english_phonemes(text: str, style: str) -> list[str]:
    """English text -> ARPAbet phoneme list, restyled."""
    from text.english import g2p
    from text.symbols2 import symbols

    ph = restyle_vowels(g2p(text), style)
    # Fail here with a clear message rather than deep inside
    # cleaned_text_to_sequence with a bare KeyError.
    vocab = set(symbols)
    bad = [p for p in ph if p not in vocab]
    if bad:
        raise SystemExit(
            f"phoneme(s) not in the model vocabulary: {bad}\n"
            f"  style={style}  full stream: {' '.join(ph)}"
        )
    return ph


def install_phoneme_injection() -> None:
    """Make text_lang='en' accept a literal phoneme string.

    WHY THIS EXISTS: ``text_lang='en'`` runs the text through ``g2p_en``, which
    expects *spelling*. Handing it 'DH AH1 V OY1 S' does not pass the phonemes
    through -- it re-reads them as letters and silently produces garbage
    (verified: it became 'D IY1 EY1 CH EY1 EY1 CH V IY1 OW1 ...'). So a phoneme
    route needs to bypass g2p entirely.

    Safe because the English path never uses BERT: TextPreprocessor.get_bert_inf
    returns torch.zeros for anything that is not zh/yue. The only thing that has
    to be right is the phone sequence and the length of its feature.
    """
    import torch as _torch
    # NOTE: GPT-SoVITS inserts its own repo root on sys.path and imports
    # 'TTS_infer_pack.TextPreprocessor', while we sit one level up. Patching
    # 'GPT_SoVITS.TTS_infer_pack.TextPreprocessor' would modify a *different*
    # module object and silently have no effect. Resolve the one in sys.modules
    # that the engine actually uses, and fall back to the qualified name.
    import importlib
    tp_mod = None
    for name in ("TTS_infer_pack.TextPreprocessor",
                 "GPT_SoVITS.TTS_infer_pack.TextPreprocessor"):
        if name in sys.modules:
            tp_mod = sys.modules[name]
            break
    if tp_mod is None:
        tp_mod = importlib.import_module("TTS_infer_pack.TextPreprocessor")
    from text.cleaner import cleaned_text_to_sequence

    original = tp_mod.TextPreprocessor.get_phones_and_bert

    def patched(self, text, language, version, final=False):
        # A '#' marks "these are raw ARPAbet phonemes, do not re-spell them".
        # pre_seg_text may prepend '.' when the first segment is short, so the
        # marker is not necessarily at index 0.
        if language == "en" and "#" in text:
            i = text.index("#")
            head, body = text[:i], text[i + 1:]
            phones = [p for p in body.split() if p]
            # keep any real punctuation that was sitting before the marker
            phones = [p for p in head.strip() if p in ",.!?-;:"] + phones
            ids = cleaned_text_to_sequence(phones, version)
            bert = _torch.zeros((1024, len(ids)), dtype=_torch.float32).to(self.device)
            return ids, bert, body
        return original(self, text, language, version, final)
        return original(self, text, language, version, final)

    tp_mod.TextPreprocessor.get_phones_and_bert = patched


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------

def load_torch() -> None:  # pragma: no cover - import side effects only
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", nargs="*", help="lines to synthesize")
    ap.add_argument("--mode", default="phonemes",
                    choices=["phonemes", "loanword", "japanese"])
    ap.add_argument("--vowels", default="full",
                    choices=["natural", "full", "mash"],
                    help="vowel character, phonemes mode only")
    ap.add_argument("--ref", default=os.path.join(ROOT, "data/ref/miku_prompt.wav"))
    ap.add_argument("--ref-text",
                    default="今日、ライブに来てくれたみんなにも お礼を言わせてほしいな")
    ap.add_argument("--out", default=os.path.join(ROOT, "out"))
    ap.add_argument("--tag", default="", help="filename prefix")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="speed_factor. The phoneme route over-articulates at 1.0 "
                         "because it has no mora constraint; 1.3-1.6 is the usual fix.")
    ap.add_argument("--version", default="v2ProPlus", choices=sorted(PROFILES))
    args = ap.parse_args()

    lines = args.text or ["The world is mine."]

    os.chdir(os.path.join(ROOT, "vendor", "GPT-SoVITS"))
    sys.path.insert(0, ".")
    sys.path.insert(0, "GPT_SoVITS")

    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    # must run after the engine modules exist in sys.modules (see install_phoneme_injection)
    install_phoneme_injection()

    t2s, vits = PROFILES[args.version]
    for p in (t2s, vits, args.ref):
        if not os.path.exists(p):
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    print(f"loading {args.version} on CPU …", flush=True)
    t0 = time.time()
    pipe = TTS(TTS_Config({"device": "cpu", "is_half": False,
                           "t2s_weights_path": t2s, "vits_weights_path": vits,
                           "version": args.version}))
    print(f"  loaded in {time.time()-t0:.1f}s\n", flush=True)

    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or (args.mode if args.mode != "phonemes" else f"ph-{args.vowels}")
    total_audio = total_proc = 0.0

    for i, line in enumerate(lines, 1):
        # --- decide what text and language actually go to the model ---
        if args.mode == "phonemes":
            ph = english_phonemes(line, args.vowels)
            # '#' marks a literal phoneme run. pre_seg_text splits on '.' and ',',
            # and each fragment is preprocessed on its own, so the marker goes in
            # after every punctuation token, not just at the start.
            token_segs, cur = [], []
            for p_ in ph:
                cur.append(p_)
                if p_ in (",", ".", "!", "?", ";"):
                    token_segs.append("#" + " ".join(cur))
                    cur = []
            if cur:
                token_segs.append("#" + " ".join(cur))
            payload = " ".join(token_segs)
            lang, shown = "en", " ".join(ph)
        elif args.mode == "loanword":
            payload = transliterate(line)
            lang, shown = "ja", payload
        else:
            payload, lang, shown = line, "ja", line

        print(f"[{i}] {args.mode}/{args.vowels if args.mode=='phonemes' else '-'}")
        print(f"    in  : {line}")
        print(f"    fed : {shown}   (text_lang={lang})")

        req = {
            "text": payload, "text_lang": lang,
            "ref_audio_path": args.ref, "prompt_text": args.ref_text,
            "prompt_lang": "ja",
            "top_k": 15, "top_p": 1.0, "temperature": 1.0,
            "text_split_method": "cut5",
            "batch_size": 1,
            "speed_factor": args.speed, "return_fragment": False,
            "seed": 42, "parallel_infer": False, "repetition_penalty": 1.35,
        }

        t0 = time.time()
        audio = sr = None
        for sr, audio in pipe.run(req):
            pass
        dt = time.time() - t0
        if audio is None:
            print("    -> no audio returned\n", file=sys.stderr)
            continue

        # GPT-SoVITS yields int16 here. Writing that as float32 clamps to +/-1
        # and yields a square wave that looks like audio (peak 1.00, rms 0.9,
        # only 3 unique values). Normalise explicitly.
        if audio.dtype not in (np.float32, np.float64) or np.abs(audio).max() > 1.5:
            audio = audio.astype(np.float32) / 32768.0
        peak = float(np.abs(audio).max())
        if peak > 1.0:
            audio = audio / peak

        dur = len(audio) / sr
        total_audio += dur
        total_proc += dt

        path = os.path.join(args.out, f"{tag}_{i}.wav")
        sf.write(path, audio.astype(np.float32), sr)
        print(f"    -> {path}  {dur:.2f}s in {dt:.2f}s (RTF {dt/dur:.2f})\n")

    if total_audio:
        print(f"total: {total_audio:.1f}s audio, {total_proc:.1f}s CPU "
              f"(RTF {total_proc/total_audio:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
