# Milestone 2 — corrected: Miku-only corpus

Date: 2026-10-30
Status: **complete for card voices.** Corpus is smaller than reported and that report is retracted.

## Correction: the earlier "147 minutes of Miku" was wrong

Two errors, both caught by the user:

1. **The text is Japanese**, and
2. **most of those voices are not Miku.**

My previous note claimed 2,138 clips / 147 min of Miku. That was wrong. Downloading every
WAV under `sound/card_scenario/voice/021*` gets every voice *in Miku's card episodes* —
but an episode is a conversation. Verified speaker breakdown of one card dir
(`021001_miku01`): **Miku 5, Ichika 14, Saki 17**. Across all 126 Miku card assets the true
split is:

| | count | share |
|---|---|---|
| Miku (`Character2dId == 21`) | **129** | **2.8 %** |
| other cast | 4,509 | 97.2 % |

So the 779 MB / 2,138-file download was ~97 % the wrong speaker and has been **deleted**.
A model fine-tuned on it would have learned a mixture of ~30 voices, not Miku.

## The actual dataset

| | |
|---|---|
| Files | **50** (`data/miku21/`) |
| Audio | **2.6 min**, mono 44.1 kHz |
| Disk | 14 MB |
| Transcript | **exact Japanese**, from the game's own `Body` field — no ASR needed |
| Duration | median 1.36 s, max 10.4 s, min 0.19 s |
| Median f0 | 364 Hz (consistent with one speaker) |

2.6 min is **below** the plan's 10–15 min target, but it is genuine, correctly-labelled,
single-speaker data — which is worth more than 147 minutes of the wrong thing.

## The text problem is solved, and not by ASR

Card `.asset` files are **plain JSON, not Unity bundles** (verified: `file` reports JSON),
served at `character/member/res021_no<NNN>/<card>_miku<nn>.asset`. Each holds a `TalkData`
list where every record carries both `Body` (the Japanese line) and `Voices[]` with the
speaker's `Character2dId`. So the transcript and the speaker id come from the same record.

This supersedes the "ASR with Whisper" recommendation in the previous note.

**Japanese is not a problem here — it is the input format the katakana pipeline wants.**
§C of the plan feeds katakana-transliterated English through OpenJTalk. Training on native
Japanese text with `text_lang='ja'` keeps everything in the Japanese frontend's
distribution, which is exactly the design.

## How to reproduce

`scripts/extract_miku_lines.py` — filters on `Voices[].Character2dId == 21` (not on the
card), resolves each `VoiceId` against the real bucket listing, validates the `RIFF` magic
before writing, and emits `manifest.list` as `relpath|text`.

**Bug worth remembering:** a `VoiceId` is reused across cards and a card ships multiple
variant dirs (`021001_miku01` / `021001_miku02`), so the asset basename is *not* the voice
directory. Guessing it produced 18 spurious 404s; resolving against the listing fixed all 50.

## Where more Miku speech could come from

Checked and ruled out as quick wins:

- `scenario/*` story assets: 400 sampled files, 1,095 voices, **0 Miku** — she is not a
  story character in that set.
- Other card-voice prefixes: no Miku dirs exist outside `021*`.
- Solo songs: sung, not speech — excluded by plan.

## Next decision

2.6 min is thin but usable for zero-shot, and marginal for fine-tuning. Priority is
therefore **zero-shot first** (plan §D) with these 50 clips, which costs ~45 min and answers
whether the katakana-English aesthetic lands. Fine-tuning only if that disappoints — and
before that, expanding the corpus would need sources outside the card assets
(area talks / event scenarios as separate passes, or sung material as a blend).
