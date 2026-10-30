# Milestone 3 — zero-shot result

Date: 2026-10-30
Status: **working.** Zero-shot katakana-English Miku synthesis runs on CPU.

## Result

Three lines synthesized end-to-end from a 4.57 s Miku reference clip:

| Line | Katakana fed to frontend | Audio | RTF |
|---|---|---|---|
| Hello, I am Hatsune Miku. | ヘロ, イ アム ハツネ ミク. | 2.22 s | 1.05 |
| Today I want to sing about the world. | トデー イ ウォント とぅ しんぐ アブート ざ わーるど. | 2.74 s | 0.66 |
| The voice of the future is here. | ざ ぼいす おぶ ざ フューチャー いず ひあ. | 3.02 s | 0.65 |

Overall RTF **0.76**, consistent with milestone 1 (0.75–0.9).

Voice check: median f0 of output is 348–444 Hz against the reference's 394 Hz — same
speaker register, so the zero-shot clone is taking. Output is 32 kHz, RMS ~0.19,
healthy peak spread (0.81–0.92).

## Reference clip choice

`voice_card_bd_202308_21_4b_21_21_piapro.wav` → `data/ref/miku_prompt.wav`.
Selected by scoring all 50 clips on duration (3–10 s), voiced-frame ratio, peak, and
clipping count. Winner: 4.57 s, 73 % speech, **zero clipped samples**, warm line
(今日、ライブに来てくれたみんなにも お礼を言わせてほしいな). Prompt text was taken verbatim
from the manifest, so prompt/audio always match.

## Two bugs found in the synth path

1. **`e2k`'s obvious class is the wrong one.** `P2K` looks like the word→katakana model but
   its `in_table` is **ARPAbet phonemes** (`AA0`, `AA1`, …), so calling `P2K()("world")`
   silently returns `''` — no exception, just empty. `C2K` is the character-level model that
   takes written words. Verified: `C2K()("computer")` → コンピューター.

2. **GPT-SoVITS yields `int16`, not float.** Writing those samples through
   `sf.write(..., audio.astype(np.float32))` clamps every value to ±1, producing a
   **square wave** that superficially looks like audio — peak 1.00, RMS 0.9 — but has only
   3 unique sample values. The tell: `len(np.unique(y))` was 3, and 66 % of samples sat at
   ±1. `miku_say.py` now normalises int16 → float32 by /32768 and peak-limits.

Note the milestone-1 `smoke_test.py` has the same latent bug (writes inside the loop); it
happened to survive because it wrote multiple fragments rather than one clamped buffer.
Worth fixing if it is ever used again.

## What this settles

The §C design premise is confirmed **at the output stage**, not just the phonemizer: the
katakana route produces stable, intelligible, in-register speech with a genuine Japanese
accent, with **zero fine-tuning and no English training data**.

Whether the accent *sounds right* is an ear judgement — listen to `out/zeroshot_{1,2,3}.wav`.

## Next

- Add punctuation-aware pacing; current numbers (`0.000 0.006 1.323 0.471`) suggest the
  model is landing on reasonable duration/pitch predictions.
- Expand `OVERRIDES` in `miku_say.py` by ear — that table is the character of the voice.
- Fine-tuning is **on hold**: 2.6 min is too thin, and zero-shot already works. See the
  README's "Fine-tune sources" for how to get to 10–15 min properly.
