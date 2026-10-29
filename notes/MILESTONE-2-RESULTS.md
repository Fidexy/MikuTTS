# Milestone 2 — Miku spoken corpus acquired

Date: 2026-10-30
Status: **complete** (dataset in hand, QA'd, text alignment still to build)

## Result

| | |
|---|---|
| Source | Project Sekai asset bucket (`sekai-jp-assets`), Miku character id **21** |
| Files | **2,138** card voices + 10 profile voices |
| Audio | **147 minutes**, all mono 44.1 kHz PCM 16-bit |
| Disk | 779 MB (`data/raw/miku_cards/`) |
| BGM | **none** — median noise floor 0.00005, every file has silence gaps |
| Clipping | 472 files touch full scale, but median only **3 samples** (of ~170k) — cosmetic |
| Duration | median 3.9 s, max 15.4 s, min 0.17 s |

Well beyond the plan's 10–15 min target by ~10x, which changes the training strategy
for the better (see below).

## The tool was broken — two real bugs found

`FindUrVoicesPJSK` produced **1,530 files of which 1,424 were XML, not audio**. It also
only ever exposed 106 usable clips. Two separate defects:

1. **It writes error bodies to disk as `.wav`.** `_fetch_and_save` never checks the HTTP
   status, so every 404 `NoSuchKey` response body was saved verbatim. Caught by
   `file`-checking the outputs — `soundfile` refused to open them (`Format not recognised`).
2. **Its card-voice URL scheme is stale.** It builds
   `sound/card_scenario/voice/<scenario_id>/<VoiceId>.wav`, which 404s in all regions
   (jp/en/tw/kr/cn). Its card `.asset` URLs (`/character/member/<assetbundle>/…`) 404 too.
   The bucket actually serves
   `sound/card_scenario/voice/021<card>_miku<nn>/partvoice_<n>_021_<unit>.wav`.
   Because the tool aborts its card loop on the first 404, it stopped after the ~14 cards
   whose ids happened to line up — that's the 106.

**Fix:** the bucket is publicly listable, so enumerate the prefix instead of reconstructing
paths. `scripts/download_pjsk_voices.py` (stdlib only) pages the S3 listing and downloads
what genuinely exists, validating the `RIFF` magic before writing.

**Gotcha:** the CDN 403s `urllib`'s default User-Agent. Every request needs an explicit
`User-Agent` header. (This is why `curl` worked and Python didn't.)

## What still blocks fine-tuning

Downloads give WAVs but **the Japanese transcript text is not recovered** — the broken
tool was also supposed to write `manifest.list` (`path|text`). Without text, GPT-SoVITS
cannot train (it needs utterance→sentence alignment).

Naming does encode some timing/metadata (`voice_card_*`), but the spoken line itself must
come from the card-episode scenario data. Next step options:

- rebuild the scenario→voice mapping from `cardEpisodes.json` + scenario `.asset` files
  (the `.asset` files are Unity bundles; `partvoice_*` names appear to be indexed inside
  the card `.asset`'s `TalkData` — needs a bundle reader), or
- **ASR**: transcribe with Whisper (available locally via faster-whisper / whisper.cpp).
  Given no ASR was needed in the original plan, this is a new cost — but on 2,138 short
  clips it is cheap and side-steps bundle parsing entirely.

For the *katakana-English* goal, the transcript matters less than one might think: we
synthesize from ad-hoc katakana strings, so text is only needed to satisfy the trainer's
label format and to let it learn duration/prosody. ASR text is adequate for that.

## Effect on the training plan

The plan assumed a curated 10–15 min subset because that was the data ceiling. At
147 min the constraint inverts: **the bottleneck is CPU training time, not data.** So:

- Stage 1: pick ~10–15 min of the *cleanest, longest, most speech-like* clips
  (favours 4–8 s lines, the 863-clip / 82-min bucket).
- Stage 2: scale to the remainder if Stage 1 underfits.
- 313 clips under 1 s are near-useless for prosody learning — exclude, or keep only as
  punctuation-scale exclamations.
