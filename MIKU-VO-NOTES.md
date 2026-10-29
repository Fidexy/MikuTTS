# Miku voiceover for manim videos — options & pipeline

Research note: how to get **Hatsune Miku to narrate** a manim-rendered video, when
piracy of the voice is acceptable (2026-10-24).

TL;DR — **GPT-SoVITS v2+ with a Miku fine-tune (or zero-shot from Miku
spokes) is the recommended pipeline.** It is self-hosted, free, runs comfortably
on the local RTX 3080 (20 GB), does *spoken* text freely (not just singing),
and eats a plain WAV out. NijiVoice (Algomatic) is the best *licensed* option
if you ever need commercial clearance.

---

## 1. What "a Miku voice" requires

Miku's timbre is Saki Fujita's recordings, reprocessed:

- The Vocaloid voicebanks she sings with are **phoneme grids of sung vowel
  samples**, not free speech. They give timbre + pitch range; they are *not*
  usable as narration TTS data on their own (wrong register, no natural
  prosody). Source: Fujita interview on the recording process (J-CAST,
  2008): https://web.archive.org/web/20080102085851/http:/www.j-cast.com/2008/01/01015179.html
- The **spoken** versions of her voice come from vocal games/consoles —
  Project DIVA series voice lines, Miku: Stage Pass, etc. This is the data
  that makes her *talk*.
- Precedent: the community already does exactly "Miku → RVC" by mixing
  voicebank acapellas + game speech clips:
  - aple/HatsuneMikuRVC — "a voice using hatsune miku's original and english
    voicebanks": https://huggingface.co/aple/HatsuneMikuRVC
  - HidekoHaruna/RVC_V2_Hatsune_Miku — "22 minutes … includes some clips of
    Saki Fujita's voice taken from Project Diva X":
    https://huggingface.co/HidekoHaruna/RVC_V2_Hatsune_Miku
  - javinfamous/infamous_miku_v2 — 38 min dataset, RVC v2, 1000 epochs:
    https://huggingface.co/javinfamous/infamous_miku_v2

**Consequence:** for natural narration you want a **TTS model fine-tuned on
Miku *speech*** (or zero-shot with Miku-speech reference clips). RVC
models of Miku are *singing voice converters* — great for sung covers, but
RVC-driven TTS requires generating source speech first and loses prosody.

---

## 2. Route A — GPT-SoVITS (recommended)

`RVC-Boss/GPT-SoVITS`, MIT: https://github.com/RVC-Boss/GPT-SoVITS

- Zero-shot TTS: 5 s reference clip. Few-shot: **~1 min of clean training
  audio** for a fine-tune (features list, repo README).
- Languages: v2+ supports zh/en/ja/ko/yue with cross-lingual inference:
  https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90features(Latest%E2%80%90Including%E2%80%90v5)
  (v2 table: "Chinese, Japanese, English, Korean, Cantonese").
- Inference speed (v2 ProPlus): **RTF 0.014 on a 4090** — a full 4-minute
  narration synthesises in seconds; a 3080 is fine:
  repo README.
- Trained-JP exemplar proving the recipe works end-to-end: a Japanese
  character fine-tune runs on CPU/CUDA with 3-s–6-s reference clips and a
  one-command `infer.py --text "…" --reference ref1 --output out.wav`:
  https://huggingface.co/Skttttt/GPT-SoVITS-Miku-Nakano
- Known pitfalls from that same card: English targets via a JP-trained model
  "can sound accented, repeat, or mispronounce words. Short sentences work
  better." → for English manim narration, either train on English-adjacent
  data or do **zero-shot** (JP reference, EN target) and expect a light
  L2 accent, which arguably fits Miku.

### Why GPT-SoVITS over the alternatives

| Route | What it is | Miku-likeness | Free-speech narration? | Notes |
|---|---|---|---|---|
| **GPT-SoVITS v2/v3 (+Miku ref)** | TTS, few-shot | high with 1–10 min Miku speech | **yes** | recommended; self-hosts |
| RVC (aple/HidekoHaruna/javinfamous miku) | *singing* voice conversion | high for singing | indirectly (convert TTS voice) | two-stage: neutral TTS → RVC; prosody comes from the TTS voice, Miku timbre layered on top; cv4/so-vits-svc 4.x are the engines (https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI; so-vits-svc 4.1 stable is archived: https://github.com/svc-develop-team/so-vits-svc) |
| so-vits-svc 4 "Hatsune Miku" (Civitai) | singing SVC, 461 MB zip, 3000 epochs, trained on acapellas | high for singing | no — sings only | https://civitai.com/models/74595 |
| zeta-apps "Miku" | hosted zeta TTS character | medium | in-chat only, hosted | https://zeta-ai.io/en/plots/8e70682d-ada8-4645-8530-850b24f7e4b9/profile — no self-hosted weights |
| F5-TTS / CosyVoice / OpenVoice | general TTS | needs own Miku fine-tune | yes | fully viable, but no ready Miku artefact to start from; GPT-SoVITS has the most Miku-adjacent community data |

RVC note: RVC is also the *back-end* option — generate the narration with
voicecloning-lite (e.g. GPT-SoVITS zero-shot or any TTS), pitch-shift up,
then RVC-convert with a Miku model to stamp the timbre. More work, less
coherent prosody; keep as plan B.

---

## 3. Route B — licensed: NijiVoice (Algomatic), comparison only

NijiVoice is an AI voice platform by Algomatic (DMM Group), in
commercial release since Dec 2024, with a public API:

- Pricing: prepaid, **¥825 / 10,000 characters**, 5,000-char free trial
  (https://gamemakers.jp/article/2024_12_11_87974/,
  https://docs.nijivoice.com/docs/getting-started).
- API returns generated audio (base64 WAV endpoint added per
  https://docs.nijivoice.com/changelog/); example request pattern at
  https://harusdia.hatenablog.com/entry/2025/07/05/190000.
  A community MCP + AIAvatarKit wiring exist
  (https://github.com/ryoooo/nijivoice-mcp,
  https://gist.github.com/uezo/4e4e8f965075e7d96d1e187d37ddf0f8).
- **But** NijiVoice's characters are original voice-actor-based personas —
  there is **no actual Hatsune Miku** (Crypton doesn't license her voice to
  it), so you get "anime-style girl", not Miku. Also requires
  "NIJI Voice" credit on output. This is the route you take only when you
  want clean legal speech; not the route for *Miku*.

Official Crypton side for completeness:

- **VOCALOID 5/6 Miku** can export narration-quality WAV (44.1 kHz/16-bit,
  master/multitrack; https://vocaloid.com/en/support/faq/503, /608), but it's
  sing-for-mediaphase: narration means MIDI + per-note tuning with no
  natural prosody engine; you'd hand-tune every word.
- 4xPIA / Piapro: **no Miku voicebank for PIA** (Miku's PIA offerings are
  song-licensed); the closest first-party "Miku speaks" is game dialogue.
- Nijiflow/Nijivocal web tools: no Miku character.

**Verdict:** the official stack either doesn't include Miku or is
phoneme-grid song synthesis — for video *narration* you'll always be better
served by a TTS model.

---

## 4. Where to get Miku speech data (piracy allowed)

Ranked for GPT-SoVITS use:

1. **Game spoken dialogue (best timbre match to Miku-as-person):** Project
   DIVA / Miku: Stage Pass / Project SEKAI (Miku) voice lines. Game
   extraction (dumped dialogue WAVs) — SEKAI specifically has one-click
   per-character voice-bulk-downloaders:
   https://github.com/GuangChen2333/FindUrVoicesPJSK (Sekai-World database,
   card + profile + song voices, WAV export, auto-annotation format).
   Even ~1–5 minutes of clean dialogue is enough for a GPT-SoVITS v2
   fine-tune (repo's own target: ~1 min).
2. **Vocaloid song acapellas (timbre + her high register):** vocaloid.net /
   r/VocaTorrent; DDB (V4) voicebank sample extraction is scripted
   (https://github.com/RigoLigoRLC/vocaloid-ddb-extractor); V5 banks use the
   newer DVIQ/DVQM packaging (installer/extractor scripts, e.g.
   https://pastebin.com/ziRhTupk; .vpr tooling
   https://github.com/TheerapakG/vocaloid5tools). Use as *blended* training
   set, not alone — it's sung material.
3. **Existing community Miku datasets/models as starting point:** reuse or
   fine-tune-a-fine-tune from aple/HatsuneMikuRVC, javinfamous/infamous_miku_v2
   (38-min dataset documented) or NoCrypt's miku set
   (https://huggingface.co/NoCrypt/miku_RVC — the mikuTTS space that builds
   on it: https://huggingface.co/spaces/NoCrypt/mikuTTS).

Target set for a solid Miku-speaker: **5–15 min mixed, majority spoken,
cleanly labelled** (GPT-SoVITS label format: `path|speaker|ja/en|text`),
recorded at 32 kHz.

---

## 5. Recommended pipeline (local RTX 3080, 20 GB)

```
script.txt ──> GPT-SoVITS (Miku model) ──> narration.wav
                                                        │
manim render (video, silent) ──────────────┐           │
                                           ▼           ▼
                                  ffmpeg mux: -i vid.mp4 -i narr.wav
                                             -c:v copy -c:a aac
```

Steps:

1. `bash install.sh --device CU128 --source HF` (or use the Docker image
   `xxxxrt666/gpt-sovits`), Python 3.10 env — repo README.
2. **Quick path (today):** zero-shot inference — pick 3–5 s of clean Miku
   *spoken* clip (DIVA/SEKAI line), type narration text, get WAV. RTF
   ~0.03 → 4 min narration ≈ seconds of GPU time.
3. **Good path (a weekend):** label ~10 min Miku dialogue WAVs
   (GPT-SoVITS WebUI has built-in ASR + labeller), fine-tune v2
   GPT+SoVITS pair, then batch-infer line-by-line with line-level reference
   selection for expression.
4. Mux per clip: align each segment's WAV to the manim beat via `ffmpeg
   -start_number … amix` or just concat with known offsets; keep the video
   `libx264/copy` + `aac` audio. (The hke reel has "No audio, no subtitles"
   today — this is a drop-in upgrade.)

Pitfalls:

- **Monotone narration:** TTS reads flat; inject pauses and emphasis via the
  text (commas, ！/?, shorter lines), and give each script line its own
  reference clip.
- **Uncanny valley:** a GPT-SoVITS model overfits to its reference
  register — mixed singing+speaking data reduces the "singer reading a
  script" wobble; keep lines short (≤ 1–2 s each) and keep the voice
  high-pitched but not peaky.
- **Cross-language English:** expect a Japanese-accented delivery from a
  JP-trained model (documented behaviour, Skttttt model card). Either lean
  into it (arguably more Miku) or add EN reference data if available.
- **Long-form coherence:** synthesise per-beat, not per-whole-script; this
  also gives you per-beat re-rolls and the timing control manim needs.
- **Licence hygiene (even with piracy allowed):** the *engine* weights are
  MIT; the *voice data* is Fujita/Crypton — for a private/personal video
  that's fine; for public release expect at most takedown-level risk, same
  as any fan-made Miku track.

## 6. Sources

- GPT-SoVITS features/versions:
  https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90features(Latest%E2%80%90Including%E2%80%90v5)
- GPT-SoVITS repo (install, zero/few-shot, RTF):
  https://github.com/RVC-Boss/GPT-SoVITS
- JP character fine-tune exemplar + reference-clip guidance:
  https://huggingface.co/Skttttt/GPT-SoVITS-Miku-Nakano
- Miku RVC models (timbre provenance): aple/HatsuneMikuRVC,
  HidekoHaruna/RVC_V2_Hatsune_Miku, javinfamous/infamous_miku_v2 (HF links
  above); engine: Retrieval-based-Voice-Conversion-WebUI,
  svc-develop-team/so-vits-svc (4.1 stable, archived)
- Civitai so-vits-svc Miku (singing): https://civitai.com/models/74595
- zeta-miku character (hosted): https://zeta-ai.io/en/plots/8e70682d-ada8-4645-8530-850b24f7e4b9/profile
- NoCrypt miku RVC + mikuTTS space:
  https://huggingface.co/NoCrypt/miku_RVC ·
  https://huggingface.co/spaces/NoCrypt/mikuTTS
- SEKAI dialogue downloader:
  https://github.com/GuangChen2333/FindUrVoicesPJSK
- V4 DDB extraction: https://github.com/RigoLigoRLC/vocaloid-ddb-extractor ·
  V5 project tooling: https://github.com/TheerapakG/vocaloid5tools
- NijiVoice (Algomatic): pricing/credits
  https://gamemakers.jp/article/2024_12_11_87974/ ·
  API docs https://docs.nijivoice.com/docs/getting-started ·
  WAV-base64 endpoint changelog
  https://docs.nijivoice.com/changelog/%E3%82%A8%E3%83%B3%E3%82%B3%E3%83%BC%E3%83%99%E3%83%89%E3%81%95%E3%82%8C%E3%81%9F%E9%9F%B3%E5%A3%B0%E3%81%AE%E7%94%9F%E6%88%90api%E3%81%AE%E6%96%B0%E8%A6%8F%E8%BF%BD%E5%8A%A0
- VOCALOID 5/6 WAV export: https://vocaloid.com/en/support/faq/503 · /608
- Fujita recording interview:
  https://web.archive.org/web/20080102085851/http:/www.j-cast.com/2008/01/01015179.html
