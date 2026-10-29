# Voicing manim videos with Hatsune Miku — options

Research date: 2026-10-24. Goal: a Miku-timed narration track for manim
renders. Piracy/unlicensed approaches are explicitly acceptable.

## The core problem

There is no "Miku TTS model" shipped as a product by anyone. Miku's voice is
Saki Fujita's recordings, and what exists in the wild splits into three
buckets:

1. **Singing voice conversion (SVC/RVC)** — VITS/so-vits-svc/RVC models
   trained on Miku singals. They convert an input *vocal* track to Miku.
   They do **not** take text input: you must feed them a TTS voice (or a
   singing voice) first.
2. **GPT-SoVITS character fine-tunes** — zero/few-shot TTS engines fine-tuned
   on a character's speech. These take text directly and are the best
   "Miku narrator" option.
3. **Official/licensed speech** — Nijiflood, PIAP, 4xPIA, VOCALOID
   engine-based (VOCALOID4/5 speech synthesis) — all constrained by license
   + engine quirks.

## Ranked options

### 1. GPT-SoVITS fine-tuned on Miku speech (recommended)

- Engine: [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) (MIT).
  Zero-shot: ~5 s reference clip; few-shot: fine-tune with **~1 min** of
  training data (README). v2+ supports **Chinese, Japanese, English, Korean,
  Cantonese** (cross-lingual inference), so English narration is fine
  ([features wiki](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT-SoVITSfeatures)).
- The Miku-likeness comes from the **reference/training audio**, which you
  supply. Two sub-options:
  a. Use an existing Miku GPT-SoVITS fine-tune if/when one shows up on
     HuggingFace (search filter `base_model: finetune: lj1995/GPT-SoVITS`).
     The ecosystem is chock full of character fine-tunes; a Miku one is
     plausible (there are e.g. Miku-character repos for PJSK voice data via
     [FindUrVoicesPJSK](https://github.com/GuangChen2333/FindUrVoicesPJSK)
     which bulk-downloads every Miku line in Project Sekai: solo vocals,
     profile voice, card voice — exactly the *spoken* Miku data you want).
  b. Train your own: gather **spoken** Miku audio (game dialogue! e.g.
     Project Diva / Miku: Stage Pass / PJSK cutscene lines — Miku *talking*,
     not singing), label it, train. 5–20 min of clean speech is plenty for a
     strong few-shot model. Training runs comfortably on a 3080.
  - Inference: a few minutes of audio per script is ~seconds to a minute on
    a 3080. [AdamCodd/GPTSoVITS-JP-tts](https://huggingface.co/AdamCodd/GPTSoVITS-JP-tts)
    shows an example of a well-trained JP GPT-SoVITS checkpoint pair on
    HuggingFace — same recipe, swap in Miku audio.
- Why it wins for narration: prosody/expressiveness of an LLM TTS front end,
  text in, WAV out, no license cost, self-hosted.

### 2. RVC / so-vits-svc Miku model + any TTS (two-stage)

- Feed a normal TTS engine (Piper, Coqui, XTTS, Kokoro) the script, then
  voice-convert to Miku with an RVC model.
- Models with heavy Miku training data already exist:
  - [javinfamous/infamous_miku_v2](https://huggingface.co/javinfamous/infamous_miku_v2) —
    **38 min dataset, 1000 epochs, RVC v2**.
  - [HidekoHaruna/RVC_V2_Hatsune_Miku](https://huggingface.co/HidekoHaruna/RVC_V2_Hatsune_Miku) —
    22 min dataset across multiple composers, includes clips of Saki Fujita
    (Miku's CV) from Project Diva X.
  - [aple/HatsuneMikuRVC](https://huggingface.co/aple/HatsuneMikuRVC) —
    3 flavors (default/mellow/power), 30–40 epochs each, on original/EN
    voicebank data.
  - [NoCrypt/miku_RVC](https://huggingface.co/NoCrypt/miku_RVC) — used by the
    [NoCrypt/mikuTTS](https://huggingface.co/spaces/NoCrypt/mikuTTS) HF
    Space, which is literally a "text → Miku speech" web app (duplicated by
    [John6666](https://huggingface.co/spaces/John6666/mikuTTS) and
    [terastudio](https://huggingface.co/spaces/terastudio/mikuTTS) if the
    original goes offline). That space is RVC-GSVC hybrid — TTS then
    conversion — i.e. exactly this pipeline.
  - [Civitai / Hatsune Miku so-vits-svc](https://civitai.com/models/74595)
    (461 MB zip, 3000 epochs, so-vits-svc 4.x).
- Trade-off: RVC models trained mostly on *singing* data give a "singing
  while talking" wobble on speech. Models that include *spoken* material
  (Fujita clips, game voice) behave much better for narration — prefer
  HidekoHaruna / javinfamous. The 38-min-dataset model is the safest bet.
- Note: RVC also shifts pitch/timbre well onto pre-pitched TTS, and you can
  run it in batch on a 3080 in a few minutes for a full script.
- Licensing: RVC models are community uploads, usually unlicensed or
  All-Rights-Reserved — totally fine given the piracy acceptance.

### 3. so-vits-svc 4 / Zeta (Miku character) — hosted, less controllable

- [zeta-ai.io Miku character](https://zeta-ai.io/en/plots/8e70682d-ada8-4645-8530-850b24f7e4b9/profile)
  — chatbot with Miku TTS. It works, but you're at the mercy of the service,
  queueing, and per-message limits; exportable audio quality is "good enough"
  for a demo but not for a polished manim video.
- so-vits-svc 4.1-Stable is [archived](https://github.com/svc-develop-team/so-vits-svc)
  (GNU AGPL) — the ecosystem has migrated to GPT-SoVITS-based tools, so
  Zeta is effectively a convenience wrapper around the same tech.

### 4. Synthesizer V — dead end for Miku

No official Miku voicebank (SynthV is a separate engine; confirmed by
[r/SynthesizerV](https://www.reddit.com/r/SynthesizerV/comments/1j98310/)
and [r/Vocaloid](https://www.reddit.com/r/Vocaloid/comments/1am42dx/)). The
deleted "Miku SynthV test" video (June 2024) was likely taken down for
license reasons — a useful caution flag.

### 5. Official routes (for completeness — why they lose)

- **Crypton Nijiflo (Nijidisc) AI** — Miku's official AI character;
  conversation-oriented, outputs speech in-chat, and you cannot cleanly
  export an offline WAV voice track for a video pipeline (it's designed as a
  chatbot/streamer).
- **PIAP / 4xPIA** — speech TTS ("AI Miku" talk) exists in-game; 4xPIA has
  Japanese speech output but the pipeline is PC-app-bound, per-sentence, and
  licensed for PIAP voicebanks. No clean export API, no batch.
- **VOCALOID 5 + Miku V5 bank** — you *can* have Miku speak (phoneme-based
  "chat mode" of the engine), and VOCALOID 5/6 exports 44.1 kHz/16-bit WAV
  per official FAQ ([V5 FAQ 503](https://www.vocaloid.com/en/support/faq/503),
  [V6 FAQ 608](https://www.vocaloid.com/en/support/faq/608)). But Miku V5
  speech uses *Saki Fujita's actual voicebank* as a speech TTS — the result
  is mechanical, monotone, and needs per-phrase input; plus it costs
  (~€300+ for bank + engine). Fine as a fallback "official" flavor; wrong
  tool for a narration workflow.
- Also of note: **PiaproStudio_TTS** (GitHub,
  [isanan39s/PiaproStudio_TTS](https://github.com/isanan39s/PiaproStudio_TTS))
  — a hobby project that scripts PIAP V4x to do Japanese TTS with Miku
  voicebanks via ppsf + open-jtalk. Requires a licensed bank. Same monotone
  constraint.

## Reference-audio sourcing (the important part)

To get *real* Miku, the training/reference audio must be
**Miku speaking**, not singing. Best piracy-friendly sources, in order:

1. **Game dialogue — the goldmine**: Project DIVA, Miku: Stage Pass
   (MSP), Project SEGA (PJSK/Miku), CHUNITHM, etc. All contain substantial
   Miku *spoken* lines. PJSK specifically has an automated bulk downloader
   ([FindUrVoicesPJSK](https://github.com/GuangChen2333/FindUrVoicesPJSK),
   based on the Sekai-World database, outputs clean WAVs with transcripts —
   独唱, 资料语音, 卡片语音 separately). This is the highest-value dataset:
   spoken, clean, transcribed, and Miku's on-screen personality is consistent.
2. **Live2D VTuber Miku / Nijiflo / Miku chat AI outputs** if you can record
   them: a secondhand source, but genuinely Miku-voiced prose at scale.
3. **Saki Fujita's direct speech** (interviews, Live2D motion, DIVA X voice
   clips) — timbrally *identical* Miku, but very short clips.
4. **Song lyrics/spoken parts** (Miku radio segments, radio-show interludes
   in PJSK, CHUNITHM) — medium-length spoken material.
5. What to **avoid** as primary speech data: her vocaloid singals. RVC
   singing banks capture timbre but not speech prosody — they produce the
   "sings while narrating" effect.

**Data volume targets**
- GPT-SoVITS few-shot: **1–5 min** of high-quality spoken Miku, transcribed,
  with transcripts in the target language(s). For English narration, record
  English training material (ideally a Miku who is speaking English — e.g.
  English-voicebank Miku content, or PJSK English mode, or a Miku cosplay
  VTuber) *or* use the multilingual fine-tune to cross-language, but a
  cross-lingual TTS will retain some accent; for a clean English Miku
  narrator, mix 60% English spoken Miku + 40% Japanese Miku.
- RVC: 20–40 min of singing/speaking Miku (already the norm in the models
  above).

## Recommended pipeline

Assuming a 3080-class GPU (confirm locally) and the goal of a ~1–3 min
narration for each manim video:

1. **Gather data**: bulk-download PJSK Miku lines (spoken-focused filter)
   with FindUrVoicesPJSK; add any Fujita direct-speech clips.
2. **Build the model** (one-time, ~hours on 3080):
   - Train a **GPT-SoVITS v2 (v5turbo)** checkpoint, 1–5 min speaking data,
     transcribed, with target languages set to `en` (+ `ja` if bilingual).
   - *Or* skip training and zero-shot with a 5–10 s Miku clip every time.
3. **Per video**:
   ```bash
   # manim renders the silent video
   manim -qm render.py Scene
   # TTS the script (GPT-SoVITS, ~30 s to a minute on GPU)
   ./gptsovits_infer.sh script.txt miku.ckpt reference_miku.mp3 en
   # mux
   ffmpeg -i out.mp4 -i miku_narration.wav -c:v copy -c:a aac -shortest final.mp4
   ```
4. **Fallback** if TTS TTS quality disappoints: GPT-SoVITS → RVC pipeline
   (TTS in neutral voice, RVC to Miku with
   [javinfamous/infamous_miku_v2](https://huggingface.co/javinfamous/infamous_miku_v2)
   or [HidekoHaruna/RVC_V2_Hatsune_Miku](https://huggingface.co/HidekoHaruna/RVC_V2_Hatsune_Miku)).

## Quality + effort estimate

| Route | Miku-likeness | Effort | Hardware | Notes |
|-------|---------------|--------|----------|-------|
| GPT-SoVITS fine-tune (1) | High (speaker similarity + prosody) | Low–Med (2–4 hrs to train) | 3080 fine | Best for narration; requires spoken-Miku audio set |
| RVC + TTS (2) | High for timbre, medium for prosody | Low (download model, run batch) | 3080 fine | Quickest; RVC "singing" wobble on speech is the main risk |
| Zeta / hosted Miku (3) | Medium | Very low | Any | Not for production |
| Official (5) | Very low (monotone speech mode) | High | Any | Wrong tool for this use-case |

**Bottom line**: Train a GPT-SoVITS on spoken Miku audio (PJSK lines from
FindUrVoicesPJSK are the fastest way to a 10-min clean dataset). If you
need it *today*, run one of the existing RVC models on top of a normal TTS
output.

## Open questions

1. Are there any *spoken* Miku clips in a downloadable/extractable format in
   PJSK, or does FindUrVoicesPJSK cover card/profile/duet voice only?
2. Does GPT-SoVITS v5 cross-lingual (English narration from Japanese Miku
   training data) produce an acceptable accent, or is English training data
   mandatory?
3. Confirm RTX 3080 VRAM is actually present (nvidia-smi) and check disk
   space for the dataset (~1–2 GB of compressed WAVs).

## Key sources (links)

- GPT-SoVITS repo + multilingual features:
  https://github.com/RVC-Boss/GPT-SoVITS ·
  https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT-SoVITSfeatures
- Example JP GPT-SoVITS checkpoint (recipe reference):
  https://huggingface.co/AdamCodd/GPTSoVITS-JP-tts ·
  https://huggingface.co/Skttttt/GPT-SoVITS-Miku-Nakano
- PJSK Miku bulk-downloader (spoken data):
  https://github.com/GuangChen2333/FindUrVoicesPJSK
- RVC Miku models:
  https://huggingface.co/javinfamous/infamous_miku_v2 ·
  https://huggingface.co/HidekoHaruna/RVC_V2_Hatsune_Miku ·
  https://huggingface.co/aple/HatsuneMikuRVC ·
  https://huggingface.co/NoCrypt/miku_RVC
- Miku TTS spaces (RVC-GSVC hybrid, hosted):
  https://huggingface.co/spaces/NoCrypt/mikuTTS ·
  https://huggingface.co/spaces/John6666/mikuTTS ·
  https://huggingface.co/spaces/terastudio/mikuTTS
- so-vits-svc Miku (SVC, 461 MB, Civitai):
  https://civitai.com/models/74595
- Zeta Miku (hosted chatbot TTS):
  https://zeta-ai.io/en/plots/8e70682d-ada8-4645-8530-850b24f7e4b9/profile
- SynthV has no Miku (dead end):
  https://www.reddit.com/r/SynthesizerV/comments/1j98310/
- PICAP / 4xPICAP:
  https://github.com/isanan39s/PiaproStudio_TTS
- VOCALOID 5/6 WAV export (official FAQ):
  https://www.vocaloid.com/en/support/faq/503 · /608
