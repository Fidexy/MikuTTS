# MikuTTS

Hatsune Miku text-to-speech on CPU only, built on [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS),
with a deliberate twist: **English text is transliterated into katakana and fed to the
Japanese frontend**, so she speaks English with an authentic Japanese accent
("the" → ざ, "world" → わーるど). The target use is narrating manim videos.

No GPU is used. Everything runs on a Ryzen 5 5600X at roughly real time.

---

## Status

| Milestone | State |
|---|---|
| 1. Engine on CPU | ✅ GPT-SoVITS v2ProPlus synthesizes end-to-end, RTF ~0.75–0.9 |
| 2. Miku spoken corpus | ✅ 50 clips / 2.6 min, exact Japanese transcripts |
| 3. Zero-shot | ✅ working (`scripts/miku_say.py`) |
| 4. Fine-tune | ⏸ blocked on data volume — see [Fine-tune sources](#fine-tune-sources) |
| 5. Sentence-length narration + ffmpeg mux | ⏳ not started |

---

## Quick start

### 1. Environment

```bash
cd ~/MikuTTS
/usr/bin/python3.10 -m venv .venv          # the system python is 3.14; GPT-SoVITS needs 3.10–3.12
. .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torchcodec resampy e2k        # torchaudio 2.11 needs torchcodec; resampy is not pinned upstream
```

### 2. Engine + weights

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS vendor/GPT-SoVITS
# install.sh hard-requires conda and abort on this box, so fetch the bundle directly:
curl -L -o /tmp/pm.zip https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip
unzip /tmp/pm.zip -d vendor/GPT-SoVITS/GPT_SoVITS/
pip install -r vendor/GPT-SoVITS/requirements.txt
```

See [`notes/MILESTONE-1-RESULTS.md`](notes/MILESTONE-1-RESULTS.md) for the five install traps
(`python3.10-dev`, `cmake`, `pyopenjtalk`, `onnxruntime-gpu`, `numpy<2.0`).

### 3. Voice data

```bash
.venv/bin/python scripts/extract_miku_lines.py     # -> data/miku21/ + manifest.list
```

### 4. Speak

```bash
cd vendor/GPT-SoVITS
PYTHONPATH=.:./GPT_SoVITS ../../.venv/bin/python ../../scripts/miku_say.py \
  --ref ../../data/ref/miku_prompt.wav \
  --ref-text "今日、ライブに来てくれたみんなにも お礼を言わせてほしいな" \
  --text "Hello, I am Hatsune Miku." "Today I want to sing about the world."
```

Output lands in `out/zeroshot_N.wav` (32 kHz). The script prints both the English and the
katakana it actually synthesized, so you can see what went to the frontend.

---

## How the phonetic-bash works

```
English  "The voice of the future is here."
   │
   ├─ e2k (C2K model) + a hand-tuned override table
   ▼
Katakana "ざ ぼいす おぶ ざ フューチャー いず ひあ."
   │
   ├─ OpenJTalk via pyopenjtalk, text_lang='ja'
   ▼
Phonemes "z a | b o i s u | o b u | z a | f u tS a a | i z u | h i a"
   │
   ├─ GPT-SoVITS v2ProPlus (zero-shot from a 4.6 s Miku reference)
   ▼
Audio
```

The point is that the model never leaves its training distribution. Doing cross-lingual
inference (`text_lang='en'` on a Japanese model) produces the same accent as an *artefact*;
this produces it *by construction*, and keeps prosody stable.

**`scripts/miku_say.py`** holds the `OVERRIDES` table. Add entries to steer pronunciation —
it is the whole character of the voice, and it is worth tuning by ear.

---

## Layout

```
MikuTTS/
├── scripts/
│   ├── extract_miku_lines.py   # Miku-only corpus + manifest (filter by speaker id!)
│   ├── miku_say.py             # katakana-English synthesis, the main entry point
│   └── smoke_test.py           # CPU sanity check from milestone 1
├── data/
│   ├── miku21/                 # 50 Miku clips + manifest.list  (relpath|text)
│   └── ref/miku_prompt.wav     # 4.6 s reference for zero-shot
├── notes/                      # milestone findings
├── out/                        # generated audio
└── vendor/GPT-SoVITS/          # engine + 4.6 GB weights (not in git)
```

Only code and docs are committed. Venvs, `vendor/`, voice data and generated audio are
gitignored.

---

## Gotchas that cost real time

- **A Miku card contains a whole conversation.** Downloading every voice in her card
  episodes yields ~2.8 % Miku; the rest is Ichika, Saki, Honami and ~25 others. Filter on
  `Voices[].Character2dId == 21`, never on the card owner.
- **`miku_say.py` must be run from `vendor/GPT-SoVITS`** with `PYTHONPATH=.:./GPT_SoVITS`,
  otherwise you get `ModuleNotFoundError: tools` / `AR`.
- **GPT-SoVITS yields int16, not float.** Writing it as float32 clamps to ±1 and gives you a
  square wave that looks like audio (peak 1.00, RMS 0.9) but isn't. The script normalises.
- **The asset CDN 403s urllib's default User-Agent.** Any request needs an explicit header.
- **The shipped `inference_cli.py` cannot do `all_ja`**, only 中文/英文/日文 — so the katakana
  path goes through the Python API.
- **`FindUrVoicesPJSK` is broken** against the current asset bucket: it writes HTTP error
  bodies to disk as `.wav`, and its URL scheme 404s in every region. Use the scripts here.

---

## Performance

Measured on Ryzen 5 5600X (AVX2, no AVX512), fp32, 6 threads:

| Config | RTF |
|---|---|
| 4 threads | 0.837 |
| **6 threads (default)** | **0.751** |
| 12 threads | 0.914 (oversubscription) |
| 6 threads + fp16 | 0.826 (no AVX2 half-precision gain) |

RTF ≈ 0.75–0.9 means roughly **real time**: a 3-minute narration takes 3–4 minutes, an
hour-long video 70–80 minutes. Short lines are dominated by fixed per-call overhead.

---

## Fine-tune sources

The current corpus (2.6 min) is enough for zero-shot but thin for fine-tuning, which
typically wants 10–15 min of single-speaker speech. Realistic options, in rough order of
value for *this* project:

### Tier 1 — same series, more of it

1. **PJSK event/story scenarios.** Miku appears in event stories even though the 400-file
   `scenario/` sample showed none — she does speak in some events. Needs a broader sweep of
   the `scenario/` prefix (it is large; enumerate by event, filter `Character2dId == 21`).
   *Same recording conditions as the existing 50 clips, so it blends cleanly.*
2. **PJSK "area conversations" and virtual-singer interactions.** Short, conversational,
   and already in the same asset bucket. Cheap to add if the prefix can be found.

### Tier 2 — adjacent official material

3. **Project DIVA / Mega Mix voice clips.** Miku has hundreds of short spoken lines
   (menu callouts, song intros) across the DIVA series. Different codec and room, so needs
   light denoising and a consistency check, but the speaker is the same (Saki Fujita).
4. **Miku's official spoken promos** — Magical Mirai announcements, snow-Miku messages,
   Crypton's own voice packs. Few but high-quality and professionally recorded.

### Tier 3 — community datasets (fastest to get volume)

5. **`javinfamous/infamous_miku_v2`** — ~38 min of Miku dataset used by the RVC community.
6. **`HidekoHaruna/RVC_V2_Hatsune_Miku`** — ~22 min, includes Saki Fujita Project DIVA X clips.
7. **`NoCrypt/miku_RVC`**, **`aple/HatsuneMikuRVC`** — smaller RVC training sets.

These are already-segmented and easy to load, but they are *derived* datasets of uncertain
provenance and variable quality. Worth a spot-check before trusting them into a corpus.

### Tier 4 — sung material (use with care)

8. **Vocaloid song acapellas.** The notes' original plan allows a blended set. Sung audio
   has no speech prosody and much longer vowels, so a model trained mostly on it will
   sound like she is singing spoken words. Best used as **≤20 % of the corpus** to widen
   timbre/phone coverage, never as the majority.

### Not viable

- **Miku's Vocaloid voicebank.** A `.ddb` bank holds only single-phoneme samples at one
  sustained pitch — no sentences, no prosody, no timing. It cannot supply the
  utterance→sentence alignment training needs, and there is no *speaking* bank at all.
  At most useful later as a reference for pronunciation overrides.

---

## Licensing

Engine weights are MIT. The **voice** belongs to Saki Fujita / Crypton Future Media, and
PJSK audio assets are their property. This is a personal research project; anything
published with this voice needs to respect Crypton's character guidelines.
