# Miku TTS on CPU/RAM — plan v2 (decisions integrated)

Supersedes §A–F of `PLAN-CPU-RAM.md` v1. Research base: `MIKU-VO-NOTES.md`,
`miku-voiceover.md`. Surveyed 2026-10-26.

## 0. Decisions (from you)

| # | Your answer | How it lands in the plan |
|---|---|---|
| 1 | "No idea" (zero-shot vs fine-tune) | **Zero-shot first, fine-tune second.** Zero-shot costs nothing, needs no training, and — critically — it *tests the katakana-English trick in minutes*. Fine-tune is only worth a night of CPU if zero-shot's similarity disappoints. |
| 2 | **Full** dataset | **Download all** Miku PJSK spoken data (all card + profile + pure voices). **Train in stages**: curated ~10–15 min for the first fine-tune; remainder kept for a second pass. Rationale: "full" download is cheap and gives better reference-clip choice; "full" *training* on CPU is ~linear in audio length and 30–60 min of audio = days, not a night. |
| 3 | Wants the **"phonetic-bashed English"** Miku accent | This is the **primary design constraint** and it changes the architecture — see §C. Short version: **don't fight the accent, synthesise it deliberately.** Train on Japanese, feed English pre-transliterated to katakana, use `text_lang='ja'`. English training data becomes unnecessary. |
| 4 | GPU "decide late" | Deferred by design. **Training defaults to CPU regardless** — 4 GB VRAM means batch=1 and "pray it runs" (upstream maintainer's own words), while CPU gets batch ≈ RAM÷4 ≈ 8–11. The 1650 is only a candidate for *inference*, decided after step 3. |
| 5 | Layout "don't know" | Plan picks (veto freely): everything under `~/MikuTTS` — engine in `vendor/GPT-SoVITS`, data in `data/`, outputs in `out/`, one entry point `miku_say.py`. |

## 1. Machine (verified)

| | |
|---|---|
| CPU | Ryzen 5 5600X, 6c/12t, **AVX2 only**, 4.65 GHz |
| RAM | 45 GB (≈39 free), 8 GB swap |
| GPU | GTX 1650 4 GB — idle, healthy, **not relied on** |
| Disk | 130 GB free |
| OS/py | Ubuntu 26.04, `python3.10` + pip + venv ✓, `ffmpeg` ✓, docker ✓ (≈10 idle containers) |

CPU-only is an officially supported GPT-SoVITS target.

**MEASURED (milestone 1, supersedes the estimate):** RTF **0.75–0.9** (~1× realtime) on
v2ProPlus. The projected 0.3–0.8 was optimistic — the official 0.526 is an M4 CPU on long
text. Practical: 3-min narration ≈ **3–4 min** CPU; 1-hour video ≈ 70–80 min.
Config: `set_num_threads(6)` (12 threads is *slower*), `is_half=False` (fp16 slower on AVX2),
model load ~1 s. Short lines are dominated by fixed overhead — batch per paragraph, not
per word. See `notes/MILESTONE-1-RESULTS.md`.

Inference RAM: 3–5 GB fp32. No pressure.

## 2. Verified install gotchas (pre-solved)

These are the things that would otherwise burn an evening:

| Blocker | Finding | Fix |
|---|---|---|
| Japanese frontend | `pyopenjtalk` 0.4.1 is **source-only** (sdist) → compiles OpenJTalk; needs CMake, which is **not installed** | Use **`pyopenjtalk-plus` 0.4.1.post10** — has a **prebuilt cp310 manylinux wheel** (verified downloadable). No CMake, no build. Fallback: `apt install cmake` + build. |
| ONNX runtime | `requirements.txt` pins **`onnxruntime-gpu`** on x86_64 → drags CUDA libs onto a CPU box | Override with plain `onnxruntime` before install, or install requirements then `pip uninstall onnxruntime-gpu && pip install onnxruntime` |
| numpy | repo pins `numpy<2.0` | keep the pin inside the venv; don't share with system Python 3.14 |
| Python version | system 3.14 is too new; 3.10 present | use `python3.10 -m venv .venv` (no uv/conda needed — both absent) |
| Kana transliteration | `e2k` v0.8.1 ships model **release assets** (safetensors), not in the wheel | fetch `model-p2k-turbo.safetensors` from the GitHub release; runs on CPU |

## 3. §A — Engine install (CPU)

```bash
cd ~/MikuTTS && mkdir -p vendor data out
python3.10 -m venv .venv && . .venv/bin/activate
git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS vendor/GPT-SoVITS && cd vendor/GPT-SoVITS
pip install -U pip wheel
# CPU torch first, then requirements without the gpu onnx pin
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r extra-req.txt --no-deps
pip install -r requirements.txt          # then fix onnxruntime-gpu -> onnxruntime
pip install pyopenjtalk-plus             # prebuilt wheel, replaces pyopenjtalk
pip install -r extra-req.txt             # faster-whisper: OPTIONAL, we have transcripts
bash install.sh --device CPU --source HF # CPU weights + pretrained bundle
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"   # expect CPU wheel, False
```

Verify the JA frontend actually works before anything else:

```bash
python -c "import pyopenjtalk;print(pyopenjtalk.g2p('ハツネ ミク'))"
```

**Weights — version choice.** v1→v2→v3→v4→v2Pro plus. Pick:
- **`v2ProPlus`** (default): most CPU-tested path, has the only official CPU RTF number, fastest.
- **`v4`** (optional upgrade for final renders): native 48 kHz, fixes v3's metallic artefacts; heavier.

Plan: install **v2ProPlus** for iteration; keep `v4` weights as an optional second profile for the final pass. Same repo, version is a launch parameter.

CPU tuning: `is_half=False`, `torch.set_num_threads(6)`, `set_num_interop_threads(1)`
(leave cores for the docker services), batch 1 at inference.

## 4. §B — Voice source acquisition (full)

**Tool:** `GuangChen2333/FindUrVoicesPJSK` (alive, master, pushed 2026-04-20).
Miku = **`characterId 21`** (verified: `characterProfiles.json` → `self_miku_2nd`).

```bash
git clone --depth 1 https://github.com/GuangChen2333/FindUrVoicesPJSK data/tools/fuv
cd data/tools/fuv && poetry install   # or: pip install requests httpx loguru tqdm
poetry run python main.py            # mode → Miku; card cap: high (e.g. 800+)
```

Content mix (full download):

| Source | Content | Verdict |
|---|---|---|
| `download_character_cards_voices(21, N)` | per-card spoken lines, ~1 line/WAV | ✅ **core corpus** |
| `download_character_profile_voices(21)` | profile lines | ✅ small, very clean |
| "pure voices" | spoken shorts | ✅ include |
| `download_solo_songs(21)` | solo **song** vocals | ❌ sung (+music) — exclude from a speech set |

**Labels come free.** The tool writes `manifest.list` with transcripts; convert:

```bash
python scripts/manifest_for_gpt_sovits.py data/tools/fuv/output/dataset_21/manifest.list data/miku_21 21
# → data/miku_21/*.list  (path|21|ja|text)
```

⇒ **No ASR pass.** This is the single biggest CPU-cost eliminator (Whisper labelling
of an hour of audio on 6 cores would be hours), and it's why CPU-only is viable here.

**QA (script it, don't eyeball 1000 files):** keep 2–13 s; drop clips with music/SFX
(detect by energy-flatness / non-speech), clipping, silence-padding, fades; de-dupe by
audio hash (alt card versions repeat lines); assert every label has non-empty text.
Target: **all downloaded**, curated **10–15 min** high-quality subset for training
(plus a handful of hand-picked "reference clips" for zero-shot).

Licence: engine MIT; voice data Fujita/Crypton — private use accepted risk (per your notes).

## 5. §C — English delivery: **SUPERSEDED, see `PLAN-V3-PHONEME-MASH.md`**

> ⚠️ **This section was wrong and is kept only as a record.** It conflated *loanword
> transliteration* (what a Japanese speaker does reading English: `world` → わーるど)
> with *Vocaloid phoneme mashing* (English phonemes produced by a Japanese-trained voice:
> `/w 3: r l d/`). Those are different targets and the second one is the actual goal.
> Plan v3 replaces this section; milestone 3 built the first thing, not the second.

The original (incorrect) framing follows.

You want the **"Miku singing/talking English by bashing the phonetics"** sound — the
classic JPN-Vocaloid English: `the` → *za*, `world` → *waarudo*, `with you` → *wizu yuu*.
That is not a quality problem to be fixed; it's **a timbre-identical reproduction of how
the Japanese frontend maps English onto Japanese phonemes**.

**How to get it deliberately in GPT-SoVITS:**

1. Model is trained on **Japanese speech** (`ja` labels) — in its native distribution.
2. At inference set **`text_lang='ja'`** (or `'all_ja'`, which forces the Japanese
   frontend for the whole string — verified present in `TextPreprocessor.get_phones_and_bert`,
   alongside `all_zh`, `all_ko`, `all_yue`, `auto`).
3. Feed the text **pre-transliterated to katakana**. The Japanese frontend
   (`pyopenjtalk` → OpenJTalk) then produces true Japanese phonemes, so the accent is
   authentic *by construction* rather than an artefact of cross-lingual drift.
4. Fix the leftovers with **`GPT_SoVITS/text/ja_userdic`** — OpenJTalk's custom
   pronunciation dictionary — for words where the machine transliteration is ugly or
   off-brand (`the`, `a`, `R`, `-ing` endings, proper nouns).

**Tooling:**
- `e2k` (v0.8.1) — English→katakana RNN, numpy+torch(cpu), models from release assets.
  `model-p2k-turbo.safetensors` is the fast one.
- Hand-curated override table (`data/kana_overrides.tsv`) layered on top — this is where
  the *character* of the accent gets tuned, and it will be the main knob we iterate on.
- Optional A/B: also synthesise with `text_lang='en'` to hear the difference; keep
  whichever the ear prefers (expect `en` to drift, `ja` to sound like the reference aesthetic).

**Consequence for data:** ~**zero English training data needed**. That removes the entire
"find English Miku speech" problem, which v1 of this plan flagged as the hard constraint.
Training target stays pure Japanese spoken Miku.

English script for narration ⇒ pipeline is `script.txt` → katakana bash → TTS(`ja`) → WAV.

## 6. §D — Training (staged, CPU)

CPU batch size auto-formula is `RAM ÷ 4` (~8–11 here) — the RAM-only box is genuinely
*better* at training than the 4 GB 1650.

Estimates (3080 datum: 1 min audio ≈ 78 s SoVITS@8ep + 60 s GPT@15ep; CPU multiplier 20–50×):

| Corpus | SoVITS@8 ep | GPT@15 ep | Total |
|---|---|---|---|
| 5 min | 2–4 h | 1.5–3 h | **4–7 h** |
| 10–15 min | 4–10 h | 3–7 h | **7–17 h** |
| 30–60 min (full) | 12–40 h | 9–28 h | **20–68 h** — staged, not first |

Plan: stage 1 on the curated 10–15 min overnight; `save_every_epoch=1` so intermediate
checkpoints are usable; validate similarity vs zero-shot before launching stage 2.
Caveat: upstream warns CPU-trained models are lower quality than GPU-trained — this is
why zero-shot goes first and why "does the fine-tune actually beat zero-shot?" is a
real, measurable gate, not a formality.

## 7. §E — Inference & delivery

```bash
# one line
python miku_say.py --text "Welcome to the channel, everyone!" --out out/beat01.wav
#   internally: e2k → kana bash → GPT-SoVITS(text_lang=ja, ref=..., device=cpu)
# persistent API for manim
python vendor/GPT-SoVITS/api_v2.py --device cpu     # :9880
# mux
ffmpeg -i video.mp4 -i narr.wav -c:v copy -c:a aac -shortest final.mp4
```

Per the notes' pitfalls: synthesise **per beat**, not per whole script (re-rolls + timing);
short lines; pauses via punctuation. `miku_say.py` owns: kana bash → ref-clip choice →
TTS → loudness normalise → optional concat with offsets.

## 8. §F — Milestones

| # | Step | Time | Gate |
|---|---|---|---|
| 0 | machine survey | done | — |
| 1 | venv + engine + CPU weights + `pyopenjtalk` hello-world | 1–2 h | `g2p()` prints katakana phonemes |
| 2 | Full Miku download + label conversion + QA script | 1–2 h (mostly download) | N clips, N labels, all valid |
| 3 | **Zero-shot Miku narration** ← first real output | 30–45 min | you listen: does the accent land? |
| 3b | **Phoneme-mash route (`text_lang='en'`)** — see `PLAN-V3-PHONEME-MASH.md` | 1 h | does it sound like Vocaloid English, not a Japanese speaker? |
| 4 | Tune the bash: `e2k` + `ja_userdic` overrides | iterative | accent matches the aesthetic |
| 5 | Overnight fine-tune on curated 10–15 min | 7–17 h unattended | beats zero-shot on similarity? |
| 6 | (optional) v4 48 kHz profile for final renders; (optional) 1650 inference test | — | quality/speed call |

Decision points: after **3** we know the aesthetic works; after **5** we know whether the
extra hours bought anything. GPU question (4) gets answered with real numbers at 6.

## 9. §G — Sources

- GPT-SoVITS repo, install matrix (`--device CPU`), version notes, dataset format ·
  https://github.com/RVC-Boss/GPT-SoVITS
- Language modes (`all_ja` etc.) · `GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py`
- CPU inference fork (optional) · https://github.com/baicai-1145/GPT-SoVITS-CPUFast
- PJSK single-character voice downloader · https://github.com/GuangChen2333/FindUrVoicesPJSK
- English→katakana · https://github.com/Patchethium/e2k (v0.8.1 release weights)
- Katakana-English / phoneme-bashing background (Vocaloid practice) ·
  https://www.studiorag.com/en-us/vocaloid-english-lyrics · https://ec.crypton.co.jp/support/faq/240
- 4 GB VRAM training reality · https://github.com/RVC-Boss/GPT-SoVITS/issues/577
- Predecessor notes · `MIKU-VO-NOTES.md`, `miku-voiceover.md`
