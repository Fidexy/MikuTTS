# Plan v3 — English by phoneme mashing, not loanword transliteration

Supersedes §C of `PLAN-CPU-RAM.md` v2. Milestone 3 output showed the v2 target was
**wrong**, and you named the error precisely.

## 0. The correction

v2 §C said: *"the classic JPN-Vocaloid English: `the` → za, `world` → waarudo"* and framed
the goal as *"a timbre-identical reproduction of how the Japanese frontend maps English
onto Japanese phonemes"*.

That is **loanword transcription** — the mapping a Japanese *speaker* applies when reading
English. It is what I built in milestone 3. It produces "a Japanese person speaking
English".

You want the **Vocaloid** condition, which is a different and stranger thing:

| | Loanword (what M3 does) | Vocaloid phoneme-mash (what you want) |
|---|---|---|
| `world` | わーるど → /w a a r u d o/ | /w 3: r l d/ — keeps **/r/ AND /l/**, keeps the final cluster |
| `the` | ざ → /z a/ | /D @/ — **/D/**, which Japanese does not have at all |
| `computer` | こんぴゅーたー | /k @ m p j u: t @ r/ — final /r/ as an approximant, not ー |
| `mine` | まいん | /m aI n/ — a **diphthong**, not two mora |
| Why it sounds wrong | vowels collapsed to 5, `-er`→ー, clusters broken by う | vowels came from the wrong inventory; consonants are *foreign on purpose* |

The Vocaloid sound is not "English filtered through Japanese rules". It is **correct
English phonemes produced by a voice that learned Japanese** — so the vowels come out
slightly wrong and the consonants carry too much weight. Miku's iconic `/w 3: r l d/` is
not わーるど, and that is the whole point.

**Consequence:** the katakana step must be *removed*, not refined. Tuning `OVERRIDES` in
`miku_say.py` is now tuning the wrong thing.

## 1. What the engine can actually do (verified)

The v2ProPlus phoneme inventory is `GPT_SoVITS/text/symbols2.py` — pinyin initials +
Japanese/Chinese vowels. Measured against real English ARPAbet:

```
inventory has : AA B CH D DH ER F G HH IH JH K L M N NG P R S SH T TH V W Y Z ZH
MISSING (vow): AE AH AO AW AY EH EY IY OW OY UH UW
```

**27 of 39 English phonemes are present, including /TH/ /DH/ /R/ /L/ /V/ /Z/ /ZH/.** What's
missing is almost entirely **vowels**. That is exactly the Vocaloid condition: consonants
land, vowels bend. We do not need to fake this — the engine imposes it.

Stress digits are a separate axis (`AH0` vs `AH1`), handled by the English frontend.

Verified end-to-end at the text layer:

```
g2p('the world is mine') → DH AH0 W ER1 L D IH1 Z M AY1 N     (nothing dropped)
```

So `text_lang='en'` is **live and lossless** on the model already installed. This is the
single most important fact in this plan: the target is reachable with **no training, no
data, and no new weights**.

## 2. Architecture change

```
BEFORE (M3, wrong)                      AFTER (target)
English                                English
  │ e2k C2K → katakana                  │ CMUdict g2p → ARPAbet (keeps /DH/, /3:/, clusters)
  ▼                                      ▼
ざ ぼいす おぶ ざ ふゅーちゃー           DH AH0 V OY1 S AH0 V DH AH0 F Y UW1 CH ER0
  │ text_lang='ja' (OpenJTalk)          │ text_lang='en'
  ▼                                      ▼
/k o N p j u u t a a/   ← loanword     /k AH0 M P Y UW1 T ER0/  ← English, model-accented
```

Both routes feed the **same Japanese-trained v2ProPlus model**. The difference is purely
which phonemes go in. Route 2 keeps English's own segmental structure and lets the
model's Japanese training supply the accent — which is the authentic Vocaloid mechanism.

### 2a. The three knobs, in order of impact

1. **`text_lang`** — `'en'` vs `'ja'` on already-katakana'd text. `'en'` is the target.
2. **Stress digits** — `AH0`(reduced schwa) vs `AH1`. The reduced vowels are what make
   English sound natural; a Vocaloid often *under*-reduces. Stripping stress to force
   fuller vowels is a real aesthetic lever (MIKU English famously over-articulates).
3. **`ja_userdic`** — no longer the main knob. Useful only for proper nouns.

### 2b. What replaces the katakana table

A **phoneme override table** keyed on ARPAbet instead of katakana:

```
data/phoneme_overrides.tsv
word<TAB>ARPAbet     # e.g.  world<TAB>W ER1 L D
                            miku<TAB>M IY1 K UW0
```

`ja_userdic`/katakana stay in the repo behind a flag — the A/B is genuinely informative,
and the ear decides.

## 3. Acceptance test

The gate is not "does it sound like a Japanese person". It is:

| Check | Pass condition |
|---|---|
| `world` contains /r/ and /l/ | yes — `W ER1 L D`, never わーるど |
| `the` contains /DH/ | present in the phoneme stream, not `/z/` |
| Final clusters survive | `-ld`, `-sks`, `-nts` not broken by inserted う |
| Diphthongs are single segments | `mine` = `M AY1 N`, not `MA I N` |
| A/B is audibly **different** from M3 | the two files must not sound like the same voice reading the same words |
| Same speaker | median f0 within ~±60 Hz of the 394 Hz reference |

Automated: assert the phoneme stream for a fixed sentence contains `ER1`, `L`, `DH`, and
`AY1` (this catches any silent regression to the katakana route). Ear: blind A/B of the
same sentence from both routes.

## 4. Milestones

| # | Step | Time | Gate |
|---|---|---|---|
| M4a | `miku_say.py --mode phonemes`: `text_lang='en'`, ARPAbet pass-through, log the phoneme stream | 30 min | phoneme stream prints with `ER1`/`DH`/`L` intact |
| M4b | Synthesize the M3 sentence set through both routes into `out/ab/` | 15 min | 3+3 wavs, same reference, comparable loudness |
| M4c | **Your ear verdict** | — | does `en` route sound like Vocaloid English, not a Japanese speaker? |
| M4d | Tune: stress handling, then phoneme overrides for words the ear rejects | iterative | aesthetic locks |
| M4e | Sentence-length narration + ffmpeg mux for manim | 1–2 h | 30 s narration muxed onto a test clip |

Fine-tuning stays **on hold** — and this is now a stronger position: the accent comes from
the phoneme route, not from data volume, so 2.6 min remains sufficient to test the
aesthetic. Fine-tuning would only sharpen timbre later.

## 5. Open questions I need you on

1. **How wrong should the vowels be?** Options: (a) unstressed → schwa (natural English,
   least Vocaloid), (b) keep full vowels, under-reduce (over-articulated Miku English —
   my read of "phoneme mashing"), (c) deliberately force Japanese vowels onto English
   phonemes. I'd start at **(b)**, since it is the sound of her actual English voicebank.
2. **Latin-only or mixed-language lines?** Affects whether `auto` splitting is needed.
3. **Should the katakana route stay as a `--mode loanword` fallback?** Costs nothing and
   makes the A/B reproducible.

## 6. What I got wrong, plainly

I treated "phonetic bashing" as equivalent to "loanword transliteration" and wrote it into
the plan as if settled, then built it and reported success. The pipeline works — that part
was real — but it answers a different question than the one you asked, and the plan's own
example (`world` → waarudo) was the tell I should have caught before writing code. Your
correction is the right one, and it is cheaper to implement than what I already built.
