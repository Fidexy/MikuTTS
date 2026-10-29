# Milestone 1 results — CPU inference verified (2026-10-29)

**Status: DONE.** GPT-SoVITS v2ProPlus runs end-to-end on CPU + RAM, no GPU.

## Environment built

| Item | Result |
|---|---|
| venv | `~/MikuTTS/.venv`, Python **3.10.22** |
| torch | **2.14.1+cpu**, `cuda_available=False` |
| numpy | **1.26.4** (repo's `<2.0` pin satisfied) |
| onnxruntime | 1.23.2, CPUExecutionProvider only |
| pyopenjtalk | 0.4.1 (built from source after `apt install cmake python3.10-dev`) |
| weights | `vendor/GPT-SoVITS/GPT_SoVITS/pretrained_models` — **4.6 GB**, v2/v2Pro/v2ProPlus/v3/v4 all present |

## Install gotchas actually hit (all resolved)

1. `install.sh` **hard-requires conda** → "Conda Not Found". Bypassed by doing its
   steps manually: download `pretrained_models.zip` (4.2 GB) from HF and unzip into
   `GPT_SoVITS/`.
2. `requirements.txt` pins **`onnxruntime-gpu`** on x86_64 → patched to `onnxruntime`
   (original kept as `requirements.txt.orig`).
3. `pyopenjtalk` / `jieba_fast` / `opencc` failed to build → fixed by
   `sudo apt install python3.10-dev cmake` (Python.h + CMake). `opencc` also needed
   the newer wheel rather than the pinned source build.
4. **torchaudio 2.11 removed its old loader** → `ImportError: TorchCodec is required`.
   Fix: `pip install torchcodec --index-url https://download.pytorch.org/whl/cpu`.
5. Missing runtime dep not in requirements: **`resampy`**.
6. Module paths: the API needs `PYTHONPATH=.:./GPT_SoVITS` from the repo root
   (both `AR` and `tools` imports are resolved that way).

## The katakana-English mechanism — PROVEN

`pyopenjtalk.g2p('ザ ワールド イズ マイ ステージ')`
→ `z a | w a a r u d o | i z u | m a i | s U t e e j i`

Exactly the target aesthetic (`the`→*za*, `world`→*waarudo*, `stage`→*suteeji*).
This is the load-bearing assumption of §C of the plan, and it holds on CPU with no GPU.

## CPU performance (measured, Ryzen 5 5600X)

Model load: **0.9–1.8 s**.

| Config | RTF | Verdict |
|---|---|---|
| 4 threads, fp32 | 0.837 | — |
| **6 threads, fp32** | **0.751** | ✅ **use this** |
| 12 threads, fp32 | 0.914 | oversubscription hurts |
| 6 threads, fp16 | 0.826 | no CPU fp16 accel; slower |

Variance between identical runs: 0.751–0.906.

**Realistic planning number: RTF ≈ 0.75–0.9 (~0.8–1.3× realtime).**
The plan's 0.3–0.8 projection was optimistic; the official 0.526 figure is an M4 CPU
on long text. Practical consequence: a 3-min narration ≈ 3–4 min CPU (not 1–3 min),
and a 1-hour video ≈ 70–80 min. Fine, but must not be planned as "seconds".

Short lines are dominated by fixed overhead (RTF 2.4 at 0.9 s → 0.77 at 2.5 s), so
**batch per-paragraph rather than per-word** where timing allows.

## Reproduce

```bash
cd ~/MikuTTS/vendor/GPT-SoVITS && . ../../.venv/bin/activate
PYTHONPATH=.:./GPT_SoVITS python ../../scripts/smoke_test.py
```

## Still open

- Reference audio is a **synthetic placeholder** — real Miku timbre needs milestone 2 data.
- `text_lang='ja'` path verified; the `all_ja` variant untested.
- No `ja_userdic` overrides added yet (that's the accent-tuning knob, milestone 4).
