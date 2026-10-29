#!/usr/bin/env python
"""Milestone-1 smoke test: verify GPT-SoVITS runs on CPU and the katakana-English
phoneme path produces the intended phonetic-bash accent.

Run from the repo root:
    cd vendor/GPT-SoVITS && . ../../.venv/bin/activate
    PYTHONPATH=.:./GPT_SoVITS python ../../scripts/smoke_test.py
"""
import os
import time

import torch

os.environ.setdefault("version", "v2ProPlus")
# 6 threads measured fastest on Ryzen 5 5600X; 12 threads oversubscribes (see notes).
torch.set_num_threads(6)
torch.set_num_interop_threads(1)

import soundfile as sf  # noqa: E402

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

REPO = "GPT_SoVITS/pretrained_models"
REF_AUDIO = "../../data/ref/placeholder.wav"
OUT_DIR = "../../out"

TEXTS = [
    "ザ ワールド イズ マイ ステージ",
    "ヘロー エブリワン アンド ウェルカム トゥ ザ チャンネル",
    "トゥデイ ウィ アー ゴーイング トゥ ラーン アバウト ベクトル スペーシズ",
]


def main() -> None:
    cfg = TTS_Config(
        {
            "device": "cpu",
            "is_half": False,  # fp16 is slower on CPU (no AVX2 fp16 accel)
            "version": "v2ProPlus",
            "t2s_weights_path": (
                f"{REPO}/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
            ),
            "vits_weights_path": f"{REPO}/v2Pro/s2Gv2ProPlus.pth",
        }
    )
    t0 = time.time()
    tts = TTS(cfg)
    print(f">> pipeline loaded in {time.time() - t0:.1f}s (device={cfg.device})")

    os.makedirs(OUT_DIR, exist_ok=True)
    total_audio = total_wall = 0.0

    for i, text in enumerate(TEXTS, 1):
        req = {
            "text": text,
            "text_lang": "ja",  # katakana-English: JP frontend, phonetic-bashed input
            "ref_audio_path": REF_AUDIO,
            "prompt_text": "ハツネ ミク",
            "prompt_lang": "ja",
            "top_k": 15,
            "top_p": 0.8,
            "temperature": 0.8,
            "text_split_method": "cut5",
            "batch_size": 1,
            "speed_factor": 1.0,
            "seed": 42,
            "parallel_infer": False,
            "repetition_penalty": 1.35,
        }
        t0 = time.time()
        dur = 0.0
        for sr, audio in tts.run(req):
            dur += len(audio) / sr
            sf.write(f"{OUT_DIR}/smoke_{i}.wav", audio, sr)
        wall = time.time() - t0
        total_audio += dur
        total_wall += wall
        print(f"   line{i}: {dur:5.2f}s audio  {wall:5.2f}s wall  RTF {wall / dur:.3f}")

    print(
        f">> TOTAL {total_audio:.2f}s audio / {total_wall:.2f}s wall "
        f"=> RTF {total_wall / total_audio:.3f}"
    )
    print(f">> wrote {OUT_DIR}/smoke_*.wav")


if __name__ == "__main__":
    main()
