# Data, models and licences

Every dataset, pretrained model, checkpoint and vendored file Earmark touches, with its
source and licence. Licences marked **checked** were read from the primary source on
2026-09-11 (the Hugging Face Hub licence tag, the Zenodo record metadata, the OpenSLR page
or the upstream repository). Re-check a row before a public release; licences can change.

**Ground rules**

- No dataset audio is committed to git. Prepared training data lives in **private** Kaggle
  datasets, derived by the notebooks in `notebooks/`.
- Evaluation code stores numbers only (JSON, JSONL and transcripts). Enhanced audio is never
  written, except at most 10 gallery clips (`GALLERY_MAX_CLIPS`). Each gallery clip will be
  listed with its source speaker, chapter and licence in `docs/ATTRIBUTION.md`.
- Anything CC BY-NC (ESC-50) is evaluation-only and never appears in training or in the public
  gallery.
- Downloads are pinned: the evaluation assets and checkpoints by SHA-256 in
  `python/earmark/eval/assets.tsv`, and the WeSpeaker ONNX file by SHA-256 in
  `python/earmark/data/embeddings.py`.

## Training data

| Dataset | Used for | Source | Licence | Notes |
| --- | --- | --- | --- | --- |
| LibriTTS-R train-clean-100, train-clean-360 (capped at 3 min per speaker) | target and interfering talkers; enrolment pools | HF `mythicinfinity/libritts_r` | CC BY 4.0 (**checked**) | Each speaker is split into enrolment and target pools by chapter. The dev/test splits (`devtest_16k`) are used only for Earmark-Synth. |
| VCTK 0.92 | extra accents for talkers and interferers | Edinburgh DataShare 10283/3443 | CC BY 4.0 | p232 and p257 are excluded because they are the VoiceBank+DEMAND test speakers. |
| MUSAN music | music bed under TV/podcast interferers | OpenSLR 17 | CC BY 4.0 (**checked**) | Split into train and held-out music. Individual tracks carry their own licences inside the archive. |
| RIRS_NOISES | simulated RIRs (training); point-source noises | OpenSLR 28 | Apache 2.0 (**checked**) | Real RIRs are held out for evaluation. Point-source noises go through the content filter below. |
| DEMAND | training noise environments | Zenodo record 1227121 | CC BY 4.0 (**checked**) | The VoiceBank+DEMAND test environments are held out (`VB_HELDOUT_ENVIRONMENTS`). |
| Kokoro-82M synthetic "agent voice" | agent-voice interferer (training voices only) | HF `hexgrad/Kokoro-82M` | Apache 2.0 (**checked**) | 28 English voices, split once with seed 0; the 7 test voices never reach training. The prompt text comes from LibriTTS-R. |

**Models used while preparing the data (their outputs shape the data, but they are not part of Earmark)**

| Model | Role | Source | Licence | Notes |
| --- | --- | --- | --- | --- |
| WeSpeaker ResNet34-LM (VoxCeleb) | frozen 256-d enrolment embeddings, 8 per speaker | HF `Wespeaker/wespeaker-voxceleb-resnet34-LM` | CC BY 4.0 (**checked**) | ONNX file SHA-256 pinned. Front end: Kaldi fbank with a Hamming window (see DECISIONS.md). |
| AST fine-tuned on AudioSet | content filter that drops noises in excluded classes (aircraft, propellers, engines) | HF `MIT/ast-finetuned-audioset-10-10-0.4593` | BSD-3-Clause (**checked**) | Only decides which noise clips to keep; see `noise_rir_16k/content_filter_log.json`. |

## Evaluation data

| Dataset | Suite | Source | Licence | Notes |
| --- | --- | --- | --- | --- |
| VoiceBank+DEMAND test set (824 utterances) | B (reproduction gates) | Edinburgh DataShare 10283/2791 | CC BY 4.0 | SHA-256 pinned; fetched by `scripts/fetch_eval_data.sh vbd`. |
| LibriTTS-R dev/test | A (Earmark-Synth) | HF `mythicinfinity/libritts_r` | CC BY 4.0 (**checked**) | Speaker-disjoint from training (a leak check runs when the mixer is built). |
| LibriCSS | R (real rooms) | GitHub `chenzhuo1011/libri_css` (audio on Google Drive) | **unclear**: the repository has no licence file (GitHub reports NOASSERTION); the source speech is LibriSpeech, CC BY 4.0 | Used for evaluation only. Do not redistribute LibriCSS audio or put it in the gallery until the licence is confirmed with the authors. |
| LibriSpeech test-clean | R (enrolment for LibriCSS speakers) | OpenSLR 12 | CC BY 4.0 (**checked**) | |
| ESC-50 | A (`esc50_real_rir` condition) | GitHub `karolpiczak/ESC-50` | CC BY-NC 3.0; the ESC-10 subset is CC BY | Evaluation only (non-commercial); the licence is recorded per clip in the manifest. |
| Earmark-Real (own recordings) | R | recorded by the author with volunteers | chosen by each volunteer on `docs/CONSENT_TEMPLATE.md` (CC BY 4.0 proposed) | No recordings exist yet. |
| Svarah (Indian-English slice) | A (planned) | gated dataset | check before use | Named as a condition only; no exporter yet. |
| DNS Challenge 5 | C (v1.1) | Microsoft DNS Challenge | check before use | After launch. |

## Baselines and checkpoints

| System | Status | Source | Licence | Notes |
| --- | --- | --- | --- | --- |
| GTCRN model code | vendored: `python/earmark/eval/baselines/gtcrn_model.py` | GitHub `Xiaobin-Rong/gtcrn` @ `502ebfab` | MIT, Copyright (c) 2024 Rong Xiaobin (**checked**) | Notice in `GTCRN_LICENSE.txt`. Three documented edits: einops replaced by a reshape, `h is None`, demo removed. |
| GTCRN checkpoints (VoiceBank+DEMAND, DNS3) | fetched, not committed | same commit | MIT (**checked**) | SHA-256 pinned in `assets.tsv`. |
| DeepFilterNet3 (deep-filter 0.5.6 CLI) | planned | GitHub `Rikorose/DeepFilterNet` | MIT or Apache 2.0 | The binary will be SHA-256 pinned. |
| RNNoise | planned | xiph/rnnoise | BSD-3-Clause | |
| WebRTC APM (webrtc-noise-gain 1.3.0) | planned | PyPI | check when added | manylinux wheels only, so it runs in CI or Colab. |
| Whisper (WER) | planned | openai/whisper | MIT | |
| onnxruntime-web (browser enrolment) | planned | microsoft/onnxruntime | MIT | Self-hosted in the Space. |

## Vendored code

| File | Upstream | Version | Licence |
| --- | --- | --- | --- |
| `engine/third_party/pocketfft_hdronly.hpp` | github.com/mreineck/pocketfft (`cpp` branch) | commit `c90e55b3`, plus the no-alloc patch in `engine/third_party/patches/` | BSD-3-Clause (`POCKETFFT_LICENSE.md`) |
| `engine/third_party/catch2/` (tests only) | github.com/catchorg/Catch2 | v3.16.0 | BSL-1.0 (`catch2/LICENSE.txt`) |
| `python/earmark/eval/baselines/gtcrn_model.py` | github.com/Xiaobin-Rong/gtcrn | commit `502ebfab` | MIT (`GTCRN_LICENSE.txt`) |

SHA-256 values for the vendored engine files are in `engine/third_party/README.md`.

## Python dependencies (installed, not vendored)

As declared in each package's metadata: torch (BSD-3-Clause and bundled licences), numpy
(BSD-3-Clause), scipy (BSD-3-Clause), soundfile (BSD-3-Clause), soxr (LGPL-2.1-or-later;
used as an unmodified, dynamically loaded library), PyYAML (MIT), huggingface_hub
(Apache 2.0), pyarrow (Apache 2.0), pesq (MIT wrapper around the ITU-T P.862 reference code,
which the ITU licenses separately; used for evaluation only), pystoi (MIT), pytest (MIT).

## Training-data contamination by system

This table records which data each system was trained on, so that any overlap with an
evaluation suite is visible. It is filled in as systems are added.

| System | Trained on | Overlap to watch |
| --- | --- | --- |
| Earmark (M, S-GRU, S-SSM) | LibriTTS-R train, VCTK (without p232/p257), DEMAND (without the VB test environments), simulated RIRs, MUSAN train music, Kokoro training voices; WeSpeaker embeddings (VoxCeleb) | None by construction: dev/test speakers, VB speakers, held-out environments, real RIRs, held-out music and test voices are rejected by the mixer's leak check. |
| GTCRN-VB | VoiceBank+DEMAND train (28 VCTK speakers, DEMAND) | In-domain on Suite B; its test speakers are disjoint from its training set. |
| GTCRN-DNS3 | DNS Challenge 3 (LibriVox read speech and others) | LibriVox is also the source of LibriSpeech, LibriTTS-R and LibriCSS, so Suite A and R speakers may overlap. Check before comparing. |
| DeepFilterNet3 | DNS Challenge 4 and other corpora (see its paper) | Same LibriVox caveat. To be confirmed when the row is added. |
| WeSpeaker ResNet34-LM | VoxCeleb 1 and 2 | Speaker verification only; no overlap with the enhancement suites is expected. |
