# UniEE

**Unified Regression-Classification Engagement Estimation via Hierarchical Multimodal Fusion and Multi-Party Interaction**

## Table of Contents

- [Background](#background)
- [Contributions](#contributions)
- [Challenge Domains](#challenge-domains)
- [Method Overview](#method-overview)
- [Feature Set](#feature-set)
- [Results](#results)
- [Source Code Tutorial](#source-code-tutorial)
  - [1. Clone and Set the Package Path](#1-clone-and-set-the-package-path)
  - [2. Prepare Raw Data](#2-prepare-raw-data)
  - [3. Build Per-Session Feature Caches](#3-build-per-session-feature-caches)
  - [4. Compute Feature Statistics](#4-compute-feature-statistics)
  - [5. Train UniEE](#5-train-uniee)
  - [6. Run Inference](#6-run-inference)
  - [7. Validate and Zip Submission](#7-validate-and-zip-submission)
- [Project Layout](#project-layout)
- [License](#license)

This repository contains the source code for UniEE, our MultiMediate'26 multi-domain engagement estimation system. UniEE is designed for the new MultiMediate'26 setting where engagement estimation is no longer only continuous regression under ordinary domain shift: the benchmark also introduces heterogeneous label semantics, heterogeneous feature families, and heterogeneous interaction structures.

## Background

Engagement estimation predicts how engaged each participant is at every frame of an interaction. Earlier MultiMediate systems mainly treated engagement as a continuous regression problem, but MultiMediate'26 combines several transfer axes in a single benchmark:

- Language shift: English/French/German/Japanese/Chinese in training, plus Arabic/Italian/Indonesian/Spanish zero-shot test languages.
- Age and social setting shift: adult dyadic conversations, adult group discussions, and child free-play sessions.
- Interaction-structure shift: dyadic partner modeling for NoXi-style data and multi-party partner modeling for MPIIGI.
- Label-semantics shift: CCC regression for NoXi, NoXi-Additional, NoXi-J, and MPIIGI; Cohen's Kappa classification for PInSoRo task/social engagement.

UniEE addresses these shifts by keeping DAPA's reactive/anticipatory interaction backbone and extending it with regression-classification unification, hierarchical multimodal fusion, and attention-based multi-party pooling.

## Contributions

1. **Label-semantics unification.** UniEE jointly models continuous CCC domains and categorical PInSoRo domains. A Learnable Bridge maps PInSoRo task/social class probabilities into a pseudo-continuous engagement space so classification supervision can contribute to the shared representation.

2. **Hierarchical multimodal fusion.** The final model uses an 11-feature full preset with 7,490 input dimensions. Features are organized into five semantic groups and fused by intra-group and inter-group Transformer encoders.

3. **Multi-party interaction modeling.** UniEE extends dyadic DAPA-style target-partner modeling to MPIIGI group discussions through learnable attention pooling over multiple partners.

## Challenge Domains

| Domain | Languages | Setting | Label | Metric |
|---|---|---|---|---|
| NoXi | en / fr / de | Dyadic adult conversation | Continuous [0, 1] | CCC |
| NoXi-Additional | ar / it / id / es | Zero-shot dyadic adult conversation | Continuous [0, 1] | CCC |
| NoXi-J | ja / zh | Dyadic adult conversation | Continuous [0, 1] | CCC |
| MPIIGI | de | 3-4 person group discussion | Continuous [0, 1] | CCC |
| PInSoRo CC / CR | child free play | Child-human/robot interaction | 4 task + 5 social classes | Cohen's Kappa |

Manifest statistics in this repository:

| Domain | Train sessions | Val sessions | Test sessions | Labeled train roles |
|---|---:|---:|---:|---:|
| NoXi | 38 | 10 | 16 | 76 |
| NoXi-Additional | 0 | 0 | 12 | 0 |
| NoXi-J | 31 | 10 | 10 | 62 |
| MPIIGI | 4* | 2* | 6 | 14 |
| PInSoRo CC | 20 | 7 | 6 | 40 |
| PInSoRo CR | 12 | 5 | 6 | 24 |

`*` MPIIGI has no official training split, so the code uses a 4-session train / 2-session held-validation split from its validation set. Across all splits, the manifests contain 195 source sessions and 470 session-role entries, of which 287 are labeled. PInSoRo's `env` stream is used as contextual input but has no engagement label.

## Method Overview

```text
11 aligned feature streams
  -> per-feature projection: Linear + LayerNorm + GELU
  -> hierarchical ModalityGroupFusion
  -> HierarchicalDomainPrompt
  -> MultiPartnerPooling
  -> 3 x DAPA reactive/anticipatory cross-attention layers
  -> regression head + PInSoRo classification heads + LearnableBridge
```

Model settings:

| Item | Value |
|---|---:|
| Hidden dimension | 384 |
| DAPA layers | 3 |
| Attention heads | 4 |
| Dropout | 0.15 |
| Prompt tokens | 4 coarse + 8 fine |
| Coarse prompt groups | conversational_adult, group_discussion, child_freeplay |
| Classification classes | 4 task + 5 social |
| Parameters, 11-feature model | 18.017M |

The bridge uses hand-coded ordinal priors to create a pseudo-continuous PInSoRo target:

```text
pseudo = 0.5 * task_prior[task_class] + 0.5 * social_prior[social_class]
```

Task priors are `(0.80, 0.40, 0.10, 0.55)` for `goaloriented, aimless, noplay, adultseeking`; social priors are `(0.10, 0.30, 0.50, 0.70, 0.90)` for `solitary, onlooker, parallel, associative, cooperative`.

## Feature Set

The paper's final model uses the 11-feature full preset (`mm26_whisper_full`), totaling 7,490 dimensions after alignment.

| Group | Feature | Dim |
|---|---|---:|
| Audio | w2vbert2 | 1024 |
| Audio | egemapsv2 | 88 |
| Audio | whisper | 1280 |
| Text | xlmr | 768 |
| Visual behavior | openface2 | 714 |
| Visual behavior | openface3 | 21 |
| Visual behavior | openpose | 139 |
| Visual semantic | videomae | 1408 |
| Visual semantic | dino | 768 |
| Visual semantic | swin | 768 |
| Cross-modal | clip | 512 |

Other presets are defined in `configs/feature_specs.yaml`:

| Preset | Number of features | Total dim | Use |
|---|---:|---:|---|
| `mm26_main` | 8 | 4034 | compact baseline |
| `mm26_whisper` | 9 | 5314 | adds Whisper |
| `official_full` | 10 | 6210 | official features without Whisper |
| `mm26_whisper_full` | 11 | 7490 | final paper preset |
| 11 + `qwen3vl_emb` | 12 | 8514 | optional VLM experiment |

The optional Qwen3-VL experiment is implemented in `scripts/stage3a_vlm_experiment.sh`. It is analyzed as a negative/diagnostic feature experiment rather than used in the final model.

## Results

Regression test CCC:

| Method | NoXi | NoXi-Add | MPIIGI | NoXi-J | CCC Avg |
|---|---:|---:|---:|---:|---:|
| UniEE best | 0.807 | 0.759 | 0.747 | 0.636 | 0.737 |

PInSoRo test Cohen's Kappa:

| Dimension | UniEE |
|---|---:|
| CC Task | 0.202 |
| CC Social | 0.230 |
| CR Task | 0.235 |
| CR Social | 0.230 |
| Average | 0.224 |

Component ablation on CCC domains:

| Configuration | NoXi | NoXi-Add | MPIIGI | NoXi-J | CCC Avg |
|---|---:|---:|---:|---:|---:|
| Full UniEE | 0.807 | 0.759 | 0.747 | 0.636 | 0.737 |
| w/o ModalityGroupFusion | 0.794 | 0.762 | 0.643 | 0.608 | 0.702 |
| w/o MultiPartnerPooling | 0.799 | 0.755 | 0.697 | 0.595 | 0.712 |

## Source Code Tutorial

### 1. Clone and Set the Package Path

The code imports itself as `multimediate26`. Clone the repository into a directory named `multimediate26`, and run module commands from the parent directory:

```bash
git clone git@github.com:Yuefeng-Zou/UniEE.git multimediate26
pip install -r multimediate26/requirements.txt
```

If you clone into a different directory name, either rename it to `multimediate26` or create an equivalent symlink.

### 2. Prepare Raw Data

Place the official MultiMediate'26 data under a local data root. The expected split layout is documented in `manifests/dataset_overview.md` and encoded in `data/feature_extractor/build_session_npz.py`.

The main processed cache used by the 11-feature pipeline is:

```text
multimediate26/data_processed/npz_v4
```

This directory is intentionally not tracked by git.

### 3. Build Per-Session Feature Caches

Build 11-feature aligned `.npy` caches and manifests:

```bash
DATA_ROOT=/path/to/mm_26/data
FEATURES_PRESET=mm26_whisper_full

python -m multimediate26.data.feature_extractor.build_session_npz \
  --data-root "$DATA_ROOT" \
  --out-root multimediate26/data_processed/npz_v4 \
  --preset "$FEATURES_PRESET" \
  --manifest-dir multimediate26/manifests \
  --skip-done
```

For a labeled-only training cache, add `--labeled-only`. For test inference, build without `--labeled-only` so unlabeled test sessions are included.

Important preprocessing handled by the builder:

| Issue | Handling |
|---|---|
| `w2vbert2` 40 Hz header mismatch | override to 25 Hz where required |
| PInSoRo 30 Hz visual streams | interpolate to 25 Hz |
| MPIIGI DINO 2304-dim streams | slice to 768 dims |
| NaN / missing values | normalize, clip, and zero-fill |
| PInSoRo categorical labels | save `label_task.npy`, `label_social.npy`, and `label_pseudo_cont.npy` |

### 4. Compute Feature Statistics

Compute z-score normalization statistics over the training manifests:

```bash
mkdir -p multimediate26/experiments/_feature_stats

python -m multimediate26.data.feature_extractor.compute_feature_stats \
  --npz-root multimediate26/data_processed/npz_v4 \
  --manifests \
    multimediate26/manifests/noxi_train.jsonl \
    multimediate26/manifests/noxi_j_train.jsonl \
    multimediate26/manifests/mpiigi_train.jsonl \
    multimediate26/manifests/pinsoro_cc_train.jsonl \
    multimediate26/manifests/pinsoro_cr_train.jsonl \
  --features openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,videomae,dino,swin,clip \
  --max-partners 3 \
  --out multimediate26/experiments/_feature_stats/feature_stats_v4_whisper_full.npz
```

### 5. Train UniEE

The standard curriculum is:

| Phase | Domains | Purpose | Script |
|---|---|---|---|
| Phase 1 | NoXi + NoXi-J | regression pretraining | `scripts/stage2_phase1_v2arch.sh` |
| Phase 2 | NoXi + NoXi-J + MPIIGI | add multi-party regression | `scripts/stage3_phase2_v2arch.sh` |
| Phase 3 | all five domains | add PInSoRo, bridge, ordinal loss | `scripts/stage4_phase3_v2arch.sh` |

Main training settings:

| Setting | Value |
|---|---:|
| Window length | 512 frames |
| Training stride | 64 frames |
| Batch size | 32 |
| EMA decay | 0.999 |
| Gradient clip | 1.0 |
| Warmup | 1000 steps |
| Precision | bf16 |
| Sampling | single-domain batches with sqrt-N domain weighting |

Example 11-feature run:

```bash
FEATURES="openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,videomae,dino,swin,clip"
NPZ_ROOT=multimediate26/data_processed/npz_v4
FEATURE_STATS=multimediate26/experiments/_feature_stats/feature_stats_v4_whisper_full.npz

# Phase 1
EXP_NAME=phase1_11feat \
FEATURES="$FEATURES" \
NPZ_ROOT="$NPZ_ROOT" \
FEATURE_STATS="$FEATURE_STATS" \
bash multimediate26/scripts/stage2_phase1_v2arch.sh

# Phase 2
EXP_NAME=phase2_11feat \
FEATURES="$FEATURES" \
NPZ_ROOT="$NPZ_ROOT" \
FEATURE_STATS="$FEATURE_STATS" \
INIT_FROM=multimediate26/output/phase1_11feat_seed0/best.pt \
bash multimediate26/scripts/stage3_phase2_v2arch.sh

# Phase 3
EXP_NAME=phase3_11feat \
FEATURES="$FEATURES" \
NPZ_ROOT="$NPZ_ROOT" \
FEATURE_STATS="$FEATURE_STATS" \
INIT_FROM=multimediate26/output/phase2_11feat_seed0/best.pt \
bash multimediate26/scripts/stage4_phase3_v2arch.sh
```

The scripts expose `SEED`, `GPU`, `EPOCHS`, `BATCH`, `WINDOW_LEN`, `TRAIN_STRIDE`, `FEATURES`, `NPZ_ROOT`, `FEATURE_STATS`, and `INIT_FROM` as environment-variable overrides.

### 6. Run Inference

Standalone inference:

```bash
python -m multimediate26.train.inference \
  --checkpoint multimediate26/output/phase3_11feat_seed0/best.pt \
  --out-dir predictions/phase3_11feat_seed0 \
  --features openface2,openface3,openpose,w2vbert2,egemapsv2,whisper,xlmr,videomae,dino,swin,clip \
  --feature-stats multimediate26/experiments/_feature_stats/feature_stats_v4_whisper_full.npz \
  --npz-root multimediate26/data_processed/npz_v4 \
  --use-group-fusion
```

Paper inference uses 512-frame windows, Hann-window overlap-add, regression smoothing, three-seed ensembling, TTA over LayerNorm/domain-prompt parameters, and domain-specific checkpoint selection.

### 7. Validate and Zip Submission

```bash
python -m multimediate26.submission.zip_and_check \
  predictions/phase3_11feat_seed0 \
  --zip predictions/phase3_11feat_seed0.zip \
  --strict
```

## Project Layout

```text
multimediate26/
├── configs/                 # model, domain, feature, instruction configs
├── data/                    # dataset, samplers, label loading, feature builders
├── inference/               # prompt-only TTA and full test helpers
├── losses/                  # CCC/MSE/smoothness and ordinal contrastive losses
├── manifests/               # split manifests and dataset documentation
├── manifests_p8/            # language-split manifests
├── models/                  # UniEE model, DAPA layers, prompts, heads, pooling
├── scripts/                 # preprocessing, training, retraining, evaluation
├── submission/              # official writer and zip validator
├── train/                   # trainer, inference, ensemble, TTA, baselines
├── README.md
└── requirements.txt
```

Large local artifacts are not tracked: `.codegraph/`, processed `.npy/.npz` caches, checkpoints, logs, and OS/editor files.

## License

Research code for the MultiMediate'26 challenge. Please follow the challenge terms for dataset access, redistribution, and submission use.
