# MultiMediate'26 Engagement Estimation — MD-DAPA

**Multi-Domain Dual-Aspect Partner Attention (MD-DAPA)** for the [MultiMediate'26](https://multimediate-challenge.org/) cross-cultural multi-domain engagement estimation challenge.

## Overview

Frame-wise engagement prediction across 5 test domains spanning four simultaneous transfer axes: unseen languages, age, social setting, and label semantics (regression vs classification). The final score is the simple mean of per-domain metrics (CCC for regression domains, Cohen's Kappa for PInSoRo classification domains).

| Domain | Language | Setting | Label | Metric |
|---|---|---|---|---|
| NoXi | en / fr / de | Dyadic adult | Continuous [0,1] | CCC |
| NoXi-additional | ar / it / id / es | Dyadic adult (zero-shot) | Continuous [0,1] | CCC |
| NoXi-J | ja / zh | Dyadic adult | Continuous [0,1] | CCC |
| MPIIGroupInteraction | de | Group discussion | Continuous [0,1] | CCC |
| PInSoRo (CC+CR) | — | Child freeplay | Categorical (4 task + 5 social classes) | Cohen's Kappa |

## Architecture

```
Input Features (12 modalities)
  → ModalityProjector (per-feature Linear + LN + GELU)
  → ModalityGroupFusion (6 semantic groups, two-level Transformer)
  → HierarchicalDomainPrompt (3 coarse groups × 5 fine domains)
  → DAPALayer × N (BiLSTM → Reactive/Anticipatory cross-attention)
  → MultiPartnerPooling (learnable query attention over partners)
  → DualTaskHead + LearnableBridge
```

### Key Design Choices

- **25 Hz common time grid**: All modalities resampled to 25 Hz regardless of native sample rate.
- **ModalityGroupFusion**: Two-level Transformer — intra-group encoders fuse modalities within each semantic group (audio / text / visual-behavior / visual-semantic / cross-modal / VLM), then an inter-group encoder merges group representations.
- **HierarchicalDomainPrompt**: Coarse-level (4 tokens, shared within group) + fine-level (8 tokens, per-domain). Three coarse groups: `conversational_adult`, `group_discussion`, `child_freeplay`. Unseen-language fallback: noxi-add → noxi prompt.
- **DAPALayer**: BiLSTM on target/partner streams → split reactive (forward) / anticipatory (backward) → 4-way cross-attention (Reactive T←P, Reactive P←T, Anticipatory T←P, Anticipatory P←T).
- **LearnableBridge**: Maps PInSoRo classification softmax → pseudo-continuous [0,1] so regression head receives gradient even on classification frames. Initialized to output ≈ 0.5 (training mean engagement).
- **VLM gate**: `vlm_gate` initialized to `sigmoid(-3.0) ≈ 0.05` (nearly off), left for the model to open during Phase 3a.

## Features

12 precomputed + optional VLM features (dimensions at 25 Hz):

| Feature | Dim | Modality Group | Source |
|---|---|---|---|
| w2vbert2 | 1024 | audio | WavLM-BERT fused |
| egemapsv2 | 88 | audio | OpenSMILE eGeMAPS |
| whisper | 1280 | audio | Whisper-large-v3 encoder |
| xlmr | 768 | text | XLM-RoBERTa |
| openface2 | 714 | visual-behavior | OpenFace 2.0 AU |
| openface3 | 21 | visual-behavior | OpenFace 3.0 head pose |
| openpose | 139 | visual-behavior | OpenPose keypoints |
| videomae | 1408 | visual-semantic | VideoMAE ViT-B/16 |
| dino | 768 | visual-semantic | DINOv2 (first 768 dim) |
| swin | 768 | visual-semantic | Swin Transformer |
| clip | 512 | cross-modal | CLIP ViT-B/16 |
| qwen3vl_emb | 1024 | vlm | Qwen3-VL-Embedding-8B (Phase 3a) |

Feature presets in `configs/feature_specs.yaml`: `minimal`, `audio_text_visual`, `mm26_main` (8 features, primary), `mm26_whisper`, `mm26_whisper_full`, `official_full`, `official_full_whisper`, `official_full_plus_vlm`.

## Training Pipeline

### 3-Phase Curricular Training

**Phase 1 — Regression-only pretrain (NoXi + NoXi-J)**
- Continuous-only, max_partners=1, lr=5e-5
- Loss: CCC(1.0) + MSE(0.3) + Smooth(0.05)
- 40 epochs, 200 steps/epoch, EMA 0.999

**Phase 2 — Joint with MPIIGI**
- Init from Phase 1 best.pt, lr=3e-5, max_partners=3
- Adds MPIIGI val sessions as quasi-train data
- Same loss structure (bridge/ordinal still disabled)

**Phase 3 — Full 5-domain with PInSoRo**
- Init from Phase 2 best.pt, lr=2e-5
- Enables bridge_ccc=0.3, ordinal=0.1
- Classification CE with label_smoothing=0.1

**Phase 3a — VLM injection (optional)**
- Init from Phase 2 best.pt with `init_new_modality()`, small std=0.01
- Gated `qwen3vl_emb` (1024d) added as modality group
- PInSoRo excluded (no video, ethics-restricted)

### PInSoRo Specialist Models
- **MD-DAPA fine-tuned**: Phase 2 init → PInSoRo-only training with FocalLoss + inverse-freq class weights
- **MLP baseline**: Per-frame MLP classifier mirroring official baseline, MinMaxScaler, independent task/social heads per domain

### Training Features
- Layer-wise LR groups (backbone lower, heads higher)
- Cosine annealing with linear warmup
- bf16 mixed precision autocast
- Gradient clip 1.0
- EMA (0.999, eval with EMA weights)
- DomainBalancedBatchSampler (sqrt-N domain weighting, single-domain batches)
- 8× sliding window augmentation (stride=64, window=512)
- Resume from checkpoint (model + EMA only, fresh optimizer)

### Loss Functions

| Loss | Weight (Phase 1/2) | Weight (Phase 3) | Description |
|---|---|---|---|
| CCC | 1.0 | 1.0 | Lin's Concordance Correlation Coefficient (1−CCC) |
| MSE | 0.3 | 0.3 | Masked mean squared error |
| Smooth | 0.05 | 0.05 | L1 of temporal differences |
| Bridge CCC | 0.0 | 0.3 | PInSoRo pseudo-continuous target supervises regression head |
| Ordinal | 0.0 | 0.1 | Margin-based pairwise contrastive (cross-domain feature alignment) |

## Inference & Submission

### Multi-seed Ensemble
- Val-CCC-weighted averaging across seeds
- Hann-window overlap-add for temporal consistency
- Optional Savitzky-Golay post-smoothing

### Test-Time Adaptation (TTA) — two approaches

1. **Prompt-only TTA** (`inference/tta.py`): Freezes entire model, optimizes 8-token domain prompt only. Smoothness + entropy minimization + range penalty. ~5–15s per session (10 steps).

2. **LayerNorm + prompt TTA** (`train/tta.py`): Adapts LayerNorm + domain_prompt on unlabeled test data. Consistency + smoothness for regression, entropy minimization for classification. 3 epochs.

### Submission Format
- Regression domains: one float per line (`%.6f`)
- PInSoRo classification: one class label string per line (task + social)
- Validated and zipped by `submission/zip_and_check.py`

## Project Layout

```
multimediate26/
├── configs/
│   ├── base.yaml                     # Model + train hyperparameters
│   ├── feature_specs.yaml            # 12 feature dims + presets
│   ├── domain_config.yaml            # 5 fine domains + 3 coarse groups
│   └── instruction_templates.yaml    # Qwen3-VL per-domain prompts
├── data/
│   ├── feature_extractor/
│   │   ├── ssi_reader.py             # SSI XML header + binary mmap loader
│   │   ├── align_features.py         # Resample all streams to 25 Hz
│   │   ├── build_session_npz.py      # Main pipeline: raw → per-session .npy
│   │   ├── compute_feature_stats.py  # NaN-safe per-channel mean/std
│   │   ├── extract_whisper.py        # Whisper-large-v3 encoder extraction
│   │   ├── extract_vlm_embedding.py  # Qwen3-VL-Embedding-8B extraction
│   │   └── probe_official.py         # Verify actual feature dimensions
│   ├── dataset.py                    # SessionDataset (z-score + clip + nan_to_num)
│   ├── sampler.py                    # DomainBalancedBatchSampler
│   └── label_loader.py              # Domain-dispatched label loading
├── models/
│   ├── md_dapa.py                    # Main model: MD-DAPA v2
│   ├── modality_proj.py             # ModalityProjector + ModalityGroupFusion
│   ├── domain_prompt.py             # HierarchicalDomainPrompt (coarse+fine)
│   ├── domain_prompt_flat.py        # Ablation: flat single-level prompt
│   ├── dapa_layer.py               # DAPA layer (reactive/anticipatory)
│   ├── partner_pool.py             # MultiPartnerPooling (attention-based)
│   ├── partner_pool_sum.py         # Ablation: parameter-free sum pooling
│   ├── heads.py                    # RegressionHead + ClassificationHeads + LearnableBridge
│   └── pinsoro_model.py            # PInSoRo specialist (BiLSTM + cross-attn)
├── losses/
│   ├── ccc_loss.py                 # CCC + MSE + Smoothness
│   └── ordinal_contrastive.py      # Margin-based pairwise contrastive
├── train/
│   ├── trainer.py                  # 3-phase curricular trainer
│   ├── pinsoro_trainer.py          # PInSoRo specialist trainer
│   ├── pinsoro_mlp.py             # PInSoRo MLP baseline trainer
│   ├── inference.py               # Standalone sliding-window inference
│   ├── tta.py                     # LayerNorm + prompt TTA
│   ├── tta_inference.py           # Combined TTA + inference per seed
│   └── ensemble.py                # Multi-seed/window ensemble + smoothing
├── inference/
│   ├── run_test.py                # Full inference pipeline (multi-seed + TTA)
│   └── tta.py                     # Prompt-only TTA
├── submission/
│   ├── writer.py                  # Official format CSV writer
│   └── zip_and_check.py           # Validate + create submission ZIP
├── scripts/
│   ├── stage2_phase1_pretrain.sh   # Phase 1 training
│   ├── stage2_phase1_v2arch.sh     # Phase 1 with v2arch
│   ├── stage2_phase1_whisper.sh    # Phase 1 with whisper features
│   ├── stage3_phase2_joint.sh     # Phase 2 joint training
│   ├── stage3_phase2_v2arch.sh    # Phase 2 v2arch
│   ├── stage4_phase3_v2arch.sh    # Phase 3 v2arch
│   ├── stage4_phase3_pinsoro.sh   # Phase 3 PInSoRo
│   ├── stage3a_vlm_experiment.sh  # VLM injection experiment
│   ├── final_retrain_pipeline.sh  # Final retrain pipeline
│   ├── final_retrain_fulldata.sh  # Final retrain with full data
│   ├── pipeline_11feat.sh         # Full 11-feature pipeline
│   ├── pinsoro_specialist.sh      # PInSoRo specialist training
│   ├── launch_v2arch_3seed.sh     # Multi-seed launcher
│   └── ...                        # Various utility scripts
├── manifests/                      # Per-(domain, split) JSONL manifests
│   ├── dataset_overview.md         # Dataset documentation
│   ├── notes_from_authors.md      # Notes from challenge authors
│   ├── noxi_train/val/test.jsonl  # Standard train/val/test splits
│   ├── full_*_train/val.jsonl     # Merged train+val for final retrain
│   └── ...
├── manifests_p8/                   # Language-split manifests (P8 experiment)
├── data_processed/                 # Aligned per-session .npy cache (~1.2 TB, NOT included)
├── output/                         # Training checkpoints (~13 GB, NOT included)
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# Optional for VLM extraction:
# pip install decord transformers flash-attn
```

### 2. Prepare Data

```bash
# Step 1: Probe official feature dimensions (verify before trust)
python -m data.feature_extractor.probe_official

# Step 2: Align all features to 25 Hz
python -m data.feature_extractor.align_features

# Step 3: Build per-session .npy cache
python -m data.feature_extractor.build_session_npz

# Step 4: Compute normalization statistics
python -m data.feature_extractor.compute_feature_stats

# Optional: Extract Whisper features
python -m data.feature_extractor.extract_whisper --gpus 0,1,2,3

# Optional: Extract VLM embeddings (Phase 3a)
python -m data.feature_extractor.extract_vlm_embedding --gpus 0,1
```

### 3. Train

```bash
# Phase 1: Regression pretrain on NoXi + NoXi-J
bash scripts/stage2_phase1_v2arch.sh

# Phase 2: Joint training with MPIIGI
bash scripts/stage3_phase2_v2arch.sh

# Phase 3: Full 5-domain with PInSoRo
bash scripts/stage4_phase3_v2arch.sh

# Optional: VLM experiment
bash scripts/stage3a_vlm_experiment.sh
```

### 4. Inference & Submit

```bash
# Multi-seed ensemble inference
python -m inference.run_test --checkpoint-dir output/ --seeds 0,1,2

# Or with TTA
python -m train.tta_inference --checkpoint output/phase3_v2arch_11feat_seed0/best.pt

# Validate and create submission ZIP
python -m submission.zip_and_check --pred-dir predictions/
```

## Known Data Issues & Workarounds

| Issue | Affected | Workaround |
|---|---|---|
| **w2vbert2 40 Hz header bug** | NoXi, NoXi-add | SSI headers declare sr=40 but actual data is 25 Hz. `build_session_npz.py` overrides `src_fps=25`. If trusted at face value, resampling would truncate 37.5% of every session. |
| **MPII DINO 2304-dim anomaly** | MPIIGI | DINOv2-Large ships at 2304 (3 ViT scales concatenated) vs 768 elsewhere. `align_features.py` slices first 768 columns. |
| **XLM-R 2 Hz segment-level** | All | XLM-R outputs at 2 Hz. `align_features.py` broadcasts segment-level features to 25 Hz. |
| **PInSoRo dual-microphone egemapsv2** | PInSoRo | Two microphones packed consecutively (2× dim). `ssi_reader.py` auto-detects and slices first microphone. |
| **PInSoRo NaN in egemapsv2** | PInSoRo | Consistent NaN channels in egemapsv2 features. `dataset.py` replaces NaN → 0 after normalization. |

## Results (Phase 2 Test Submissions)

| Submission | NoXi | NoXi-add | NoXi-J | MPIIGI | PInSoRo | Avg |
|---|---|---|---|---|---|---|
| v2arch-3seed | 0.800 | 0.660 | 0.730 | 0.520 | 0.857 | 0.7135 |

PInSoRo Kappa is the primary bottleneck for overall score improvement.

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.3 + CUDA
- NumPy, SciPy, pandas, PyYAML, tqdm, einops, scikit-learn
- ~170 GB disk for processed features (data_processed/)
- ~13 GB disk for checkpoints (output/)
- GPU: A100 80GB recommended (bf16 training)

## License

Research code for the MultiMediate'26 challenge. Please refer to the challenge's terms of use for data and submission guidelines.