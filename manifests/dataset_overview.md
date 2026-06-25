# MultiMediate'26 Dataset Overview

**Source**: `/mnt/pro-dtai/moe-lite/fenghui.zyf/mm_26/data`
**Last inspected**: 2026-05-30 (re-run `multimediate26/scripts/inspect_datasets.py`)
**Machine-readable mirror**: `dataset_schema.yaml` (per-session, per-role detail)

## Session × role coverage

Each `(session, role)` pair becomes one NPZ in `data_processed/npz_v3/`.
That is the unit the training dataset iterates over.

| domain        | split | sessions | (sess,role) | labeled | notes                                |
|---------------|-------|---------:|------------:|--------:|--------------------------------------|
| noxi          | train |    38   |    76       |   76    | expert + novice                      |
| noxi          | val   |    10   |    20       |   20    |                                      |
| noxi          | test  |    16   |    32       |    0    | challenge eval, no labels            |
| noxi_add      | test  |    12   |    24       |    0    | zero-shot (4 new languages)          |
| noxi_j        | train |    31   |    62       |   62    |                                      |
| noxi_j        | val   |    10   |    20       |   20    |                                      |
| noxi_j        | test  |    10   |    20       |    0    |                                      |
| pinsoro_cc    | train |    20   |    60       |   40    | purple+yellow+env; env has no label  |
| pinsoro_cc    | val   |     7   |    21       |   14    |                                      |
| pinsoro_cc    | test  |     6   |    18       |    0    |                                      |
| pinsoro_cr    | train |    12   |    36       |   24    |                                      |
| pinsoro_cr    | val   |     5   |    15       |   10    |                                      |
| pinsoro_cr    | test  |     6   |    18       |    0    |                                      |
| mpiigi        | val   |     6   |    24       |   21    | 2 of 6 sessions are 3-person         |
| mpiigi        | test  |     6   |    24       |    0    | session 001 is 3-person              |
| **TOTAL**     |       | **195** |   **470**   | **287** | 287 trainable + 183 unlabelled test  |

## Roles per domain

| domain        | roles                                                                            |
|---------------|----------------------------------------------------------------------------------|
| noxi family   | expert, novice                                                                   |
| noxi_j        | expert, novice                                                                   |
| pinsoro_cc/cr | purple, yellow, env (environment camera, shared view; no engagement label)       |
| mpiigi        | subjectPos1, subjectPos2, subjectPos3, subjectPos4 (3 of 4 in 3-person sessions) |

## mpiigi 3-person session handling (confirmed by author)

From Huajian Qiu (VIS Stuttgart, 2026 email):
> "the group 001 has only 3 participants so there is no person at the
> position of SubjectPos04. In the dataset, there are some groups with 3
> persons, while others have 4 persons. The reason why there are
> precomputed visual features for SubjectPos04 of test group 001 while no
> audio features is unknown for me."

Per-session audit:

| split | session | n participants | empty position | audio missing for | label missing for |
|-------|---------|---------------:|---------------|-------------------|-------------------|
| val   | 008, 009, 010, 028 | 4 | —              | —                 | —                 |
| val   | 026, 027  | 3              | Pos1          | Pos1              | Pos1              |
| test  | 001       | 3              | Pos4          | Pos4              | (no labels)       |
| test  | 002,003,004| 4             | —              | —                 | (no labels)       |
| test  | 005, 006  | 4              | —              | —                 | (no labels)       |

**Implication**: at build time, skip role slots whose source dir has
zero `{role}.*.stream` files. The dataset.py reads `partner_roles.npy`
to know which slots are real.

**openface3 is universally absent from mpiigi** (not session-dependent).
The feature was added in a later OpenFace release than the mpii preprocessing
batch. Treat as a NaN-everywhere modality for mpii — projector forward
should mask it out.

`mpii test/002-004 also contain a "s." prefix` — likely a "scene" shared
view (analogous to PInSoRo `env`). We currently ignore it.

## Feature dim & sample-rate matrix

Per `dataset_schema.yaml` per-session reading.

| feature   | noxi  | noxi_j | noxi_add | pinsoro (cc/cr) | mpii val (per-role) | mpii test (per-role) |
|-----------|------:|-------:|---------:|----------------:|--------------------:|---------------------:|
| w2vbert2  | 1024 (40Hz) | 1024 (25Hz) | 1024 (40Hz) | 1024 (25Hz)   | 1024 (40Hz)         | 1024 (25Hz)          |
| egemapsv2 | 88   (25Hz) | 88   (25Hz) | 88   (25Hz) | 88   (25Hz)   | 88   (25Hz)         | 88   (25Hz)          |
| xlmr      | 768  (25Hz, occasionally 2Hz)| 768  (25Hz) | 768  (25Hz) | 768  (25Hz)   | 768  (25Hz)         | 768  (25Hz)          |
| openface2 | 714  (25Hz) | 714  (25Hz) | 714  (25Hz) | 714 (**30Hz**) | 714 (25Hz)          | 714 (25Hz)           |
| openface3 | 21   (25Hz) | 21   (25Hz) | 21   (25Hz) | 21  (**30Hz**) | **absent**          | **absent**           |
| openpose  | 139  (25Hz) | 139  (25Hz) | 139  (25Hz) | 139 (**30Hz**) | 139 (25Hz)          | 139 (25Hz)           |
| videomae  | 1408 (25Hz) | 1408 (25Hz) | 1408 (25Hz) | 1408 (**30Hz**)| 1408 (25Hz)         | 1408 (25Hz)          |
| dino      | 768  (25Hz) | 768  (25Hz) | 768  (25Hz) | 768 (**30Hz**) | **2304** (25Hz)     | **2304** (25Hz)      |
| swin      | 768  (25Hz) | 768  (25Hz) | 768  (25Hz) | 768 (**30Hz**) | 768 (25Hz)          | 768 (25Hz)           |
| clip      | 512  (25Hz) | 512  (25Hz) | 512  (25Hz) | 512 (**30Hz**) | 512 (25Hz)          | 512 (25Hz)           |

### Resampling required at build time

1. **w2vbert2 40 Hz → 25 Hz** (NoXi en/fr/de, NoXi-add, mpii val) — linear interp
2. **xlmr 2 Hz → 25 Hz** for the 1-2 NoXi sessions where it shipped as
   segment-broadcast — header-driven, not a per-domain constant
3. **PInSoRo all visual 30 Hz → 25 Hz** — linear interp
4. **mpii dino 2304 → 768 columns** (DINOv2-L → take first 768) — channel slice

### Label format

| domain        | format                            | location relative to features                    |
|---------------|-----------------------------------|--------------------------------------------------|
| noxi family   | continuous (0-1), one float/line  | `{role}.engagement.annotation.csv` in feature dir|
| noxi_j        | same                              | same                                             |
| pinsoro_cc/cr | categorical task + social         | `{role}.task_engagement.annotation.csv` + `social_…` |
| mpiigi        | continuous (0-1)                  | `engagement-annotations-{split}/{session}/{role}.engagement.annotation.csv` (parallel tree) |
| any test split| —                                 | absent                                            |

PInSoRo task classes (4): `goaloriented, aimless, noplay, adultseeking`
PInSoRo social classes (5): `solitary, onlooker, parallel, associative, cooperative`

There are duplicate `*.1.annotation.csv` files in PInSoRo (a second
rater); we currently use the primary rater only.

## File naming inside a session dir

```
{session_id}/
  {role}.engagement.annotation.csv            ← continuous label (NoXi family)
  {role}.task_engagement.annotation.csv       ← PInSoRo task class
  {role}.social_engagement.annotation.csv     ← PInSoRo social class
  {role}.task_engagement.1.annotation.csv     ← duplicate rater 2 (ignored)
  {role}.audio.transcript.annotation.csv      ← raw transcript text
  {role}.audio.wav                            ← raw audio
  {role}.video.mp4                            ← raw video
  {role}.{feature}.stream                     ← SSI XML header (~200 B)
  {role}.{feature}.stream~                    ← float32 binary, shape (num, dim)
  {role}.audio.{audio_feature}.stream(~)      ← w2vbert2 / egemapsv2 / xlmr
  language.annotation.csv                     ← NoXi-add: declared language per session
```

**`.stream~` is the data file, NOT a backup.** Always read the XML header
first to learn `(dim, sample_rate_hz, num_frames)`, then mmap the binary
as `np.memmap(stream~, dtype=float32, shape=(num, dim))`.

## mpii sub-layout (different from rest)

```
mpii/MultiMediate25/{val,test}/
  engagement-annotations-{val,test}/{session}/{role}.engagement.annotation.csv
  precomputed-features-{val,test}/{session}/{role}.{feature}.stream(~)
  originalAudioVideo-{val,test}/{session}/{role}.{wav,mp4}
```

Labels live in a parallel directory tree, not next to features. The build
script looks for them via the sibling dir starting with `engagement-annotations-`.
