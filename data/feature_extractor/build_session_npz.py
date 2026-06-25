"""Build per-session NPZ caches the training loop can mmap.

For each (domain, session, target_role) we:

  1. read every available official feature stream (XML + binary),
  2. resample each onto a 25 Hz grid,
  3. dim-align where needed (mpii dino 2304 → 768),
  4. load the engagement label (continuous or PInSoRo categorical),
  5. resample the label onto the same 25 Hz grid,
  6. truncate every array to the same shortest length T,
  7. write each array as its own ``.npy`` file under a per-session
     directory + a ``_DONE`` marker.

Per-session directory layout (matches the existing dataset.py expectations,
just with renamed feature keys for the v3 release):

    {npz_root}/{domain}/{session_id}/{target_role}/
        T.npy                    int scalar (number of 25 Hz frames)
        label.npy                (T,) float32 (zero for masked frames)
        label_mask.npy           (T,) bool
        label_type.npy           object ("continuous" | "pinsoro")
        label_task.npy           (T,) int64  (pinsoro only)
        label_social.npy         (T,) int64  (pinsoro only)
        partner_roles.npy        object ([str, ...])
        target_{feat}.npy        (T, dim)
        partner{i}_{feat}.npy    (T, dim)  for i in 0 .. n_partners-1
        _DONE                    marker file

The roles array tracks which partner is at each index so downstream we can
inspect / debug who's at slot 0.

The ``--manifest`` writer emits a JSONL of every successfully-built
(domain, session, target_role) with a few quick stats — used by trainer.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

# Adjust path so we can run as either ``python -m …`` or directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from multimediate26.data.feature_extractor.ssi_reader import (
    load_stream, parse_header,
)
from multimediate26.data.feature_extractor.align_features import (
    resample_to_25hz, broadcast_segments, align_feature_dim, TARGET_FPS,
)
from multimediate26.data.label_loader import (
    load_labels, LabelBundle,
    PINSORO_TASK_CONT_PRIOR, PINSORO_SOCIAL_CONT_PRIOR,
)


# ── Domain → role list + relative dir under data_root ─────────────────────
DOMAIN_LAYOUT: dict[str, dict] = {
    "noxi":       {"roles": ["expert", "novice"],            "splits": {
                       "train": "NoXi/train",
                       "val":   "NoXi/val",
                       "test":  "NoXi/test-base",
                   }},
    "noxi_add":   {"roles": ["expert", "novice"],            "splits": {
                       "test":  "NoXi/test-additional",
                   }},
    "noxi_j":     {"roles": ["expert", "novice"],            "splits": {
                       "train": "Noxi-J/train",
                       "val":   "Noxi-J/val",
                       "test":  "Noxi-J/test",
                   }},
    "pinsoro_cc": {"roles": ["purple", "yellow", "env"],            "splits": {
                       "train": "Pinsoro/train-cc",
                       "val":   "Pinsoro/val-cc",
                       "test":  "Pinsoro/test-cc",
                   }},
    "pinsoro_cr": {"roles": ["purple", "yellow", "env"],            "splits": {
                       "train": "Pinsoro/train-cr",
                       "val":   "Pinsoro/val-cr",
                       "test":  "Pinsoro/test-cr",
                   }},
    "mpiigi":     {"roles": ["subjectPos1", "subjectPos2",
                              "subjectPos3", "subjectPos4"], "splits": {
                       # mpii features sit a level deeper.
                       "val":  "mpii/MultiMediate25/val/precomputed-features-val",
                       "test": "mpii/MultiMediate25/test/precomputed-features-test",
                   }},
}


def _resolve_stream_path(session_dir: Path, role: str, feat: str,
                         template: str) -> Path | None:
    p = session_dir / template.format(role=role)
    return p if p.exists() else None


def _load_feature(session_dir: Path, role: str, feat: str,
                  raw_paths: dict, expected_dim: int) -> np.ndarray | None:
    """Read one feature's (T_src, D_src) array, resample to 25 Hz, dim-align."""
    template = raw_paths.get(feat)
    if template is None:
        return None
    p = _resolve_stream_path(session_dir, role, feat, template)
    if p is None:
        return None
    # Whisper features were extracted offline by extract_whisper.py into a
    # plain ``.npy`` already pooled to 25 Hz — no SSI header, no resample.
    if feat == "whisper":
        arr = np.load(p, mmap_mode=None).astype(np.float32, copy=False)
        return align_feature_dim(arr, expected_dim)
    arr, hdr = load_stream(p, mmap=False)  # materialize so resample can copy
    src_fps = hdr.sample_rate_hz
    # Known data defect: NoXi-family w2vbert2 streams declare sr=40 in the
    # SSI header, but the actual binary contains the model's 25 Hz pooled
    # output (one row per 40 ms ≈ 25 Hz). Trusting the 40 Hz claim makes
    # resample_to_25hz shorten T by 25/40 = 62.5%, throwing away the second
    # half of every NoXi session and corrupting label alignment.
    #
    # Detection: w2vbert2 with sr=40 on NoXi/NoXi-add (PInSoRo and NoXi-J
    # ship the same feature at sr=25 correctly; mpii also at sr=25).
    # We rewrite src_fps to 25 in those cases.
    if feat == "w2vbert2" and abs(src_fps - 40.0) < 0.1:
        src_fps = 25.0
    # NoXi XLM-R ships at 2 Hz segment-level; everything else is plain
    # per-frame resample. We treat the 2 Hz xlmr as still uniform-rate (no
    # explicit segment times shipped beside it).
    arr_25 = resample_to_25hz(arr, src_fps)
    arr_25 = align_feature_dim(arr_25, expected_dim)
    return arr_25.astype(np.float32, copy=False)


def _load_label_aligned(session_dir: Path, role: str, domain: str,
                        target_T: int) -> LabelBundle | None:
    """Load labels then resample / pad to ``target_T`` frames at 25 Hz."""
    bundle = load_labels(session_dir, role, domain)
    if bundle is None:
        return None
    if bundle.labels_cont is not None:
        cont = resample_to_25hz(
            bundle.labels_cont[:, None], bundle.src_fps, target_T=target_T
        ).squeeze(-1).astype(np.float32)
        mask_arr = bundle.label_mask.astype(np.float32)[:, None]
        mask_25 = resample_to_25hz(mask_arr, bundle.src_fps,
                                   target_T=target_T).squeeze(-1)
        # Resampled mask is fractional; treat anything >= 0.5 as valid.
        mask_bool = mask_25 >= 0.5
        cont = np.where(mask_bool, cont, 0.0).astype(np.float32)
        return LabelBundle(
            labels_cont=cont,
            labels_task=None, labels_social=None,
            label_mask=mask_bool, src_fps=TARGET_FPS,
        )
    # PInSoRo: int64 class labels → use nearest-neighbour (resample by
    # repeating). Mask is resampled the same way.
    src_fps = bundle.src_fps
    n_src = len(bundle.labels_task)
    src_times = np.arange(n_src) / src_fps
    tgt_times = np.arange(target_T) / TARGET_FPS
    # Nearest-neighbour index lookup
    idxs = np.clip(np.searchsorted(src_times, tgt_times), 0, n_src - 1)
    # searchsorted gives insertion index; correct to nearest neighbour
    left_idx = np.maximum(idxs - 1, 0)
    use_left = (np.abs(src_times[left_idx] - tgt_times) <
                np.abs(src_times[idxs] - tgt_times))
    nn_idx = np.where(use_left, left_idx, idxs)
    task_25 = bundle.labels_task[nn_idx].astype(np.int64)
    social_25 = bundle.labels_social[nn_idx].astype(np.int64)
    mask_25 = bundle.label_mask[nn_idx]
    return LabelBundle(
        labels_cont=None,
        labels_task=task_25, labels_social=social_25,
        label_mask=mask_25, src_fps=TARGET_FPS,
    )


def build_one(session_dir: Path, target_role: str, partner_roles: list[str],
              domain: str, feature_dims: dict[str, int], raw_paths: dict,
              out_session_dir: Path,
              features_to_build: list[str]) -> dict:
    """Build NPZ for one (session, target_role). Returns stats dict."""
    # 1. Read all target features and discover the common T.
    target_feats: dict[str, np.ndarray] = {}
    Ts = []
    for feat in features_to_build:
        if feat not in feature_dims:
            continue
        arr = _load_feature(session_dir, target_role, feat, raw_paths,
                            feature_dims[feat])
        if arr is None:
            continue
        target_feats[feat] = arr
        Ts.append(arr.shape[0])

    if not target_feats:
        raise RuntimeError(f"no target features for {session_dir}/{target_role}")

    T = min(Ts)

    # 2. Try labels at the 25 Hz resampled length.
    label_bundle = _load_label_aligned(session_dir, target_role, domain, T)

    # 3. Build partner features (best-effort; missing partners → zero-fill).
    partner_blob: dict[str, np.ndarray] = {}
    partner_roles_present: list[str] = []
    for i, prole in enumerate(partner_roles):
        any_present = False
        for feat in features_to_build:
            if feat not in feature_dims:
                continue
            arr = _load_feature(session_dir, prole, feat, raw_paths,
                                feature_dims[feat])
            if arr is None:
                continue
            # Pad/truncate partner streams to the same T as target.
            if arr.shape[0] >= T:
                arr = arr[:T]
            else:
                pad = np.zeros((T - arr.shape[0], arr.shape[1]),
                               dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
            partner_blob[f"partner{i}_{feat}"] = arr.astype(np.float32, copy=False)
            any_present = True
        partner_roles_present.append(prole if any_present else "")

    # 4. Write outputs (one .npy per array + DONE marker).
    out_session_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_session_dir / "T.npy", np.int64(T))
    np.save(out_session_dir / "partner_roles.npy",
            np.array(partner_roles_present, dtype=object), allow_pickle=True)

    # Labels (always write, with empty arrays + mask=False if missing).
    if label_bundle is None:
        # Test-set / no-label session. Write zeros + all-False mask so
        # downstream code reading `label.npy` doesn't have to special-case.
        np.save(out_session_dir / "label.npy",
                np.zeros(T, dtype=np.float32))
        np.save(out_session_dir / "label_mask.npy",
                np.zeros(T, dtype=bool))
        np.save(out_session_dir / "label_type.npy",
                np.array("none", dtype=object), allow_pickle=True)
    elif label_bundle.labels_cont is not None:
        np.save(out_session_dir / "label.npy",
                label_bundle.labels_cont.astype(np.float32))
        np.save(out_session_dir / "label_mask.npy",
                label_bundle.label_mask.astype(bool))
        np.save(out_session_dir / "label_type.npy",
                np.array("continuous", dtype=object), allow_pickle=True)
    else:
        # PInSoRo categorical. Also write a "label" alias as 0-1 derived
        # from task class (used by ordinal_contrastive). Trainer can ignore.
        np.save(out_session_dir / "label.npy",
                np.zeros(T, dtype=np.float32))   # filled by bridge at train time
        np.save(out_session_dir / "label_mask.npy",
                label_bundle.label_mask.astype(bool))
        np.save(out_session_dir / "label_type.npy",
                np.array("pinsoro", dtype=object), allow_pickle=True)
        np.save(out_session_dir / "label_task.npy",
                label_bundle.labels_task.astype(np.int64))
        np.save(out_session_dir / "label_social.npy",
                label_bundle.labels_social.astype(np.int64))
        # Pseudo-continuous target for the Phase 3 bridge loss. Per-frame
        # mean of the two hand-tuned priors; masked frames stay at 0 so the
        # loss masking still drops them.
        task_prior   = np.asarray(PINSORO_TASK_CONT_PRIOR,   dtype=np.float32)
        social_prior = np.asarray(PINSORO_SOCIAL_CONT_PRIOR, dtype=np.float32)
        pseudo = 0.5 * task_prior[label_bundle.labels_task] \
               + 0.5 * social_prior[label_bundle.labels_social]
        pseudo = np.where(label_bundle.label_mask, pseudo, 0.0).astype(np.float32)
        np.save(out_session_dir / "label_pseudo_cont.npy", pseudo)

    for feat, arr in target_feats.items():
        np.save(out_session_dir / f"target_{feat}.npy", arr[:T])
    for key, arr in partner_blob.items():
        np.save(out_session_dir / f"{key}.npy", arr)

    (out_session_dir / "_DONE").write_text("ok\n")

    return {
        "T": T,
        "n_features": len(target_feats),
        "features": sorted(target_feats),
        "n_partners": len([r for r in partner_roles_present if r]),
        "has_label": label_bundle is not None,
        "label_type":
            "continuous" if (label_bundle and label_bundle.labels_cont is not None)
            else "pinsoro" if (label_bundle and label_bundle.labels_task is not None)
            else "none",
    }


# ── Iteration helpers ─────────────────────────────────────────────────────

def iter_sessions(data_root: Path,
                  selected_domains: list[str] | None,
                  selected_splits: list[str] | None) -> Iterable[tuple[str, str, Path]]:
    """Yield (domain, split, session_dir) for every (domain, split, session)."""
    for domain, layout in DOMAIN_LAYOUT.items():
        if selected_domains and domain not in selected_domains:
            continue
        for split, rel in layout["splits"].items():
            if selected_splits and split not in selected_splits:
                continue
            root = data_root / rel
            if not root.exists():
                continue
            for sess in sorted(root.iterdir()):
                if not sess.is_dir():
                    continue
                # mpii has a few non-session directories named
                # ``engagement-annotations-*`` / ``originalAudioVideo-*`` —
                # the per-session probe filters them by checking that at
                # least one .stream file lives there.
                if not any(sess.glob("*.stream")):
                    continue
                yield domain, split, sess


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True,
                    help="…/multimediate26/data_processed/npz_v3")
    ap.add_argument("--feature-specs", type=Path,
                    default=Path("multimediate26/configs/feature_specs.yaml"))
    ap.add_argument("--preset", type=str, default="official_full",
                    help="Which preset under feature_specs.presets to build")
    ap.add_argument("--domains", type=str, default="",
                    help="comma-separated subset; empty = all")
    ap.add_argument("--splits", type=str, default="",
                    help="comma-separated subset; empty = all")
    ap.add_argument("--manifest-dir", type=Path,
                    default=Path("multimediate26/manifests"))
    ap.add_argument("--skip-roles", type=str,
                    default="env,s",
                    help="comma-separated role names to skip as target_role "
                         "(but still allowed as partner). PInSoRo's 'env' is "
                         "an environment camera with no label; mpii 's.' is "
                         "a scene-shared stream. Default skips both.")
    ap.add_argument("--labeled-only", action="store_true",
                    help="only build (session, role) pairs with engagement "
                         "labels. Skips test splits + PInSoRo 'env'. Use this "
                         "for Phase 1/2 training; rebuild without it for test "
                         "inference.")
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, build at most this many (session, role) pairs "
                         "across all domains — debug")
    ap.add_argument("--skip-done", action="store_true", default=True,
                    help="skip session dirs that already have _DONE")
    ap.add_argument("--no-skip-done", dest="skip_done", action="store_false")
    args = ap.parse_args()

    specs = yaml.safe_load(args.feature_specs.read_text())
    feature_dims = specs["feature_dims"]
    raw_paths = specs["raw_paths"]
    features_to_build = specs["presets"][args.preset]

    selected_domains = [d.strip() for d in args.domains.split(",") if d.strip()] or None
    selected_splits  = [s.strip() for s in args.splits.split(",")  if s.strip()] or None
    skip_roles = {s.strip() for s in args.skip_roles.split(",") if s.strip()}

    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    # Per-domain JSONL manifests, written incrementally.
    manifest_fps = {}

    n_built = n_skipped = n_failed = n_no_label = 0
    for domain, split, sess_dir in iter_sessions(args.data_root,
                                                 selected_domains,
                                                 selected_splits):
        roles = DOMAIN_LAYOUT[domain]["roles"]
        for target_role in roles:
            if target_role in skip_roles:
                continue
            # Some mpii sessions have empty seats — if this role has no
            # *.stream files for this session, skip silently (3-person case).
            if not any(sess_dir.glob(f"{target_role}.*.stream")):
                continue
            partner_roles = [r for r in roles if r != target_role]
            sess_id = sess_dir.name
            out_dir = args.out_root / domain / sess_id / target_role
            done_marker = out_dir / "_DONE"
            if args.skip_done and done_marker.exists():
                n_skipped += 1
                continue

            # Cheap label-presence pre-check before building features —
            # honored only with --labeled-only. We open & resample
            # everything anyway inside build_one, but for unlabelled
            # sessions we can short-circuit.
            if args.labeled_only:
                from multimediate26.data.label_loader import load_labels
                pre = load_labels(sess_dir, target_role, domain)
                if pre is None:
                    n_no_label += 1
                    continue

            try:
                stats = build_one(
                    sess_dir, target_role, partner_roles, domain,
                    feature_dims, raw_paths, out_dir, features_to_build,
                )
            except Exception as e:
                n_failed += 1
                print(f"  FAIL {domain}/{sess_id}/{target_role}: {e}",
                      file=sys.stderr)
                continue
            n_built += 1
            # Append to manifest.
            row = {
                "domain": domain, "split": split, "session_id": sess_id,
                "target_role": target_role,
                "partner_roles": partner_roles,
                "T": stats["T"], "n_features": stats["n_features"],
                "label_type": stats["label_type"],
                "has_label": stats["has_label"],
                "out_dir": str(out_dir),
            }
            mf_key = f"{domain}_{split}"
            if mf_key not in manifest_fps:
                manifest_fps[mf_key] = (
                    args.manifest_dir / f"{mf_key}.jsonl"
                ).open("w")
            manifest_fps[mf_key].write(json.dumps(row) + "\n")
            manifest_fps[mf_key].flush()
            if n_built % 10 == 0:
                print(f"  built={n_built} skipped={n_skipped} failed={n_failed} "
                      f"(latest: {domain}/{sess_id}/{target_role} T={stats['T']})",
                      file=sys.stderr)
            if args.limit and n_built >= args.limit:
                break
        if args.limit and n_built >= args.limit:
            break

    for fp in manifest_fps.values():
        fp.close()

    print()
    print(f"=== DONE ===")
    print(f"  built  = {n_built}")
    print(f"  skipped (already done) = {n_skipped}")
    print(f"  skipped (no label, with --labeled-only) = {n_no_label}")
    print(f"  failed = {n_failed}")
    print(f"  manifests in {args.manifest_dir}/")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
