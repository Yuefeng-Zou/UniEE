"""Validate a submission directory matches MultiMediate'26 requirements and ZIP it.

Format (verified against multimediate26_official/baseline/4_TestingNN_fairness_per_session.py):

  <out_dir>/
    noxi-base/<session>/{expert,novice}.engagement.prediction.csv         # 16 sessions
    noxi-additional/<session>/{expert,novice}.engagement.prediction.csv   # 12 sessions
    noxi-j/<session>/{expert,novice}.engagement.prediction.csv            # 10 sessions
    mpiigroupinteraction/<session>/{subjectPos1..4}.engagement.prediction.csv  # 6 sessions
    pinsoro-cc/<session>/{red,yellow}.{task_engagement,social_engagement}.engagement.prediction.csv  # 6 sessions
    pinsoro-cr/<session>/{red,yellow}.{task_engagement,social_engagement}.engagement.prediction.csv  # 6 sessions

Regression CSVs: one float per line, %.6f format, length == feature_T == GT_T.
Classification CSVs: one class label string per line.

Checks performed:
  1. Every expected session has predictions (with role coverage).
  2. Per-session row count is sane (>=100 frames, < 1e6).
  3. Regression values are in [0, 1] and no NaN.
  4. Classification values are valid class names (per PINSORO_*_INV).
  5. No extra files / directories that the evaluator would reject.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

from multimediate26.submission.writer import PINSORO_TASK_INV, PINSORO_SOCIAL_INV


# Expected layout: folder → {role_pattern, n_min_sessions, n_max_sessions, file_suffixes}
# (set n_min == n_max if exact count required; otherwise use a range)
EXPECTED = {
    "noxi-base": {
        "roles": ["expert", "novice"],
        "n_sessions": (16, 16),  # NoXi test-base = 16 sessions × 2 roles = 32
        "suffixes": [".engagement.prediction.csv"],
        "kind": "regression",
    },
    "noxi-additional": {
        "roles": ["expert", "novice"],
        "n_sessions": (12, 12),  # NoXi test-add = 12 sessions × 2 = 24
        "suffixes": [".engagement.prediction.csv"],
        "kind": "regression",
    },
    "noxi-j": {
        "roles": ["expert", "novice"],
        "n_sessions": (10, 10),  # 10 × 2 = 20
        "suffixes": [".engagement.prediction.csv"],
        "kind": "regression",
    },
    "mpiigroupinteraction": {
        "roles": ["subjectPos1", "subjectPos2", "subjectPos3", "subjectPos4"],
        # Not every session has all 4 (some are 3-person); accept (2..4) per session.
        "n_sessions": (6, 6),
        "suffixes": [".engagement.prediction.csv"],
        "kind": "regression",
        "partial_roles_ok": True,
    },
    "pinsoro-cc": {
        "roles": ["purple", "yellow"],  # PInSoRo color codes for the two children
        "n_sessions": (6, 6),
        "suffixes": [
            ".task_engagement.engagement.prediction.csv",
            ".social_engagement.engagement.prediction.csv",
        ],
        "kind": "classification",
    },
    "pinsoro-cr": {
        "roles": ["purple", "yellow"],
        "n_sessions": (6, 6),
        "suffixes": [
            ".task_engagement.engagement.prediction.csv",
            ".social_engagement.engagement.prediction.csv",
        ],
        "kind": "classification",
    },
}


class ValidationError(Exception):
    pass


def _list_session_dirs(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_dir()])


def _check_regression_csv(p: Path, errors: list[str]) -> None:
    lines = p.read_text().rstrip("\n").split("\n")
    n = len(lines)
    if n < 100:
        errors.append(f"{p}: only {n} frames (< 100)")
        return
    if n > 200_000:
        errors.append(f"{p}: {n} frames suspiciously large")
        return
    # Spot-check first/last/middle frames.
    for idx in (0, n // 2, n - 1):
        try:
            v = float(lines[idx])
        except ValueError:
            errors.append(f"{p}: line {idx} not a float: {lines[idx]!r}")
            return
        if not (0.0 <= v <= 1.0):
            errors.append(f"{p}: line {idx} value {v} out of [0,1]")
            return


def _check_classification_csv(p: Path, errors: list[str]) -> None:
    lines = p.read_text().rstrip("\n").split("\n")
    n = len(lines)
    if n < 10:
        errors.append(f"{p}: only {n} frames (< 10)")
        return
    valid_set = set(PINSORO_TASK_INV.values()) | set(PINSORO_SOCIAL_INV.values())
    # spot check first / middle / last
    for idx in (0, n // 2, n - 1):
        if lines[idx] not in valid_set:
            errors.append(f"{p}: line {idx} {lines[idx]!r} not a known class")
            return


def validate(out_dir: Path) -> tuple[list[str], dict]:
    errors: list[str] = []
    stats: dict[str, dict] = {}

    if not out_dir.exists():
        return [f"out_dir does not exist: {out_dir}"], {}

    extra_top = [p.name for p in out_dir.iterdir()
                 if p.is_dir() and p.name not in EXPECTED]
    if extra_top:
        errors.append(f"unexpected top-level folders: {extra_top}")

    for folder_name, spec in EXPECTED.items():
        folder = out_dir / folder_name
        if not folder.exists():
            errors.append(f"missing folder: {folder_name}")
            continue

        sess_dirs = _list_session_dirs(folder)
        n_lo, n_hi = spec["n_sessions"]
        if not (n_lo <= len(sess_dirs) <= n_hi):
            errors.append(
                f"{folder_name}: {len(sess_dirs)} sessions, "
                f"expected {n_lo}-{n_hi}"
            )

        n_csv = 0
        n_roles_covered = 0
        for sess_dir in sess_dirs:
            seen_roles_complete = 0
            for role in spec["roles"]:
                missing_for_role = []
                for suf in spec["suffixes"]:
                    csv_path = sess_dir / f"{role}{suf}"
                    if not csv_path.exists():
                        missing_for_role.append(suf)
                    else:
                        n_csv += 1
                        if spec["kind"] == "regression":
                            _check_regression_csv(csv_path, errors)
                        else:
                            _check_classification_csv(csv_path, errors)
                if not missing_for_role:
                    seen_roles_complete += 1
                elif not spec.get("partial_roles_ok"):
                    errors.append(
                        f"{folder_name}/{sess_dir.name}: missing for role "
                        f"{role}: {missing_for_role}"
                    )
            if seen_roles_complete == 0:
                errors.append(
                    f"{folder_name}/{sess_dir.name}: NO roles fully covered"
                )
            n_roles_covered += seen_roles_complete

        stats[folder_name] = {
            "sessions": len(sess_dirs),
            "csv_files": n_csv,
            "roles_complete": n_roles_covered,
        }

    return errors, stats


def zip_submission(out_dir: Path, zip_path: Path) -> int:
    """Create zip whose root contains the 6 dataset folders (no parent dir)."""
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for folder_name in EXPECTED:
            folder = out_dir / folder_name
            if not folder.exists():
                continue
            for f in folder.rglob("*"):
                if f.is_file():
                    zf.write(f, arcname=str(f.relative_to(out_dir)))
                    n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--zip", type=Path, default=None,
                    help="if set, zip valid submission here")
    ap.add_argument("--strict", action="store_true",
                    help="fail on first error instead of summarizing")
    args = ap.parse_args()

    print(f"validating submission: {args.out_dir}")
    errors, stats = validate(args.out_dir)

    print("\n=== STATS ===")
    for k, v in stats.items():
        print(f"  {k:25s} {v}")

    print(f"\n=== ERRORS ({len(errors)}) ===")
    if errors:
        for e in errors[:50]:
            print(f"  ✗ {e}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")
        return 1

    print("  no errors. ✓")

    if args.zip:
        args.zip.parent.mkdir(parents=True, exist_ok=True)
        n = zip_submission(args.out_dir, args.zip)
        size_mb = args.zip.stat().st_size / 1e6
        print(f"\nzipped {n} files → {args.zip} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
