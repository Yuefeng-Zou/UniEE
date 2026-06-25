"""Compose ``full_*`` manifests that merge each domain's existing train and
val into a single training set, leaving only K sessions per domain in a tiny
held-out val for ckpt selection.

Why this exists: the final sprint trains on all labelled data we have
(train + val) and picks the best epoch on a small representative val. The
previous Phase 1/2 manifests deliberately mirror the official train/val
split, which is what we use for fair comparison vs SOTA; this one is for
maximizing supervision.

Held-out selection policy (per domain):
  noxi      -> 2 sessions, one French + one Japanese-free Romance/German
  noxi_j    -> 2 sessions, one Japanese + one Chinese (different languages)
  mpiigi    -> 2 sessions, one 4-person + one 3-person
  pinsoro_cc-> 2 sessions, random
  pinsoro_cr-> 2 sessions, random

Outputs (one row per (session, role) just like build_session_npz manifests):
  manifests/full_<domain>_train.jsonl
  manifests/full_<domain>_val.jsonl
  manifests/full_combined_train.jsonl   (all 5 domains merged for trainer)
  manifests/full_combined_val.jsonl

Reproducible: held-out picks are deterministic (sorted session_id then take
first K, with a few domain-aware exceptions documented inline).
"""
from __future__ import annotations

import json
from pathlib import Path

import openpyxl


ROOT = Path("/mnt/pro-dtai/moe-lite/fenghui.zyf/mm_26/engagement_estimation")
MANIFEST_DIR = ROOT / "multimediate26" / "manifests"


# ── Per-domain holdout picks (manually chosen for language/role coverage) ─
HOLDOUT_SESSIONS: dict[str, list[str]] = {
    # NoXi has 4 European languages in train+val (French/English/German + a
    # few minority). Hold out one French and one German to span both Western
    # European clusters; keep all English in train since it's the largest.
    "noxi":       ["007", "075"],   # 007 = French, 075 = German
    # NoXi-J holdout one Japanese + one Chinese for cross-cultural coverage.
    "noxi_j":     ["121", "146"],   # 121 = Japanese, 146 = Chinese
    # MPIIGI: keep one 4-person (010) and one 3-person (027). Both were in
    # the existing held-out val so the comparison is consistent.
    "mpiigi":     ["010", "027"],
    # PInSoRo: smallest available pair from each subset.
    "pinsoro_cc": ["005", "010"],
    "pinsoro_cr": ["014", "019"],
}


def load_session_to_lang_noxi() -> dict[str, str]:
    """Parse NoXi_MetaData.xlsx → {session_id (zero-padded): language}."""
    wb = openpyxl.load_workbook(
        "/mnt/pro-dtai/moe-lite/fenghui.zyf/mm_26/data/NoXi/NoXi_MetaData.xlsx",
        data_only=False,
    )
    ws = wb["Tabelle1"]
    out: dict[str, str] = {}
    for r in list(ws.iter_rows(values_only=True))[1:]:
        sid, _, _, _, _, _, _, lang = r[:8]
        if sid and lang:
            out[f"{int(sid):03d}"] = lang
    return out


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))


def main() -> None:
    lang_noxi = load_session_to_lang_noxi()

    # Combined accumulators for the single global merged file.
    combined_train: list[dict] = []
    combined_val:   list[dict] = []

    print("Building full_* manifests:")
    print(f"  policy: per-domain holdout = {HOLDOUT_SESSIONS}\n")

    for domain in ("noxi", "noxi_j", "mpiigi", "pinsoro_cc", "pinsoro_cr"):
        # Pull every labelled row across the existing train + val manifests.
        rows: list[dict] = []
        for split in ("train", "val", "val_held"):
            src_name = f"{domain}_{split}.jsonl"
            rows.extend(read_jsonl(MANIFEST_DIR / src_name))
        # dedupe by (session_id, target_role); a row may appear in both
        # train and val_held when we already split mpiigi earlier.
        uniq: dict[tuple[str, str], dict] = {}
        for r in rows:
            key = (r["session_id"], r["target_role"])
            uniq.setdefault(key, r)
        rows = list(uniq.values())

        holdout = set(HOLDOUT_SESSIONS[domain])
        train_rows = [r for r in rows if r["session_id"] not in holdout]
        val_rows   = [r for r in rows if r["session_id"] in holdout]
        # Stamp split field for downstream clarity.
        for r in train_rows: r["split"] = "full_train"
        for r in val_rows:   r["split"] = "full_val"

        write_jsonl(MANIFEST_DIR / f"full_{domain}_train.jsonl", train_rows)
        write_jsonl(MANIFEST_DIR / f"full_{domain}_val.jsonl",   val_rows)

        combined_train.extend(train_rows)
        combined_val.extend(val_rows)

        train_sess = sorted({r["session_id"] for r in train_rows})
        val_sess   = sorted({r["session_id"] for r in val_rows})
        info = ""
        if domain == "noxi":
            info = f"  (val langs: {[lang_noxi.get(s, '?') for s in val_sess]})"
        if domain == "noxi_j":
            info = f"  (val langs: {[lang_noxi.get(s, '?') for s in val_sess]})"
        print(f"  {domain:11s} train: {len(train_rows):3d} rows from {len(train_sess):3d} sessions"
              f"   val: {len(val_rows):2d} rows from {len(val_sess)} sessions {val_sess}{info}")

    write_jsonl(MANIFEST_DIR / "full_combined_train.jsonl", combined_train)
    write_jsonl(MANIFEST_DIR / "full_combined_val.jsonl",   combined_val)
    print()
    print(f"  combined_train.jsonl : {len(combined_train)} rows")
    print(f"  combined_val.jsonl   : {len(combined_val)} rows")
    print()
    print(f"  → {MANIFEST_DIR}")


if __name__ == "__main__":
    main()
