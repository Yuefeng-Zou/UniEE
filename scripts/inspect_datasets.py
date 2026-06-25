"""Inventory the raw MultiMediate'26 release into a machine-readable schema.

Writes:
    multimediate26/manifests/dataset_schema.yaml   (machine-readable)

Re-run whenever the raw data layout changes. The companion document
``manifests/dataset_overview.md`` is hand-maintained alongside it.

Usage:
    python -m multimediate26.scripts.inspect_datasets \\
        --data-root /mnt/pro-dtai/moe-lite/fenghui.zyf/mm_26/data \\
        --out multimediate26/manifests/dataset_schema.yaml
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

import yaml


# Make safe_dump emit OrderedDict in insertion order (instead of refusing
# with RepresenterError). The output yaml is human-edited rarely so block
# style is fine.
yaml.SafeDumper.add_representer(
    OrderedDict,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map", data.items()
    ),
)


# Domain layout: which roles, which split-relative paths under data_root.
DOMAINS = {
    "noxi": {
        "roles":  ["expert", "novice"],
        "splits": {"train": "NoXi/train", "val": "NoXi/val", "test": "NoXi/test-base"},
    },
    "noxi_add": {
        "roles":  ["expert", "novice"],
        "splits": {"test": "NoXi/test-additional"},
    },
    "noxi_j": {
        "roles":  ["expert", "novice"],
        "splits": {"train": "Noxi-J/train", "val": "Noxi-J/val", "test": "Noxi-J/test"},
    },
    "pinsoro_cc": {
        "roles":  ["purple", "yellow", "env"],
        "splits": {"train": "Pinsoro/train-cc",
                   "val":   "Pinsoro/val-cc",
                   "test":  "Pinsoro/test-cc"},
    },
    "pinsoro_cr": {
        "roles":  ["purple", "yellow", "env"],
        "splits": {"train": "Pinsoro/train-cr",
                   "val":   "Pinsoro/val-cr",
                   "test":  "Pinsoro/test-cr"},
    },
    "mpiigi": {
        "roles":  ["subjectPos1", "subjectPos2", "subjectPos3", "subjectPos4"],
        "splits": {
            "val":  "mpii/MultiMediate25/val/precomputed-features-val",
            "test": "mpii/MultiMediate25/test/precomputed-features-test",
        },
    },
}

FEATURE_FILES = {
    "w2vbert2":  "{role}.audio.w2vbert2_embeddings.stream",
    "egemapsv2": "{role}.audio.egemapsv2.stream",
    "xlmr":      "{role}.audio.xlm_roberta_embeddings.stream",
    "openface2": "{role}.openface2.stream",
    "openface3": "{role}.openface3.stream",
    "openpose":  "{role}.openpose.stream",
    "videomae":  "{role}.videomae.stream",
    "dino":      "{role}.dino.stream",
    "swin":      "{role}.swin.stream",
    "clip":      "{role}.clip.stream",
}

_DIM_RE = re.compile(r'\bdim="(\d+)"')
_SR_RE  = re.compile(r'\bsr="([\d.]+)"')
_NUM_RE = re.compile(r'\bnum="(\d+)"')


def parse_header(p: Path) -> tuple[int, float, int]:
    t = p.read_text(errors="ignore")
    return (int(_DIM_RE.search(t).group(1)),
            float(_SR_RE.search(t).group(1)),
            int(_NUM_RE.search(t).group(1)))


def has_label(session_dir: Path, role: str, domain: str) -> bool:
    if domain == "mpiigi":
        # Look for sibling engagement-annotations-{val,test}/<session>/<role>.engagement.annotation.csv
        feat_parent = session_dir.parent           # precomputed-features-{val,test}
        split_root = feat_parent.parent            # …/val or …/test
        for cand in split_root.iterdir():
            if cand.is_dir() and cand.name.startswith("engagement-annotations"):
                if (cand / session_dir.name /
                        f"{role}.engagement.annotation.csv").exists():
                    return True
        return False
    if domain.startswith("pinsoro"):
        return (session_dir / f"{role}.task_engagement.annotation.csv").exists()
    return (session_dir / f"{role}.engagement.annotation.csv").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("multimediate26/manifests/dataset_schema.yaml"))
    args = ap.parse_args()

    out = OrderedDict()
    for domain, info in DOMAINS.items():
        out[domain] = OrderedDict()
        for split, rel in info["splits"].items():
            sroot = args.data_root / rel
            if not sroot.exists():
                continue
            sessions = sorted(
                d for d in sroot.iterdir()
                if d.is_dir() and any(d.glob("*.stream"))
            )
            if not sessions:
                continue

            # Per-session inventory — needed because mpii has 3-person
            # sessions where one seat is empty, so the feature/audio/label
            # coverage genuinely varies session-to-session.
            per_session = OrderedDict()
            for sess in sessions:
                entry = OrderedDict()
                # Build per-role coverage map
                roles_info = OrderedDict()
                for role in info["roles"]:
                    # Skip silently when this role has zero files in this
                    # session (e.g. mpii 3-person seat 4 has no streams).
                    role_files = list(sess.glob(f"{role}.*"))
                    if not role_files:
                        continue
                    feats = OrderedDict()
                    for fname, tmpl in FEATURE_FILES.items():
                        p = sess / tmpl.format(role=role)
                        if not p.exists():
                            continue
                        try:
                            d, sr, n = parse_header(p)
                            feats[fname] = {"dim": d, "sr_hz": sr, "num": n}
                        except Exception as e:
                            feats[fname] = {"error": str(e)}
                    roles_info[role] = OrderedDict([
                        ("features", feats),
                        ("has_label", has_label(sess, role, domain)),
                    ])
                entry["roles"] = roles_info
                per_session[sess.name] = entry

            # Domain/split summary stats
            all_role_count = sum(len(s["roles"]) for s in per_session.values())
            feat_union = set()
            for s in per_session.values():
                for ri in s["roles"].values():
                    feat_union.update(ri["features"].keys())
            n_labeled_role = sum(
                1 for s in per_session.values()
                for ri in s["roles"].values()
                if ri["has_label"]
            )

            out[domain][split] = OrderedDict([
                ("n_sessions",     len(sessions)),
                ("n_role_present", all_role_count),
                ("n_labeled_role", n_labeled_role),
                ("roles_declared", info["roles"]),
                ("features_seen_anywhere", sorted(feat_union)),
                ("sessions",       per_session),
            ])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "# Auto-generated by scripts/inspect_datasets.py — DO NOT edit by hand.\n"
        "# Re-run after every data refresh. Companion doc: dataset_overview.md\n"
        + yaml.safe_dump(out, sort_keys=False, default_flow_style=False)
    )
    print(f"wrote {args.out}")
    print(f"  {sum(len(v) for v in out.values())} (domain,split) entries")
    total_role = sum(s["n_role_present"]
                     for d in out.values() for s in d.values())
    total_labeled = sum(s["n_labeled_role"]
                        for d in out.values() for s in d.values())
    print(f"  {total_role} (session,role) pairs total; "
          f"{total_labeled} have engagement labels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
