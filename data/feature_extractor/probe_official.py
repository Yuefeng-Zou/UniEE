"""Probe official feature shapes across a sample of sessions.

The TechPlan reserves ``FEATURE_DIMS`` as configurable so the model never
hardcodes a width that might differ between feature releases. This script
walks a few sessions per domain, parses each SSI header, and reports per-
feature ``(dim, sample_rate_hz)`` along with whether the dim is consistent
across sessions.

Output: a YAML fragment that can be pasted into ``configs/feature_specs.yaml``
under ``feature_dims`` and ``native_rate_hz``.

Run example:
    python -m multimediate26.data.feature_extractor.probe_official \\
        --data-root /mnt/pro-dtai/moe-lite/fenghui.zyf/mm_26/data \\
        --feature-specs multimediate26/configs/feature_specs.yaml \\
        --max-sessions-per-domain 3
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# Allow running both as `python -m multimediate26.data...` and as a script
# from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from multimediate26.data.feature_extractor.ssi_reader import parse_header


# Where to look per domain. The role names per domain are tracked so we can
# correctly find one valid stream file (we don't need every role for shape
# probing — one is enough).
DOMAIN_ROLES = {
    "noxi":       (["expert", "novice"],            "NoXi/train"),
    "noxi_j":     (["expert", "novice"],            "Noxi-J/train"),
    "noxi_add":   (["expert", "novice"],            "NoXi/test-additional"),
    "pinsoro_cc": (["purple", "yellow"],            "Pinsoro/train-cc"),
    "pinsoro_cr": (["purple", "yellow"],            "Pinsoro/train-cr"),
    # mpii val is the only labelled mpii split; features sit one extra level
    # deep under precomputed-features-val/.
    "mpiigi":     (["subjectPos1", "subjectPos2",
                    "subjectPos3", "subjectPos4"],
                   "mpii/MultiMediate25/val/precomputed-features-val"),
}


def find_session_dirs(data_root: Path, domain: str, max_n: int) -> list[Path]:
    rel = DOMAIN_ROLES[domain][1]
    root = data_root / rel
    if not root.exists():
        print(f"  [{domain}] root missing: {root}", file=sys.stderr)
        return []
    # Each session is itself a directory of {role}.{feature}.stream files.
    sessions = sorted([p for p in root.iterdir() if p.is_dir()])
    return sessions[:max_n]


def probe_one(session_dir: Path, role: str, raw_paths: dict) -> dict:
    """Parse every available feature header for one (session, role).

    Returns a dict mapping feature_name → (dim, sample_rate_hz, num_frames)
    for features that exist; missing ones are simply omitted.
    """
    out = {}
    for feat, tmpl in raw_paths.items():
        if feat == "qwen3vl_emb":
            continue  # not in the official release; produced later
        p = session_dir / tmpl.format(role=role)
        if not p.exists():
            continue
        try:
            hdr = parse_header(p)
        except Exception as e:
            print(f"  WARN parse fail {p}: {e}", file=sys.stderr)
            continue
        out[feat] = (hdr.dim, hdr.sample_rate_hz, hdr.num_frames)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True,
                    help="…/mm_26/data")
    ap.add_argument("--feature-specs", type=Path,
                    default=Path("multimediate26/configs/feature_specs.yaml"))
    ap.add_argument("--max-sessions-per-domain", type=int, default=3)
    args = ap.parse_args()

    specs = yaml.safe_load(args.feature_specs.read_text())
    raw_paths = specs["raw_paths"]

    # Aggregate observed (dim, sr) values across all probed sessions.
    observed: dict[str, dict[str, set]] = defaultdict(
        lambda: {"dims": set(), "srs": set(), "n_sessions": 0,
                 "domains_seen": set()}
    )

    print("=" * 72)
    print("Probing feature headers across domains …")
    print("=" * 72)

    for domain, (roles, _rel) in DOMAIN_ROLES.items():
        sess_dirs = find_session_dirs(args.data_root, domain,
                                      args.max_sessions_per_domain)
        if not sess_dirs:
            continue
        for sd in sess_dirs:
            # Try the first role that yields any feature data.
            for role in roles:
                seen = probe_one(sd, role, raw_paths)
                if seen:
                    break
            else:
                continue
            for feat, (d, sr, n) in seen.items():
                observed[feat]["dims"].add(d)
                observed[feat]["srs"].add(sr)
                observed[feat]["n_sessions"] += 1
                observed[feat]["domains_seen"].add(domain)
            print(f"  [{domain}] {sd.name}/{role}: "
                  f"{len(seen)} features")

    # ── Report ────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("OBSERVED feature_dims & native_rate_hz")
    print("=" * 72)
    print(f"{'feature':<14} {'dim':>8} {'sr_hz':>8} "
          f"{'#sess':>6} {'domains':<30}")
    out_dims = {}
    out_rates = {}
    any_warn = False
    for feat in raw_paths:
        if feat == "qwen3vl_emb":
            continue
        info = observed.get(feat)
        if not info or not info["dims"]:
            print(f"{feat:<14} {'(none observed)'}")
            any_warn = True
            continue
        if len(info["dims"]) > 1:
            print(f"{feat:<14} ! DIM MISMATCH: {sorted(info['dims'])}")
            any_warn = True
            continue
        if len(info["srs"]) > 1:
            print(f"{feat:<14} ! SR MISMATCH: {sorted(info['srs'])}")
            any_warn = True
            continue
        d = next(iter(info["dims"]))
        sr = next(iter(info["srs"]))
        out_dims[feat] = d
        out_rates[feat] = sr
        print(f"{feat:<14} {d:>8d} {sr:>8.3f} {info['n_sessions']:>6d} "
              f"{','.join(sorted(info['domains_seen'])):<30}")

    print()
    print("=" * 72)
    print("YAML to paste into configs/feature_specs.yaml")
    print("=" * 72)
    print("feature_dims:")
    for feat, d in out_dims.items():
        print(f"  {feat:<10}: {d}")
    print()
    print("native_rate_hz:")
    for feat, sr in out_rates.items():
        # Round to int when the rate cleanly divides — 25.000 → 25
        sr_repr = int(sr) if abs(sr - round(sr)) < 1e-6 else sr
        print(f"  {feat:<10}: {sr_repr}")

    if any_warn:
        print()
        print("⚠ Some features had inconsistent or missing dims; "
              "check above before updating yaml.")


if __name__ == "__main__":
    main()
