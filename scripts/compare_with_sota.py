"""Compare all our trained checkpoints against the published SOTA.

Walks /ossfs/workspace/run_logs/<exp>/train.log, parses every
"new best combined_ccc=" line with the most recent per-domain val numbers,
and prints a Markdown table juxtaposing our scores with:

* DAPA (HFUT-LMC, MM25 winner) — val numbers from arXiv:2509.xxxxx Table 2-3
* USTC-IAT 2025 (8x sliding paper) — val + test
* MM26 official baseline (best per-feature from README.md) — test only,
  shown for reference (we can't compare to our val directly).

Usage:
    python -m multimediate26.scripts.compare_with_sota
    python -m multimediate26.scripts.compare_with_sota --runs-dir /ossfs/workspace/run_logs

Output: prints a Markdown table to stdout. Pipe to a file or paste into a doc.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# Published SOTA (val unless noted)
SOTA = {
    # paper:                noxi    noxi_j  mpiigi  pinsoro_cc_task pinsoro_cc_social pinsoro_cr_task pinsoro_cr_social
    "DAPA (HFUT-LMC val)":   {"noxi": 0.855, "noxi_j": 0.722, "mpiigi": None},
    "USTC-IAT 2025 (val)":   {"noxi": 0.830, "noxi_j": 0.670, "mpiigi": None},
    "MM26 baseline (test)":  {"noxi": 0.554, "noxi_j": 0.313, "mpiigi": 0.454,
                              "noxi_add": 0.491,
                              "pinsoro_cc/task_kappa":   0.135,
                              "pinsoro_cc/social_kappa": 0.213,
                              "pinsoro_cr/task_kappa":   0.711,
                              "pinsoro_cr/social_kappa": 0.172},
}


VAL_LINE = re.compile(
    r"val: combined_ccc=([0-9.]+)\s+"
    r"(?:mpiigi/ccc=([0-9.\-]+)\s+)?"
    r"(?:noxi/ccc=([0-9.\-]+)\s+)?"
    r"(?:noxi_j/ccc=([0-9.\-]+)\s*)?"
    r"(?:pinsoro_cc/task_kappa=([0-9.\-]+)\s+)?"
    r"(?:pinsoro_cc/social_kappa=([0-9.\-]+)\s+)?"
    r"(?:pinsoro_cr/task_kappa=([0-9.\-]+)\s+)?"
    r"(?:pinsoro_cr/social_kappa=([0-9.\-]+)\s*)?"
)
NEW_BEST  = re.compile(r"new best combined_ccc=([0-9.]+)")


def parse_run(log: Path) -> dict:
    """Returns best snapshot from a train.log:
        {"combined_ccc": float, "per_domain": {domain: float}, "ep_at_best": int}
    """
    if not log.exists():
        return {}
    txt = log.read_text(errors="ignore")
    lines = txt.split("\n")

    best = {"combined_ccc": 0.0, "per_domain": {}, "ep_at_best": -1}
    # Walk line by line, tracking the latest 'val:' line so when we hit
    # 'new best', we know which per-domain numbers belong to it.
    last_val_dict: dict[str, float] = {}
    epoch = -1
    for ln in lines:
        ep_match = re.search(r"epoch\s+(\d+)\s+lr=", ln)
        if ep_match:
            epoch = int(ep_match.group(1))
        m = VAL_LINE.search(ln)
        if m:
            last_val_dict = {}
            combined = float(m.group(1))
            last_val_dict["combined_ccc"] = combined
            for key, g in (("mpiigi", 2), ("noxi", 3), ("noxi_j", 4),
                           ("pinsoro_cc/task_kappa", 5),
                           ("pinsoro_cc/social_kappa", 6),
                           ("pinsoro_cr/task_kappa", 7),
                           ("pinsoro_cr/social_kappa", 8)):
                raw = m.group(g)
                if raw is not None:
                    last_val_dict[key] = float(raw)
            continue
        nb = NEW_BEST.search(ln)
        if nb:
            v = float(nb.group(1))
            if v > best["combined_ccc"] and last_val_dict:
                best["combined_ccc"] = v
                best["per_domain"] = {k: vv for k, vv in last_val_dict.items()
                                      if k != "combined_ccc"}
                best["ep_at_best"] = epoch
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path,
                    default=Path("/ossfs/workspace/run_logs"),
                    help="parent directory containing <exp_name>/train.log")
    ap.add_argument("--filter",
                    help="only experiments containing this substring")
    args = ap.parse_args()

    # Discover runs
    runs: list[tuple[str, dict]] = []
    if args.runs_dir.exists():
        for exp_dir in sorted(args.runs_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            if args.filter and args.filter not in exp_dir.name:
                continue
            log = exp_dir / "train.log"
            best = parse_run(log)
            if best.get("combined_ccc", 0) > 0:
                runs.append((exp_dir.name, best))

    # Gather all column keys actually seen.
    all_cols: list[str] = []
    seen = set()
    for _, b in runs:
        for k in b["per_domain"]:
            if k not in seen:
                all_cols.append(k); seen.add(k)
    # If empty (no per-domain), at least show combined.
    if not all_cols:
        all_cols = ["combined_ccc"]

    # Stable column order: ccc domains first, then kappa.
    ccc_first = sorted([c for c in all_cols if "ccc" in c or c in ("noxi", "noxi_j", "mpiigi", "noxi_add")])
    kappa = sorted([c for c in all_cols if "kappa" in c])
    others = [c for c in all_cols if c not in set(ccc_first) and c not in set(kappa)]
    cols = ccc_first + others + kappa

    # ── Markdown table ────────────────────────────────────────────────
    print(f"# Engagement Estimation — our runs vs SOTA")
    print()
    headers = ["run", "ep", "combined"] + cols
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for name, b in runs:
        row = [f"`{name}`", str(b["ep_at_best"]),
               f"{b['combined_ccc']:.4f}"]
        for c in cols:
            v = b["per_domain"].get(c)
            row.append(f"{v:.4f}" if v is not None else "—")
        print("| " + " | ".join(row) + " |")

    # SOTA rows
    print("|" + "|".join("***" for _ in headers) + "|")
    for sota_name, sota in SOTA.items():
        row = [f"**{sota_name}**", "—", "—"]
        for c in cols:
            v = sota.get(c)
            row.append(f"**{v:.4f}**" if v is not None else "—")
        print("| " + " | ".join(row) + " |")

    print()
    print("Notes:")
    print("- ep = epoch at which best `combined_ccc` was reached.")
    print("- DAPA / USTC are val numbers (compare directly to our runs above).")
    print("- MM26 baseline is **test** (lower bound; expect ours val > baseline test).")
    print("- mpiigi is evaluated on the 7-row val_held subset during training; "
          "use scripts/eval_full_val.py on the full 21-row val for a paper-comparable number.")


if __name__ == "__main__":
    main()
