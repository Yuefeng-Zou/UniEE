#!/usr/bin/env python3
"""Archive all completed experiments to /ossfs/workspace/engagement_experiments/.

For each experiment, creates a directory with:
  - best.pt (symlink to original)
  - train.log (copy)
  - summary.json (parsed metrics: best epoch, best val scores, final val, config)
  - config.yaml (training config snapshot)
"""
import json, re, shutil, os
from pathlib import Path
from datetime import datetime

ARCHIVE_ROOT = Path("experiments")
OUTPUT_ROOT = Path("multimediate26/output")
LOG_ROOT = Path("/ossfs/workspace/run_logs")


def parse_train_log(log_path: Path) -> dict:
    """Extract key metrics from a train.log."""
    text = log_path.read_text()

    info = {
        "log_path": str(log_path),
        "total_epochs": 0,
        "best_epoch": None,
        "best_combined": None,
        "best_per_domain": {},
        "final_val": {},
        "model_params": None,
        "features": None,
        "init_from": None,
    }

    # Model params
    m = re.search(r"model params: ([\d.]+) M", text)
    if m:
        info["model_params"] = f"{m.group(1)}M"

    # Features
    m = re.search(r"features:\s*\[([^\]]+)\]", text)
    if m:
        info["features"] = [f.strip().strip("'\"") for f in m.group(1).split(",")]

    # Init from
    m = re.search(r"init from (.+?) \(", text)
    if m:
        info["init_from"] = m.group(1).strip()

    # Count epochs
    info["total_epochs"] = len(re.findall(r"^  epoch  ", text, re.MULTILINE))

    # Parse all "new best" lines
    best_lines = re.findall(r"new best combined_ccc=([\d.]+)", text)
    if best_lines:
        info["best_combined"] = float(best_lines[-1])
        info["best_epoch"] = len(best_lines) - 1  # approximate

    # Parse last val line for final scores
    val_lines = re.findall(r"val: (.+)", text)
    if val_lines:
        last_val = val_lines[-1]
        for kv in re.findall(r"([\w/]+)=([\d.]+)", last_val):
            info["final_val"][kv[0]] = float(kv[1])

    # Parse best val line (the one just before "new best")
    for i, line in enumerate(text.splitlines()):
        if "new best" in line and i > 0:
            prev = text.splitlines()[i-1] if i > 0 else ""
            for kv in re.findall(r"([\w/]+)=([\d.]+)", prev):
                info["best_per_domain"][kv[0]] = float(kv[1])

    return info


def archive_experiment(exp_name: str):
    """Archive one experiment."""
    output_dir = OUTPUT_ROOT / exp_name
    if not output_dir.exists():
        return None

    best_pt = output_dir / "best.pt"
    if not best_pt.exists():
        return None

    archive_dir = ARCHIVE_ROOT / exp_name
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Symlink best.pt
    link = archive_dir / "best.pt"
    if not link.exists():
        abs_target = best_pt.resolve()
        link.symlink_to(abs_target)

    # Copy train.log
    log_dir = LOG_ROOT / exp_name
    train_log = log_dir / "train.log"
    if train_log.exists():
        dst = archive_dir / "train.log"
        if not dst.exists():
            shutil.copy2(train_log, dst)

    # Parse and save summary
    summary = {"experiment": exp_name, "archived_at": datetime.now().isoformat()}
    if train_log.exists():
        summary.update(parse_train_log(train_log))
    summary["best_pt_size_mb"] = round(best_pt.stat().st_size / 1e6, 1)
    summary["best_pt_mtime"] = datetime.fromtimestamp(
        best_pt.stat().st_mtime).isoformat()

    (archive_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    return summary


def main():
    os.chdir("/mnt/pro-dtai/moe-lite/fenghui.zyf/mm_26/engagement_estimation")

    # Find all experiments with best.pt
    experiments = sorted(d.name for d in OUTPUT_ROOT.iterdir()
                        if d.is_dir() and (d / "best.pt").exists())

    print(f"Found {len(experiments)} experiments with best.pt\n")

    all_summaries = []
    for exp in experiments:
        s = archive_experiment(exp)
        if s:
            best = s.get("best_combined", "?")
            ep = s.get("total_epochs", "?")
            params = s.get("model_params", "?")
            print(f"  ✓ {exp:<50s} best={best}  ep={ep}  params={params}")
            all_summaries.append(s)

    # Also archive eval_full_val and TTA logs
    misc_dir = ARCHIVE_ROOT / "_eval_logs"
    misc_dir.mkdir(parents=True, exist_ok=True)
    for log_name in [
        "eval_full_val_phase3.log",
        "sota_comparison.txt",
        "tta_noxi_val.log",
        "tta_noxi_j_val.log",
        "tta_mpiigi_val.log",
        "tta_pinsoro_cc_val.log",
        "tta_pinsoro_cr_val.log",
    ]:
        src = LOG_ROOT / log_name
        dst = misc_dir / log_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ {log_name} → _eval_logs/")

    # Also archive feature stats
    stats_dir = ARCHIVE_ROOT / "_feature_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    for f in Path("/ossfs/workspace/mm26_stats").glob("*.npz"):
        dst = stats_dir / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
            print(f"  ✓ {f.name} → _feature_stats/")

    # Write master index
    (ARCHIVE_ROOT / "experiments_index.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False) + "\n")
    print(f"\n=== Archived {len(all_summaries)} experiments → {ARCHIVE_ROOT}")
    print(f"    Master index: {ARCHIVE_ROOT}/experiments_index.json")


if __name__ == "__main__":
    main()
