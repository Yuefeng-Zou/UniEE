"""Generate language-split manifests for P8 prompt split.

Reads existing manifests (noxi_train.jsonl, noxi_j_train.jsonl, etc.) and
splits them by language → per-language-domain manifest.

New domains:
  noxi_fr, noxi_en, noxi_de        (from noxi)
  noxi_j_ja, noxi_j_zh             (from noxi_j)
  noxi_add_it, noxi_add_ar, noxi_add_es, noxi_add_id  (from noxi_add test)

For training, we keep each language-domain as a separate prompt in
DomainPromptPool; for inference, noxi_add_* uses fallback to the closest
noxi language prompt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LANG_MAP = {
    "French":     "fr",
    "English":    "en",
    "German":     "de",
    "Japanese":   "ja",
    "Chinese":    "zh",
    "Italian":    "it",
    "Arabic":     "ar",
    "Spanish":    "es",
    "Indonesian": "id",
}

# domain → language code → new domain name
DOMAIN_SPLIT = {
    "noxi":     {"fr": "noxi_fr",  "en": "noxi_en",  "de": "noxi_de"},
    "noxi_j":   {"ja": "noxi_j_ja", "zh": "noxi_j_zh"},
    "noxi_add": {"it": "noxi_add_it", "ar": "noxi_add_ar",
                 "es": "noxi_add_es", "id": "noxi_add_id"},
}

# fallback for inference: noxi_add_* → closest noxi_* prompt
FALLBACKS = {
    "noxi_add_it": "noxi_fr",   # Italian ≈ French (Romance family)
    "noxi_add_ar": "noxi_en",   # Arabic interviews conducted in English context
    "noxi_add_es": "noxi_fr",   # Spanish ≈ French (Romance family)
    "noxi_add_id": "noxi_en",   # Indonesian interviews in English context
    "noxi_add":    "noxi_en",   # generic fallback
}


def get_session_language(data_root: Path, domain: str, session_id: str) -> str | None:
    """Read language.annotation.csv and return the language code."""
    from multimediate26.data.feature_extractor.build_session_npz import DOMAIN_LAYOUT
    layout = DOMAIN_LAYOUT.get(domain)
    if not layout:
        return None
    for split, rel in layout["splits"].items():
        csv = data_root / rel / session_id / "language.annotation.csv"
        if csv.exists():
            line = csv.read_text().strip().split("\n")[0]
            lang_full = line.split(";")[2].strip()
            return LANG_MAP.get(lang_full)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path,
                    default=Path("/mnt/pro-dtai/moe-lite/fenghui.zyf/mm_26/data"))
    ap.add_argument("--manifest-dir", type=Path,
                    default=Path("multimediate26/manifests"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("multimediate26/manifests_p8"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # For each source manifest, split by language.
    for src_file in sorted(args.manifest_dir.glob("*.jsonl")):
        rows = [json.loads(l) for l in src_file.read_text().splitlines() if l.strip()]
        if not rows:
            continue
        domain = rows[0]["domain"]
        if domain not in DOMAIN_SPLIT:
            # Non-splittable domain — copy as-is
            dst = args.out_dir / src_file.name
            dst.write_text(src_file.read_text())
            print(f"  copy: {src_file.name} ({len(rows)} rows)")
            continue

        # Split
        lang_groups: dict[str, list] = {}
        n_unknown = 0
        for row in rows:
            lang_code = get_session_language(args.data_root, domain,
                                             row["session_id"])
            if lang_code is None:
                n_unknown += 1
                continue
            new_domain = DOMAIN_SPLIT[domain].get(lang_code)
            if new_domain is None:
                n_unknown += 1
                continue
            row_copy = dict(row)
            row_copy["domain"] = new_domain
            lang_groups.setdefault(new_domain, []).append(row_copy)

        for new_domain, group_rows in lang_groups.items():
            split_tag = src_file.stem.replace(domain, new_domain)
            dst = args.out_dir / f"{split_tag}.jsonl"
            with open(dst, "w") as f:
                for r in group_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  split: {dst.name} ({len(group_rows)} rows)")

        if n_unknown:
            print(f"    [warn] {n_unknown} rows with unknown language in {src_file.name}")

    # Write fallback config
    fb_path = args.out_dir / "_fallbacks.json"
    fb_path.write_text(json.dumps(FALLBACKS, indent=2))
    print(f"\n  fallbacks → {fb_path}")
    print(f"\n=== done. New manifests in {args.out_dir}")


if __name__ == "__main__":
    main()
