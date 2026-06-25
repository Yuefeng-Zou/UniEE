"""Extract Qwen3-VL-Embedding-8B features for every NoXi / NoXi-J / mpii video.

Per video:
  1. Open mp4 / avi via decord; get duration.
  2. Slice into non-overlapping ``segment_seconds`` segments.
  3. For each segment, sample ``frames_per_segment`` frames uniformly.
  4. Build a Qwen3VLEmbedder input dict: {video: [frames], instruction: ...}.
  5. Batch-process N segments at a time → (B, 4096) embedding → MRL truncate
     to ``output_dim`` → L2 normalize.
  6. Broadcast each segment's embedding to its 25Hz frame range → save
     ``{role}.audio.vlm.npy`` (T_25hz, output_dim) float32.

Why per-role? In NoXi/NoXi-J each role has its own front-facing camera; the
embedding describes that role's view of the interaction (face, gaze, posture).
For mpii the same group video covers all participants — but we save per-role
copies so the dataset.py path is consistent.

PInSoRo has no raw video (ethics-restricted) → skipped entirely.

Sharding: each process takes a slice of sessions by ``(seed + rank) % world``.
Run 4 instances with --shard 0..3 / --world 4 on GPU 0..3 to parallelize.

Output layout (next to the input wav):
  <session_dir>/<role>.audio.vlm.npy           # (T_25hz, 1024) fp32
  <session_dir>/.vlm.done                       # marker; skip if exists

Skipping fully-built sessions makes this safe to re-run.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F


# Where coarse-domain prompts live (must match dataset_config naming).
INSTRUCTION_BY_DOMAIN = {
    "noxi":     "Represent this two-person conversation for engagement estimation. Focus on participant attention, eye contact, listening behavior, and turn-taking dynamics.",
    "noxi_j":   "Represent this two-person conversation for engagement estimation. Focus on participant attention, eye contact, listening behavior, and turn-taking dynamics.",
    "noxi_add": "Represent this two-person conversation for engagement estimation. Focus on participant attention, eye contact, listening behavior, and turn-taking dynamics.",
    "mpiigi":   "Represent this group meeting for engagement estimation. Focus on speaker dominance, attention shifts, listener responsiveness, and group focus.",
}


VIDEO_GLOB_BY_DOMAIN = {
    # NoXi train/val/test/test-additional
    "noxi":     [("NoXi/train", "*.video.mp4"), ("NoXi/val", "*.video.mp4"),
                 ("NoXi/test-base", "*.video.mp4")],
    "noxi_add": [("NoXi/test-additional", "*.video.mp4")],
    "noxi_j":   [("Noxi-J/train", "*.video.mp4"), ("Noxi-J/val", "*.video.mp4"),
                 ("Noxi-J/test", "*.video.mp4")],
    "mpiigi":   [("mpii/MultiMediate25/val/originalAudioVideo-val", "*.video.avi"),
                 ("mpii/MultiMediate25/test/originalAudioVideo-test", "*.video.avi")],
}


def iter_videos(data_root: Path, domains: list[str]) -> Iterator[tuple[Path, str]]:
    """Yields (mp4_path, domain) for every video to process."""
    for d in domains:
        for sub, glob in VIDEO_GLOB_BY_DOMAIN.get(d, []):
            root = data_root / sub
            if not root.exists():
                continue
            for sess in sorted(root.iterdir()):
                if not sess.is_dir():
                    continue
                for v in sorted(sess.glob(glob)):
                    yield v, d


def load_embedder(model_path: Path, dtype: torch.dtype = torch.bfloat16):
    """Load Qwen3VLEmbedder in IMAGE mode (low-res, fast)."""
    sys.path.insert(0, str(Path(model_path) / "scripts"))
    from qwen3_vl_embedding import Qwen3VLEmbedder, IMAGE_FACTOR
    return Qwen3VLEmbedder(
        model_name_or_path=str(model_path),
        dtype=dtype,
        min_pixels=64 * IMAGE_FACTOR * IMAGE_FACTOR,
        max_pixels=128 * IMAGE_FACTOR * IMAGE_FACTOR,
    )


def video_duration_s(path: Path) -> float:
    """Returns duration in seconds using decord (fast, no decode)."""
    import decord
    vr = decord.VideoReader(str(path))
    return len(vr) / vr.get_avg_fps()


def sample_frame_indices(num_total: int, num_pick: int, t0: float, t1: float,
                         fps: float) -> list[int]:
    """Pick ``num_pick`` evenly spaced frame indices inside [t0, t1) at fps."""
    f0 = max(0, int(t0 * fps))
    f1 = min(num_total - 1, int(t1 * fps))
    if f1 <= f0:
        return [f0]
    idx = np.linspace(f0, f1, num_pick, dtype=int)
    return idx.tolist()


def extract_one_video(video_path: Path, domain: str, embedder,
                      segment_seconds: float, frames_per_segment: int,
                      output_dim: int, target_hz: float,
                      batch_size: int, log, resize: int = 256) -> np.ndarray:
    """Returns (T_25hz, output_dim) float32. Caller saves it.

    Uses IMAGE mode (single center frame per segment, resized to ``resize``²)
    for ~4× speedup over VIDEO mode (4 frames per segment). Benchmark:
    IMAGE bs=16 → 1.5s/seg vs VIDEO bs=4 → 6.5s/seg.
    """
    import decord
    from PIL import Image
    vr = decord.VideoReader(str(video_path))
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    total_s = total_frames / fps

    instruction = INSTRUCTION_BY_DOMAIN[domain]

    n_seg = int(np.ceil(total_s / segment_seconds))
    segments = [(i * segment_seconds,
                 min((i + 1) * segment_seconds, total_s))
                for i in range(n_seg)]

    seg_embeddings = np.zeros((n_seg, output_dim), dtype=np.float32)
    for b_start in range(0, n_seg, batch_size):
        b_end = min(b_start + batch_size, n_seg)
        batch_inputs = []
        for t0, t1 in segments[b_start:b_end]:
            mid_frame = int(((t0 + t1) / 2) * fps)
            mid_frame = min(mid_frame, total_frames - 1)
            frame = Image.fromarray(vr[mid_frame].asnumpy())
            if resize:
                frame = frame.resize((resize, resize), Image.LANCZOS)
            batch_inputs.append({"image": frame, "instruction": instruction})
        embs = embedder.process(batch_inputs, normalize=False)
        embs = embs[:, :output_dim]
        embs = F.normalize(embs.float(), p=2, dim=-1)
        seg_embeddings[b_start:b_end] = embs.cpu().numpy()

    T_25 = int(round(total_s * target_hz))
    out = np.zeros((T_25, output_dim), dtype=np.float32)
    for (t0, t1), emb in zip(segments, seg_embeddings):
        f0 = int(round(t0 * target_hz))
        f1 = min(T_25, int(round(t1 * target_hz)))
        out[f0:f1] = emb
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path,
                    default=Path("/mnt/pro-dtai/moe-lite/fenghui.zyf/models/Qwen/Qwen3-VL-Embedding-8B"))
    ap.add_argument("--data-root", type=Path,
                    default=Path("/mnt/pro-dtai/moe-lite/fenghui.zyf/mm_26/data"))
    ap.add_argument("--domains", default="noxi,noxi_add,noxi_j,mpiigi",
                    help="comma-separated; PInSoRo has no video — skipped")
    ap.add_argument("--segment-seconds", type=float, default=10.0)
    ap.add_argument("--frames-per-segment", type=int, default=1,
                    help="IMAGE mode = 1 center frame (fast); >1 uses VIDEO mode (slow)")
    ap.add_argument("--output-dim", type=int, default=1024,
                    help="MRL truncate from 4096 to this dim")
    ap.add_argument("--target-hz", type=float, default=25.0)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="segments per Qwen forward — 16 fits ~20GB VRAM")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="only process first N videos (debug)")
    ap.add_argument("--no-skip-done", action="store_true")
    args = ap.parse_args()

    def log(msg):
        sys.stdout.write(f"[rank{args.shard}] {msg}\n")
        sys.stdout.flush()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    log(f"domains: {domains}, world={args.world}, shard={args.shard}, "
        f"dim={args.output_dim}, seg={args.segment_seconds}s × {args.frames_per_segment}f")

    # 1) Enumerate work, then take our slice.
    all_videos = list(iter_videos(args.data_root, domains))
    my_videos = [v for i, v in enumerate(all_videos) if i % args.world == args.shard]
    if args.limit > 0:
        my_videos = my_videos[: args.limit]
    log(f"total videos={len(all_videos)}, my slice={len(my_videos)}")

    # 2) Skip already-done.
    pending = []
    for vp, dom in my_videos:
        out_path = vp.with_name(vp.name.replace(".video.mp4", ".audio.vlm.npy")
                                            .replace(".video.avi", ".audio.vlm.npy"))
        done_path = vp.parent / f".{vp.stem}.vlm.done"
        if not args.no_skip_done and done_path.exists() and out_path.exists():
            continue
        pending.append((vp, dom, out_path, done_path))
    log(f"pending after skip: {len(pending)}")

    if not pending:
        return

    # 3) Lazy-load embedder ONLY if work remaining (avoids loading 16GB to find nothing).
    log("loading Qwen3-VL-Embedding-8B ...")
    embedder = load_embedder(args.model_path)
    log("model loaded.")

    # 4) Process.
    t0_all = time.time()
    for idx, (vp, dom, out_path, done_path) in enumerate(pending):
        t0 = time.time()
        try:
            feat = extract_one_video(
                vp, dom, embedder,
                segment_seconds=args.segment_seconds,
                frames_per_segment=args.frames_per_segment,
                output_dim=args.output_dim,
                target_hz=args.target_hz,
                batch_size=args.batch_size,
                log=log,
            )
            np.save(out_path, feat)
            done_path.touch()
            log(f"[{idx+1}/{len(pending)}] {vp.relative_to(args.data_root)} "
                f"→ {feat.shape} in {time.time()-t0:.1f}s")
        except Exception as e:
            log(f"FAIL {vp}: {e}")
            traceback.print_exc()

    log(f"=== done. total {time.time()-t0_all:.0f}s, {len(pending)} videos")


if __name__ == "__main__":
    main()
