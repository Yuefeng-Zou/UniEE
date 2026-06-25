"""Extract Whisper-large-v3 encoder hidden states as audio features.

For each .audio.wav file under the data tree, run the Whisper encoder over
fixed-length 30-second mel chunks, concatenate the resulting hidden states
along time, mean-pool down to a 25 Hz grid, and save as
``{role}.audio.whisper.npy`` of shape ``(T_25, 1280)`` (float16 to halve
disk).

Whisper-large-v3 internals (from config):
  * input: 30 s × 16 kHz mono → 128-mel spectrogram → encoder
  * encoder output: 1500 timesteps × 1280 d_model per 30 s chunk
    → native frame rate = 50 Hz (1500 / 30)
  * pool 2:1 along time → 25 Hz to match our 25 Hz training grid

We chunk long audio at exact 30-second non-overlapping windows (the encoder
is permutation-equivariant inside a 30 s mel window via positional embedding,
so neighboring chunks have no cross-boundary leakage; the tail chunk is
right-padded with silence to 30 s and trimmed back to its real duration
after pooling).

Usage:
    python -m multimediate26.data.feature_extractor.extract_whisper \
        --wav-root /mnt/pro-dtai/moe-lite/fenghui.zyf/mm_26/data/NoXi \
        --model    /ossfs/workspace/models/whisper-large-v3 \
        --device   cuda:4 \
        --batch    4 \
        --dtype    float16

Files that already have a sibling ``.whisper.npy`` of correct shape are
skipped (resumable).
"""
from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperModel


TARGET_FPS_OUT = 25.0     # output grid
WHISPER_FPS    = 50.0     # encoder native (1500 frames / 30 s)
CHUNK_SECONDS  = 30.0


def load_wav_16k_mono(path: Path) -> np.ndarray:
    """Read a wav file, convert to mono 16-kHz float32 in [-1, 1].

    Uses the stdlib ``wave`` module so we don't drag in soundfile/torchaudio.
    Re-samples by linear interpolation if the source sample rate differs.
    """
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        n_samp = w.getnframes()
        sampwidth = w.getsampwidth()
        raw = w.readframes(n_samp)
    if sampwidth == 2:
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        # 24-bit packed PCM — unpack 3 bytes per sample to int32
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        i = (b[:, 0].astype(np.int32) |
             (b[:, 1].astype(np.int32) << 8) |
             (b[:, 2].astype(np.int32) << 16))
        # sign-extend 24-bit → 32-bit
        i = np.where(i & 0x800000, i - 0x1000000, i).astype(np.int32)
        a = i.astype(np.float32) / (2 ** 23)
    elif sampwidth == 4:
        a = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / (2 ** 31)
    else:
        raise ValueError(f"unsupported wav sample width {sampwidth} at {path}")
    if n_ch > 1:
        a = a.reshape(-1, n_ch).mean(axis=1)
    if sr != 16000:
        # Linear-interpolate resample. Whisper's mel front-end is robust to
        # this for sr ∈ {8k, 44.1k, 48k}; we sanity-check large drops.
        n_new = int(round(a.size * 16000 / sr))
        x_old = np.arange(a.size, dtype=np.float64)
        x_new = np.linspace(0, a.size - 1, n_new, dtype=np.float64)
        a = np.interp(x_new, x_old, a).astype(np.float32)
    return a


def pool_to_25hz(x: np.ndarray) -> np.ndarray:
    """Pool a (T, D) array at 50 Hz down to 25 Hz by pairwise mean."""
    T = x.shape[0]
    if T % 2 == 1:
        x = np.concatenate([x, x[-1:]], axis=0)
        T += 1
    return x.reshape(T // 2, 2, x.shape[1]).mean(axis=1).astype(x.dtype, copy=False)


def find_wavs(root: Path, pattern: str = "*.audio.wav") -> list[Path]:
    return sorted(root.rglob(pattern))


@torch.no_grad()
def extract_one(
    wav_path: Path,
    out_path: Path,
    model: WhisperModel,
    fe: WhisperFeatureExtractor,
    device: torch.device,
    autocast_dtype: torch.dtype,
    chunk_seconds: float = CHUNK_SECONDS,
) -> tuple[int, float]:
    """Returns (n_frames_25hz, elapsed_seconds)."""
    t0 = time.time()
    audio = load_wav_16k_mono(wav_path)               # (N,) float32, sr=16k
    sr = 16000
    chunk_samples = int(chunk_seconds * sr)           # 480_000
    total_samples = audio.size
    n_chunks = int(np.ceil(total_samples / chunk_samples))

    # True target length at 25 Hz, based on audio duration. Each 30 s chunk
    # contributes 750 frames after pooling, EXCEPT the tail chunk which only
    # contributes ceil(real_tail_seconds * 25). We compute the expected
    # global length up-front then truncate.
    target_T_25 = int(np.ceil(total_samples * TARGET_FPS_OUT / sr))

    all_chunks: list[np.ndarray] = []
    for i in range(n_chunks):
        s = i * chunk_samples
        e = s + chunk_samples
        chunk = audio[s:e]
        real_len_samples = chunk.size
        if real_len_samples < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - real_len_samples), mode="constant")
        # Whisper feature_extractor: returns log-mel (B, 128, 3000)
        feats = fe(chunk, sampling_rate=sr, return_tensors="pt")
        input_features = feats.input_features.to(device=device, dtype=autocast_dtype)
        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=device.type == "cuda"):
            enc = model.encoder(input_features).last_hidden_state    # (1, 1500, 1280)
        enc = enc.squeeze(0).float().cpu().numpy()                   # (1500, 1280)
        # Pool 50 Hz → 25 Hz.
        enc_25 = pool_to_25hz(enc)                                   # (750, 1280)
        # Trim the tail chunk's padded frames.
        real_T_25 = int(np.ceil(real_len_samples * TARGET_FPS_OUT / sr))
        all_chunks.append(enc_25[:real_T_25])

    out = np.concatenate(all_chunks, axis=0).astype(np.float16)
    # Final clamp to the precise target length.
    if out.shape[0] > target_T_25:
        out = out[:target_T_25]
    elif out.shape[0] < target_T_25:
        pad = np.zeros((target_T_25 - out.shape[0], out.shape[1]), dtype=np.float16)
        out = np.concatenate([out, pad], axis=0)
    np.save(out_path, out)
    return out.shape[0], time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-root", type=Path, required=True,
                    help="search root for *.audio.wav files (e.g. .../data/NoXi)")
    ap.add_argument("--model",    type=Path,
                    default=Path("/ossfs/workspace/models/whisper-large-v3"))
    ap.add_argument("--device",   type=str, default="cuda:0")
    ap.add_argument("--dtype",    type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--chunk-seconds", type=float, default=CHUNK_SECONDS)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, process at most this many wavs (debug)")
    ap.add_argument("--shard", type=int, nargs=2, default=None,
                    metavar=("IDX", "TOTAL"),
                    help="process only every TOTAL-th wav starting at IDX. "
                    "Used to split work across multiple GPUs.")
    args = ap.parse_args()

    device = torch.device(args.device)
    autocast_dtype = {"float16": torch.float16,
                      "bfloat16": torch.bfloat16,
                      "float32": torch.float32}[args.dtype]

    print(f"[whisper] loading model from {args.model} on {device} ({args.dtype})",
          file=sys.stderr)
    fe = WhisperFeatureExtractor.from_pretrained(str(args.model))
    model = WhisperModel.from_pretrained(str(args.model)).to(device).eval()
    if autocast_dtype != torch.float32:
        model = model.to(autocast_dtype)

    wavs = find_wavs(args.wav_root)
    if args.shard is not None:
        idx, total = args.shard
        wavs = [w for i, w in enumerate(wavs) if i % total == idx]
        print(f"[whisper] shard {idx}/{total}: {len(wavs)} wavs", file=sys.stderr)
    if args.limit:
        wavs = wavs[: args.limit]
    print(f"[whisper] {len(wavs)} wavs under {args.wav_root}", file=sys.stderr)

    n_done = n_skip = n_fail = 0
    t_start = time.time()
    for i, wav in enumerate(wavs):
        out = wav.with_suffix("").with_suffix(".whisper.npy")
        # wav like .../125/expert.audio.wav → .../125/expert.audio.whisper.npy
        out = wav.parent / (wav.stem + ".whisper.npy")
        if args.skip_existing and out.exists():
            try:
                shp = np.load(out, mmap_mode="r").shape
                if shp[1] == 1280:
                    n_skip += 1
                    continue
            except Exception:
                pass
        try:
            T, dt = extract_one(wav, out, model, fe, device, autocast_dtype,
                                args.chunk_seconds)
            n_done += 1
            elapsed = time.time() - t_start
            eta = elapsed / max(n_done, 1) * (len(wavs) - n_skip - n_done)
            print(f"  [{i + 1}/{len(wavs)}] {wav.relative_to(args.wav_root)}  "
                  f"T_25={T} ({dt:.1f}s)  ETA={eta / 60:.0f} min",
                  file=sys.stderr)
        except Exception as e:
            n_fail += 1
            print(f"  FAIL {wav}: {e}", file=sys.stderr)

    print(f"\n[whisper] done. built={n_done} skipped={n_skip} failed={n_fail}",
          file=sys.stderr)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
