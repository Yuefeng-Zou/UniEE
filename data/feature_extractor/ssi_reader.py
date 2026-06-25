"""SSI stream binary reader.

The MultiMediate'26 official feature release uses the SSI / NoVA framework
file format. Each feature is split across two files:

    foo.stream       — XML header (200 B): dim, sampling rate, num frames,
                       byte width, dtype. Always opens with `<stream` or
                       `<annotation>`.
    foo.stream~      — pure float32 binary, row-major, shape (num, dim).
                       Size in bytes = num * dim * 4.

Both files SHARE the prefix (no extension change beyond the trailing `~`).
The `~` is a literal character, not a backup marker — it is the canonical
data file.

All 9 official features ship at 25 Hz, so no resampling is needed. The only
alignment work is truncating to the shortest frame count across features
for the same (session, role) pair.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Regex over the small XML header. We don't pull a full XML parser in for
# this — the schema is tiny and fixed.
_DIM_RE  = re.compile(r'\bdim="(\d+)"')
_SR_RE   = re.compile(r'\bsr="([\d.]+)"')
_NUM_RE  = re.compile(r'\bnum="(\d+)"')
_BYTE_RE = re.compile(r'\bbyte="(\d+)"')
# `type=` must NOT match `ftype=` — the SSI header has both:
#   <info ftype="BINARY" ... type="FLOAT" />
# Use a negative-lookbehind so the leading char is not 'f'.
_TYPE_RE = re.compile(r'(?<![a-zA-Z])type="([A-Z]+)"', re.IGNORECASE)


@dataclass(frozen=True)
class SSIHeader:
    """Parsed SSI .stream XML header."""
    dim: int
    sample_rate_hz: float
    num_frames: int
    byte_width: int
    dtype_name: str   # e.g. "FLOAT" → np.float32

    @property
    def numpy_dtype(self) -> np.dtype:
        if self.dtype_name.upper() == "FLOAT" and self.byte_width == 4:
            return np.dtype(np.float32)
        if self.dtype_name.upper() == "FLOAT" and self.byte_width == 8:
            return np.dtype(np.float64)
        raise NotImplementedError(
            f"Unsupported SSI dtype ({self.dtype_name}, {self.byte_width} B)"
        )

    @property
    def expected_bytes(self) -> int:
        return self.num_frames * self.dim * self.byte_width


def parse_header(stream_xml_path: Path) -> SSIHeader:
    """Parse the small XML header file. Raises if any field is missing."""
    text = Path(stream_xml_path).read_text(errors="ignore")
    try:
        # `num` lives inside the <chunk> tag — find on the chunk line to avoid
        # accidentally matching some other "num" attribute that may appear.
        dim   = int(_DIM_RE.search(text).group(1))
        sr    = float(_SR_RE.search(text).group(1))
        num   = int(_NUM_RE.search(text).group(1))
        byte  = int(_BYTE_RE.search(text).group(1))
        tname = _TYPE_RE.search(text).group(1)
    except AttributeError as e:
        raise ValueError(f"Malformed SSI header at {stream_xml_path}: {e}") from e
    return SSIHeader(
        dim=dim, sample_rate_hz=sr, num_frames=num,
        byte_width=byte, dtype_name=tname,
    )


def load_stream(stream_xml_path: Path,
                mmap: bool = True) -> tuple[np.ndarray, SSIHeader]:
    """Load an SSI stream pair into a (T, D) ndarray.

    Parameters
    ----------
    stream_xml_path
        Path to the small XML file (e.g. ``expert.openface2.stream``). The
        sibling binary file ``…stream~`` must exist next to it.
    mmap
        If True, the binary is memory-mapped (read-only). Useful when callers
        only slice / probe a few frames. Set False to materialize into RAM.

    Returns
    -------
    (data, header)
        ``data`` has shape ``(num_frames, dim)`` and dtype as declared in
        the header (typically float32). Columns are exactly as the upstream
        extractor wrote them — no preprocessing.
    """
    p = Path(stream_xml_path)
    hdr = parse_header(p)
    bin_path = p.with_name(p.name + "~")
    if not bin_path.exists():
        raise FileNotFoundError(
            f"SSI binary {bin_path} missing (header at {p} exists)."
        )
    actual = bin_path.stat().st_size
    if actual != hdr.expected_bytes:
        # Known data defect: PInSoRo egemapsv2 has TWO microphones stitched
        # into the binary (per-mic 88 dims concatenated → 176 dim), but the
        # XML header still claims dim=88. The binary is exactly 2x larger
        # than the header expects. We auto-detect and slice the first mic's
        # 88 columns. README: "audio features extracted from 2 microphones
        # on the side of each participant".
        if actual == 2 * hdr.expected_bytes:
            real_dim = 2 * hdr.dim
            arr = np.memmap(bin_path, dtype=hdr.numpy_dtype, mode="r",
                            shape=(hdr.num_frames, real_dim))
            # Use the first mic — both channels carry similar prosody since
            # they're from the same participant. Could later mean(mic0, mic1).
            if not mmap:
                arr = np.asarray(arr[:, : hdr.dim])
            else:
                arr = arr[:, : hdr.dim]
            return arr, hdr
        # Truncated binary (extractor crashed mid-write). Trim num_frames
        # down to whatever data actually exists, rather than failing the
        # whole session. Seen in the wild on NoXi/test-base/002/novice
        # w2vbert2 (82% of expected bytes).
        frame_bytes = hdr.dim * hdr.byte_width
        if actual < hdr.expected_bytes and actual % frame_bytes == 0:
            real_num = actual // frame_bytes
            arr = np.memmap(bin_path, dtype=hdr.numpy_dtype, mode="r",
                            shape=(real_num, hdr.dim))
            if not mmap:
                arr = np.asarray(arr)
            # Return a header reflecting the trimmed length.
            trimmed = SSIHeader(
                dim=hdr.dim, sample_rate_hz=hdr.sample_rate_hz,
                num_frames=real_num, byte_width=hdr.byte_width,
                dtype_name=hdr.dtype_name,
            )
            return arr, trimmed
        raise ValueError(
            f"SSI size mismatch at {bin_path}: header expects "
            f"{hdr.expected_bytes} bytes ({hdr.num_frames} × {hdr.dim} × "
            f"{hdr.byte_width}), got {actual} bytes"
        )
    if mmap:
        arr = np.memmap(bin_path, dtype=hdr.numpy_dtype, mode="r",
                        shape=(hdr.num_frames, hdr.dim))
    else:
        arr = np.fromfile(bin_path, dtype=hdr.numpy_dtype).reshape(
            hdr.num_frames, hdr.dim
        )
    return arr, hdr
