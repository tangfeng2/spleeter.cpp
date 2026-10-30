"""Definitive parity test.

Both Python Spleeter and the C++ port write 16-bit PCM WAV files, so the
WAV quantization floor limits the measurable SNR. To compare the algorithms
rather than the container, this script:

  1. quantizes the Python reference stems to 16-bit PCM exactly as ffmpeg does,
  2. decodes the C++ WAV back to float,
  3. compares them.

If the C++ port is numerically equivalent to Python, the remaining difference
is only the rounding of the shared 16-bit representation.
"""

import argparse
import json
import os
import subprocess

import numpy as np


def decode_wav(path, sample_rate=44100):
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2",
         "-ar", str(sample_rate), "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(result.stdout, dtype="<f4").reshape(-1, 2)


def quantize_16bit(samples):
    """Rounds float samples the way ffmpeg's pcm_s16le encoder does."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (np.round(clipped * 32767.0)).astype(np.int16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp_dir", required=True)
    parser.add_argument("--ref_dir", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    with open(os.path.join(args.ref_dir, "_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    instruments = [
        k for k in meta
        if not k.startswith("_") and k not in ("sample_rate", "n_samples")
    ]

    report = {}
    for instrument in instruments:
        ref = np.load(os.path.join(args.ref_dir, f"{instrument}.npy"))
        cpp = decode_wav(os.path.join(args.cpp_dir, f"{instrument}.wav"))

        n = min(ref.shape[0], cpp.shape[0])
        ref_q = quantize_16bit(ref[:n]).astype(np.float32) / 32767.0
        cpp = cpp[:n]

        diff = cpp.astype(np.float64) - ref_q.astype(np.float64)
        max_abs = float(np.max(np.abs(diff)))
        ref_rms = float(np.sqrt(np.mean(ref_q.astype(np.float64) ** 2)))
        err_rms = float(np.sqrt(np.mean(diff**2)))
        snr = 20 * np.log10(ref_rms / err_rms) if err_rms > 0 else float("inf")

        # How many samples differ at all?
        differing = int(np.count_nonzero(diff))
        exact = differing == 0

        report[instrument] = {
            "max_abs_diff_after_quantization": max_abs,
            "snr_db": float(snr),
            "differing_samples": differing,
            "total_samples": int(n),
            "bit_exact": exact,
            "length_match": bool(ref.shape[0] == cpp.shape[0]),
        }
        print(
            f"{instrument:12s} max_abs={max_abs:.3e} snr={snr:7.1f}dB "
            f"differing={differing}/{n} exact={exact} "
            f"len_match={ref.shape[0] == cpp.shape[0]}"
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
