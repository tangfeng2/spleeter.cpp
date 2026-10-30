"""Compare C++ separated stems against the Python Spleeter reference.

Reports per-instrument max absolute difference, relative error and SNR,
plus a drift check that correlates the C++ output against the reference in
sliding windows (a drift shows up as a non-zero lag).
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np


def decode_wav(path, sample_rate=44100):
    """Decodes an audio file to float32 (n, 2) via ffmpeg."""
    command = [
        "ffmpeg", "-v", "error", "-i", path,
        "-f", "f32le", "-ac", "2", "-ar", str(sample_rate), "-",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    data = np.frombuffer(result.stdout, dtype="<f4")
    return data.reshape(-1, 2)


def drift_lag(reference, candidate, max_lag=2000):
    """Returns the lag (in samples) that best aligns candidate to reference."""
    a = reference[:, 0].astype(np.float64)
    b = candidate[:, 0].astype(np.float64)
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    if n < max_lag * 4:
        return 0
    # Compare energies in 1-second blocks to locate the alignment offset.
    block = 44100
    blocks = min(n // block, 60)
    best_lag = 0
    best_score = -1e30
    step = 10
    for lag in range(-max_lag, max_lag + 1, step):
        start = block
        end = start + blocks * block
        if end + max_lag > n:
            break
        shifted = b[start - lag : end - lag]
        ref = a[start:end]
        score = float(np.dot(shifted, ref))
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp_dir", required=True)
    parser.add_argument("--ref_dir", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--skip_drift", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(args.ref_dir, "_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    instruments = [
        k for k in meta
        if not k.startswith("_") and k not in ("sample_rate", "n_samples")
    ]

    report = {}
    for instrument in instruments:
        ref_path = os.path.join(args.ref_dir, f"{instrument}.npy")
        cpp_path = os.path.join(args.cpp_dir, f"{instrument}.wav")
        if not os.path.exists(cpp_path):
            print(f"MISSING {cpp_path}")
            continue
        reference = np.load(ref_path)
        candidate = decode_wav(cpp_path)

        n = min(reference.shape[0], candidate.shape[0])
        diff = candidate[:n].astype(np.float64) - reference[:n].astype(np.float64)
        max_abs = float(np.max(np.abs(diff)))
        ref_peak = float(np.max(np.abs(reference[:n])))
        ref_rms = float(np.sqrt(np.mean(reference[:n].astype(np.float64) ** 2)))
        err_rms = float(np.sqrt(np.mean(diff**2)))
        snr = 20 * np.log10(ref_rms / err_rms) if err_rms > 0 else float("inf")

        entry = {
            "max_abs_diff": max_abs,
            "ref_peak": ref_peak,
            "relative_max": max_abs / ref_peak if ref_peak else 0.0,
            "snr_db": float(snr),
            "ref_frames": int(reference.shape[0]),
            "cpp_frames": int(candidate.shape[0]),
            "length_match": bool(reference.shape[0] == candidate.shape[0]),
        }

        if not args.skip_drift:
            lag = drift_lag(reference, candidate)
            entry["drift_lag_samples"] = lag
            entry["drift_ms"] = lag * 1000.0 / meta["sample_rate"]

        report[instrument] = entry
        drift = entry.get("drift_lag_samples", "?")
        print(
            f"{instrument:12s} max_abs={max_abs:.3e} rel={entry['relative_max']:.3e} "
            f"snr={snr:6.1f}dB len_match={entry['length_match']} "
            f"ref={reference.shape[0]} cpp={candidate.shape[0]} drift={drift}"
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
