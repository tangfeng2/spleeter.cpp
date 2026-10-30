"""Prove the C++ port is numerically equivalent to the Python pipeline.

Two comparisons are made:

  1. C++ output vs the NumPy pipeline output, both quantized to 16-bit PCM.
     If the algorithms agree, the quantized arrays are identical.

  2. C++ output (CUDA EP) vs C++ output (CPU EP), which isolates the
     numerical difference contributed by the GPU execution provider.
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
    return np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)


def compare(a, b, label):
    n = min(a.shape[0], b.shape[0])
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    diff = a - b
    max_abs = float(np.max(np.abs(diff)))
    ref_rms = float(np.sqrt(np.mean(b**2)))
    err_rms = float(np.sqrt(np.mean(diff**2)))
    snr = 20 * np.log10(ref_rms / err_rms) if err_rms > 0 else float("inf")
    differing = int(np.count_nonzero(diff))
    print(
        f"  {label}: max_abs={max_abs:.3e} snr={snr:7.1f}dB "
        f"differing={differing}/{n} bit_exact={differing == 0}"
    )
    return {"max_abs_diff": max_abs, "snr_db": float(snr),
            "differing_samples": differing, "total_samples": int(n),
            "bit_exact": differing == 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp_cuda", required=True)
    parser.add_argument("--cpp_cpu", default="")
    parser.add_argument("--numpy_dir", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    meta_path = os.path.join(args.numpy_dir, "_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        instruments = [
            k for k in meta
            if not k.startswith("_") and k not in ("sample_rate", "n_samples")
        ]
    else:
        # Fall back to the stem files present in the directory.
        instruments = sorted(
            f[:-4] for f in os.listdir(args.numpy_dir)
            if f.endswith(".npy") and not f.startswith("_")
        )

    report = {}
    for instrument in instruments:
        print(f"{instrument}:")
        npy = np.load(os.path.join(args.numpy_dir, f"{instrument}.npy"))
        cuda = decode_wav(os.path.join(args.cpp_cuda, f"{instrument}.wav"))

        # The NumPy pipeline output quantized to 16-bit is what a WAV file of
        # that signal contains, so it is the fair comparison for the C++ WAV.
        npy_q = quantize_16bit(npy).astype(np.float32) / 32767.0
        report[instrument] = {
            "cpp_vs_numpy": compare(cuda, npy_q, "C++(CUDA) vs NumPy(CPU), 16-bit"),
        }
        if args.cpp_cpu:
            cpu = decode_wav(os.path.join(args.cpp_cpu, f"{instrument}.wav"))
            report[instrument]["cpp_cpu_vs_numpy"] = compare(
                cpu, npy_q, "C++(CPU)  vs NumPy(CPU), 16-bit")
            report[instrument]["cpp_cuda_vs_cpp_cpu"] = compare(
                cuda, cpu, "C++(CUDA) vs C++(CPU)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
