"""Find where the C++ port and the reference NumPy/ONNX pipeline diverge.

Runs both on the same input for a range of durations and reports the SNR per
instrument. Durations are chosen around the 512-frame tile boundary
(512 frames * 1024 samples = 11.86 s of audio).
"""

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_masks import stft_frames  # noqa: E402
from perturb_test import run_pipeline  # noqa: E402

EXE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "build", "spleeter_cpp.exe")


def decode(path):
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2",
         "-ar", "44100", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(result.stdout, dtype="<f4").reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="test/clip_5min.mp4")
    parser.add_argument("--model", default="5stems")
    parser.add_argument("--durations", default="2,12,24,36,60")
    parser.add_argument("--work", default="test/sweep")
    args = parser.parse_args()

    os.makedirs(args.work, exist_ok=True)
    onnx = f"models/onnx/{args.model}.onnx"

    for duration in [float(x) for x in args.durations.split(",")]:
        wav = os.path.join(args.work, "in.wav")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", "0", "-t", str(duration),
             "-i", args.source, "-f", "wav", "-ac", "2", "-ar", "44100", wav, "-y"],
            check=True,
        )
        out_dir = os.path.join(args.work, f"out_{duration}")
        subprocess.run(
            [EXE, "-o", out_dir, "-p", args.model, "--model_dir", "models/onnx",
             "-c", "f32", "--cpu", wav],
            check=True,
        )

        mix = decode(wav)
        reference = run_pipeline(onnx, mix.astype(np.float32),
                                 np.abs(stft_frames(mix.astype(np.float32))).astype(np.float32))

        stem_dir = os.path.join(out_dir, "in")
        line = [f"{duration:6.1f}s frames={int(np.ceil((len(mix) + 4096) / 1024)):6d}"]
        for instrument, expected in reference.items():
            candidate = decode(os.path.join(stem_dir, f"{instrument}.wav"))
            n = min(len(expected), len(candidate))
            diff = candidate[:n].astype(np.float64) - expected[:n].astype(np.float64)
            rms = np.sqrt(np.mean(expected[:n].astype(np.float64) ** 2))
            err = np.sqrt(np.mean(diff**2))
            snr = 20 * np.log10(rms / max(err, 1e-30))
            line.append(f"{instrument}={snr:6.1f}dB")
        print("  ".join(line), flush=True)


if __name__ == "__main__":
    main()
