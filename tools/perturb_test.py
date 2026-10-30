"""Sensitivity probe: how much does the separation move when the magnitude
spectrogram is perturbed at the float32 rounding level?

Runs the Spleeter DSP (ONNX model + masks + inverse STFT) three times on the
same audio:

  * numpy    : spectrogram computed with NumPy's float64 rfft
  * cpp      : spectrogram produced by the C++ port's float32 FFT
  * perturbed: NumPy spectrogram with a relative 1e-7 perturbation

and reports the SNR between the results. This separates "the port computes
something different" from "the model amplifies float32-level differences".
"""

import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_masks import hann_periodic, inverse_stft, pad_and_partition, stft_frames  # noqa: E402

EPSILON = 1e-10


def run_pipeline(onnx_path, mix, magnitude):
    with open(onnx_path.replace(".onnx", "_meta.json"), "r", encoding="utf-8") as stream:
        meta = json.load(stream)
    instruments = meta["instrument_list"]
    sep_exp = meta["separation_exponent"]
    f_used = meta["F"]
    frame_length = meta["frame_length"]

    spec = stft_frames(mix)
    num_frames = spec.shape[0]
    tiles = pad_and_partition(magnitude.astype(np.float32), meta["T"])[:, :, :f_used, :]

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input": tiles.astype(np.float32)})

    n_extra_row = frame_length // 2 + 1 - f_used
    output_sum = np.zeros(tiles.shape, np.float32)
    for out in outputs:
        output_sum += (out.astype(np.float32) ** sep_exp).astype(np.float32)
    output_sum += np.float32(EPSILON)

    result = {}
    for index, instrument in enumerate(instruments):
        out = outputs[index].astype(np.float32)
        mask = (out**sep_exp).astype(np.float32) + np.float32(EPSILON / len(outputs))
        mask = (mask / output_sum).astype(np.float32)
        extension = np.zeros(
            (mask.shape[0], mask.shape[1], n_extra_row, mask.shape[3]), np.float32
        )
        mask = np.concatenate([mask, extension], axis=2)
        mask = mask.reshape(mask.shape[0] * mask.shape[1], mask.shape[2], mask.shape[3])
        mask = mask[:num_frames].astype(np.float32)

        waveforms = []
        for channel in range(mix.shape[1]):
            masked = mask[:, :, channel] * spec[:, :, channel]
            wave = inverse_stft(masked)
            waveforms.append(wave[frame_length : frame_length + mix.shape[0]])
        result[instrument] = np.stack(waveforms, axis=1).astype(np.float32)
    return result


def snr(a, b):
    n = min(a.shape[0], b.shape[0])
    x = a[:n].astype(np.float64)
    y = b[:n].astype(np.float64)
    diff = y - x
    rms = float(np.sqrt(np.mean(x**2)))
    err = float(np.sqrt(np.mean(diff**2)))
    return (20 * np.log10(rms / err) if err > 0 else float("inf")), float(np.max(np.abs(diff)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--mix", required=True)
    parser.add_argument("--cpp_mag", required=True)
    parser.add_argument("--epsilon", type=float, default=1e-7)
    args = parser.parse_args()

    mix = np.load(args.mix).astype(np.float32)
    base = np.abs(stft_frames(mix)).astype(np.float32)
    cpp = np.fromfile(args.cpp_mag, dtype=np.float32).reshape(-1, 2049, 2)
    n = min(base.shape[0], cpp.shape[0])
    base = base[:n]
    cpp = cpp[:n]

    rng = np.random.RandomState(7)
    noise = (rng.randn(*base.shape) * args.epsilon).astype(np.float32)
    perturbed = (base * (1.0 + noise)).astype(np.float32)

    print(f"model: {args.onnx}")
    results = {}
    for label, magnitude in (("numpy", base), ("cpp", cpp), ("perturbed", perturbed)):
        results[label] = run_pipeline(args.onnx, mix, magnitude)

    for instrument in results["numpy"]:
        line = [f"{instrument:14s}"]
        for label in ("cpp", "perturbed"):
            value, max_abs = snr(results["numpy"][instrument], results[label][instrument])
            line.append(f"{label}={value:7.1f}dB (max {max_abs:.2e})")
        print("  ".join(line))


if __name__ == "__main__":
    main()
