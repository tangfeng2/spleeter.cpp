"""Validate the ONNX model + DSP pipeline against real Python Spleeter output.

This reproduces Spleeter's predict path exactly (see spleeter/model/__init__.py):
  waveform -> front zero pad -> STFT -> abs -> pad_and_partition -> UNet
  -> masks -> extend mask -> masked STFT -> iSTFT -> crop

and compares the result sample-for-sample against the reference stems produced
by the real Spleeter package.
"""

import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort

FRAME_LENGTH = 4096
FRAME_STEP = 1024
T = 512
F = 1024
EPSILON = 1e-10
WINDOW_COMPENSATION_FACTOR = 2.0 / 3.0


def hann_periodic(length):
    """tf.signal.hann_window(length, periodic=True)."""
    n = np.arange(length, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)).astype(np.float32)


def stft_frames(waveform):
    """Reproduce tf.signal.stft(..., pad_end=True) on the front-padded signal.

    waveform: (n_samples, n_channels) float32
    returns complex64 (num_frames, 2049, n_channels)
    """
    n_channels = waveform.shape[1]
    padded = np.concatenate(
        [np.zeros((FRAME_LENGTH, n_channels), np.float32), waveform], axis=0
    )
    total = padded.shape[0]
    num_frames = -(-total // FRAME_STEP)  # ceil division, as TF does with pad_end
    needed = (num_frames - 1) * FRAME_STEP + FRAME_LENGTH
    if needed > total:
        padded = np.concatenate(
            [padded, np.zeros((needed - total, n_channels), np.float32)], axis=0
        )
    window = hann_periodic(FRAME_LENGTH)[:, None]  # (4096, 1) broadcasts over channels
    out = np.zeros((num_frames, FRAME_LENGTH // 2 + 1, n_channels), np.complex64)
    for i in range(num_frames):
        frame = padded[i * FRAME_STEP : i * FRAME_STEP + FRAME_LENGTH]
        out[i] = np.fft.rfft(frame * window, axis=0)
    return out


def pad_and_partition(tensor, segment_len):
    """spleeter.utils.tensor.pad_and_partition (axis 0)."""
    size = tensor.shape[0]
    pad = (-size) % segment_len
    if pad:
        tensor = np.concatenate(
            [tensor, np.zeros((pad,) + tensor.shape[1:], tensor.dtype)], axis=0
        )
    n_seg = tensor.shape[0] // segment_len
    return tensor.reshape(n_seg, segment_len, *tensor.shape[1:])


def inverse_stft(masked):
    """Reproduce spleeter's _inverse_stft for one channel.

    masked: complex64 (num_frames, 2049)
    returns float32 waveform
    """
    window = hann_periodic(FRAME_LENGTH)
    frames = np.fft.irfft(masked, n=FRAME_LENGTH, axis=1).astype(np.float32)
    frames *= window
    num_frames = frames.shape[0]
    total = (num_frames - 1) * FRAME_STEP + FRAME_LENGTH
    out = np.zeros(total, np.float32)
    for i in range(num_frames):
        out[i * FRAME_STEP : i * FRAME_STEP + FRAME_LENGTH] += frames[i]
    out *= WINDOW_COMPENSATION_FACTOR
    return out


def run(onnx_path, mix_path, out_dir):
    with open(onnx_path.replace(".onnx", "_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    instruments = meta["instrument_list"]
    sep_exp = meta["separation_exponent"]
    frame_length = meta["frame_length"]
    frame_step = meta["frame_step"]

    mix = np.load(mix_path)
    n_samples = mix.shape[0]
    print(f"mix: {mix.shape}")

    spec_complex = stft_frames(mix)
    num_frames = spec_complex.shape[0]
    print(f"stft frames: {num_frames}")

    magnitude = np.abs(spec_complex).astype(np.float32)
    tiles = pad_and_partition(magnitude, T)[:, :, :F, :]
    print(f"tiles: {tiles.shape}")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input": tiles.astype(np.float32)})
    print(f"onnx outputs: {[o.shape for o in outputs]}")

    # Masks, exactly as spleeter's _build_masks.
    n_extra_row = frame_length // 2 + 1 - F
    # TF computes the whole mask chain in float32.
    output_sum = np.zeros(tiles.shape, np.float32)
    for out in outputs:
        output_sum += (out.astype(np.float32) ** sep_exp).astype(np.float32)
    output_sum += np.float32(EPSILON)

    results = {}
    for index, instrument in enumerate(instruments):
        out = outputs[index].astype(np.float32)
        mask = ((out**sep_exp).astype(np.float32) + np.float32(EPSILON / len(outputs)))
        mask = (mask / output_sum).astype(np.float32)
        # Extend mask with zeros over the high frequency band (axis 2 = freq).
        extension = np.zeros(
            (mask.shape[0], mask.shape[1], n_extra_row, mask.shape[3]), np.float32
        )
        mask = np.concatenate([mask, extension], axis=2)
        mask = mask.reshape(mask.shape[0] * mask.shape[1], mask.shape[2], mask.shape[3])
        mask = mask[:num_frames].astype(np.float32)

        waveforms = []
        for channel in range(mix.shape[1]):
            masked = mask[:, :, channel] * spec_complex[:, :, channel]
            wave = inverse_stft(masked)
            waveforms.append(wave[frame_length : frame_length + n_samples])
        results[instrument] = np.stack(waveforms, axis=1).astype(np.float32)
        print(f"{instrument}: {results[instrument].shape}")

    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for instrument, data in results.items():
        np.save(os.path.join(out_dir, f"{instrument}.npy"), data)
        ref_path = os.path.join(os.path.dirname(mix_path), f"{instrument}.npy")
        if os.path.exists(ref_path):
            ref = np.load(ref_path)
            n = min(ref.shape[0], data.shape[0])
            diff = data[:n] - ref[:n]
            max_abs = float(np.max(np.abs(diff)))
            denom = float(np.max(np.abs(ref[:n])))
            signal_rms = float(np.sqrt(np.mean(ref[:n].astype(np.float64) ** 2)))
            noise_rms = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
            snr = 20 * np.log10(signal_rms / noise_rms) if noise_rms > 0 else float("inf")
            report[instrument] = {
                "max_abs_diff": max_abs,
                "max_abs_ref": denom,
                "relative_max": max_abs / denom if denom else 0.0,
                "snr_db": float(snr),
                "ref_len": int(ref.shape[0]),
                "out_len": int(data.shape[0]),
            }
            print(
                f"PARITY {instrument}: max_abs_diff={max_abs:.3e} "
                f"rel={max_abs/denom if denom else 0:.3e} snr={snr:.1f}dB"
            )

    with open(os.path.join(out_dir, "_parity.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--mix", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.onnx, args.mix, args.out)
