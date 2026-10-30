"""Run the Spleeter DSP over the exported ONNX model, tile by tile.

The C++ port feeds one 512-frame tile per forward pass; this script does the
same so that the two can be compared without the batch composition being a
confounding factor.
"""

import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_masks import inverse_stft, pad_and_partition, stft_frames  # noqa: E402

EPSILON = 1e-10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--mix", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    with open(args.onnx.replace(".onnx", "_meta.json"), "r", encoding="utf-8") as stream:
        meta = json.load(stream)
    instruments = meta["instrument_list"]
    sep_exp = meta["separation_exponent"]
    f_used = meta["F"]
    frame_length = meta["frame_length"]

    mix = np.load(args.mix).astype(np.float32)
    spec = stft_frames(mix)
    num_frames = spec.shape[0]
    tiles = pad_and_partition(np.abs(spec).astype(np.float32), meta["T"])[:, :, :f_used, :]

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    collected = []
    for start in range(0, tiles.shape[0], args.batch):
        chunk = tiles[start : start + args.batch].astype(np.float32)
        collected.append(session.run(None, {"input": chunk}))
    outputs = [
        np.concatenate([chunk[i] for chunk in collected], axis=0)
        for i in range(len(collected[0]))
    ]

    n_extra_row = frame_length // 2 + 1 - f_used
    output_sum = np.zeros(tiles.shape, np.float32)
    for out in outputs:
        output_sum += (out.astype(np.float32) ** sep_exp).astype(np.float32)
    output_sum += np.float32(EPSILON)

    os.makedirs(args.out, exist_ok=True)
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
        wave = np.stack(waveforms, axis=1).astype(np.float32)
        np.save(os.path.join(args.out, f"{instrument}.npy"), wave)
        print(f"{instrument}: {wave.shape}")


if __name__ == "__main__":
    main()
