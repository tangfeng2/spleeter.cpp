"""Compare upstream Spleeter's masks (and waveforms) against the port's masks.

Builds upstream's *full* predict graph (waveform -> STFT -> UNet -> masks ->
masked STFT -> iSTFT), restores the pretrained checkpoint into it, and dumps

  * the per-instrument masks          (n_frames, 2049, 2)
  * the final waveforms               (n_samples, 2)

for the reference mix. The same masks are then recomputed from the exported
ONNX model with a NumPy reproduction of the Spleeter DSP, so the two can be
compared element-wise.

Modes (run in separate processes - Keras layer naming is process-global):
    upstream : dump upstream masks + waveforms
    numpy    : dump masks computed from ONNX + NumPy DSP
    compare  : diff the two
"""

import argparse
import json
import os
import sys

import numpy as np

# Use the Spleeter package installed in the environment; set $SPLEETER_SRC to
# a source checkout when it is not installed.
_SPLEETER_SRC = os.environ.get("SPLEETER_SRC", "")
if _SPLEETER_SRC and os.path.isdir(_SPLEETER_SRC):
    sys.path.insert(0, _SPLEETER_SRC)

FRAME_LENGTH = 4096
FRAME_STEP = 1024
EPSILON = 1e-10
WINDOW_COMPENSATION_FACTOR = 2.0 / 3.0


def hann_periodic(length):
    n = np.arange(length, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)).astype(np.float32)


def stft_frames(waveform):
    n_channels = waveform.shape[1]
    padded = np.concatenate(
        [np.zeros((FRAME_LENGTH, n_channels), np.float32), waveform], axis=0
    )
    total = padded.shape[0]
    num_frames = -(-total // FRAME_STEP)
    needed = (num_frames - 1) * FRAME_STEP + FRAME_LENGTH
    if needed > total:
        padded = np.concatenate(
            [padded, np.zeros((needed - total, n_channels), np.float32)], axis=0
        )
    window = hann_periodic(FRAME_LENGTH)[:, None]
    out = np.zeros((num_frames, FRAME_LENGTH // 2 + 1, n_channels), np.complex64)
    for i in range(num_frames):
        frame = padded[i * FRAME_STEP : i * FRAME_STEP + FRAME_LENGTH]
        out[i] = np.fft.rfft(frame * window, axis=0)
    return out


def pad_and_partition(tensor, segment_len):
    size = tensor.shape[0]
    pad = (-size) % segment_len
    if pad:
        tensor = np.concatenate(
            [tensor, np.zeros((pad,) + tensor.shape[1:], tensor.dtype)], axis=0
        )
    return tensor.reshape(tensor.shape[0] // segment_len, segment_len, *tensor.shape[1:])


def inverse_stft(masked):
    window = hann_periodic(FRAME_LENGTH)
    frames = np.fft.irfft(masked, n=FRAME_LENGTH, axis=1).astype(np.float32)
    frames *= window
    total = (frames.shape[0] - 1) * FRAME_STEP + FRAME_LENGTH
    out = np.zeros(total, np.float32)
    for i in range(frames.shape[0]):
        out[i * FRAME_STEP : i * FRAME_STEP + FRAME_LENGTH] += frames[i]
    out *= WINDOW_COMPENSATION_FACTOR
    return out


def run_upstream(config, checkpoint, mix_path, out_dir):
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    from spleeter.model import EstimatorSpecBuilder

    n_channels = config["n_channels"]
    wave_ph = tf.compat.v1.placeholder(
        tf.float32, shape=(None, n_channels), name="waveform"
    )
    builder = EstimatorSpecBuilder({"waveform": wave_ph}, dict(config))
    masks = builder.masks
    outputs = builder.outputs
    model_outputs = builder.model_outputs
    stft_feature = builder.stft_feature

    mix = np.load(mix_path).astype(np.float32)
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as session:
        saver.restore(session, checkpoint)
        mask_values = session.run(list(masks.values()), feed_dict={wave_ph: mix})
        wave_values = session.run(list(outputs.values()), feed_dict={wave_ph: mix})
        model_values = session.run(
            list(model_outputs.values()), feed_dict={wave_ph: mix}
        )
        stft_values = session.run(stft_feature, feed_dict={wave_ph: mix})

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "_stft.npy"), stft_values)
    instruments = config["instrument_list"]
    for i, instrument in enumerate(instruments):
        np.save(os.path.join(out_dir, f"{instrument}_mask.npy"), mask_values[i])
        np.save(os.path.join(out_dir, f"{instrument}_wave.npy"), wave_values[i])
        np.save(os.path.join(out_dir, f"{instrument}_modelout.npy"), model_values[i])
        print(f"upstream {instrument}: mask {mask_values[i].shape} wave {wave_values[i].shape}")


def run_numpy(config, onnx_path, mix_path, out_dir, stft_path=""):
    import onnxruntime as ort

    with open(onnx_path.replace(".onnx", "_meta.json"), "r", encoding="utf-8") as stream:
        meta = json.load(stream)
    instruments = meta["instrument_list"]
    sep_exp = meta["separation_exponent"]
    f_used = meta["F"]

    mix = np.load(mix_path).astype(np.float32)
    if stft_path:
        spec = np.load(stft_path).astype(np.complex64)
    else:
        spec = stft_frames(mix)
    num_frames = spec.shape[0]
    magnitude = np.abs(spec).astype(np.float32)
    tiles = pad_and_partition(magnitude, meta["T"])[:, :, :f_used, :]

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input": tiles.astype(np.float32)})

    n_extra_row = FRAME_LENGTH // 2 + 1 - f_used
    output_sum = np.zeros(tiles.shape, np.float32)
    for out in outputs:
        output_sum += (out.astype(np.float32) ** sep_exp).astype(np.float32)
    output_sum += np.float32(EPSILON)

    os.makedirs(out_dir, exist_ok=True)
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
        np.save(os.path.join(out_dir, f"{instrument}_mask.npy"), mask)

        waveforms = []
        for channel in range(mix.shape[1]):
            masked = mask[:, :, channel] * spec[:, :, channel]
            wave = inverse_stft(masked)
            waveforms.append(wave[FRAME_LENGTH : FRAME_LENGTH + mix.shape[0]])
        wave = np.stack(waveforms, axis=1).astype(np.float32)
        np.save(os.path.join(out_dir, f"{instrument}_wave.npy"), wave)
        print(f"numpy {instrument}: mask {mask.shape} wave {wave.shape}")


def run_compare(config, dir_a, dir_b):
    for instrument in config["instrument_list"]:
        for kind in ("mask", "wave"):
            a = np.load(os.path.join(dir_a, f"{instrument}_{kind}.npy"))
            b = np.load(os.path.join(dir_b, f"{instrument}_{kind}.npy"))
            n = min(a.shape[0], b.shape[0])
            x = a[:n].astype(np.float64)
            y = b[:n].astype(np.float64)
            diff = y - x
            max_abs = float(np.max(np.abs(diff)))
            peak = float(np.max(np.abs(x)))
            rms = float(np.sqrt(np.mean(x**2)))
            err = float(np.sqrt(np.mean(diff**2)))
            snr = 20 * np.log10(rms / err) if err > 0 else float("inf")
            print(
                f"{instrument:12s} {kind:5s} max_abs={max_abs:.3e} "
                f"rel={max_abs / peak if peak else 0:.3e} snr={snr:7.1f}dB "
                f"shape_up={a.shape} shape_np={b.shape}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["upstream", "numpy", "compare"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--onnx", default="")
    parser.add_argument("--mix", default="")
    parser.add_argument("--dir_a", default="")
    parser.add_argument("--dir_b", default="")
    parser.add_argument("--stft", default="")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as stream:
        config = json.load(stream)

    if args.mode == "upstream":
        run_upstream(config, args.checkpoint, args.mix, args.dir_a)
    elif args.mode == "numpy":
        run_numpy(config, args.onnx, args.mix, args.dir_b, args.stft)
    else:
        run_compare(config, args.dir_a, args.dir_b)


if __name__ == "__main__":
    main()
