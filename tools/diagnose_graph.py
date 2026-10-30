"""Compare upstream Spleeter's graph against the rebuilt/exported graph.

Upstream (spleeter/model/__init__.py) builds the network on top of the
magnitude spectrogram produced by `_build_stft_feature`. This script compares
the per-instrument *masked spectrograms* (the model outputs) produced by

  * upstream's own EstimatorSpecBuilder graph, with the pretrained checkpoint
    restored into it (exactly what `spleeter.separator.Separator` does), and
  * the rebuilt Keras model used by export_onnx.py,

for the very same input spectrogram.

The two graphs must be built in *separate processes*: Keras auto-generates
layer names from a global counter, so building both in one process would shift
the names and break the checkpoint mapping.

Usage:
    python diagnose_graph.py --mode upstream --config <cfg> --checkpoint <ckpt> \
        --name 2stems --out upstream.npy
    python diagnose_graph.py --mode rebuilt  --config <cfg> --checkpoint <ckpt> \
        --name 2stems --out rebuilt.npy
    python diagnose_graph.py --mode compare  --out upstream.npy --out2 rebuilt.npy \
        --name 2stems
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

SEED = 20240607

# Use the Spleeter package installed in the environment; set $SPLEETER_SRC to
# a source checkout when it is not installed.
_SPLEETER_SRC = os.environ.get("SPLEETER_SRC", "")
if _SPLEETER_SRC and os.path.isdir(_SPLEETER_SRC):
    sys.path.insert(0, _SPLEETER_SRC)


def load_export_module():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_onnx.py")
    spec = importlib.util.spec_from_file_location("export_onnx", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_probe(frames, f_full, n_channels):
    rng = np.random.RandomState(SEED)
    return (
        rng.randn(frames, f_full, n_channels)
        + 1j * rng.randn(frames, f_full, n_channels)
    ).astype(np.complex64) * np.float32(0.05)


def numpy_stft(waveform, frame_length=4096, frame_step=1024):
    """NumPy reproduction of tf.signal.stft on the front-padded signal."""
    n_channels = waveform.shape[1]
    padded = np.concatenate(
        [np.zeros((frame_length, n_channels), np.float32), waveform], axis=0
    )
    total = padded.shape[0]
    num_frames = -(-total // frame_step)
    needed = (num_frames - 1) * frame_step + frame_length
    if needed > total:
        padded = np.concatenate(
            [padded, np.zeros((needed - total, n_channels), np.float32)], axis=0
        )
    n = np.arange(frame_length, dtype=np.float64)
    window = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / frame_length)).astype(np.float32)
    out = np.zeros((num_frames, frame_length // 2 + 1, n_channels), np.complex64)
    for i in range(num_frames):
        frame = padded[i * frame_step : i * frame_step + frame_length]
        out[i] = np.fft.rfft(frame * window[:, None], axis=0)
    return out


def load_input(args, config, f_full, n_channels):
    """Returns the complex STFT used as the shared network input."""
    if args.mix:
        mix = np.load(args.mix).astype(np.float32)
        return numpy_stft(mix, config["frame_length"], config["frame_step"])
    return make_probe(config["T"], f_full, n_channels)


def run_upstream(config, checkpoint, out_path, args):
    """Build upstream's graph and restore the checkpoint into it."""
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    from spleeter.model import EstimatorSpecBuilder

    n_channels = config["n_channels"]
    f_full = config["frame_length"] // 2 + 1
    stft_ph = tf.compat.v1.placeholder(
        tf.complex64, shape=(None, f_full, n_channels), name="mix_stft"
    )
    builder = EstimatorSpecBuilder({"mix_stft": stft_ph}, dict(config))
    outputs = builder.model_outputs

    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as session:
        saver.restore(session, checkpoint)
        probe = load_input(args, config, f_full, n_channels)
        values = session.run(list(outputs.values()), feed_dict={stft_ph: probe})
    np.save(out_path, np.array(values))
    print(f"upstream outputs saved to {out_path}: {[v.shape for v in values]}")


def run_rebuilt(config, checkpoint, out_path, args):
    """Build the rebuilt Keras model, assign the checkpoint, run the probe."""
    import tensorflow as tf

    export_onnx = load_export_module()
    model = export_onnx.build_model(config, os.path.splitext(os.path.basename(out_path))[0])
    assigned, missing = export_onnx.assign_checkpoint_weights(model, checkpoint)
    print(f"checkpoint: assigned={assigned} missing={missing}")
    if missing:
        raise SystemExit("checkpoint variable mismatch")

    n_channels = config["n_channels"]
    f_full = config["frame_length"] // 2 + 1
    f_used = config["F"]
    probe = load_input(args, config, f_full, n_channels)
    magnitude = np.abs(probe).astype(np.float32)
    pad = (-magnitude.shape[0]) % config["T"]
    if pad:
        magnitude = np.concatenate(
            [magnitude, np.zeros((pad,) + magnitude.shape[1:], np.float32)], axis=0
        )
    tiles = magnitude[:, :f_used, :].reshape(
        -1, config["T"], f_used, n_channels
    ).astype(np.float32)
    values = model.predict(tiles, verbose=0)
    if not isinstance(values, list):
        values = [values]
    np.save(out_path, np.array(values))
    print(f"rebuilt outputs saved to {out_path}: {[v.shape for v in values]}")


def run_compare(config, path_a, path_b):
    a = np.load(path_a)
    b = np.load(path_b)
    print(f"{'instrument':14s} {'max_abs':>12s} {'rel_max':>12s} {'snr_db':>8s}")
    for i, instrument in enumerate(config["instrument_list"]):
        x = a[i].astype(np.float64)
        y = b[i].astype(np.float64)
        diff = y - x
        max_abs = float(np.max(np.abs(diff)))
        peak = float(np.max(np.abs(x)))
        rms = float(np.sqrt(np.mean(x**2)))
        err = float(np.sqrt(np.mean(diff**2)))
        snr = 20 * np.log10(rms / err) if err > 0 else float("inf")
        print(
            f"{instrument:14s} {max_abs:12.3e} "
            f"{(max_abs / peak if peak else 0):12.3e} {snr:8.1f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["upstream", "rebuilt", "compare"])
    parser.add_argument("--config", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--name", default="model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--out2", default="")
    parser.add_argument("--mix", default="")
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as stream:
            config = json.load(stream)

    if args.mode == "upstream":
        run_upstream(config, args.checkpoint, args.out, args)
    elif args.mode == "rebuilt":
        run_rebuilt(config, args.checkpoint, args.out, args)
    else:
        run_compare(config, args.out, args.out2)


if __name__ == "__main__":
    main()
