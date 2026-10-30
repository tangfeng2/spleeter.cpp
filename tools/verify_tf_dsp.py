"""Verify the exact TF STFT/iSTFT semantics against a NumPy re-implementation.

Critical case: signal lengths that are NOT multiples of frame_step, so the
ceil() frame-count rule of tf.signal.frame(pad_end=True) is actually
exercised.
"""

import os
import sys

import numpy as np
import tensorflow as tf

FRAME_LENGTH = 4096
FRAME_STEP = 1024

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def hann_periodic(length):
    n = np.arange(length, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)).astype(np.float32)


def np_stft(waveform):
    """(n_samples, n_channels) -> complex64 (num_frames, 2049, n_channels)."""
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


def np_istft(masked):
    """(num_frames, 2049) complex -> float32 waveform (no compensation factor)."""
    window = hann_periodic(FRAME_LENGTH)
    frames = np.fft.irfft(masked, n=FRAME_LENGTH, axis=1).astype(np.float32)
    frames *= window
    num_frames = frames.shape[0]
    total = (num_frames - 1) * FRAME_STEP + FRAME_LENGTH
    out = np.zeros(total, np.float32)
    for i in range(num_frames):
        out[i * FRAME_STEP : i * FRAME_STEP + FRAME_LENGTH] += frames[i]
    return out


def main():
    rng = np.random.RandomState(7)
    ok = True
    for n in (100000, 123457, 200001, 65537, 300000):
        wave = (rng.randn(n, 2) * 0.1).astype(np.float32)

        padded = tf.concat(
            [tf.zeros((FRAME_LENGTH, 2)), tf.constant(wave)], 0
        )
        tf_stft = tf.signal.stft(
            tf.transpose(padded),
            FRAME_LENGTH,
            FRAME_STEP,
            window_fn=lambda f, dtype: tf.signal.hann_window(
                f, periodic=True, dtype=dtype
            ),
            pad_end=True,
        )
        tf_stft = tf.transpose(tf_stft, perm=[1, 2, 0]).numpy()

        my_stft = np_stft(wave)
        same_shape = tf_stft.shape == my_stft.shape
        err = (
            float(np.max(np.abs(tf_stft - my_stft))) if same_shape else float("nan")
        )
        print(
            f"n={n:>7}  tf_frames={tf_stft.shape[0]:>5} "
            f"np_frames={my_stft.shape[0]:>5}  shape_ok={same_shape}  stft_max_err={err:.3e}"
        )
        ok = ok and same_shape and err < 1e-3

        # inverse stft round trip on the (unmasked) spectrum
        tf_inv = tf.signal.inverse_stft(
            tf.transpose(tf.constant(tf_stft), perm=[2, 0, 1]),
            FRAME_LENGTH,
            FRAME_STEP,
            window_fn=lambda fl, dtype: tf.signal.hann_window(
                fl, periodic=True, dtype=dtype
            ),
        ).numpy()
        my_inv = np_istft(tf_stft[:, :, 0])
        err_inv = float(np.max(np.abs(tf_inv[0] - my_inv)))
        print(
            f"          istft tf_len={tf_inv.shape[1]:>8} "
            f"np_len={my_inv.shape[0]:>8}  istft_max_err={err_inv:.3e}"
        )
        ok = ok and err_inv < 1e-3

    # Hann window check.
    tf_win = tf.signal.hann_window(FRAME_LENGTH, periodic=True, dtype=tf.float32).numpy()
    win_err = float(np.max(np.abs(tf_win - hann_periodic(FRAME_LENGTH))))
    print(f"hann periodic max_err={win_err:.3e}")
    ok = ok and win_err < 1e-7

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
