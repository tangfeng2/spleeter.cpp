"""Probe exact TensorFlow signal semantics used by Spleeter.

Spleeter predict path:
  waveform (T, C) -> pad front with frame_length zeros -> stft -> transpose
  -> abs -> pad_and_partition(T) -> slice [:, :, :F, :]

We dump reference arrays so the C++ port can be validated bit-for-bit.
The output directory defaults to `test/ref` (git-ignored) so the dumped
arrays never end up next to the sources.
"""

import os
import sys

import numpy as np
import tensorflow as tf

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join("test", "ref")
os.makedirs(OUT, exist_ok=True)

FRAME_LENGTH = 4096
FRAME_STEP = 1024
F = 1024
T = 512
EPS = 1e-10

rng = np.random.RandomState(1234)
N = 300000  # samples
ch = 2
wave = (rng.randn(N, ch) * 0.1).astype(np.float32)

# ---------------------------------------------------------------- window
win = tf.signal.hann_window(FRAME_LENGTH, periodic=True, dtype=tf.float32).numpy()
np.save(os.path.join(OUT, "hann_periodic.npy"), win)

# periodic=False for reference
win_np = tf.signal.hann_window(FRAME_LENGTH, periodic=False, dtype=tf.float32).numpy()
np.save(os.path.join(OUT, "hann_symmetric.npy"), win_np)

# ---------------------------------------------------------------- stft
padded = tf.concat(
    [tf.zeros((FRAME_LENGTH, ch)), tf.constant(wave)], 0
)  # (N+4096, 2)

stft_c = tf.signal.stft(
    tf.transpose(padded),
    FRAME_LENGTH,
    FRAME_STEP,
    window_fn=lambda f, dtype: tf.signal.hann_window(f, periodic=True, dtype=dtype),
    pad_end=True,
)  # (2, frames, 2049) complex64

stft_t = tf.transpose(stft_c, perm=[1, 2, 0])  # (frames, 2049, 2)
print("stft_t shape:", stft_t.shape, stft_t.dtype)

np.save(
    os.path.join(OUT, "stft_complex.npy"),
    tf.stack([tf.math.real(stft_t), tf.math.imag(stft_t)], -1).numpy(),
)

# frame count check
print("expected frames (ceil(N+4096)/1024):", -(-(N + FRAME_LENGTH) // FRAME_STEP))

# ---------------------------------------------------------- pad_and_partition
spec = tf.abs(stft_t)  # (frames, 2049, 2)


def pad_and_partition(tensor, segment_len):
    tensor_size = tf.math.floormod(tf.shape(tensor)[0], segment_len)
    pad_size = tf.math.floormod(segment_len - tensor_size, segment_len)
    padded = tf.pad(tensor, [[0, pad_size]] + [[0, 0]] * (len(tensor.shape) - 1))
    split = (tf.shape(padded)[0] + segment_len - 1) // segment_len
    return tf.reshape(
        padded, tf.concat([[split, segment_len], tf.shape(padded)[1:]], axis=0)
    )


parts = pad_and_partition(spec, T)
print("partitioned:", parts.shape)
parts = parts[:, :, :F, :]
np.save(os.path.join(OUT, "spec_partition.npy"), parts.numpy())

# ---------------------------------------------------------------- inverse stft
# Reproduce spleeter _inverse_stft exactly.
def inverse_stft_like_spleeter(stft_t):
    inversed = (
        tf.signal.inverse_stft(
            tf.transpose(stft_t, perm=[2, 0, 1]),
            FRAME_LENGTH,
            FRAME_STEP,
            window_fn=lambda frame_length, dtype: tf.signal.hann_window(
                frame_length, periodic=True, dtype=dtype
            ),
        )
        * (2.0 / 3.0)
    )
    reshaped = tf.transpose(inversed)
    time_crop = N
    return reshaped[FRAME_LENGTH : FRAME_LENGTH + time_crop, :]


out = inverse_stft_like_spleeter(stft_t)
print("inverse_stft out shape:", out.shape)
np.save(os.path.join(OUT, "istft_identity.npy"), out.numpy())

# ---------------------------------------------------------------- envelope probe
# Determine the exact normalization envelope TF uses for inverse_stft:
# compare overlap-add of (irfft(frames) * window) against TF output.
frames = tf.signal.irfft(tf.transpose(stft_t, perm=[2, 0, 1]), FRAME_LENGTH)
print("irfft frames shape:", frames.shape)

# manual overlap-add with window, then derive envelope = oa / tf_out
def overlap_add(x, frame_step):
    n = x.shape[0]
    total = (n - 1) * frame_step + x.shape[1]
    out = np.zeros(total, dtype=np.float64)
    for i in range(n):
        out[i * frame_step : i * frame_step + x.shape[1]] += x[i]
    return out


oa = overlap_add(frames.numpy()[0].astype(np.float64), FRAME_STEP)
w2 = overlap_add(np.tile((win.astype(np.float64) ** 2), (1, 1)), FRAME_STEP)
tf_out = out.numpy()[:, 0].astype(np.float64)
crop = slice(FRAME_LENGTH, FRAME_LENGTH + N)

# envelope candidates
cand = {
    "sum_w2": w2[crop],
    "const_1.5": np.full(N, 1.5),
    "const_1.0": np.full(N, 1.0),
}
for name, env in cand.items():
    pred = oa[crop] / env * (2.0 / 3.0)
    err = np.max(np.abs(pred - tf_out))
    rel = err / max(1e-12, np.max(np.abs(tf_out)))
    print(f"envelope {name}: max_abs_err={err:.3e} rel={rel:.3e}")

# also try: oa without windowing divided by sum of w2
print("first 8 tf_out:", tf_out[:8])
print("first 8 oa   :", oa[crop][:8])
print("ratio        :", (tf_out / np.where(np.abs(oa[crop]) > 1e-9, oa[crop], 1))[:8])
