"""Verify TF 'same' Conv2DTranspose == VALID Conv2DTranspose + Cropping2D((1,2),(1,2)).

TensorFlow implements "same" padding for a transposed convolution by computing
the VALID result and cropping it. For an even kernel (5) and stride 2 the crop
removes one sample at the start and two at the end of each spatial axis.

The ONNX Runtime CUDA execution provider cannot run ConvTranspose with
asymmetric padding, so the exported model uses VALID + an explicit crop. This
script proves the two are numerically identical.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Cropping2D

rng = np.random.RandomState(0)

for height, width in [(8, 16), (16, 32), (64, 128), (128, 256), (256, 512)]:
    x = rng.randn(1, height, width, 3).astype(np.float32)

    same = Conv2DTranspose(7, (5, 5), strides=(2, 2), padding="same")
    valid = Conv2DTranspose(7, (5, 5), strides=(2, 2), padding="valid")
    same.build(x.shape)
    valid.build(x.shape)
    valid.set_weights(same.get_weights())

    a = same(x).numpy()
    b = Cropping2D(cropping=((1, 2), (1, 2)))(valid(x)).numpy()

    print(
        f"in=({height},{width}) same_out={a.shape} valid+crop={b.shape} "
        f"max_abs_diff={np.max(np.abs(a - b)):.3e}"
    )
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.max(np.abs(a - b)) < 1e-6

print("RESULT: PASS")
