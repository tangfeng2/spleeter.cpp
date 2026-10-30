"""Export Spleeter pretrained UNet models to ONNX.

The layer construction mirrors `spleeter/model/functions/unet.py` *exactly*
(same order, same padding, same activation placement) so that the exported
ONNX graph is numerically identical to the TensorFlow estimator path.

The pretrained checkpoints store variables under Keras layer names
(`conv2d/kernel`, `batch_normalization/gamma`, ...), so we rebuild the layers
in the same order to obtain identical auto-generated names, then assign the
checkpoint tensors by name.

Usage:
    python export_onnx.py --config spleeter/resources/4stems.json \
        --checkpoint models/4stems/model --out models/onnx/4stems.onnx
"""

import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    ELU,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Cropping2D,
    LeakyReLU,
    Multiply,
    ReLU,
    Softmax,
)
from tensorflow.compat.v1.keras.initializers import he_uniform

# Constants taken from spleeter/model/__init__.py
EPSILON = 1e-10
WINDOW_COMPENSATION_FACTOR = 2.0 / 3.0

CONV_N_FILTERS = [16, 32, 64, 128, 256, 512]


def get_conv_activation_layer(params):
    conv_activation = str(params.get("conv_activation"))
    if conv_activation == "ReLU":
        return ReLU()
    elif conv_activation == "ELU":
        return ELU()
    return LeakyReLU(0.2)


def get_deconv_activation_layer(params):
    deconv_activation = str(params.get("deconv_activation"))
    if deconv_activation == "LeakyReLU":
        return LeakyReLU(0.2)
    elif deconv_activation == "ELU":
        return ELU()
    return ReLU()


def apply_unet(input_tensor, output_name="output", params=None, output_mask_logit=False):
    """Mirror of spleeter.model.functions.unet.apply_unet.

    Differences from upstream, all of them inference-mode only:
      * BatchNormalization is built with training=False so that moving
        statistics are used (identical to the estimator predict path).
      * Dropout is a no-op in inference mode, so it is omitted entirely.
    """
    if params is None:
        params = {}
    kernel_initializer = he_uniform(seed=50)
    conv_activation_layer = get_conv_activation_layer(params)
    deconv_activation_layer = get_deconv_activation_layer(params)

    def conv2d_factory(filters, kernel_size):
        return Conv2D(
            filters,
            kernel_size,
            strides=(2, 2),
            padding="same",
            kernel_initializer=kernel_initializer,
        )

    def conv2d_transpose_factory(filters, kernel_size, strides=(2, 2)):
        # TensorFlow's "same" padding for a transposed convolution with an
        # even kernel and stride 2 is implemented as a VALID convolution whose
        # result is cropped by 1 sample at the start and 2 at the end.
        #
        # ONNX models that "same" padding as asymmetric ConvTranspose pads,
        # which the ONNX Runtime CUDA execution provider refuses to run (it
        # falls back to the CPU for every such node). Expressing the same
        # operation as VALID + an explicit crop keeps the arithmetic identical
        # while producing symmetric padding that the CUDA EP supports.
        return Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,
            padding="valid",
            kernel_initializer=kernel_initializer,
        )

    def batch_norm(tensor):
        # `training=False` reproduces the estimator predict path: moving
        # statistics are used instead of batch statistics.
        return BatchNormalization(axis=-1)(tensor, training=False)

    def upsample(tensor, filters, kernel_size=(5, 5)):
        """Transposed convolution with TensorFlow's "same" semantics.

        TF computes the VALID result and crops one sample at the start and
        two at the end of each spatial axis. Doing the crop explicitly keeps
        the ONNX ConvTranspose padding symmetric so the CUDA execution
        provider can run it.
        """
        valid = conv2d_transpose_factory(filters, kernel_size)(tensor)
        return Cropping2D(cropping=((1, 2), (1, 2)))(valid)

    # --- Encoder -------------------------------------------------------
    conv1 = conv2d_factory(CONV_N_FILTERS[0], (5, 5))(input_tensor)
    batch1 = batch_norm(conv1)
    rel1 = conv_activation_layer(batch1)

    conv2 = conv2d_factory(CONV_N_FILTERS[1], (5, 5))(rel1)
    batch2 = batch_norm(conv2)
    rel2 = conv_activation_layer(batch2)

    conv3 = conv2d_factory(CONV_N_FILTERS[2], (5, 5))(rel2)
    batch3 = batch_norm(conv3)
    rel3 = conv_activation_layer(batch3)

    conv4 = conv2d_factory(CONV_N_FILTERS[3], (5, 5))(rel3)
    batch4 = batch_norm(conv4)
    rel4 = conv_activation_layer(batch4)

    conv5 = conv2d_factory(CONV_N_FILTERS[4], (5, 5))(rel4)
    batch5 = batch_norm(conv5)
    rel5 = conv_activation_layer(batch5)

    conv6 = conv2d_factory(CONV_N_FILTERS[5], (5, 5))(rel5)
    batch6 = batch_norm(conv6)
    # NOTE: upstream computes `conv_activation_layer(batch6)` and discards the
    # result (`_ = ...`). `up1` is therefore built from the *raw* conv6 output,
    # bypassing batch6 entirely. We must reproduce this exactly.
    conv6_out = conv_activation_layer(batch6)

    # --- Decoder -------------------------------------------------------
    up1 = upsample(conv6, CONV_N_FILTERS[4])
    up1 = deconv_activation_layer(up1)
    batch7 = batch_norm(up1)
    merge1 = Concatenate(axis=-1)([conv5, batch7])

    up2 = upsample(merge1, CONV_N_FILTERS[3])
    up2 = deconv_activation_layer(up2)
    batch8 = batch_norm(up2)
    merge2 = Concatenate(axis=-1)([conv4, batch8])

    up3 = upsample(merge2, CONV_N_FILTERS[2])
    up3 = deconv_activation_layer(up3)
    batch9 = batch_norm(up3)
    merge3 = Concatenate(axis=-1)([conv3, batch9])

    up4 = upsample(merge3, CONV_N_FILTERS[1])
    up4 = deconv_activation_layer(up4)
    batch10 = batch_norm(up4)
    merge4 = Concatenate(axis=-1)([conv2, batch10])

    up5 = upsample(merge4, CONV_N_FILTERS[0])
    up5 = deconv_activation_layer(up5)
    batch11 = batch_norm(up5)
    merge5 = Concatenate(axis=-1)([conv1, batch11])

    up6 = upsample(merge5, 1)
    up6 = deconv_activation_layer(up6)
    batch12 = batch_norm(up6)

    if not output_mask_logit:
        up7 = Conv2D(
            2,
            (4, 4),
            dilation_rate=(2, 2),
            activation="sigmoid",
            padding="same",
            kernel_initializer=kernel_initializer,
        )(batch12)
        return Multiply(name=output_name)([up7, input_tensor])

    return Conv2D(
        2,
        (4, 4),
        dilation_rate=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(batch12)


def build_model(config, name):
    """Build the multi-instrument model exactly like spleeter does."""
    instruments = config["instrument_list"]
    params = config.get("model", {}).get("params", {}) or {}
    model_type = config.get("model", {}).get("type", "unet.unet")

    tf_input = tf.keras.Input(
        shape=(config["T"], config["F"], config["n_channels"]),
        batch_size=None,
        name="input",
    )

    if model_type == "unet.softmax_unet":
        logit_mask_list = []
        for instrument in instruments:
            out_name = f"{instrument}_spectrogram"
            logit_mask_list.append(
                apply_unet(tf_input, out_name, params, output_mask_logit=True)
            )
        masks = Softmax(axis=4)(tf.stack(logit_mask_list, axis=4))
        outputs = []
        for i, instrument in enumerate(instruments):
            out_name = f"{instrument}_spectrogram"
            outputs.append(Multiply(name=out_name)([masks[..., i], tf_input]))
    else:
        outputs = []
        for instrument in instruments:
            out_name = f"{instrument}_spectrogram"
            outputs.append(
                apply_unet(tf_input, out_name, params, output_mask_logit=False)
            )

    model = tf.keras.Model(tf_input, outputs, name=name)
    return model


def assign_checkpoint_weights(model, checkpoint_path):
    """Assign pretrained checkpoint tensors onto the rebuilt Keras model."""
    reader = tf.train.load_checkpoint(checkpoint_path)
    available = set(reader.get_variable_to_shape_map().keys())

    assigned, missing = 0, []
    for layer in model.layers:
        if not layer.weights:
            continue
        values = []
        for weight in layer.weights:
            key = weight.name.split(":")[0]
            if key not in available:
                missing.append(key)
                values.append(np.zeros(weight.shape, dtype=np.float32))
            else:
                values.append(reader.get_tensor(key))
                assigned += 1
        layer.set_weights(values)
    return assigned, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as stream:
        config = json.load(stream)

    name = os.path.splitext(os.path.basename(args.out))[0]
    model = build_model(config, name)

    assigned, missing = assign_checkpoint_weights(model, args.checkpoint)
    print(f"assigned {assigned} checkpoint tensors, missing={missing}")
    if missing:
        raise SystemExit("checkpoint variable mismatch -- aborting export")

    # Deterministic verification input.
    rng = np.random.RandomState(args.seed)
    probe = rng.randn(1, config["T"], config["F"], config["n_channels"]).astype(
        np.float32
    )
    tf_outputs = model.predict(probe, verbose=0)
    if not isinstance(tf_outputs, list):
        tf_outputs = [tf_outputs]
    np.save(os.path.splitext(args.out)[0] + "_tf_probe.npy", np.array(tf_outputs))

    import tf2onnx

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=[
            tf.TensorSpec(
                (None, config["T"], config["F"], config["n_channels"]),
                tf.float32,
                name="input",
            )
        ],
        opset=args.opset,
        target="cpu",
    )
    print("onnx inputs :", [i.name for i in model_proto.graph.input])
    print("onnx outputs:", [o.name for o in model_proto.graph.output])
    with open(args.out, "wb") as stream:
        stream.write(model_proto.SerializeToString())
    print(f"wrote {args.out}")

    # Numeric check ONNX Runtime (CPU) vs TensorFlow.
    import onnxruntime as ort

    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {"input": probe})
    worst = 0.0
    for i, (a, b) in enumerate(zip(tf_outputs, onnx_outputs)):
        err = float(np.max(np.abs(a - b)))
        worst = max(worst, err)
        print(f"output {i}: max_abs_diff={err:.3e}")
    print(f"worst max_abs_diff={worst:.3e}")

    # Persist the config next to the model so the C++ port uses the same values.
    meta = {
        "instrument_list": config["instrument_list"],
        "sample_rate": config["sample_rate"],
        "frame_length": config["frame_length"],
        "frame_step": config["frame_step"],
        "T": config["T"],
        "F": config["F"],
        "n_channels": config["n_channels"],
        "separation_exponent": config["separation_exponent"],
        "mask_extension": config["mask_extension"],
        "model_type": config.get("model", {}).get("type", "unet.unet"),
    }
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
