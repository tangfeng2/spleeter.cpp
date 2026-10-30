"""Run the reference Python Spleeter and dump ground truth for the C++ port.

Produces, for a given input file and model config:
  <out>/python/<instrument>.npy   float32 (n_samples, 2) separated waveform
  <out>/python/_mix.npy           float32 (n_samples, 2) input waveform
  <out>/python/_meta.json         lengths / sample rate

The separated waveform is captured *before* the codec round trip so the C++
implementation can be compared sample-for-sample.
"""

import argparse
import json
import os
import sys

import numpy as np

# The Spleeter package is used as the reference implementation. Use the one
# installed in the environment; if it is not installed, point $SPLEETER_SRC at
# a source checkout (the directory that contains the `spleeter` package).
_SPLEETER_SRC = os.environ.get("SPLEETER_SRC", "")
if _SPLEETER_SRC and os.path.isdir(_SPLEETER_SRC):
    sys.path.insert(0, _SPLEETER_SRC)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="4stems")
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--offset", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from spleeter.separator import Separator
    from spleeter.audio.adapter import AudioAdapter

    adapter = AudioAdapter.default()
    waveform, sample_rate = adapter.load(
        args.input,
        offset=args.offset,
        duration=args.duration,
        sample_rate=44100,
    )
    print(f"loaded waveform {waveform.shape} @ {sample_rate} Hz")

    separator = Separator(f"spleeter:{args.model}")
    sources = separator.separate(waveform, args.input)

    np.save(os.path.join(args.out, "_mix.npy"), waveform)
    meta = {"sample_rate": sample_rate, "n_samples": int(waveform.shape[0])}
    for instrument, data in sources.items():
        data = np.asarray(data)
        np.save(os.path.join(args.out, f"{instrument}.npy"), data)
        meta[instrument] = {"n_samples": int(data.shape[0]), "dtype": str(data.dtype)}
        print(f"{instrument}: {data.shape} {data.dtype}")

    with open(os.path.join(args.out, "_meta.json"), "w", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2)


if __name__ == "__main__":
    main()
