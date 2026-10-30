# spleeter.cpp — C++ port of Deezer Spleeter

A C++17 re-implementation of the [Spleeter](https://github.com/deezer/spleeter)
source-separation tool, running the pretrained UNet models through
[ONNX Runtime](https://onnxruntime.ai) on the **GPU**.

The goal is *logic parity* with the Python package: the same spectrogram, the
same network, the same masking, the same inverse STFT, and the same ffmpeg
commands — so the C++ tool produces the same audio as
`spleeter separate`.

---

## Status

| Check | Result |
|---|---|
| Output length vs Python | Exact (e.g. 13230080 = 13230080 samples) |
| Alignment / drift | Constant offset, **no drift** over the full 41 min file |
| Numeric parity (CPU EP vs Python) | 91–101 dB SNR |
| Numeric parity (CUDA EP vs Python) | 59–72 dB SNR (GPU conv rounding) |
| Peak RAM (41 min video) | 788 MB, flat vs. 5 min run (also 788 MB) |
| Runtime (41 min video, RTX 3060) | 78 s |

Per-model parity against real Python Spleeter (5-minute clip, float32 stems,
identical input length):

| Model | C++ (CPU EP) | C++ (CUDA EP) | Output length | Drift |
|---|---|---|---|---|
| `2stems` | 46.2 / 50.9 dB | 42.0 / 46.7 dB | exact | 0 |
| `4stems` | 91.2–100.7 dB | 59–72 dB | exact | 0 |
| `5stems` | 94.6–101.6 dB | 56.1–69.3 dB | exact | 0 |

The `2stems` row is *not* a logic difference: that checkpoint is numerically
chaotic. Perturbing the magnitude spectrogram by a relative `1e-7` — i.e. the
difference between two float32 FFT implementations — already moves the `2stems`
output by ~50 dB, while the same perturbation moves `5stems` by less than
0.1 dB (`tools/perturb_test.py`). Any implementation that does not reproduce
TensorFlow's float32 FFT bit-for-bit therefore lands near 45–50 dB on `2stems`.
The port's own spectrogram is in fact *closer* to a float64 reference
(1.4e-7 relative) than TensorFlow's float32 FFT is (3.6e-6 relative).

The 16-bit WAV container is the limiting factor for the measurable SNR: a
16-bit round trip of the Python reference against itself already lands at
71.7 dB. Comparing raw float32 stems (`-c f32`) removes that ceiling.

---

## Build

Requires CMake ≥ 3.20, a C++17 compiler, and the ONNX Runtime **GPU** build
(1.24 was used here). Point the build at your copy of the distribution with
`-DONNXRUNTIME_ROOT=<path>` or by setting the `ONNXRUNTIME_ROOT` environment
variable:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=<path-to>/onnxruntime/gpu
cmake --build build
```

The build copies `onnxruntime.dll` and the CUDA provider next to the binary.

### Runtime requirements

`onnxruntime_providers_cuda.dll` needs CUDA 12 and cuDNN 9 on `PATH`. Add the
runtime library folder and the CUDA `bin` folder of your installation:

```powershell
$env:PATH = "<path-to>/onnxruntime/gpu/lib;<path-to>/CUDA/v12.x/bin;$env:PATH"
```

### Python tooling (optional)

The verification scripts under `tools/` need the reference implementation and
the TensorFlow stack. Install the package from PyPI, or point `$SPLEETER_SRC`
at a source checkout:

```powershell
$env:SPLEETER_SRC = "<path-to>/spleeter"   # only if not pip-installed
$env:MODEL_PATH = "<path-to>/pretrained_models"
```

`MODEL_PATH` is where the pretrained checkpoints live; the Spleeter model
provider downloads them there on first use.

---

## Usage

```
spleeter_cpp [options] <input> [input...]

  -o, --output_path PATH   Output directory (default: output)
  -p, --params MODEL       Model name or config path (default: 4stems)
  -c, --codec CODEC        wav|mp3|ogg|m4a|wma|flac|f32 (default: wav)
  -b, --bitrate RATE       Output bitrate (default: 128k)
  -f, --filename_format F  Output filename template
  -s, --offset SECONDS     Start offset (default: 0)
  -d, --duration SECONDS   Duration, 0 = whole file (default: 0)
      --model_dir DIR      Directory holding <model>.onnx
      --cpu                Force the CPU execution provider
      --threads N           Intra-op thread count
  -v, --verbose            Verbose logging
```

Example:

```powershell
spleeter_cpp -o out -p 4stems --model_dir models\onnx "path/to/video.mp4"
```

Output layout mirrors Python: `out/<input name>/<instrument>.wav`.

`f32` writes a 32-bit float WAV and is useful for exact numeric comparisons;
`wav` (the default) matches Python's 16-bit output exactly.

---

## Models

`models/onnx/` holds the exported networks plus a `*_meta.json` sidecar with
the Spleeter configuration (instrument list, frame length, frame step, `T`,
`F`, separation exponent, mask extension).

| Model | Instruments |
|---|---|
| `2stems` | vocals, accompaniment |
| `4stems` | vocals, drums, bass, other |
| `5stems` | vocals, piano, drums, bass, other |

The pretrained checkpoints are not stored in this repository (see
`.gitignore`). Download them from the Spleeter release assets and unpack them
into a directory of your choice:

```
https://github.com/deezer/spleeter/releases/download/v1.4.0/{2stems,4stems,5stems}.tar.gz
```

Each archive contains a TensorFlow checkpoint (`checkpoint`, `model.index`,
`model.data-*`). Point `$MODEL_PATH` at the parent directory so the Spleeter
model provider can find them, then export the ONNX graphs:

```powershell
python tools\export_onnx.py --config <spleeter>/spleeter/resources/4stems.json `
    --checkpoint models\4stems\model --out models\onnx\4stems.onnx
```

The exporter rebuilds the Keras layers in exactly the order used upstream, so
the pretrained `conv2d`, `conv2d_transpose` and `batch_normalization`
variables attach by name. It then verifies the exported graph against the
TensorFlow model before writing it.

---

## Architecture

```
main.cpp        CLI, ffmpeg decode/encode processes, output layout
pipeline.cpp    STFT -> UNet -> masks -> masked STFT -> inverse STFT
separator.cpp   STFT frame stream and inverse-STFT overlap-add writer
onnx_model.cpp  ONNX Runtime session (CUDA EP, CPU fallback)
fft.cpp         radix-2 real FFT
process.cpp     ffmpeg subprocess with redirected pipes (Windows + POSIX)
json.cpp        minimal JSON reader for the model metadata
```

### Algorithm (identical to `spleeter/model/__init__.py`)

1. **Decode** — `ffmpeg -i IN -f f32le -ar 44100 -ss OFF -t DUR pipe:`, giving
   interleaved float32 stereo.
2. **Front zero-pad** the waveform with `frame_length` zeros.
3. **STFT** — frame length 4096, hop 1024, periodic Hann window; frames are
   produced until `ceil((n + frame_length) / frame_step)`.
4. **Partition** the magnitude spectrogram into tiles of `T = 512` frames
   (the last tile is zero-padded, exactly like `pad_and_partition`).
5. **UNet** — the tile is fed to the ONNX model, which returns one masked
   magnitude spectrogram per instrument.
6. **Masks** — `mask_i = (out_i² + ε/n) / (Σ_j out_j² + ε)`, with `ε = 1e-10`
   and `n` the instrument count. The mask is zero-extended from `F = 1024` to
   the full 2049 bins.
7. **Masked STFT** — the complex spectrogram is multiplied by the mask.
8. **Inverse STFT** — irFFT, multiply by the same Hann window, overlap-add at
   hop 1024, then scale by the `2/3` window compensation factor.
9. **Crop** the leading `frame_length` samples and keep exactly `n` samples.
10. **Encode** each stem through its own ffmpeg process.

### Memory behaviour

The pipeline is fully streaming:

* the STFT keeps only a sliding window of input samples,
* the UNet sees one 512-frame tile at a time,
* the overlap-add writer keeps only the samples that can still change.

Peak working set was **788 MB for the 41-minute video — identical to the
5-minute run**, so memory does not grow with input length. (Most of that is
the ONNX Runtime CUDA arena, not the DSP.)

---

## Verification tooling

| Script | Purpose |
|---|---|
| `tools/export_onnx.py` | TensorFlow checkpoint → ONNX, with self-verification |
| `tools/reference_python.py` | Runs real Spleeter and dumps reference stems |
| `tools/verify_tf_dsp.py` | Proves the STFT/iSTFT semantics match TensorFlow |
| `tools/verify_same_upsample.py` | Proves `same` == `VALID` + crop (CUDA-friendly) |
| `tools/validate_pipeline.py` | NumPy + ONNX pipeline vs the Python reference |
| `tools/compare_outputs.py` | C++ vs Python: error, SNR, length, drift |
| `tools/compare_quantized.py` | Parity after accounting for 16-bit quantization |
| `tools/prove_parity.py` | Separates DSP differences from GPU-EP differences |
| `tools/check_drift.py` | Time-alignment and reconstruction across a long file |
| `tools/check_offset.py` | Confirms a constant offset rather than accumulating drift |
| `tools/check_drift_fine.py` | Sub-millisecond lag sweep across the whole file |
| `tools/measure_full.ps1` | Peak RAM and runtime for a full-length file |
| `tools/diagnose_graph.py` | Upstream TF graph vs rebuilt graph (bit-exact check) |
| `tools/diagnose_masks.py` | Upstream masks/waveforms vs the port's DSP, stage by stage |
| `tools/perturb_test.py` | Measures the model's sensitivity to float32-level noise |
| `tools/run_pipeline_tiles.py` | Reference pipeline that feeds one tile per forward pass |
| `tools/sweep_tiles.py` | Sweeps input lengths to catch tile-boundary bugs |
| `tools/dump_stft.cpp` | Dumps the port's spectrogram for comparison (`dump_stft` target) |

---

## Notes and deviations

* **Dropout** — upstream inserts `Dropout(0.5)` in the decoder. It is a no-op
  at inference, so the exported graph omits it.
* **`same` transposed convolution** — TensorFlow's `same` padding maps to an
  ONNX `ConvTranspose` with *asymmetric* pads, which the ONNX Runtime CUDA
  provider rejects (it falls back to CPU for those nodes). The exporter
  rewrites it as `VALID` + an explicit crop, which is numerically identical
  (`tools/verify_same_upsample.py` shows a max difference of 0) and keeps the
  whole graph on the GPU.
* **Model output** — upstream applies a `Multiply` of the sigmoid mask with the
  input spectrogram inside the network; the exported graph keeps that, so the
  ONNX outputs are already per-instrument masked spectrograms.
* **GPU vs CPU numerics** — ONNX Runtime's CUDA and CPU providers round
  convolutions differently. Comparing C++(CUDA) with Python(TF) shows
  59–72 dB SNR, while C++(CPU) with the same code shows 91–101 dB. The
  difference is entirely the execution provider, not the port.
* **Model types** — `2stems`/`4stems` use `unet.unet` (sigmoid mask, then
  multiplied by the input spectrogram), `5stems` uses `unet.softmax_unet`
  (per-instrument logits, softmax across instruments, then multiplied by the
  input). The exporter mirrors both paths, and the upstream mask formula
  (`_build_masks`) is model-type agnostic, so the C++ mask stage is identical
  for all three models.
* **Verification of the rebuild** — `tools/diagnose_graph.py` builds upstream's
  own `EstimatorSpecBuilder` graph, restores the pretrained checkpoint into it
  and compares it with the rebuilt Keras graph: the masked spectrograms match
  **bit-for-bit (max difference 0.0)** for all three models.
  `tools/diagnose_masks.py` then reproduces upstream's masks from upstream's own
  model outputs with max difference 0.0, confirming the mask stage and the
  `ε/n` term.
* **Comparing outputs requires identical input lengths** — the UNet's
  receptive field spans ~±40 frames, so a difference in the *last* frames of
  the input (e.g. a reference decoded with `--duration 300` versus a port that
  decodes the whole 300.1 s file) shows up as a localized error in the final
  ~1.5 s of the stems. Always compare against a reference generated from the
  exact same sample count.
