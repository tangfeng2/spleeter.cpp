// Core Spleeter separation pipeline, mirroring spleeter/model/__init__.py.
#pragma once

#include <complex>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "fft.h"
#include "json.h"

namespace spleeter {

// Parameters taken from the Spleeter model configuration JSON.
struct ModelParams {
  std::vector<std::string> instrument_list;
  int sample_rate = 44100;
  int frame_length = 4096;
  int frame_step = 1024;
  int segment_length = 512;   // T
  int frequency_bins = 1024;  // F
  int n_channels = 2;
  double separation_exponent = 2.0;
  std::string mask_extension = "zeros";

  bool LoadFromJson(const std::string& path, std::string* error);
};

// Hann window, matching tf.signal.hann_window(length, periodic=True).
std::vector<float> HannPeriodic(int length);

// Produces STFT frames from a stream of interleaved samples.
//
// Reproduces Spleeter's front zero padding plus tf.signal.stft(..., pad_end=True):
//   * the waveform is prefixed with `frame_length` zeros,
//   * frame k covers padded samples [k*frame_step, k*frame_step + frame_length),
//   * the number of frames is ceil((n_samples + frame_length) / frame_step),
//   * samples past the end are zero padded.
//
// Frames are delivered through a callback so that no per-batch frame buffer is
// ever accumulated; memory stays flat regardless of the audio length.
class StftStream {
 public:
  // Receives one frame: bins * n_channels complex values, [bin][channel].
  using FrameCallback = std::function<void(const std::complex<float>*)>;

  explicit StftStream(const ModelParams& params);

  // Pushes interleaved stereo samples and emits every frame that became
  // fully determined.
  void Push(const float* interleaved, std::size_t sample_count,
            const FrameCallback& callback);

  // Emits the trailing frames that reach past the end of the signal; those are
  // zero padded, exactly like tf.signal.frame(pad_end=True).
  void Finish(const FrameCallback& callback);

  int64_t TotalFrames() const;

 private:
  // Emits the next frame if its data is already known.
  bool EmitFrame(const FrameCallback& callback);

  ModelParams params_;
  RealFft fft_;
  std::vector<float> window_;
  std::vector<float> pending_;      // unconsumed real samples (interleaved)
  int64_t pending_start_ = 0;       // real sample index of pending_[0]
  int64_t samples_seen_ = 0;
  int64_t next_frame_ = 0;
  bool finished_ = false;
  std::vector<float> frame_;
  std::vector<std::complex<float>> spectrum_;
  std::vector<std::complex<float>> frame_scratch_;
};

// Overlap-add inverse STFT for one instrument, reproducing
// tf.signal.inverse_stft with a periodic Hann window.
//
// The window is applied to the real frames and the result is accumulated with
// a hop of `frame_step`. Spleeter then crops the first `frame_length` samples
// and keeps exactly `n_samples` samples.
class InverseStftWriter {
 public:
  InverseStftWriter(const ModelParams& params);

  // Adds one masked complex frame (bins * n_channels values, [bin][channel]).
  void Push(const std::complex<float>* masked_frame, std::vector<float>* out);

  // Flushes the remaining tail.
  void Finish(std::vector<float>* out);

  // Number of sample frames emitted so far (after the frame_length crop).
  int64_t emitted_frames() const { return emitted_frames_; }

 private:
  // Number of interleaved sample frames currently held in accum_.
  int64_t accum_frames() const {
    return static_cast<int64_t>(accum_.size() /
                                static_cast<std::size_t>(params_.n_channels));
  }
  // Absolute position just past the last frame held in accum_.
  int64_t accum_end() const { return accum_base_ + accum_frames(); }

  ModelParams params_;
  RealFft fft_;
  std::vector<float> window_;
  std::vector<float> accum_;     // overlap-add accumulator (interleaved)
  int64_t accum_base_ = 0;       // absolute frame index of accum_[0]
  int64_t next_frame_ = 0;
  int64_t emit_pos_ = 0;         // next absolute frame index to consider
  int64_t emitted_frames_ = 0;
  bool finished_ = false;
};

// Runs the exported ONNX model over tiles of the magnitude spectrogram.
class ModelRunner {
 public:
  ModelRunner(const ModelParams& params, const std::string& onnx_path,
              bool use_cuda, int intra_threads, std::string* error);
  ~ModelRunner();

  // `tile` holds T*F*C magnitudes laid out as [time][freq][channel].
  // `outputs` receives one T*F*C magnitude estimate per instrument.
  bool Run(const std::vector<float>& tile,
           std::vector<std::vector<float>>* outputs);

  const std::vector<std::string>& instrument_names() const {
    return instrument_names_;
  }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::vector<std::string> instrument_names_;
};

}  // namespace spleeter
