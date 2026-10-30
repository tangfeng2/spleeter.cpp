// Full separation pipeline: STFT -> UNet -> masks -> masked STFT -> iSTFT.
//
// Mirrors spleeter/model/__init__.py exactly:
//   _build_stft_feature  -> StftStream
//   _build_model_outputs -> ModelRunner
//   _build_masks         -> mask computation
//   _build_masked_stfts  -> mask * complex STFT
//   _inverse_stft        -> InverseStftWriter
#pragma once

#include <complex>
#include <memory>
#include <string>
#include <vector>

#include "separator.h"

namespace spleeter {

// Receives separated samples per instrument as they are produced.
class SeparationSink {
 public:
  virtual ~SeparationSink() = default;
  virtual void OnSamples(std::size_t instrument, const float* samples,
                         std::size_t count) = 0;
};

class Separator {
 public:
  Separator(const ModelParams& params, const std::string& onnx_path,
            bool use_cuda, int intra_threads, std::string* error);
  ~Separator();

  // Pushes interleaved stereo samples.
  bool PushSamples(const float* samples, std::size_t count, std::string* error);

  // Flushes the tail and emits remaining samples.
  bool Finish(std::string* error);

  void set_sink(SeparationSink* sink) { sink_ = sink; }

  int64_t total_frames() const { return total_frames_; }
  int64_t processed_frames() const { return processed_frames_; }

 private:
  // Runs one T-frame tile through the network and the mask stage.
  // `real_frames` is the number of frames in the tile that carry real audio.
  bool ProcessTile(std::string* error, int64_t real_frames);

  ModelParams params_;
  std::unique_ptr<StftStream> stft_;
  std::unique_ptr<ModelRunner> model_;
  std::vector<std::unique_ptr<InverseStftWriter>> writers_;

  std::vector<float> tile_;                   // T * F * C magnitudes
  std::vector<std::complex<float>> tile_spec_;  // T * bins * C complex STFT
  std::vector<std::vector<float>> outputs_;    // per instrument
  std::vector<float> mask_;                    // F * C per (time) in tile
  std::vector<std::vector<float>> tails_;      // per instrument holdback

  SeparationSink* sink_ = nullptr;
  int64_t total_frames_ = 0;
  int64_t processed_frames_ = 0;
  int64_t frames_in_tile_ = 0;
  int64_t total_samples_ = 0;
  bool finished_ = false;
};

}  // namespace spleeter
