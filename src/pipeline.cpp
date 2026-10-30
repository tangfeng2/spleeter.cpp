#include "pipeline.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace spleeter {

namespace {

// spleeter.model.EstimatorSpecBuilder.EPSILON
constexpr float kEpsilon = 1e-10f;

// Number of samples kept back at the end of each stem so the exact output
// length can be enforced once the total input length is known.
//
// The overlap-add tail can exceed the wanted length by up to `frame_length`
// frames, so the holdback must comfortably exceed that.
constexpr std::size_t kTailHoldback = 16384;

}  // namespace

Separator::Separator(const ModelParams& params, const std::string& onnx_path,
                     bool use_cuda, int intra_threads, std::string* error)
    : params_(params) {
  stft_.reset(new StftStream(params));
  model_.reset(new ModelRunner(params, onnx_path, use_cuda, intra_threads, error));
  if (!error->empty()) return;

  const std::size_t instruments = params_.instrument_list.size();
  writers_.reserve(instruments);
  for (std::size_t i = 0; i < instruments; ++i) {
    writers_.emplace_back(new InverseStftWriter(params_));
    tails_.emplace_back();
  }

  const std::size_t tile_frames = static_cast<std::size_t>(params_.segment_length);
  const std::size_t tile_f = static_cast<std::size_t>(params_.frequency_bins);
  const std::size_t channels = static_cast<std::size_t>(params_.n_channels);
  const std::size_t bins = static_cast<std::size_t>(params_.frame_length) / 2 + 1;

  tile_.assign(tile_frames * tile_f * channels, 0.0f);
  tile_spec_.assign(tile_frames * bins * channels, std::complex<float>(0.0f, 0.0f));
  mask_.assign(tile_f * channels, 0.0f);
}

Separator::~Separator() = default;

bool Separator::PushSamples(const float* samples, std::size_t count,
                            std::string* error) {
  const std::size_t channels = static_cast<std::size_t>(params_.n_channels);
  const std::size_t bins = static_cast<std::size_t>(params_.frame_length) / 2 + 1;
  const std::size_t tile_frames = static_cast<std::size_t>(params_.segment_length);
  const std::size_t tile_f = static_cast<std::size_t>(params_.frequency_bins);

  total_samples_ += count;

  // Frames are consumed as soon as they are produced so that no per-batch
  // frame buffer is needed.
  auto on_frame = [this, &tile_frames, &tile_f, &channels, &bins, error](
                      const std::complex<float>* frame) -> bool {
    const std::size_t slot = static_cast<std::size_t>(frames_in_tile_);

    // Keep the complex spectrum for the masked-STFT stage.
    std::copy(frame, frame + bins * channels,
              tile_spec_.begin() +
                  static_cast<std::ptrdiff_t>(slot * bins * channels));

    // Magnitude spectrogram for the first F bins (network input).
    for (std::size_t b = 0; b < tile_f; ++b) {
      for (std::size_t c = 0; c < channels; ++c) {
        const float mag = std::abs(frame[b * channels + c]);
        tile_[(slot * tile_f + b) * channels + c] = mag;
      }
    }

    ++frames_in_tile_;
    if (frames_in_tile_ == static_cast<int64_t>(tile_frames)) {
      if (!ProcessTile(error, static_cast<int64_t>(tile_frames))) return false;
      frames_in_tile_ = 0;
      std::fill(tile_.begin(), tile_.end(), 0.0f);
      std::fill(tile_spec_.begin(), tile_spec_.end(),
                std::complex<float>(0.0f, 0.0f));
    }
    return true;
  };

  stft_->Push(samples, count, on_frame);
  return true;
}

bool Separator::ProcessTile(std::string* error, int64_t real_frames) {
  const std::size_t channels = static_cast<std::size_t>(params_.n_channels);
  const std::size_t tile_frames = static_cast<std::size_t>(params_.segment_length);
  const std::size_t tile_f = static_cast<std::size_t>(params_.frequency_bins);
  const std::size_t bins = static_cast<std::size_t>(params_.frame_length) / 2 + 1;
  const std::size_t instruments = params_.instrument_list.size();

  // Run the network on the whole tile.
  if (!model_->Run(tile_, &outputs_)) {
    *error = "ONNX inference failed";
    return false;
  }

  // spleeter's _build_masks:
  //   output_sum = sum_i(output_i ** separation_exponent) + EPSILON
  //   mask_i     = (output_i ** separation_exponent + EPSILON / n) / output_sum
  const float eps_per_instrument = kEpsilon / static_cast<float>(instruments);

  std::vector<float> mask_values(bins * channels);
  std::vector<std::complex<float>> masked(bins * channels);
  std::vector<float> emitted;

  // Only the frames that carry real audio are pushed to the inverse STFT:
  // Python truncates the mask to the real frame count before the masked STFT
  // is built, so the zero padded tail frames never reach the reconstruction.
  for (int64_t t = 0; t < real_frames; ++t) {
    // Denominator for this frame: sum over instruments.
    for (std::size_t k = 0; k < tile_f * channels; ++k) {
      float sum = 0.0f;
      for (std::size_t i = 0; i < instruments; ++i) {
        const float value = outputs_[i][(t * tile_f) * channels + k];
        sum += static_cast<float>(
            std::pow(static_cast<double>(value), params_.separation_exponent));
      }
      mask_[k] = sum + kEpsilon;
    }

    for (std::size_t i = 0; i < instruments; ++i) {
      const std::vector<float>& out = outputs_[i];

      for (std::size_t b = 0; b < bins; ++b) {
        for (std::size_t c = 0; c < channels; ++c) {
          float mask_value = 0.0f;
          if (b < tile_f) {
            const std::size_t k = b * channels + c;
            const float value = out[(t * tile_f) * channels + k];
            const float power = static_cast<float>(
                std::pow(static_cast<double>(value), params_.separation_exponent));
            mask_value = (power + eps_per_instrument) / mask_[k];
          }
          // Above the F bins the mask extension is zero ("zeros" extension),
          // so the masked spectrum there is zero as well.
          mask_values[b * channels + c] = mask_value;
        }
      }

      // Apply the mask to the complex spectrum of this frame.
      const std::complex<float>* spec =
          tile_spec_.data() + static_cast<std::size_t>(t) * bins * channels;
      for (std::size_t k = 0; k < bins * channels; ++k) {
        masked[k] = spec[k] * mask_values[k];
      }
      writers_[i]->Push(masked.data(), &emitted);
      if (sink_ != nullptr && !emitted.empty()) {
        // Keep the tail back so the exact output length can be enforced.
        std::vector<float>& tail = tails_[i];
        tail.insert(tail.end(), emitted.begin(), emitted.end());
        emitted.clear();
        if (tail.size() > kTailHoldback) {
          const std::size_t flush = tail.size() - kTailHoldback;
          sink_->OnSamples(i, tail.data(), flush);
          tail.erase(tail.begin(),
                     tail.begin() + static_cast<std::ptrdiff_t>(flush));
        }
      }
    }
  }

  processed_frames_ += static_cast<int64_t>(tile_frames);
  return true;
}

bool Separator::Finish(std::string* error) {
  if (finished_) return true;
  finished_ = true;

  const std::size_t channels = static_cast<std::size_t>(params_.n_channels);
  const std::size_t tile_frames = static_cast<std::size_t>(params_.segment_length);
  const std::size_t tile_f = static_cast<std::size_t>(params_.frequency_bins);
  const std::size_t bins = static_cast<std::size_t>(params_.frame_length) / 2 + 1;

  // Flush the trailing STFT frames that extend past the end of the signal.
  auto on_frame = [this, &tile_frames, &tile_f, &channels, &bins, error](
                      const std::complex<float>* frame) -> bool {
    const std::size_t slot = static_cast<std::size_t>(frames_in_tile_);
    std::copy(frame, frame + bins * channels,
              tile_spec_.begin() +
                  static_cast<std::ptrdiff_t>(slot * bins * channels));
    for (std::size_t b = 0; b < tile_f; ++b) {
      for (std::size_t c = 0; c < channels; ++c) {
        const float mag = std::abs(frame[b * channels + c]);
        tile_[(slot * tile_f + b) * channels + c] = mag;
      }
    }
    ++frames_in_tile_;
    if (frames_in_tile_ == static_cast<int64_t>(tile_frames)) {
      if (!ProcessTile(error, static_cast<int64_t>(tile_frames))) return false;
      frames_in_tile_ = 0;
      std::fill(tile_.begin(), tile_.end(), 0.0f);
      std::fill(tile_spec_.begin(), tile_spec_.end(),
                std::complex<float>(0.0f, 0.0f));
    }
    return true;
  };
  stft_->Finish(on_frame);
  total_frames_ = stft_->TotalFrames();

  // The trailing partial tile is zero padded, exactly like Spleeter's
  // pad_and_partition does. Only the frames that carry real audio are
  // reconstructed; the padded ones are dropped, matching the mask truncation
  // in _build_masks.
  if (frames_in_tile_ > 0) {
    if (!ProcessTile(error, frames_in_tile_)) return false;
    frames_in_tile_ = 0;
  }

  // Flush the inverse STFT tails.
  std::vector<float> emitted;
  for (std::size_t i = 0; i < writers_.size(); ++i) {
    writers_[i]->Finish(&emitted);
    std::vector<float>& tail = tails_[i];
    tail.insert(tail.end(), emitted.begin(), emitted.end());
    emitted.clear();
  }

  // Python crops the output to exactly n_samples; drop the excess produced by
  // the overlap-add tail.
  const std::size_t instruments = writers_.size();
  for (std::size_t i = 0; i < instruments; ++i) {
    std::vector<float>& tail = tails_[i];
    const int64_t excess =
        writers_[i]->emitted_frames() - static_cast<int64_t>(total_samples_);
    if (excess > 0 && static_cast<std::size_t>(excess) * channels <= tail.size()) {
      const std::size_t drop = static_cast<std::size_t>(excess) * channels;
      tail.resize(tail.size() - drop);
    }
    if (sink_ != nullptr && !tail.empty()) {
      sink_->OnSamples(i, tail.data(), tail.size());
    }
    tail.clear();
  }
  return true;
}

}  // namespace spleeter
