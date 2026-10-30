#include "separator.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "onnx_model.h"

namespace spleeter {

namespace {

// spleeter.model.EstimatorSpecBuilder.WINDOW_COMPENSATION_FACTOR
constexpr float kWindowCompensationFactor = 2.0f / 3.0f;

}  // namespace

std::vector<float> HannPeriodic(int length) {
  std::vector<float> window(static_cast<std::size_t>(length));
  const double two_pi = 6.283185307179586476925286766559;
  for (int i = 0; i < length; ++i) {
    const double angle =
        two_pi * static_cast<double>(i) / static_cast<double>(length);
    window[static_cast<std::size_t>(i)] =
        static_cast<float>(0.5 - 0.5 * std::cos(angle));
  }
  return window;
}

bool ModelParams::LoadFromJson(const std::string& path, std::string* error) {
  JsonValue root;
  if (!ReadJsonFile(path, &root, error)) return false;

  const JsonValue* instruments = root.Find("instrument_list");
  if (instruments == nullptr || !instruments->IsArray()) {
    *error = "Config is missing 'instrument_list': " + path;
    return false;
  }
  instrument_list.clear();
  for (const JsonValue& item : instruments->items) {
    instrument_list.push_back(item.GetString(""));
  }

  auto number = [&root](const char* key, double fallback) {
    const JsonValue* value = root.Find(key);
    return value != nullptr ? value->GetNumber(fallback) : fallback;
  };
  sample_rate = static_cast<int>(number("sample_rate", 44100));
  frame_length = static_cast<int>(number("frame_length", 4096));
  frame_step = static_cast<int>(number("frame_step", 1024));
  segment_length = static_cast<int>(number("T", 512));
  frequency_bins = static_cast<int>(number("F", 1024));
  n_channels = static_cast<int>(number("n_channels", 2));
  separation_exponent = number("separation_exponent", 2.0);
  const JsonValue* extension = root.Find("mask_extension");
  mask_extension =
      extension != nullptr ? extension->GetString("zeros") : "zeros";
  return true;
}

// ---------------------------------------------------------------------------
// StftStream
// ---------------------------------------------------------------------------

StftStream::StftStream(const ModelParams& params)
    : params_(params), fft_(static_cast<std::size_t>(params.frame_length)) {
  window_ = HannPeriodic(params.frame_length);
  frame_.resize(static_cast<std::size_t>(params.frame_length));
  const std::size_t bins = static_cast<std::size_t>(params.frame_length) / 2 + 1;
  spectrum_.resize(bins);
  frame_scratch_.resize(bins * static_cast<std::size_t>(params.n_channels));
}

int64_t StftStream::TotalFrames() const {
  const int64_t total = samples_seen_ + params_.frame_length;
  return (total + params_.frame_step - 1) / params_.frame_step;
}

bool StftStream::EmitFrame(const FrameCallback& callback) {
  const std::size_t channels = static_cast<std::size_t>(params_.n_channels);
  const std::size_t frame_length = static_cast<std::size_t>(params_.frame_length);
  const std::size_t bins = frame_length / 2 + 1;

  // Never emit more frames than tf.signal.frame(pad_end=True) would produce.
  if (next_frame_ >= TotalFrames()) return false;

  // Frame k covers padded samples [k*frame_step, k*frame_step+frame_length).
  // Padded sample p corresponds to real sample p - frame_length, so the frame
  // needs real samples [k*frame_step - frame_length, k*frame_step).
  const int64_t frame_start = next_frame_ * static_cast<int64_t>(params_.frame_step);
  const int64_t need_from = frame_start - static_cast<int64_t>(frame_length);
  const int64_t need_to = frame_start;

  // A frame can only be produced once its real samples have been seen.
  if (!finished_ && need_to > samples_seen_) return false;

  for (std::size_t c = 0; c < channels; ++c) {
    for (std::size_t i = 0; i < frame_length; ++i) {
      const int64_t pos = need_from + static_cast<int64_t>(i);
      float value = 0.0f;
      if (pos >= pending_start_ &&
          pos < pending_start_ + static_cast<int64_t>(pending_.size() / channels)) {
        value = pending_[static_cast<std::size_t>(pos - pending_start_) * channels + c];
      }
      frame_[i] = value * window_[i];
    }
    fft_.Forward(frame_.data(), spectrum_.data());
    // The callback receives [bin][channel] layout, so transpose here.
    for (std::size_t b = 0; b < bins; ++b) {
      frame_scratch_[b * channels + c] = spectrum_[b];
    }
  }
  callback(frame_scratch_.data());

  ++next_frame_;
  return true;
}

void StftStream::Push(const float* interleaved, std::size_t sample_count,
                      const FrameCallback& callback) {
  const std::size_t channels = static_cast<std::size_t>(params_.n_channels);
  pending_.insert(pending_.end(), interleaved,
                  interleaved + sample_count * channels);
  samples_seen_ += static_cast<int64_t>(sample_count);

  while (EmitFrame(callback)) {
    // Release samples no future frame needs.
    const int64_t keep_from =
        next_frame_ * static_cast<int64_t>(params_.frame_step) -
        static_cast<int64_t>(params_.frame_length);
    if (keep_from > pending_start_) {
      const std::size_t drop =
          static_cast<std::size_t>(keep_from - pending_start_) * channels;
      if (drop >= pending_.size()) {
        pending_.clear();
      } else {
        pending_.erase(pending_.begin(),
                       pending_.begin() + static_cast<std::ptrdiff_t>(drop));
      }
      pending_start_ = keep_from;
    }
  }
}

void StftStream::Finish(const FrameCallback& callback) {
  if (finished_) return;
  finished_ = true;
  // Emit the trailing frames that reach past the end of the signal; those are
  // zero padded, exactly like tf.signal.frame(pad_end=True).
  while (EmitFrame(callback)) {
  }
}

// ---------------------------------------------------------------------------
// InverseStftWriter
// ---------------------------------------------------------------------------

InverseStftWriter::InverseStftWriter(const ModelParams& params)
    : params_(params), fft_(static_cast<std::size_t>(params.frame_length)) {
  window_ = HannPeriodic(params.frame_length);
}

void InverseStftWriter::Push(const std::complex<float>* masked_frame,
                             std::vector<float>* out) {
  const std::size_t frame_length = static_cast<std::size_t>(params_.frame_length);
  const std::size_t frame_step = static_cast<std::size_t>(params_.frame_step);
  const std::size_t bins = frame_length / 2 + 1;
  const std::size_t channels = static_cast<std::size_t>(params_.n_channels);

  const int64_t offset = next_frame_ * static_cast<int64_t>(frame_step);

  for (std::size_t c = 0; c < channels; ++c) {
    std::vector<std::complex<float>> spectrum(bins);
    for (std::size_t b = 0; b < bins; ++b) {
      spectrum[b] = masked_frame[b * channels + c];
    }
    std::vector<float> frame(frame_length);
    fft_.Inverse(spectrum.data(), frame.data());

    // tf.signal.inverse_stft multiplies real frames by the same window.
    for (std::size_t i = 0; i < frame_length; ++i) {
      frame[i] *= window_[i];
    }

    // Overlap-add at the frame offset. accum_ holds interleaved frames for
    // the absolute range [accum_base_, accum_end()).
    const int64_t relative = offset - accum_base_;
    if (relative < 0) {
      accum_.clear();
      accum_base_ = offset;
    }
    const int64_t rel = offset - accum_base_;
    const std::size_t needed = static_cast<std::size_t>(rel) + frame_length;
    if (accum_frames() < static_cast<int64_t>(needed)) {
      accum_.resize(needed * channels, 0.0f);
    }
    for (std::size_t i = 0; i < frame_length; ++i) {
      accum_[(static_cast<std::size_t>(rel) + i) * channels + c] += frame[i];
    }
  }
  ++next_frame_;

  // After frame (next_frame_ - 1) is complete, every position below
  // next_frame_ * frame_step is final: the next frame only starts writing at
  // next_frame_ * frame_step.
  const int64_t limit = next_frame_ * static_cast<int64_t>(frame_step);
  if (limit > emit_pos_) {
    const int64_t from = emit_pos_;
    const int64_t to = std::min<int64_t>(limit, accum_end());
    for (int64_t pos = from; pos < to; ++pos) {
      // Spleeter crops the first `frame_length` samples of the inverse STFT.
      if (pos < static_cast<int64_t>(frame_length)) continue;
      const std::size_t index = static_cast<std::size_t>(pos - accum_base_);
      for (std::size_t c = 0; c < channels; ++c) {
        out->push_back(accum_[index * channels + c] * kWindowCompensationFactor);
      }
      ++emitted_frames_;
    }
    emit_pos_ = to;
    // Drop the consumed prefix to keep memory bounded.
    if (emit_pos_ > accum_base_) {
      const std::size_t drop =
          static_cast<std::size_t>(emit_pos_ - accum_base_) * channels;
      if (drop >= accum_.size()) {
        accum_.clear();
      } else {
        accum_.erase(accum_.begin(),
                     accum_.begin() + static_cast<std::ptrdiff_t>(drop));
      }
      accum_base_ = emit_pos_;
    }
  }
}

void InverseStftWriter::Finish(std::vector<float>* out) {
  if (finished_) return;
  finished_ = true;
  const std::size_t channels = static_cast<std::size_t>(params_.n_channels);
  const int64_t frame_length = params_.frame_length;
  const int64_t end = accum_end();
  for (int64_t pos = accum_base_; pos < end; ++pos) {
    if (pos < frame_length) continue;
    const std::size_t index = static_cast<std::size_t>(pos - accum_base_);
    for (std::size_t c = 0; c < channels; ++c) {
      out->push_back(accum_[index * channels + c] * kWindowCompensationFactor);
    }
    ++emitted_frames_;
  }
  accum_.clear();
}

// ---------------------------------------------------------------------------
// ModelRunner
// ---------------------------------------------------------------------------

struct ModelRunner::Impl {
  std::unique_ptr<OnnxModel> model;
  std::vector<std::int64_t> input_shape;
  std::vector<std::int64_t> output_shape;
};

ModelRunner::ModelRunner(const ModelParams& params,
                         const std::string& onnx_path, bool use_cuda,
                         int intra_threads, std::string* error)
    : impl_(new Impl) {
  impl_->model.reset(new OnnxModel(onnx_path, use_cuda, intra_threads, error));
  if (!error->empty()) return;

  const std::size_t count = impl_->model->OutputCount();
  for (std::size_t i = 0; i < count; ++i) {
    instrument_names_.push_back(impl_->model->OutputName(i));
  }
  if (instrument_names_.size() != params.instrument_list.size()) {
    *error = "ONNX model exposes " + std::to_string(instrument_names_.size()) +
             " outputs but the config lists " +
             std::to_string(params.instrument_list.size()) + " instruments";
    return;
  }
  impl_->input_shape = {1, params.segment_length, params.frequency_bins,
                        params.n_channels};
  impl_->output_shape = {1, params.segment_length, params.frequency_bins,
                         params.n_channels};
}

ModelRunner::~ModelRunner() = default;

bool ModelRunner::Run(const std::vector<float>& tile,
                       std::vector<std::vector<float>>* outputs) {
  // The exported graph evaluates every instrument in one forward pass.
  return impl_->model->Run(tile, impl_->input_shape, impl_->output_shape,
                           outputs);
}

}  // namespace spleeter
