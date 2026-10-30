// Thin wrapper around the ONNX Runtime C API.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace spleeter {

class OnnxModel {
 public:
  OnnxModel(const std::string& path, bool use_cuda, int intra_threads,
            std::string* error);
  ~OnnxModel();

  std::size_t OutputCount() const { return output_names_.size(); }
  std::string OutputName(std::size_t index) const {
    return output_names_[index];
  }

  // Runs the model once and stores every output into `results`, in output
  // order. The exported graph computes all instruments in a single pass, so
  // running it once per instrument would be needlessly expensive.
  bool Run(const std::vector<float>& input,
           const std::vector<std::int64_t>& input_shape,
           const std::vector<std::int64_t>& output_shape,
           std::vector<std::vector<float>>* results);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::vector<std::string> output_names_;
};

}  // namespace spleeter
