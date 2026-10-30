#include "onnx_model.h"

#include <onnxruntime_c_api.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace spleeter {

namespace {

const OrtApi* Api() { return OrtGetApiBase()->GetApi(ORT_API_VERSION); }

std::string StatusMessage(OrtStatus* status) {
  if (status == nullptr) return "unknown error";
  std::string message = Api()->GetErrorMessage(status);
  Api()->ReleaseStatus(status);
  return message;
}

// Converts a UTF-8 path to UTF-16 for the wide-char ORT APIs.
std::wstring Utf8ToWide(const std::string& text) {
  if (text.empty()) return std::wstring();
  const int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(),
                                       static_cast<int>(text.size()), nullptr, 0);
  std::wstring wide(static_cast<std::size_t>(size), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()),
                      &wide[0], size);
  return wide;
}

}  // namespace

struct OnnxModel::Impl {
  OrtEnv* env = nullptr;
  OrtSession* session = nullptr;
  OrtMemoryInfo* memory_info = nullptr;
  std::vector<const char*> output_name_ptrs;
};

OnnxModel::OnnxModel(const std::string& path, bool use_cuda, int intra_threads,
                     std::string* error) {
  impl_.reset(new Impl());

  OrtStatus* status = Api()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "spleeter",
                                       &impl_->env);
  if (status != nullptr) {
    *error = "CreateEnv failed: " + StatusMessage(status);
    return;
  }

  OrtSessionOptions* options = nullptr;
  status = Api()->CreateSessionOptions(&options);
  if (status != nullptr) {
    *error = "CreateSessionOptions failed: " + StatusMessage(status);
    return;
  }
  Api()->SetIntraOpNumThreads(options, intra_threads);
  Api()->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL);
  // Keep the console readable: only report errors and above.
  Api()->SetSessionLogSeverityLevel(options, 3);
  Api()->SetSessionLogVerbosityLevel(options, 3);

  if (use_cuda) {
    OrtCUDAProviderOptions cuda_options = {};
    cuda_options.device_id = 0;
    status = Api()->SessionOptionsAppendExecutionProvider_CUDA(options,
                                                             &cuda_options);
    if (status != nullptr) {
      // Fall back to CPU so the tool still works without a usable GPU.
      std::fprintf(stderr, "Warning: CUDA provider unavailable (%s), using CPU\n",
                   StatusMessage(status).c_str());
      Api()->ReleaseSessionOptions(options);
      options = nullptr;
      status = Api()->CreateSessionOptions(&options);
      if (status != nullptr) {
        *error = "CreateSessionOptions failed: " + StatusMessage(status);
        return;
      }
      Api()->SetIntraOpNumThreads(options, intra_threads);
      Api()->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL);
      Api()->SetSessionLogSeverityLevel(options, 3);
      Api()->SetSessionLogVerbosityLevel(options, 3);
    }
  }

  status = Api()->CreateSession(impl_->env, Utf8ToWide(path).c_str(), options,
                                &impl_->session);
  if (status != nullptr) {
    *error = "Failed to load model " + path + ": " + StatusMessage(status);
    Api()->ReleaseSessionOptions(options);
    return;
  }
  Api()->ReleaseSessionOptions(options);

  OrtAllocator* allocator = nullptr;
  Api()->GetAllocatorWithDefaultOptions(&allocator);
  std::size_t count = 0;
  status = Api()->SessionGetOutputCount(impl_->session, &count);
  if (status != nullptr) {
    *error = "SessionGetOutputCount failed: " + StatusMessage(status);
    return;
  }
  for (std::size_t i = 0; i < count; ++i) {
    char* name = nullptr;
    status = Api()->SessionGetOutputName(impl_->session, i, allocator, &name);
    if (status != nullptr) {
      *error = "SessionGetOutputName failed: " + StatusMessage(status);
      return;
    }
    // Copy the name before releasing the allocator-owned buffer.
    output_names_.emplace_back(name);
    Api()->AllocatorFree(allocator, name);
  }
  impl_->output_name_ptrs.reserve(output_names_.size());
  for (const std::string& name : output_names_) {
    impl_->output_name_ptrs.push_back(name.c_str());
  }

  status = Api()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                      &impl_->memory_info);
  if (status != nullptr) {
    *error = "CreateCpuMemoryInfo failed: " + StatusMessage(status);
  }
}

OnnxModel::~OnnxModel() {
  if (impl_ == nullptr) return;
  if (impl_->session != nullptr) {
    Api()->ReleaseSession(impl_->session);
  }
  if (impl_->memory_info != nullptr) {
    Api()->ReleaseMemoryInfo(impl_->memory_info);
  }
}

bool OnnxModel::Run(const std::vector<float>& input,
                    const std::vector<std::int64_t>& input_shape,
                    const std::vector<std::int64_t>& output_shape,
                    std::vector<std::vector<float>>* results) {
  std::size_t input_count = 1;
  for (std::int64_t dim : input_shape) {
    input_count *= static_cast<std::size_t>(dim);
  }

  OrtValue* input_tensor = nullptr;
  OrtStatus* status = Api()->CreateTensorWithDataAsOrtValue(
      impl_->memory_info, const_cast<float*>(input.data()),
      input_count * sizeof(float), input_shape.data(), input_shape.size(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
  if (status != nullptr) {
    throw std::runtime_error("CreateTensorWithDataAsOrtValue failed: " +
                             StatusMessage(status));
  }

  const std::size_t output_count = impl_->output_name_ptrs.size();
  std::vector<OrtValue*> output_tensors(output_count, nullptr);
  const char* input_names[] = {"input"};
  status = Api()->Run(impl_->session, nullptr, input_names, &input_tensor, 1,
                      impl_->output_name_ptrs.data(), output_count,
                      output_tensors.data());
  Api()->ReleaseValue(input_tensor);
  if (status != nullptr) {
    throw std::runtime_error("ONNX Runtime Run failed: " + StatusMessage(status));
  }

  std::size_t element_count = 1;
  for (std::int64_t dim : output_shape) {
    element_count *= static_cast<std::size_t>(dim);
  }

  results->assign(output_count, {});
  for (std::size_t i = 0; i < output_count; ++i) {
    float* data = nullptr;
    status = Api()->GetTensorMutableData(
        output_tensors[i], reinterpret_cast<void**>(&data));
    if (status != nullptr) {
      Api()->ReleaseValue(output_tensors[i]);
      throw std::runtime_error("GetTensorMutableData failed: " +
                               StatusMessage(status));
    }
    (*results)[i].assign(data, data + element_count);
    Api()->ReleaseValue(output_tensors[i]);
  }
  return true;
}

}  // namespace spleeter
