// Spleeter C++ port - radix-2 FFT matching TensorFlow's numeric behavior.
#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace spleeter {

// Real-input FFT of size N (power of two).
// TF's tf.signal.stft uses Eigen's FFT; the radix-2 implementation below is
// numerically equivalent to within float32 rounding (~1e-6 relative).
class RealFft {
 public:
  explicit RealFft(std::size_t size);

  std::size_t size() const { return size_; }

  // Forward transform: takes `size` real samples, produces size/2 + 1 complex
  // bins (matching tf.signal.rfft with fft_length == size).
  void Forward(const float* input, std::complex<float>* output) const;

  // Inverse transform: takes size/2 + 1 complex bins (RFFT layout) and
  // produces `size` real samples (matching tf.signal.irfft).
  void Inverse(const std::complex<float>* input, float* output) const;

 private:
  void Transform(std::complex<float>* data, bool inverse) const;

  std::size_t size_;
  std::vector<std::size_t> reverse_;
  std::vector<std::complex<float>> roots_;   // twiddle factors, len = size_/2
};

}  // namespace spleeter
