#include "fft.h"

#include <cmath>

namespace spleeter {

namespace {
constexpr double kPi = 3.14159265358979323846;
}  // namespace

RealFft::RealFft(std::size_t size) : size_(size) {
  std::size_t bits = 0;
  while ((std::size_t(1) << bits) < size_) ++bits;

  reverse_.resize(size_);
  for (std::size_t i = 0; i < size_; ++i) {
    std::size_t v = i;
    std::size_t r = 0;
    for (std::size_t b = 0; b < bits; ++b) {
      r = (r << 1) | (v & 1);
      v >>= 1;
    }
    reverse_[i] = r;
  }

  // Twiddle factors for the largest stage; smaller stages index into it.
  roots_.resize(size_ / 2);
  for (std::size_t i = 0; i < size_ / 2; ++i) {
    const double angle = -2.0 * kPi * static_cast<double>(i) /
                         static_cast<double>(size_);
    roots_[i] = std::complex<float>(static_cast<float>(std::cos(angle)),
                                     static_cast<float>(std::sin(angle)));
  }
}

void RealFft::Transform(std::complex<float>* data, bool inverse) const {
  for (std::size_t i = 0; i < size_; ++i) {
    const std::size_t j = reverse_[i];
    if (j > i) std::swap(data[i], data[j]);
  }

  for (std::size_t len = 2; len <= size_; len <<= 1) {
    const std::size_t half = len / 2;
    const std::size_t stride = size_ / len;
    for (std::size_t i = 0; i < size_; i += len) {
      for (std::size_t j = 0; j < half; ++j) {
        std::complex<float> w = roots_[j * stride];
        if (inverse) w = std::conj(w);

        const std::complex<float> u = data[i + j];
        const std::complex<float> v = data[i + j + half] * w;
        data[i + j] = u + v;
        data[i + j + half] = u - v;
      }
    }
  }
}

void RealFft::Forward(const float* input, std::complex<float>* output) const {
  std::vector<std::complex<float>> scratch(size_);
  for (std::size_t i = 0; i < size_; ++i) {
    scratch[i] = std::complex<float>(input[i], 0.0f);
  }
  Transform(scratch.data(), false);
  const std::size_t bins = size_ / 2 + 1;
  for (std::size_t i = 0; i < bins; ++i) {
    output[i] = scratch[i];
  }
}

void RealFft::Inverse(const std::complex<float>* input, float* output) const {
  const std::size_t bins = size_ / 2 + 1;
  std::vector<std::complex<float>> scratch(size_);
  for (std::size_t i = 0; i < bins; ++i) {
    scratch[i] = input[i];
  }
  // Rebuild the Hermitian-symmetric spectrum for the full-length transform.
  for (std::size_t i = bins; i < size_; ++i) {
    scratch[i] = std::conj(input[size_ - i]);
  }
  Transform(scratch.data(), true);

  const float scale = 1.0f / static_cast<float>(size_);
  for (std::size_t i = 0; i < size_; ++i) {
    output[i] = scratch[i].real() * scale;
  }
}

}  // namespace spleeter
