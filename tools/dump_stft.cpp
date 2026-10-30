// Diagnostic tool: dumps the magnitude spectrogram produced by the port's
// STFT implementation so it can be compared against TensorFlow / NumPy.
//
// Usage: dump_stft <input.f32le> <output.f32> [frames]
//   input.f32le : interleaved stereo float32 samples (ffmpeg -f f32le)
//   output.f32  : frames * 2049 * 2 float32 magnitudes, [frame][bin][channel]

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "separator.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: dump_stft <input.f32le> <output.f32>\n");
    return 2;
  }

  std::FILE* in = std::fopen(argv[1], "rb");
  if (in == nullptr) {
    std::fprintf(stderr, "cannot open %s\n", argv[1]);
    return 1;
  }
  std::fseek(in, 0, SEEK_END);
  const long size = std::ftell(in);
  std::fseek(in, 0, SEEK_SET);
  std::vector<float> samples(static_cast<std::size_t>(size) / sizeof(float));
  if (!samples.empty() &&
      std::fread(samples.data(), sizeof(float), samples.size(), in) !=
          samples.size()) {
    std::fprintf(stderr, "short read\n");
    return 1;
  }
  std::fclose(in);

  spleeter::ModelParams params;
  params.frame_length = 4096;
  params.frame_step = 1024;
  params.n_channels = 2;

  const std::size_t bins = 2049;
  const std::size_t channels = 2;

  std::FILE* out = std::fopen(argv[2], "wb");
  if (out == nullptr) {
    std::fprintf(stderr, "cannot open %s\n", argv[2]);
    return 1;
  }

  spleeter::StftStream stream(params);
  std::vector<float> magnitudes(bins * channels);
  int64_t frames = 0;

  auto on_frame = [&](const std::complex<float>* frame) {
    for (std::size_t b = 0; b < bins; ++b) {
      for (std::size_t c = 0; c < channels; ++c) {
        magnitudes[b * channels + c] = std::abs(frame[b * channels + c]);
      }
    }
    std::fwrite(magnitudes.data(), sizeof(float), magnitudes.size(), out);
    ++frames;
  };

  const std::size_t chunk = 4096;  // in sample frames
  for (std::size_t offset = 0; offset < samples.size(); offset += chunk * channels) {
    const std::size_t count =
        std::min(chunk, (samples.size() - offset) / channels);
    if (count == 0) break;
    stream.Push(samples.data() + offset, count, on_frame);
  }
  stream.Finish(on_frame);
  std::fclose(out);
  std::fprintf(stderr, "wrote %lld frames\n", static_cast<long long>(frames));
  return 0;
}
