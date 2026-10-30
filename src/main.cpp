// spleeter_cpp - a C++ port of the Deezer Spleeter source separation tool.
//
// Mirrors the `spleeter separate` command: decodes audio with ffmpeg, runs the
// exported ONNX UNet, and writes separated stems back through ffmpeg.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include "pipeline.h"
#include "process.h"
#include "separator.h"

namespace {

struct Options {
  std::vector<std::string> inputs;
  std::string output_path = "output";
  std::string params = "4stems";
  std::string codec = "wav";
  std::string bitrate = "128k";
  std::string filename_format = "{filename}/{instrument}.{codec}";
  std::string model_dir = "models/onnx";
  double offset = 0.0;
  double duration = 0.0;  // 0 = whole file
  bool use_cuda = true;
  int threads = 0;
  bool verbose = false;
};

void PrintUsage() {
  std::printf(
      "Usage: spleeter_cpp [options] <input> [input...]\n"
      "\n"
      "Options:\n"
      "  -o, --output_path PATH   Output directory (default: output)\n"
      "  -p, --params MODEL       Model config name or path "
      "(default: 4stems)\n"
      "  -c, --codec CODEC        wav|mp3|ogg|m4a|wma|flac|f32 (default: wav)\n"
      "  -b, --bitrate RATE       Output bitrate (default: 128k)\n"
      "  -f, --filename_format F  Output filename template\n"
      "  -s, --offset SECONDS     Start offset (default: 0)\n"
      "  -d, --duration SECONDS   Duration, 0 = whole file (default: 0)\n"
      "      --model_dir DIR      Directory holding <model>.onnx\n"
      "      --cpu                Force the CPU execution provider\n"
      "      --threads N           Intra-op thread count\n"
      "  -v, --verbose            Verbose logging\n"
      "  -h, --help               Show this message\n");
}

std::string BaseName(const std::string& path) {
  const std::size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::string DirName(const std::string& path) {
  const std::size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

std::string StripExtension(const std::string& name) {
  const std::size_t dot = name.find_last_of('.');
  return dot == std::string::npos ? name : name.substr(0, dot);
}

// Formats an HH:MM:SS.mmm timestamp the way Python's timedelta does.
std::string TimeStamp(double seconds) {
  if (seconds < 0) seconds = 0;
  char buffer[64];
  const int hours = static_cast<int>(seconds / 3600.0);
  const int minutes = static_cast<int>((seconds - hours * 3600.0) / 60.0);
  const double rest = seconds - hours * 3600.0 - minutes * 60.0;
  std::snprintf(buffer, sizeof(buffer), "%d:%02d:%09.6f", hours, minutes, rest);
  return buffer;
}

// Replaces {filename}, {instrument}, {foldername} and {codec} placeholders.
std::string ApplyFormat(const std::string& format, const std::string& filename,
                        const std::string& instrument, const std::string& folder,
                        const std::string& codec) {
  std::string result;
  result.reserve(format.size() + 32);
  for (std::size_t i = 0; i < format.size(); ++i) {
    if (format[i] != '{') {
      result.push_back(format[i]);
      continue;
    }
    const std::size_t end = format.find('}', i);
    if (end == std::string::npos) {
      result.push_back(format[i]);
      continue;
    }
    const std::string key = format.substr(i + 1, end - i - 1);
    if (key == "filename") {
      result += filename;
    } else if (key == "instrument") {
      result += instrument;
    } else if (key == "foldername") {
      result += folder;
    } else if (key == "codec") {
      result += codec;
    } else {
      result += format.substr(i, end - i + 1);
    }
    i = end;
  }
  return result;
}

std::string FfmpegCodec(const std::string& codec) {
  if (codec == "m4a") return "aac";
  if (codec == "ogg") return "libvorbis";
  if (codec == "wma") return "wmav2";
  if (codec == "f32") return "pcm_f32le";
  return codec;
}

// Writes separated stems to disk through ffmpeg, mirroring
// spleeter.audio.ffmpeg.FFMPEGProcessAudioAdapter.save.
class FfmpegSink : public spleeter::SeparationSink {
 public:
  FfmpegSink(const std::vector<std::string>& paths,
             const std::vector<std::string>& codecs, int sample_rate,
             const std::string& bitrate)
      : paths_(paths), codecs_(codecs), sample_rate_(sample_rate),
        bitrate_(bitrate) {
    for (std::size_t i = 0; i < paths_.size(); ++i) {
      processes_.emplace_back(new spleeter::Process());
      std::vector<std::string> arguments = {
          "ffmpeg", "-f", "f32le", "-ac", "2",
          "-ar", std::to_string(sample_rate_), "-i", "pipe:"};
      // Linear PCM codecs ignore a bitrate, and ffmpeg rejects the
      // combination for some encoders.
      const bool linear_pcm = codecs_[i].rfind("pcm_", 0) == 0;
      if (!bitrate_.empty() && !linear_pcm) {
        arguments.push_back("-b:a");
        arguments.push_back(bitrate_);
      }
      arguments.push_back("-ar");
      arguments.push_back(std::to_string(sample_rate_));
      if (!codecs_[i].empty() && codecs_[i] != "wav") {
        arguments.push_back("-codec");
        arguments.push_back(FfmpegCodec(codecs_[i]));
      }
      arguments.push_back("-strict");
      arguments.push_back("-2");
      arguments.push_back(paths_[i]);
      arguments.push_back("-y");
      std::string error;
      if (!processes_[i]->Start(arguments, true, false, &error)) {
        std::fprintf(stderr, "Failed to start encoder: %s\n", error.c_str());
        std::exit(1);
      }
    }
  }

  void OnSamples(std::size_t instrument, const float* samples,
                 std::size_t count) override {
    if (instrument >= processes_.size()) return;
    if (!processes_[instrument]->Write(samples, count * sizeof(float))) {
      std::fprintf(stderr, "Failed to write samples for stem %zu\n", instrument);
      std::exit(1);
    }
  }

  void Close() {
    for (auto& process : processes_) {
      process->CloseStdin();
      const int code = process->Wait();
      if (code != 0) {
        std::fprintf(stderr, "Encoder failed with code %d: %s\n", code,
                     process->StderrOutput().c_str());
      }
    }
  }

 private:
  std::vector<std::string> paths_;
  std::vector<std::string> codecs_;
  int sample_rate_;
  std::string bitrate_;
  std::vector<std::unique_ptr<spleeter::Process>> processes_;
};

bool MakeDirectories(const std::string& path) {
  std::string current;
  for (std::size_t i = 0; i <= path.size(); ++i) {
    if (i == path.size() || path[i] == '/' || path[i] == '\\') {
      if (!current.empty() && current != "." && current != "..") {
        std::string command = "mkdir \"" + current + "\" 2>nul";
        std::system(command.c_str());
      }
    }
    if (i < path.size()) current.push_back(path[i]);
  }
  return true;
}

}  // namespace

#ifdef _WIN32
// On Windows the narrow `main` entry point receives ANSI-encoded arguments,
// which corrupts non-ASCII paths. `wmain` gives the original UTF-16 command
// line, so the arguments are converted to UTF-8 here.
int wmain(int argc, wchar_t** argv);

std::string WideToUtf8(const wchar_t* text) {
  if (text == nullptr || *text == L'\0') return std::string();
  const int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0,
                                       nullptr, nullptr);
  if (size <= 1) return std::string();
  std::string out(static_cast<std::size_t>(size - 1), '\0');
  WideCharToMultiByte(CP_UTF8, 0, text, -1, out.data(), size, nullptr, nullptr);
  return out;
}
#endif

namespace {

// Runs the tool. `args` holds the arguments in UTF-8.
int RunMain(const std::vector<std::string>& args) {
  const int argc = static_cast<int>(args.size());
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = args[static_cast<std::size_t>(i)];
    auto next = [&]() -> std::string {
      return i + 1 < argc ? args[static_cast<std::size_t>(++i)] : std::string();
    };
    if (arg == "-h" || arg == "--help") {
      PrintUsage();
      return 0;
    } else if (arg == "-o" || arg == "--output_path") {
      options.output_path = next();
    } else if (arg == "-p" || arg == "--params_filename" || arg == "--params") {
      options.params = next();
    } else if (arg == "-c" || arg == "--codec") {
      options.codec = next();
    } else if (arg == "-b" || arg == "--bitrate") {
      options.bitrate = next();
    } else if (arg == "-f" || arg == "--filename_format") {
      options.filename_format = next();
    } else if (arg == "-s" || arg == "--offset") {
      options.offset = std::atof(next().c_str());
    } else if (arg == "-d" || arg == "--duration") {
      options.duration = std::atof(next().c_str());
    } else if (arg == "--model_dir") {
      options.model_dir = next();
    } else if (arg == "--cpu") {
      options.use_cuda = false;
    } else if (arg == "--threads") {
      options.threads = std::atoi(next().c_str());
    } else if (arg == "-v" || arg == "--verbose") {
      options.verbose = true;
    } else if (!arg.empty() && arg[0] == '-') {
      std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
      return 1;
    } else {
      options.inputs.push_back(arg);
    }
  }

  if (options.inputs.empty()) {
    PrintUsage();
    return 1;
  }

  // Resolve the model configuration.
  std::string config_path = options.params;
  std::string model_name = options.params;
  if (options.params.find('.') == std::string::npos) {
    config_path = options.model_dir + "/" + options.params + "_meta.json";
    model_name = options.params;
  }

  spleeter::ModelParams params;
  std::string error;
  if (!params.LoadFromJson(config_path, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }

  const std::string onnx_path = options.model_dir + "/" + model_name + ".onnx";
  if (options.verbose) {
    std::printf("Config    : %s\n", config_path.c_str());
    std::printf("Model     : %s\n", onnx_path.c_str());
    std::printf("Instruments: %zu\n", params.instrument_list.size());
  }

  for (const std::string& input : options.inputs) {
    // Build the output paths for every stem.
    const std::string filename = StripExtension(BaseName(input));
    const std::string folder = BaseName(DirName(input));

    std::vector<std::string> stem_paths;
    std::vector<std::string> stem_codecs;
    // "f32" is a 32-bit float WAV; it keeps the .wav extension so that
    // ffmpeg can detect the container.
    const std::string extension = options.codec == "f32" ? "wav" : options.codec;
    for (const std::string& instrument : params.instrument_list) {
      const std::string relative = ApplyFormat(options.filename_format, filename,
                                               instrument, folder, extension);
      const std::string full = options.output_path + "/" + relative;
      MakeDirectories(DirName(full));
      stem_paths.push_back(full);
      stem_codecs.push_back(options.codec);
    }

    // Decode with ffmpeg, using exactly the same argument order as the
    // Python audio adapter:
    //   ffmpeg -i IN -f f32le -ar SR -ss OFFSET -t DURATION pipe:
    std::vector<std::string> arguments = {"ffmpeg", "-i", input, "-f", "f32le",
                                          "-ar", std::to_string(params.sample_rate)};
    if (options.offset > 0) {
      arguments.push_back("-ss");
      arguments.push_back(TimeStamp(options.offset));
    }
    if (options.duration > 0) {
      arguments.push_back("-t");
      arguments.push_back(TimeStamp(options.duration));
    }
    arguments.push_back("pipe:");
    spleeter::Process decoder;
    if (!decoder.Start(arguments, false, true, &error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }

    spleeter::Separator separator(params, onnx_path, options.use_cuda,
                                  options.threads, &error);
    if (!error.empty()) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }

    FfmpegSink sink(stem_paths, stem_codecs, params.sample_rate,
                    options.bitrate);
    separator.set_sink(&sink);

    // Feed the decoder output into the pipeline.
    std::vector<char> raw(1 << 16);
    std::vector<float> samples;
    const std::size_t channels = static_cast<std::size_t>(params.n_channels);
    std::size_t leftover = 0;
    std::size_t total_frames = 0;

    while (true) {
      const std::size_t got = decoder.Read(raw.data() + leftover, raw.size() - leftover);
      const std::size_t available = leftover + got;
      const std::size_t float_count = available / sizeof(float);
      const std::size_t used = float_count * sizeof(float);
      samples.resize(float_count);
      std::memcpy(samples.data(), raw.data(), used);
      leftover = available - used;

      // The pipeline counts stereo frames, the decoder produces interleaved
      // floats (channels per frame).
      const std::size_t frame_count = float_count / channels;
      if (frame_count > 0 &&
          !separator.PushSamples(samples.data(), frame_count, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
      }
      total_frames += frame_count;
      if (got == 0) break;
    }
    if (leftover > 0) {
      std::fprintf(stderr, "Warning: trailing %zu bytes were not a full sample\n",
                   leftover);
    }

    if (!separator.Finish(&error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    sink.Close();

    const int decode_code = decoder.Wait();
    if (decode_code != 0) {
      std::fprintf(stderr, "ffmpeg decode failed with code %d: %s\n",
                   decode_code, decoder.StderrOutput().c_str());
      return 1;
    }
    if (total_frames == 0) {
      std::fprintf(stderr, "No audio decoded from %s\n", input.c_str());
      return 1;
    }

    if (options.verbose) {
      std::printf("Separated %s into %zu stems\n", input.c_str(),
                  params.instrument_list.size());
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> args;
  args.reserve(static_cast<std::size_t>(argc));
  for (int i = 0; i < argc; ++i) args.push_back(argv[i]);
  return RunMain(args);
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
  std::vector<std::string> args;
  args.reserve(static_cast<std::size_t>(argc));
  for (int i = 0; i < argc; ++i) args.push_back(WideToUtf8(argv[i]));
  return RunMain(args);
}
#endif
