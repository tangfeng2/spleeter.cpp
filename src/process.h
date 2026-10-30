// Cross-platform subprocess helper used to drive ffmpeg pipes.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace spleeter {

// Spawns a child process with optional stdin/stdout redirection.
class Process {
 public:
  Process();
  ~Process();

  Process(const Process&) = delete;
  Process& operator=(const Process&) = delete;

  // `arguments` must be the full argument vector (argv[0] included).
  bool Start(const std::vector<std::string>& arguments, bool redirect_stdin,
             bool redirect_stdout, std::string* error);

  // Writes bytes to the child's stdin. Returns false on error.
  bool Write(const void* data, std::size_t size);

  // Reads bytes from the child's stdout. Returns the number read (0 at EOF).
  std::size_t Read(void* data, std::size_t size);

  // Closes the child's stdin to signal end of input.
  void CloseStdin();

  // Waits for the process to exit and returns its exit code.
  int Wait();

  // Returns whatever the child wrote to stderr (best effort).
  std::string StderrOutput();

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace spleeter
