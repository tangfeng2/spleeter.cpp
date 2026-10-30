#include "process.h"

#include <cstdio>
#include <cstring>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace spleeter {

#ifdef _WIN32

struct Process::Impl {
  HANDLE process = nullptr;
  HANDLE stdin_write = nullptr;
  HANDLE stdout_read = nullptr;
  HANDLE stderr_file = INVALID_HANDLE_VALUE;
  std::wstring stderr_path;
  bool finished = false;
};

namespace {

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

Process::Process() : impl_(new Impl()) {}

Process::~Process() {
  if (impl_->process != nullptr) {
    if (impl_->stdin_write != nullptr) CloseHandle(impl_->stdin_write);
    if (impl_->stdout_read != nullptr) CloseHandle(impl_->stdout_read);
    if (impl_->stderr_file != INVALID_HANDLE_VALUE) CloseHandle(impl_->stderr_file);
    CloseHandle(impl_->process);
  }
  // The temporary stderr file must not survive the process, whether or not
  // StderrOutput() was called.
  if (!impl_->stderr_path.empty()) {
    DeleteFileW(impl_->stderr_path.c_str());
    impl_->stderr_path.clear();
  }
  delete impl_;
}

bool Process::Start(const std::vector<std::string>& arguments, bool redirect_stdin,
                    bool redirect_stdout, std::string* error) {
  if (arguments.empty()) {
    *error = "No command given";
    return false;
  }

  // Build a Windows command line with proper quoting.
  std::wstring command;
  for (std::size_t i = 0; i < arguments.size(); ++i) {
    if (i > 0) command += L" ";
    const std::string& arg = arguments[i];
    const bool needs_quotes =
        arg.empty() || arg.find_first_of(" \t\"") != std::string::npos;
    if (!needs_quotes) {
      command += Utf8ToWide(arg);
      continue;
    }
    // Convert the whole argument first so that multi-byte UTF-8 characters
    // (for example non-ASCII file names) stay intact, then escape the
    // characters that are special inside a quoted argument.
    const std::wstring wide = Utf8ToWide(arg);
    command += L'"';
    for (wchar_t ch : wide) {
      if (ch == L'"') {
        command += L"\\\"";
      } else if (ch == L'\\') {
        command += L"\\\\";
      } else {
        command += ch;
      }
    }
    command += L'"';
  }

  SECURITY_ATTRIBUTES attributes;
  attributes.nLength = sizeof(attributes);
  attributes.lpSecurityDescriptor = nullptr;
  attributes.bInheritHandle = TRUE;

  HANDLE stdin_read = nullptr;
  HANDLE stdout_write = nullptr;

  if (redirect_stdin) {
    if (!CreatePipe(&stdin_read, &impl_->stdin_write, &attributes, 0)) {
      *error = "CreatePipe(stdin) failed";
      return false;
    }
    SetHandleInformation(impl_->stdin_write, HANDLE_FLAG_INHERIT, 0);
  }
  if (redirect_stdout) {
    if (!CreatePipe(&impl_->stdout_read, &stdout_write, &attributes, 0)) {
      *error = "CreatePipe(stdout) failed";
      return false;
    }
    SetHandleInformation(impl_->stdout_read, HANDLE_FLAG_INHERIT, 0);
  }

  // Redirect stderr to a temporary file. A pipe would need a dedicated reader
  // thread to avoid deadlocking when the child writes a lot (ffmpeg progress).
  // The file lives in the system temporary directory, never in the current
  // working directory, and is removed when the process object is destroyed.
  wchar_t temp_dir[MAX_PATH + 2] = {0};
  const DWORD temp_dir_length = GetTempPathW(MAX_PATH + 1, temp_dir);
  if (temp_dir_length == 0 || temp_dir_length > MAX_PATH) {
    *error = "GetTempPath failed";
    return false;
  }
  std::vector<wchar_t> temp_file(MAX_PATH + 2, L'\0');
  if (!GetTempFileNameW(temp_dir, L"sp_", 0, temp_file.data())) {
    *error = "GetTempFileName failed";
    return false;
  }
  impl_->stderr_path = temp_file.data();
  SECURITY_ATTRIBUTES file_attributes;
  file_attributes.nLength = sizeof(file_attributes);
  file_attributes.lpSecurityDescriptor = nullptr;
  file_attributes.bInheritHandle = TRUE;
  impl_->stderr_file = CreateFileW(
      impl_->stderr_path.c_str(), GENERIC_WRITE,
      FILE_SHARE_READ | FILE_SHARE_WRITE, &file_attributes, CREATE_ALWAYS,
      FILE_ATTRIBUTE_TEMPORARY, nullptr);
  if (impl_->stderr_file == INVALID_HANDLE_VALUE) {
    *error = "Could not create temporary stderr file";
    return false;
  }

  STARTUPINFOW startup;
  ZeroMemory(&startup, sizeof(startup));
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdInput = redirect_stdin ? stdin_read : GetStdHandle(STD_INPUT_HANDLE);
  startup.hStdOutput =
      redirect_stdout ? stdout_write : GetStdHandle(STD_OUTPUT_HANDLE);
  startup.hStdError = impl_->stderr_file;
  startup.wShowWindow = SW_HIDE;

  std::vector<wchar_t> mutable_command(command.begin(), command.end());
  mutable_command.push_back(L'\0');

  PROCESS_INFORMATION info;
  ZeroMemory(&info, sizeof(info));

  if (!CreateProcessW(nullptr, mutable_command.data(), nullptr, nullptr, TRUE,
                      CREATE_NO_WINDOW, nullptr, nullptr, &startup, &info)) {
    *error = "CreateProcess failed for " + arguments[0];
    return false;
  }
  impl_->process = info.hProcess;
  CloseHandle(info.hThread);

  if (redirect_stdin) CloseHandle(stdin_read);
  if (redirect_stdout) CloseHandle(stdout_write);
  return true;
}

bool Process::Write(const void* data, std::size_t size) {
  if (impl_->stdin_write == nullptr) return false;
  const char* cursor = static_cast<const char*>(data);
  std::size_t remaining = size;
  while (remaining > 0) {
    DWORD written = 0;
    const DWORD chunk =
        static_cast<DWORD>(remaining > 0x10000000 ? 0x10000000 : remaining);
    if (!WriteFile(impl_->stdin_write, cursor, chunk, &written, nullptr)) {
      return false;
    }
    if (written == 0) return false;
    cursor += written;
    remaining -= written;
  }
  return true;
}

std::size_t Process::Read(void* data, std::size_t size) {
  if (impl_->stdout_read == nullptr) return 0;
  DWORD read = 0;
  if (!ReadFile(impl_->stdout_read, data, static_cast<DWORD>(size), &read,
                nullptr)) {
    return 0;
  }
  return read;
}

void Process::CloseStdin() {
  if (impl_->stdin_write != nullptr) {
    CloseHandle(impl_->stdin_write);
    impl_->stdin_write = nullptr;
  }
}

int Process::Wait() {
  if (impl_->finished) return 0;
  impl_->finished = true;

  WaitForSingleObject(impl_->process, INFINITE);
  DWORD code = 0;
  GetExitCodeProcess(impl_->process, &code);

  if (impl_->stderr_file != INVALID_HANDLE_VALUE) {
    CloseHandle(impl_->stderr_file);
    impl_->stderr_file = INVALID_HANDLE_VALUE;
  }
  return static_cast<int>(code);
}

std::string Process::StderrOutput() {
  if (impl_->stderr_path.empty()) return std::string();
  HANDLE file = CreateFileW(impl_->stderr_path.c_str(), GENERIC_READ,
                            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_TEMPORARY, nullptr);
  if (file == INVALID_HANDLE_VALUE) return std::string();
  std::string text;
  char buffer[4096];
  DWORD read = 0;
  while (ReadFile(file, buffer, sizeof(buffer), &read, nullptr) && read > 0) {
    text.append(buffer, read);
  }
  CloseHandle(file);
  DeleteFileW(impl_->stderr_path.c_str());
  impl_->stderr_path.clear();
  return text;
}

#else  // POSIX

struct Process::Impl {
  pid_t pid = -1;
  int stdin_fd = -1;
  int stdout_fd = -1;
  int stderr_fd = -1;
  std::string stderr_text;
  bool finished = false;
};

Process::Process() : impl_(new Impl()) {}

Process::~Process() {
  if (impl_->stdin_fd >= 0) close(impl_->stdin_fd);
  if (impl_->stdout_fd >= 0) close(impl_->stdout_fd);
  if (impl_->stderr_fd >= 0) close(impl_->stderr_fd);
  if (impl_->pid > 0) {
    int status = 0;
    waitpid(impl_->pid, &status, 0);
  }
  delete impl_;
}

bool Process::Start(const std::vector<std::string>& arguments, bool redirect_stdin,
                    bool redirect_stdout, std::string* error) {
  if (arguments.empty()) {
    *error = "No command given";
    return false;
  }
  int in_pipe[2] = {-1, -1};
  int out_pipe[2] = {-1, -1};
  int err_pipe[2] = {-1, -1};
  if (redirect_stdin && pipe(in_pipe) != 0) {
    *error = "pipe failed";
    return false;
  }
  if (redirect_stdout && pipe(out_pipe) != 0) {
    *error = "pipe failed";
    return false;
  }
  if (pipe(err_pipe) != 0) {
    *error = "pipe failed";
    return false;
  }

  const pid_t pid = fork();
  if (pid < 0) {
    *error = "fork failed";
    return false;
  }
  if (pid == 0) {
    if (redirect_stdin) {
      dup2(in_pipe[0], STDIN_FILENO);
      close(in_pipe[0]);
      close(in_pipe[1]);
    }
    if (redirect_stdout) {
      dup2(out_pipe[1], STDOUT_FILENO);
      close(out_pipe[0]);
      close(out_pipe[1]);
    }
    dup2(err_pipe[1], STDERR_FILENO);
    close(err_pipe[0]);
    close(err_pipe[1]);

    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (const std::string& arg : arguments) {
      argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);
    execvp(argv[0], argv.data());
    _exit(127);
  }

  impl_->pid = pid;
  if (redirect_stdin) {
    close(in_pipe[0]);
    impl_->stdin_fd = in_pipe[1];
  }
  if (redirect_stdout) {
    close(out_pipe[1]);
    impl_->stdout_fd = out_pipe[0];
  }
  close(err_pipe[1]);
  impl_->stderr_fd = err_pipe[0];
  return true;
}

bool Process::Write(const void* data, std::size_t size) {
  if (impl_->stdin_fd < 0) return false;
  const char* cursor = static_cast<const char*>(data);
  std::size_t remaining = size;
  while (remaining > 0) {
    const ssize_t written = write(impl_->stdin_fd, cursor, remaining);
    if (written <= 0) return false;
    cursor += written;
    remaining -= static_cast<std::size_t>(written);
  }
  return true;
}

std::size_t Process::Read(void* data, std::size_t size) {
  if (impl_->stdout_fd < 0) return 0;
  const ssize_t got = read(impl_->stdout_fd, data, size);
  return got > 0 ? static_cast<std::size_t>(got) : 0;
}

void Process::CloseStdin() {
  if (impl_->stdin_fd >= 0) {
    close(impl_->stdin_fd);
    impl_->stdin_fd = -1;
  }
}

int Process::Wait() {
  if (impl_->finished) return 0;
  impl_->finished = true;
  char buffer[4096];
  ssize_t got = 0;
  while ((got = read(impl_->stderr_fd, buffer, sizeof(buffer))) > 0) {
    impl_->stderr_text.append(buffer, static_cast<std::size_t>(got));
  }
  close(impl_->stderr_fd);
  impl_->stderr_fd = -1;
  int status = 0;
  waitpid(impl_->pid, &status, 0);
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

std::string Process::StderrOutput() { return impl_->stderr_text; }

#endif

}  // namespace spleeter
