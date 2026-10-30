cmake_minimum_required(VERSION 3.20)
project(spleeter_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
else()
  add_compile_options(-Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# ONNX Runtime (GPU build)
# ---------------------------------------------------------------------------
# Root of the ONNX Runtime distribution. Override with -DONNXRUNTIME_ROOT=<path>
# or by setting the ONNXRUNTIME_ROOT environment variable before configuring.
if(DEFINED ENV{ONNXRUNTIME_ROOT})
  set(_ort_default "$ENV{ONNXRUNTIME_ROOT}")
else()
  set(_ort_default "C:/onnxruntime/gpu")
endif()

set(ONNXRUNTIME_ROOT "${_ort_default}" CACHE PATH
    "Root directory of the ONNX Runtime distribution")

find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_c_api.h
  HINTS "${ONNXRUNTIME_ROOT}/include")

find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  HINTS "${ONNXRUNTIME_ROOT}/lib")

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "ONNX Runtime not found under ${ONNXRUNTIME_ROOT}. "
    "Set -DONNXRUNTIME_ROOT=<path>.")
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

add_executable(spleeter_cpp
  src/main.cpp
  src/fft.cpp
  src/json.cpp
  src/onnx_model.cpp
  src/pipeline.cpp
  src/process.cpp
  src/separator.cpp)

target_include_directories(spleeter_cpp PRIVATE
  "${ONNXRUNTIME_INCLUDE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(spleeter_cpp PRIVATE "${ONNXRUNTIME_LIBRARY}")

# Diagnostic tool: dumps the port's magnitude spectrogram for comparison
# against TensorFlow / NumPy.
add_executable(dump_stft
  tools/dump_stft.cpp
  src/fft.cpp
  src/json.cpp
  src/onnx_model.cpp
  src/separator.cpp)

target_include_directories(dump_stft PRIVATE
  "${ONNXRUNTIME_INCLUDE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(dump_stft PRIVATE "${ONNXRUNTIME_LIBRARY}")

# On Windows the wide `wmain` entry point is used so that non-ASCII paths
# survive the command line intact.
if(WIN32)
  if(MSVC)
    set_target_properties(spleeter_cpp PROPERTIES LINK_FLAGS "/ENTRY:wmainCRTStartup")
  else()
    # Clang passes unknown options straight to the linker driver.
    set_target_properties(spleeter_cpp PROPERTIES
      LINK_FLAGS "-Xlinker /ENTRY:wmainCRTStartup")
  endif()
endif()

# std::thread needs the MSVC threading library when building with clang-cl.
if(MSVC)
  find_package(Threads REQUIRED)
  target_link_libraries(spleeter_cpp PRIVATE Threads::Threads)
endif()

# Copy the runtime DLLs next to the executable so it runs from any directory.
if(WIN32)
  add_custom_command(TARGET spleeter_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
      "$<TARGET_FILE_DIR:spleeter_cpp>/onnxruntime.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.dll"
      "$<TARGET_FILE_DIR:spleeter_cpp>/onnxruntime_providers_cuda.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
      "$<TARGET_FILE_DIR:spleeter_cpp>/onnxruntime_providers_shared.dll"
    VERBATIM)
endif()
