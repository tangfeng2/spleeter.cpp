# Runs spleeter_cpp on a full-length file and reports peak working set and time.
#
# Used to verify that the streaming pipeline keeps memory flat regardless of
# the input length: run it on a short clip and on a long file, the peak working
# set must be the same.
#
# Example:
#   powershell -File tools\measure_full.ps1 -Source "path/to/video.mp4"
param(
  # Input file to separate (required).
  [Parameter(Mandatory = $true)]
  [string]$Source,

  # Model configuration name or path.
  [string]$Model = "4stems",

  # Directory holding <model>.onnx. Defaults to <repo>/models/onnx.
  [string]$ModelDir,

  # Output directory for the stems. Defaults to <repo>/test/cpp_mem.
  [string]$Out,

  # Path to the spleeter_cpp executable. Defaults to <repo>/build/spleeter_cpp.exe.
  [string]$Exe,

  # Optional directories prepended to PATH (ONNX Runtime and CUDA/cuDNN).
  # Existing directories only; nothing happens when they are omitted.
  [string]$OrtLib = "",
  [string]$CudaBin = ""
)

$Root = Split-Path -Parent $PSScriptRoot

if (-not $Exe) { $Exe = Join-Path $Root "build\spleeter_cpp.exe" }
if (-not $Out) { $Out = Join-Path $Root "test\cpp_mem" }
if (-not $ModelDir) { $ModelDir = Join-Path $Root "models\onnx" }

foreach ($dir in @($OrtLib, $CudaBin)) {
  if ($dir -and (Test-Path $dir)) { $env:PATH = "$dir;$env:PATH" }
}

if (-not (Test-Path $Exe)) {
  Write-Output "executable not found: $Exe"
  Write-Output "build it first: cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build"
  exit 1
}

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $Exe
$psi.Arguments = "-o `"$Out`" -p $Model --model_dir `"$ModelDir`" `"$Source`""
$psi.UseShellExecute = $false
$psi.RedirectStandardError = $true
$psi.CreateNoWindow = $true

$proc = [System.Diagnostics.Process]::Start($psi)
$stderr = $proc.StandardError.ReadToEndAsync()

$peak = 0
$sw = [Diagnostics.Stopwatch]::StartNew()
while (-not $proc.HasExited) {
  try {
    $proc.Refresh()
    if ($proc.PeakWorkingSet64 -gt $peak) { $peak = $proc.PeakWorkingSet64 }
  } catch {}
  Start-Sleep -Milliseconds 200
}
$sw.Stop()
$err = $stderr.Result

Write-Output "source         : $Source"
Write-Output "model          : $Model"
Write-Output "exit_code      : $($proc.ExitCode)"
Write-Output "elapsed_sec    : $([math]::Round($sw.Elapsed.TotalSeconds,1))"
Write-Output "peak_working_MB: $([math]::Round($peak/1MB,1))"
if ($err) {
  $lines = ($err -split "`n") | Where-Object { $_ -notmatch 'built with|configuration:|lib(a|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z|postproc)' -and $_.Trim() -ne '' }
  Write-Output "stderr_lines   : $($lines.Count)"
  $lines | Select-Object -First 5 | ForEach-Object { Write-Output "  $_" }
}

exit $proc.ExitCode
