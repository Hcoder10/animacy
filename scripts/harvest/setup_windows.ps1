# Provision a native-Windows harvest host (squaredcube1): repo clone, venv (python 3.12), ffmpeg, deno.
# Idempotent. Run:  powershell -ExecutionPolicy Bypass -File setup_windows.ps1 [-Root C:\harvest]
param([string]$Root = "C:\harvest",
      [string]$Py = "C:\Users\sarta\AppData\Roaming\uv\python\cpython-3.12.13-windows-x86_64-none\python.exe",
      [string]$Repo = "https://github.com/Hcoder10/animacy.git")
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force "$Root", "$Root\bin", "$Root\data", "$Root\data\logs" | Out-Null
if (!(Test-Path "$Root\animacy\.git")) { git clone $Repo "$Root\animacy" } else { git -C "$Root\animacy" pull --ff-only }
if (!(Test-Path "$Root\venv\Scripts\python.exe")) { & $Py -m venv "$Root\venv" }
$vpy = "$Root\venv\Scripts\python.exe"
# pip walks every PATH entry for its scripts-dir warning and dies on Windows "untrusted mount point"
# junctions (WinError 448, seen with the Codex bin dir on squaredcube1): drop them for the install.
$env:Path = (($env:Path -split ';') | Where-Object { $_ -and (Test-Path $_) -and ($_ -notlike '*OpenAI\Codex*') }) -join ';'
& $vpy -m pip install -q --no-cache-dir --no-warn-script-location -U pip wheel
# PyPI torch on Windows is the CPU build (silero-vad needs torch); mediapipe/opencv from the capture extra.
& $vpy -m pip install -q --no-cache-dir --no-warn-script-location -e "$Root\animacy[capture]" "yt-dlp[default]" silero-vad huggingface_hub soundfile
if (!(Test-Path "$Root\bin\ffmpeg.exe")) {
  curl.exe -sL -o "$Root\ffmpeg.zip" https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
  Expand-Archive "$Root\ffmpeg.zip" "$Root\ffmpeg_tmp" -Force
  Copy-Item "$Root\ffmpeg_tmp\*\bin\*.exe" "$Root\bin\"
  Remove-Item -Recurse -Force "$Root\ffmpeg_tmp", "$Root\ffmpeg.zip"
}
if (!(Test-Path "$Root\bin\deno.exe")) {   # yt-dlp >= 2025.11 needs a JS runtime for YouTube's n-challenge
  curl.exe -sL -o "$Root\deno.zip" https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip
  Expand-Archive "$Root\deno.zip" "$Root\bin" -Force
  Remove-Item "$Root\deno.zip"
}
$env:Path = "$Root\bin;" + $env:Path
& "$Root\bin\ffmpeg.exe" -version | Select-Object -First 1
& "$Root\bin\deno.exe" --version | Select-Object -First 1
& $vpy -c "import mediapipe, cv2, yt_dlp, silero_vad, torch, huggingface_hub; print('venv ok', mediapipe.__version__, cv2.__version__, yt_dlp.version.__version__, torch.__version__, torch.cuda.is_available())"
Write-Output "SETUP DONE"
