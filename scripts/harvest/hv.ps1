# Run a harvest script inside the Windows harvest venv with the right environment.
#   powershell -File C:\harvest\hv.ps1 scripts/harvest/status.py
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$env:Path = "C:\harvest\bin;" + (($env:Path -split ';') | Where-Object { $_ -and ($_ -notlike '*OpenAI\Codex*') }) -join ';'
$env:HARVEST_ROOT = "C:\harvest\data"
$env:HARVEST_BIN = "C:\harvest\bin"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:CUDA_VISIBLE_DEVICES = ""
Set-Location C:\harvest\animacy
& C:\harvest\venv\Scripts\python.exe @Rest
exit $LASTEXITCODE
