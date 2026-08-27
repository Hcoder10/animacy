# Launch the license-verified fetch (scripts/data_fetch_more.py fetch) detached, logging to data/logs/fetch.log.
$env:PYTHONIOENCODING = "utf-8"
$py = "C:\Users\sarta\reachy-duplex\.venv\Scripts\python.exe"
$root = "C:\Users\sarta\animacy"
$picks = Get-Content "$root\data\raw\picks_arg.txt" -Raw -Encoding UTF8
Set-Location $root
& $py "$root\scripts\data_fetch_more.py" fetch --pick $picks --max-mb 120 --min-width 480 *> "$root\data\logs\fetch.log"
