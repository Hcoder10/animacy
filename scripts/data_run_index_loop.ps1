# Re-write data/clips/_index.json every 2 minutes while the capture loop is alive, so
# scripts/data_report.py is always current. Exits when capture_loop.log says "done".
$env:PYTHONIOENCODING = "utf-8"
$py = "C:\Users\sarta\reachy-duplex\.venv\Scripts\python.exe"
$root = "C:\Users\sarta\animacy"
Set-Location $root
while ($true) {
    & $py "$root\scripts\data_capture_batch.py" index *> "$root\data\logs\index_last.log"
    if ((Test-Path "$root\data\logs\capture_loop.log") -and (Select-String -Path "$root\data\logs\capture_loop.log" -Pattern "^done " -Quiet)) { break }
    Start-Sleep -Seconds 120
}
& $py "$root\scripts\data_capture_batch.py" index *> "$root\data\logs\index_last.log"
