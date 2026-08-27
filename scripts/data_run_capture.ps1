# Loop scripts/data_capture_batch.py run (3 workers, 480 s cap) until the fetch has finished and
# every licensed raw video has a clip. Logs to data/logs/capture_loop.log.
$env:PYTHONIOENCODING = "utf-8"
$py = "C:\Users\sarta\reachy-duplex\.venv\Scripts\python.exe"
$root = "C:\Users\sarta\animacy"
$log = "$root\data\logs\capture_loop.log"
Set-Location $root
"start $(Get-Date)" | Out-File $log -Encoding utf8
while ($true) {
    $out = & $py "$root\scripts\data_capture_batch.py" run --jobs 3 --duration 480 2>&1 | Out-String
    $out | Out-File $log -Append -Encoding utf8
    $fetchDone = (Test-Path "$root\data\logs\fetch.log") -and (Select-String -Path "$root\data\logs\fetch.log" -Pattern "^fetched " -Quiet)
    $nothing = $out -match "0 videos to capture"
    if ($fetchDone -and $nothing) { break }
    Start-Sleep -Seconds 60
}
"done $(Get-Date)" | Out-File $log -Append -Encoding utf8
