# Re-index, write showcase motion.json for 3 varied clips, then push every kept clip to the Hub.
# Log: data/logs/push_hf.log
$env:PYTHONIOENCODING = "utf-8"
$py = "C:\Users\sarta\reachy-duplex\.venv\Scripts\python.exe"
$root = "C:\Users\sarta\animacy"
$log = "$root\data\logs\push_hf.log"
Set-Location $root
& $py "$root\scripts\data_capture_batch.py" index *> $log
foreach ($c in @("direitos_humanos_entrevista_com_fl_via_pinto", "zachary_levi_about_working_on_broadway_at_nerdhq", "anarkali_honaryar_ghani_has_pledged_to_ensure_wo")) {
    & $py -m animacy.cli clip json "$root\data\clips\$c" *>> $log
}
$idx = Get-Content "$root\data\clips\_index.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$excl = ($idx.clips | Where-Object { $_.status -ne "kept" } | ForEach-Object { $_.name }) -join ","
$excl | Out-File "$root\data\clips\_exclude.txt" -Encoding utf8
"exclude: $excl" | Out-File $log -Append -Encoding utf8
& $py "$root\scripts\push_hf.py" --repo squaredcuber/animacy-human-motion --exclude $excl *>> $log
"push exit $LASTEXITCODE $(Get-Date)" | Out-File $log -Append -Encoding utf8
