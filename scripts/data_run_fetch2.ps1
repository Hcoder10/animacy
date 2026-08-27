# Pass 2: paced fetch of the pass-1 picks that were lost to 429s (interviews first), then a plan over
# additional CC-BY interview categories -> data/raw/candidates2.json. Logs: data/logs/fetch2.log, plan2.log.
$env:PYTHONIOENCODING = "utf-8"
$py = "C:\Users\sarta\reachy-duplex\.venv\Scripts\python.exe"
$root = "C:\Users\sarta\animacy"
Set-Location $root
& $py "$root\scripts\data_fetch_more.py" fetch --pick-file "$root\data\raw\picks2.json" --max-mb 120 --min-width 480 *> "$root\data\logs\fetch2.log"
$cats = '[{"cat":"Videos by The Royal Society"},{"cat":"Videos of scientists in the 2020s"},{"cat":"Videos of scientists in the 2010s"},{"cat":"CDC video series Science Speaks - Talking to Women in Science"},{"cat":"Videos from the NIH"},{"cat":"Oral History (videos)"},{"cat":"Wikimania 2017 oral history"},{"cat":"Kiwix oral histories 2017"},{"cat":"Videos of political scientists"},{"cat":"Vlog"}]'
& $py "$root\scripts\data_fetch_more.py" plan --out "$root\data\raw\candidates2.json" --commons $cats --no-archive --limit-per-cat 60 *> "$root\data\logs\plan2.log"
