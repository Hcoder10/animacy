# web/clips — captured canonical clips for the viewer

Drop `animacy clip json <clip_dir> -o web/clips/<name>.json` output here
(the `HumanClip.to_web_json` shape: `{"schema":"animacy.human.v1","rate_hz",
"n","channels","data":{channel:[...]}}`), then run

    python web/dev/build_manifest.py

so the static site knows the file exists (GitHub Pages cannot list a
directory). On `python -m http.server` the viewer also picks up new files here
without rebuilding the manifest.
