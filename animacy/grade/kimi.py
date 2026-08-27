"""Kimi K3 through the local ``kimi`` CLI (non-interactive, read-only use).

Ported from the proven reachy-motion-studio adapter: long prompts go through a
file inside the workspace (Windows has a small command-line limit), the
response is parsed from ``--output-format stream-json`` events, and the first
JSON object in the assistant text is returned.

The judge's workspace must contain ONLY what it is allowed to see (reel(s) and
the prompt file). Nothing here ever writes anything else into it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = "kimi-code/k3"
_FALLBACK_BINS = [
    os.path.join(os.path.expanduser("~"), ".kimi-code", "bin", "kimi.exe"),
    os.path.join(os.path.expanduser("~"), ".kimi-code", "bin", "kimi"),
]


class KimiError(RuntimeError):
    pass


def kimi_binary() -> Optional[str]:
    """Path to the ``kimi`` executable (``KIMI_BIN`` env, PATH, then the standard install dir)."""
    env = os.environ.get("KIMI_BIN")
    if env and os.path.exists(env):
        return env
    found = shutil.which("kimi")
    if found:
        return found
    for cand in _FALLBACK_BINS:
        if os.path.exists(cand):
            return cand
    return None


def available() -> bool:
    return kimi_binary() is not None


def extract_json(text: str) -> Dict[str, Any]:
    """First JSON object in ``text`` (Kimi sometimes wraps it in prose or a code fence)."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    raise KimiError(f"Kimi did not return a JSON object. Output began: {text[:300]!r}")


def parse_stream_json(stdout: str) -> str:
    """Concatenate the assistant text from ``--output-format stream-json`` lines."""
    parts: List[str] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("role") == "assistant" and isinstance(event.get("content"), str):
            parts.append(event["content"])
    return "\n".join(parts)


def ask(prompt: str, workspace: Path, timeout: int = 900, model: Optional[str] = None,
        prompt_file_threshold: int = 12_000) -> Dict[str, Any]:
    """Run one non-interactive prompt with ``workspace`` as the only added directory.

    Returns ``{"text": assistant text, "stdout": raw, "stderr": raw, "seconds": float,
    "prompt_file": path or None}``. Raises :class:`KimiError` on a non-zero exit."""
    binary = kimi_binary()
    if binary is None:
        raise KimiError("the kimi executable is not available (set KIMI_BIN or install kimi-code)")
    model = model or os.environ.get("KIMI_MODEL", DEFAULT_MODEL)
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    prompt_file: Optional[Path] = None
    cli_prompt = prompt
    if len(prompt) > prompt_file_threshold or os.name == "nt" and len(prompt) > 6_000:
        prompt_dir = workspace / ".kimi-prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".md", prefix="prompt-",
                                             dir=prompt_dir, delete=False)
        with handle:
            handle.write(prompt)
        prompt_file = Path(handle.name)
        cli_prompt = (f"Read the complete task from the UTF-8 file at {prompt_file.resolve()}. "
                      "Follow it exactly and return only the requested response. Do not modify files.")
    command = [binary, "-m", model, "--add-dir", str(workspace.resolve()), "-p", cli_prompt,
               "--output-format", "stream-json"]
    t0 = time.perf_counter()
    try:
        process = subprocess.run(command, cwd=str(workspace), capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired as e:
        raise KimiError(f"kimi timed out after {timeout}s") from e
    finally:
        if prompt_file is not None:
            try:
                prompt_file.unlink()
            except OSError:
                pass
    seconds = time.perf_counter() - t0
    if process.returncode:
        raise KimiError(process.stderr[-1200:] or f"kimi exited {process.returncode}")
    text = parse_stream_json(process.stdout) or process.stdout
    return {"text": text, "stdout": process.stdout, "stderr": process.stderr, "seconds": seconds,
            "prompt_file": str(prompt_file) if prompt_file else None}


def ask_json(prompt: str, workspace: Path, timeout: int = 900, model: Optional[str] = None) -> Dict[str, Any]:
    """:func:`ask` + :func:`extract_json`; the raw text is attached under ``_raw``."""
    r = ask(prompt, workspace, timeout=timeout, model=model)
    obj = extract_json(r["text"])
    obj["_raw"] = r["text"]
    obj["_seconds"] = r["seconds"]
    return obj


def workspace_listing(workspace: Path) -> List[str]:
    """Every file under the workspace, relative, sorted (what the judge could see)."""
    workspace = Path(workspace)
    out = []
    for root, _dirs, files in os.walk(workspace):
        for f in files:
            out.append(os.path.relpath(os.path.join(root, f), workspace).replace("\\", "/"))
    return sorted(out)
