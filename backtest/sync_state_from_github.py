#!/workspace/venv/bin/python
"""Box side: pull the GitHub Actions state (public repo, anonymous) into the box paths.

GitHub Actions is the primary scanner/pusher; its ledgers, dedupe file and breaker state live in
state/ of https://github.com/JamesKu2023/hs-scanner. This copies them over the box files that the
daily / weekly reviews read (scan_out/paper_* and pushed_state.json). It never touches the
box's signals.json (box catch-up keeps working) and needs no credentials.

Usage: /workspace/venv/bin/python /workspace/trading/backtest/sync_state_from_github.py
Prints one summary line; exit 0 on success, 1 if the fetch failed (box files left unchanged).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO_URL = os.environ.get("HS_STATE_REPO", "https://github.com/JamesKu2023/hs-scanner.git")
CLONE = Path(os.environ.get("HS_STATE_CLONE", "/workspace/hs-scanner-state"))


def _git(*args: str, cwd: Path | None = None) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    env.pop("GH_TOKEN", None)
    env.pop("GITHUB_TOKEN", None)
    out = subprocess.run(["git", "-c", "credential.helper=", *args], cwd=str(cwd) if cwd else None,
                         env=env, capture_output=True, text=True, timeout=90)
    if out.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {out.stderr.strip()[:200]}")
    return out.stdout.strip()


def fetch() -> str:
    if not (CLONE / ".git").exists():
        if CLONE.exists():
            shutil.rmtree(CLONE)
        _git("clone", "-q", "--depth", "1", "--branch", "main", REPO_URL, str(CLONE))
    else:
        _git("fetch", "-q", "--depth", "1", "origin", "main", cwd=CLONE)
        _git("reset", "-q", "--hard", "FETCH_HEAD", cwd=CLONE)
    return _git("log", "-1", "--format=%h %cd", "--date=format-local:%Y-%m-%d %H:%M", cwd=CLONE)


def sync() -> int:
    from gha_runner import STATE_FILES
    try:
        head = fetch()
    except Exception as e:  # noqa: BLE001
        print(f"state sync FAILED ({type(e).__name__}: {e}); box ledgers unchanged")
        return 1
    n = 0
    for rel in STATE_FILES:
        src = CLONE / "state" / rel
        if src.exists():
            dst = HERE / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".sync_tmp")
            shutil.copy2(src, tmp)
            tmp.replace(dst)
            n += 1
    try:
        bars = json.loads((CLONE / "state" / "last_bars.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        bars = {}
    print(f"state sync ok: {n} files from GitHub commit {head} Asia/Shanghai; github last_bars={bars}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TZ", "Asia/Shanghai")
    raise SystemExit(sync())
