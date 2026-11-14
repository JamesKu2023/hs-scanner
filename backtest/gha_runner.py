#!/usr/bin/env python
"""One GitHub Actions cycle of the H&S short scanner (notify only, never places orders).

1. restore persisted state from ../state into the paths scan_runner.py uses;
2. decide what to scan: 15m when its latest closed bar is not yet scanned, 1H / 4H by the same
   catch-up rule as the box daemon (last_bar in signals.json behind the latest closed bar), so a
   late or skipped cron run is caught up on the next one;
3. run scan_runner.run_once() (journals, breaker, dedupe and webhook exactly as on the box);
4. copy state back to ../state (the workflow commits it) and back up charts of newly pushed signals.

Webhook URL / key only come from the HS_WEBHOOK_URL / HS_WEBHOOK_KEY environment (GitHub
Secrets). Missing secrets -> scan + journal still run, push is skipped with a log line.
Exit code is 0 unless the cycle itself crashed, so a missing secret never fails the workflow.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATE = ROOT / "state"
CHARTS = ROOT / "charts"
TZ = ZoneInfo("Asia/Shanghai")
sys.path.insert(0, str(HERE))

# relative to backtest/ ; journals + dedupe + breaker state (all inside the journal json files)
STATE_FILES = ["pushed_state.json"] + [
    f"scan_out/paper_{book}_{kind}"
    for book in ("journal", "watch", "v6_15m", "htf_1h", "htf_4h", "nested_15m", "nested_1h", "nested_4h")
    for kind in ("journal.json", "trades.csv", "journal.md")
]
# paper_journal.json / paper_trades.csv naming for the formal book differs:
STATE_FILES = [f.replace("paper_journal_journal.json", "paper_journal.json")
                .replace("paper_journal_trades.csv", "paper_trades.csv")
                .replace("paper_journal_journal.md", "paper_journal.md") for f in STATE_FILES]
SIGNAL_FILES = {"15m": "scan_out/signals.json", "1h": "scan_out/htf/1h/signals.json",
                "4h": "scan_out/htf/4h/signals.json"}


def log(msg: str) -> None:
    print(f"{datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')} Asia/Shanghai [gha] {msg}", flush=True)


def restore_state() -> None:
    for rel in STATE_FILES:
        src = STATE / rel
        if src.exists():
            dst = HERE / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    bars = {}
    lb = STATE / "last_bars.json"
    if lb.exists():
        try:
            bars = json.loads(lb.read_text(encoding="utf-8"))
        except ValueError:
            bars = {}
    for tf, rel in SIGNAL_FILES.items():
        dst = HERE / rel
        if bars.get(tf) and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps({"last_bar_shanghai": bars[tf], "signals": [], "restored_stub": True},
                                      ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"state restored: last_bars={bars}")


def save_state() -> dict:
    STATE.mkdir(parents=True, exist_ok=True)
    for rel in STATE_FILES:
        src = HERE / rel
        if src.exists():
            dst = STATE / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    bars = {}
    for tf, rel in SIGNAL_FILES.items():
        try:
            bars[tf] = json.loads((HERE / rel).read_text(encoding="utf-8")).get("last_bar_shanghai") or ""
        except (OSError, ValueError):
            bars[tf] = ""
    (STATE / "last_bars.json").write_text(json.dumps(bars, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return bars


def backup_charts(before: set[str], after: dict) -> int:
    """Copy the PNG of every signal first pushed in this run into charts/YYYY-MM-DD/."""
    import scan_runner as SR
    new = set(after) - before
    if not new:
        return 0
    n = 0
    day = datetime.now(TZ).strftime("%Y-%m-%d")
    for tf, rel in SIGNAL_FILES.items():
        try:
            data = json.loads((HERE / rel).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for s in data.get("signals") or []:
            if not isinstance(s, dict):
                continue
            if tf != "15m":
                s.setdefault("timeframe", tf)
            if SR.pattern_id(s) not in new or not s.get("chart"):
                continue
            src = Path(s["chart"])
            if src.exists():
                d = CHARTS / day
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, d / f"{datetime.now(TZ).strftime('%H%M')}_{tf}_{src.name}")
                n += 1
    return n


def main() -> int:
    restore_state()
    import scan_runner as SR
    now = SR.now_sh()
    need_15m = SR.bar_unscanned("15m", now=now) or os.environ.get("HS_FORCE_15M") == "1"
    htf = SR.resolve_htf_tfs([], now=now)  # catch-up rule == "a new 1H/4H bar closed since last scan"
    log(f"plan: 15m={need_15m} htf={htf or '-'} night={SR.is_quiet_hours(now)} "
        f"webhook_secrets={'set' if os.environ.get('HS_WEBHOOK_KEY') and os.environ.get('HS_WEBHOOK_URL') else 'missing'} "
        f"push_disabled={SR.push_disabled()}")
    before = set((SR.load_state().get("pushed") or {}).keys())
    rc = 0
    if need_15m or htf:
        try:
            rc = SR.run_once(htf_tfs=htf, skip_15m=not need_15m)
        except Exception:
            log("cycle crashed:\n" + traceback.format_exc())
            rc = 1
    else:
        log("nothing to scan (latest closed bars already scanned)")
    after = SR.load_state().get("pushed") or {}
    try:
        n = backup_charts(before, after)
        if n:
            log(f"backed up {n} chart(s) into charts/")
    except Exception:
        log("chart backup error:\n" + traceback.format_exc())
    bars = save_state()
    log(f"state saved: last_bars={bars} run_once_rc={rc}")
    # rc 3 = webhook not sent (e.g. secrets missing) -> not a workflow failure
    return 1 if rc == 1 else 0


if __name__ == "__main__":
    raise SystemExit(main())
