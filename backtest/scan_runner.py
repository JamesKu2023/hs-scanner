#!/workspace/venv/bin/python
"""Wrapper: run the 15m H&S short scan, POST the webhook only on new actionable signals.

Does not place orders. Does not change trading rules.
Schedule (daemon mode): Asia/Shanghai 24h at :02/:17/:32/:47.
Night slots (00:00–08:59 Beijing) still scan and POST; payload sets quiet=True and night=True so the alert is silent.
Circuit breaker still resumes at 09:02 Beijing, not at the first midnight slot.

1H / 4H (2026-11-12): after the 15m part of the :02 slot, the same daemon runs scan_htf.py
(1H every hour; 4H also at 00/04/08/12/16/20 Beijing = Binance/OKX 4H closes). Those rows carry
tier="htf" + timeframe, are unvalidated (0.5% risk), have their own paper books
(scan_out/paper_htf_1h_*, paper_htf_4h_*), and neither count toward nor are silenced by the
15m formal circuit breaker.
Webhook (2026-11-12, 三栋): only post-break 可入场 (a closed bearish bar below the neckline,
still inside the validity window). Pre-break 观望 is written to signals.json and logged, but
not POSTed; when the same head later breaks, the 可入场 alert fires normally. Nested outer
notes therefore only ride on post-break parent alerts. Nested inner and 15m formal 观望 follow
the same rule; 15m 观察 (near-miss) is already post-break via live_entries.

Catch-up (2026-11-12): if the box was paused and a 1H/4H close was missed, the next cycle
(and daemon startup) compares last_bar in scan_out/htf/<tf>/signals.json with the latest
closed bar and runs any lagging timeframe immediately. 15m already scans the latest closed
bar each slot; a missed 15m slot is also caught up on startup if last_bar is behind.
Freshness / post-break rules are unchanged: stale breaks are not emitted as 可入场, and
pre-break 观望 is still not POSTed.

Nested H&S (2026-11-12, 三栋 UNI 4H): rows with nested_signal=True (a small H&S whose head is
the outer pattern's right shoulder) are stripped out of the regular 15m / 1H / 4H flow, journaled in
their own books (scan_out/paper_nested_{15m,1h,4h}_*), and pushed in a separate payload
(nested=True). Unvalidated, 0.5% risk, never count toward nor are silenced by the 15m breaker.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

import paper_journal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VENV_PY = Path("/workspace/venv/bin/python")
if not VENV_PY.exists():  # GitHub Actions / any other host
    VENV_PY = Path(sys.executable)
SCAN_PY = HERE / "scan_15m.py"
SCAN_HTF_PY = HERE / "scan_htf.py"
HTF_OUT_DIR = HERE / "scan_out" / "htf"
BUILD_PY = HERE / "build_core_pool.py"
POOL_PATH = ROOT / "core_pool.json"
OUT_DIR = HERE / "scan_out"
SIGNALS_PATH = OUT_DIR / "signals.json"
STATE_PATH = HERE / "pushed_state.json"
LOG_PATH = HERE / "scan_runner.log"
LOCK_PATH = HERE / "scan_runner.lock"
PID_PATH = HERE / "scan_daemon.pid"
HEARTBEAT_PATH = HERE / "scan_heartbeat.json"
SLEEP_CHUNK_SEC = 10        # never sleep longer than this without re-checking the wall clock
HEARTBEAT_LOG_EVERY_SEC = 600  # also drop a heartbeat line in the log every 10 min
MISSED_SLOT_GRACE_SEC = 90  # woke this late after a slot -> log it as missed (still run once)
SLOT_MINUTES = (2, 17, 32, 47)
QUIET_END_HOUR = 9          # 00:00-08:59 Beijing: scan+POST, but payload quiet/night=True
EXCHANGE_CN = {"binance": "币安", "okx": "欧易"}
ENV_FILE = Path.home() / ".config" / "hs-scanner" / "webhook.env"
URL_FILE = Path.home() / ".config" / "hs-scanner" / "webhook_url"


def load_webhook_url() -> str:
    """HS_WEBHOOK_URL env (GitHub Secret) > HS_WEBHOOK_URL= line in webhook.env > webhook_url file.
    The URL is not committed to the repo."""
    url = os.environ.get("HS_WEBHOOK_URL", "").strip()
    if url:
        return url
    try:
        for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("HS_WEBHOOK_URL="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    except OSError:
        pass
    try:
        return URL_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


WEBHOOK_URL = load_webhook_url()
IS_GITHUB = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
PUSHER = "github" if IS_GITHUB else "box"
# Box standby switch (2026-11-13): once GitHub Actions is the primary pusher, the box daemon keeps
# scanning/journaling but stops POSTing. Enable with this flag file or HS_PUSH_DISABLED=1.
PUSH_DISABLED_FLAG = HERE / "box_push_disabled"
DATA_SOURCE_NOTE_SPOT = "Binance 数据源：现货K线（GitHub 无法访问合约接口）"
TZ = ZoneInfo("Asia/Shanghai")
POOL_MAX_AGE_SEC = 7 * 86400
SCAN_TIMEOUT_SEC = 600
STATE_KEEP_DAYS = 14
# Webhook only after a bearish close below the neckline. Pre-break 观望 stays in
# signals.json / logs so a later break can fire as 可入场. 15m 观察 (near-miss) already
# uses the post-break live_entries gate.
ACTIONABLE = {"可入场", "观察", "已破位·等回踩"}  # 15m webhook statuses (no pre-break formal 观望)
STATUS_OBSERVE = "观察"
STATUS_RETEST = "已破位·等回踩"
TIER_FORMAL = "formal"
TIER_WATCH = "watch"
TIER_HTF = "htf"
HTF_TFS = ("1h", "4h")
HTF_SLOT_MINUTE = 2          # first slot after the hourly close
HTF_4H_HOURS = (0, 4, 8, 12, 16, 20)  # Beijing; == UTC 16/20/0/4/8/12 4H closes
HTF_ACTIONABLE = {"可入场", "已破位·等回踩"}  # 1H/4H: never webhook 观望; retest is a post-break limit
HTF_TF_CN = {"1h": "1小时", "4h": "4小时"}
WATCH_DAILY_CAP = None  # 观察档不限每日条数；watch_alerts_today 仍作计数
# Try in this order when --test-webhook (or first real post if Bearer is unknown).
AUTH_STYLES = ("bearer", "x-webhook-key", "x-api-key", "raw-authorization")
PREFERRED_AUTH_FILE = Path.home() / ".config" / "hs-scanner" / "auth_style"


def now_sh() -> datetime:
    return datetime.now(TZ)


def _stdout_is_log() -> bool:
    try:
        out = os.fstat(sys.stdout.fileno())
        st = LOG_PATH.stat()
        return out.st_ino == st.st_ino and out.st_dev == st.st_dev
    except OSError:
        return False


def log(msg: str) -> None:
    line = f"{now_sh().strftime('%Y-%m-%d %H:%M:%S')} Asia/Shanghai {msg}"
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    if not _stdout_is_log():
        print(line, flush=True)


def load_webhook_key() -> str:
    key = os.environ.get("HS_WEBHOOK_KEY", "").strip()
    if key:
        return key
    if ENV_FILE.is_file():
        for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("HS_WEBHOOK_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    raise RuntimeError(
        f"HS_WEBHOOK_KEY not in environment and not readable from {ENV_FILE}"
    )


def auth_headers(style: str, key: str) -> dict[str, str]:
    if style == "bearer":
        return {"Authorization": f"Bearer {key}"}
    if style == "x-webhook-key":
        return {"x-webhook-key": key}
    if style == "x-api-key":
        return {"x-api-key": key}
    if style == "raw-authorization":
        return {"Authorization": key}
    raise ValueError(f"unknown auth style {style}")


def remembered_auth_style() -> str | None:
    if PREFERRED_AUTH_FILE.is_file():
        val = PREFERRED_AUTH_FILE.read_text(encoding="utf-8").strip()
        if val in AUTH_STYLES:
            return val
    return None


def remember_auth_style(style: str) -> None:
    PREFERRED_AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    PREFERRED_AUTH_FILE.write_text(style + "\n", encoding="utf-8")
    os.chmod(PREFERRED_AUTH_FILE, 0o600)


def push_disabled() -> bool:
    return os.environ.get("HS_PUSH_DISABLED", "").strip() in ("1", "true", "yes") or PUSH_DISABLED_FLAG.exists()


def webhook_key_or_none() -> str | None:
    try:
        return load_webhook_key()
    except RuntimeError:
        return None


def decorate_payload(payload: dict) -> dict:
    """pusher + data-source notes (in place). Never touches secrets."""
    payload["pusher"] = PUSHER
    srcs = sorted({s.get("data_source") for s in (payload.get("signals") or [])
                   if isinstance(s, dict) and s.get("data_source")})
    payload["data_sources"] = srcs
    notes = []
    if any(x == "binance_spot" for x in srcs):
        notes.append(DATA_SOURCE_NOTE_SPOT)
    if any(x == "okx_swap_proxy" for x in srcs):
        notes.append("部分币安币种用欧易永续K线代替（币安现货无此币，GitHub 无法访问币安合约接口）")
    payload["data_source_note"] = "；".join(notes)
    return payload


def post_webhook(payload: dict, styles: list[str] | None = None) -> tuple[int, str, str]:
    """POST JSON. Returns (status_code, style_used, body_excerpt). Never logs the key.

    Status 0 = not sent (error, push disabled, or webhook secret missing): callers keep the
    signals unmarked so a later run can still send them.
    """
    decorate_payload(payload)
    if push_disabled():
        log(f"push disabled (standby: {PUSH_DISABLED_FLAG.name} / HS_PUSH_DISABLED); webhook not sent "
            f"kind={payload.get('kind')} n={len(payload.get('signals') or [])}")
        return 0, "disabled", "push disabled"
    key = webhook_key_or_none()
    if not key:
        log("HS_WEBHOOK_KEY 未配置（GitHub Secrets 里加 HS_WEBHOOK_URL / HS_WEBHOOK_KEY）；跳过推送 "
            f"kind={payload.get('kind')} n={len(payload.get('signals') or [])}")
        return 0, "no-key", "webhook key missing"
    if not WEBHOOK_URL:
        log("HS_WEBHOOK_URL 未配置（GitHub Secrets 里加 HS_WEBHOOK_URL）；跳过推送")
        return 0, "no-url", "webhook url missing"
    styles = styles or [remembered_auth_style() or "bearer"]
    last_status, last_style, last_excerpt = 0, styles[0], ""
    for style in styles:
        try:
            r = requests.post(
                WEBHOOK_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    **auth_headers(style, key),
                },
                timeout=30,
            )
        except requests.RequestException as e:
            last_status, last_style, last_excerpt = 0, style, f"{type(e).__name__}"
            log(f"webhook error style={style} err={type(e).__name__}")
            continue
        excerpt = (r.text or "").replace("\n", " ")[:200]
        last_status, last_style, last_excerpt = r.status_code, style, excerpt
        log(f"webhook HTTP {r.status_code} style={style} body={excerpt!r}")
        if 200 <= r.status_code < 300:
            remember_auth_style(style)
            return r.status_code, style, excerpt
        # Auth rejection: try the next style. Other 4xx/5xx: still try next in test mode.
        if r.status_code in (401, 403, 404) or style != styles[-1]:
            if r.status_code in (401, 403):
                continue
            # 404 might be wrong URL, no point trying more styles? still report.
            if r.status_code == 404:
                return r.status_code, style, excerpt
            continue
        return r.status_code, style, excerpt
    return last_status, last_style, last_excerpt


def pool_age_sec(path: Path) -> float | None:
    """Age from the file's asof_utc (git checkouts reset mtimes), else mtime; None if missing."""
    if not path.exists():
        return None
    try:
        asof = json.loads(path.read_text(encoding="utf-8")).get("asof_utc")
        if asof:
            t = datetime.strptime(str(asof)[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=ZoneInfo("UTC"))
            return max(0.0, time.time() - t.timestamp())
    except (OSError, ValueError, AttributeError):
        pass
    return time.time() - path.stat().st_mtime


def ensure_core_pool() -> None:
    age = pool_age_sec(POOL_PATH)
    missing = age is None
    if not missing and age <= POOL_MAX_AGE_SEC:
        return
    reason = "missing" if missing else f"age={int(age // 86400)}d"
    log(f"rebuilding core pool ({reason})")
    try:
        subprocess.run(
            [str(VENV_PY), str(BUILD_PY)],
            cwd=str(HERE),
            check=True,
            timeout=300,
        )
        return
    except Exception as e:  # noqa: BLE001  (fapi blocked on GitHub runners)
        log(f"core pool rebuild via fapi failed: {type(e).__name__}; trying spot-ticker fallback")
    try:
        import binance_data
        pool = binance_data.build_pool_from_spot()
        if pool.get("symbols"):
            POOL_PATH.write_text(json.dumps(pool, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            log(f"core pool rebuilt from spot tickers: {len(pool['symbols'])} symbols (perp-only coins dropped)")
            return
    except Exception as e:  # noqa: BLE001
        log(f"spot pool fallback failed: {type(e).__name__}: {e}")
    if missing:
        raise RuntimeError("core pool missing and could not be rebuilt")
    log("keeping the stale core pool")


def run_scan() -> int:
    ensure_core_pool()
    proc = subprocess.run(
        [str(VENV_PY), str(SCAN_PY)],
        cwd=str(HERE),
        timeout=SCAN_TIMEOUT_SEC,
    )
    return proc.returncode


def signal_id(sig: dict) -> str:
    """Stable cross-runner id (box daemon and GitHub Actions): exchange|symbol|tf|head_time|status."""
    tf = sig.get("timeframe") or "15m"
    if sig.get("nested_signal"):
        tf = f"nested-{tf}"
    return "|".join([str(sig.get("exchange") or "binance"), str(sig.get("symbol") or ""), str(tf),
                     str(sig.get("H_time") or ""), str(sig.get("status") or "")])


def pattern_id(sig: dict) -> str:
    # Binance and OKX symbol names never collide (BTCUSDT vs BTC-USDT-SWAP), so the key
    # stays identical to the pre-OKX format and old pushed_state.json entries remain valid.
    # 1H/4H: one key per head (tf|symbol|status|H_time); must match paper_journal.pattern_id.
    tf = sig.get("timeframe")
    if sig.get("nested_signal"):
        return "|".join(["nested", str(tf or "15m"), str(sig.get("symbol") or ""), str(sig.get("status") or ""),
                         str(sig.get("H_time") or "")])
    if tf in HTF_TFS:
        return "|".join([str(tf), str(sig.get("symbol") or ""), str(sig.get("status") or ""),
                         str(sig.get("H_time") or "")])
    return "|".join(
        [
            str(sig.get("symbol") or ""),
            str(sig.get("status") or ""),
            str(sig.get("L_time") or ""),
            str(sig.get("H_time") or ""),
            str(sig.get("R_time") or ""),
        ]
    )


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"pushed": {}}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("pushed"), dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"pushed": {}}


def save_state(state: dict) -> None:
    cutoff = now_sh() - timedelta(days=STATE_KEEP_DAYS)
    kept = {}
    for k, v in (state.get("pushed") or {}).items():
        ts = str((v or {}).get("first_pushed") or "")
        try:
            dt = datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ)
        except ValueError:
            dt = now_sh()
        if dt >= cutoff:
            kept[k] = v
    STATE_PATH.write_text(
        json.dumps({"pushed": kept}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def tier_of(sig: dict) -> str:
    t = sig.get("tier")
    if t in (TIER_FORMAL, TIER_WATCH, TIER_HTF):
        return t
    return TIER_WATCH if sig.get("status") == STATUS_OBSERVE else TIER_FORMAL


def watch_pushed_today(pushed: dict, today: str) -> int:
    """Count 观察 alerts actually sent on this Beijing calendar day."""
    n = 0
    for v in (pushed or {}).values():
        v = v or {}
        if v.get("tier") != TIER_WATCH or v.get("suppressed"):
            continue
        if str(v.get("first_pushed") or "")[:10] == today:
            n += 1
    return n


def split_fresh(fresh: list[dict], pushed: dict, today: str, cap: int | None = None):
    """Return (formal_rows, watch_rows_to_send, watch_rows_over_cap).

    Observe/watch alerts are not count-capped, so over_cap is always empty.
    The unused `cap` argument is kept so older callers do not break.
    """
    formal = [s for s in fresh if tier_of(s) == TIER_FORMAL]
    watch = [s for s in fresh if tier_of(s) == TIER_WATCH]
    return formal, watch, []


def is_quiet_hours(t: datetime | None = None) -> bool:
    """True during 00:00–08:59 Asia/Shanghai. Scans still run; webhook is marked quiet."""
    t = t or now_sh()
    if t.tzinfo is not None:
        t = t.astimezone(TZ)  # runner clock may be UTC (GitHub); night is Beijing time
    return t.hour < QUIET_END_HOUR


def build_payload(rows: list[dict], tier: str, scan_time: str, last_bar, n_watch_today: int = 0,
                  scan_meta: dict | None = None, now: datetime | None = None) -> dict:
    scan_meta = scan_meta or {}
    exchanges = sorted({(s.get("exchange") or "binance") for s in rows})
    quiet = is_quiet_hours(now)
    payload = {
        "test": False,
        "kind": "signals",
        "tier": tier,
        "scan_time": scan_time,
        "last_bar_shanghai": last_bar,
        "quiet": quiet,
        "night": quiet,
        "n_entry": sum(1 for s in rows if s.get("status") == "可入场"),
        "n_retest": sum(1 for s in rows if s.get("status") == STATUS_RETEST),
        "n_watch": sum(1 for s in rows if s.get("status") == "观望"),
        "n_observe": sum(1 for s in rows if s.get("status") == STATUS_OBSERVE),
        "signals": [summarize(s) for s in rows],
        "exchanges_in_payload": exchanges,
        "exchanges_in_payload_cn": [EXCHANGE_CN.get(e, e) for e in exchanges],
        "exchanges_scanned": scan_meta.get("exchanges") or ["binance"],
        "okx_ok": scan_meta.get("okx_ok", None),
        "okx_error": scan_meta.get("okx_error") or "",
        "okx_note": "欧易用同一规则扫描；规则只在币安数据上回测，欧易未单独验证。",
    }
    if tier == TIER_WATCH:
        payload["validated"] = False
        payload["suggested_risk_pct"] = 0.5
        payload["watch_alerts_today"] = n_watch_today
        payload["watch_daily_cap"] = WATCH_DAILY_CAP
        payload["disclaimer"] = "观察档：差一条的 near-miss，未经样本外验证，建议风险 0.5%，自行判断；不计入正式统计与熔断。"
    else:
        payload["validated"] = True
        payload["suggested_risk_pct"] = 1.0
    return payload


def summarize(sig: dict) -> dict:
    entry = sig.get("suggest_entry")
    if entry is None:
        entry = sig.get("official_entry")
    chart = sig.get("chart")
    exchange = sig.get("exchange") or "binance"
    tf = sig.get("timeframe") or "15m"
    return {
        "signal_id": signal_id(sig),
        "timeframe": tf,
        "data_source": sig.get("data_source") or ("okx_swap" if exchange == "okx" else "binance_fapi"),
        "chart_spec": sig.get("chart_spec"),
        "exchange": exchange,
        "exchange_cn": EXCHANGE_CN.get(exchange, exchange),
        "symbol": sig.get("symbol"),
        "coin": sig.get("coin"),
        "same_coin_on_other_exchange": sig.get("same_coin_on_other_exchange"),
        "status": sig.get("status"),
        "tier": tier_of(sig),
        "suggested_risk_pct": sig.get("suggested_risk_pct", 0.5 if tier_of(sig) == TIER_WATCH else 1.0),
        "missed_condition": sig.get("missed_condition"),
        "missed_condition_cn": sig.get("missed_condition_cn"),
        "missed_value": sig.get("missed_value"),
        "missed_by": sig.get("missed_by"),
        "missed_text": sig.get("missed_text"),
        "neck": sig.get("neck"),
        "neck_s2": sig.get("neck_s2"),
        "neck_class_cn": sig.get("neck_class_cn"),
        "neck_angle_deg": sig.get("neck_angle_deg"),
        "prop_ls_time": sig.get("prop_ls_time"),
        "prop_rs_time": sig.get("prop_rs_time"),
        "prop_ls_height": sig.get("prop_ls_height"),
        "prop_rs_height": sig.get("prop_rs_height"),
        "prop_class_cn": sig.get("prop_class_cn"),
        "prop_text": sig.get("prop_text"),
        "entry": entry,
        "official_entry": sig.get("official_entry"),
        "official_entry_time": sig.get("official_entry_time"),
        "break_level": sig.get("break_level"),
        "valid_until_shanghai": sig.get("valid_until_shanghai"),
        "leverage_cap": sig.get("leverage_cap"),
        "leverage_note": sig.get("leverage_note"),
        "stop": sig.get("stop"),
        "target_1h": sig.get("target_1h"),
        "target_15h": sig.get("target_15h"),
        "current": sig.get("current"),
        "rr": sig.get("rr"),
        "L_time": sig.get("L_time"),
        "H_time": sig.get("H_time"),
        "R_time": sig.get("R_time"),
        "note": sig.get("note"),
        "chart": chart,
        # 2026-11-12 嵌套头肩（display only; None on ordinary rows）
        "nested": bool(sig.get("nested")),
        "nested_role": sig.get("nested_role"),
        "nested_text": sig.get("nested_text"),
        "parent": sig.get("parent"),
        "child": sig.get("child"),
        "parent_neck_now": sig.get("parent_neck_now"),
        "parent_target_1h": sig.get("parent_target_1h"),
        "parent_target_15h": sig.get("parent_target_15h"),
        # 2026-11-12 fib extension + dual neckline (display / info only)
        "fib_ext": sig.get("fib_ext"),
        "fib_text": sig.get("fib_text"),
        "dual_neck_tier1": sig.get("dual_neck_tier1"),
        "dual_neck_tier2": sig.get("dual_neck_tier2"),
        "dual_neck_text": sig.get("dual_neck_text"),
    }


def acquire_lock():
    fp = open(LOCK_PATH, "a+")
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fp.close()
        return None
    fp.seek(0)
    fp.truncate()
    fp.write(str(os.getpid()) + "\n")
    fp.flush()
    return fp


def next_slot(after: datetime | None = None) -> datetime:
    now = after or now_sh()
    slots = [(h, m) for h in range(24) for m in SLOT_MINUTES]
    for h, m in slots:
        t = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if t > now + timedelta(seconds=1):
            return t
    tomorrow = (now + timedelta(days=1)).date()
    return datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, SLOT_MINUTES[0], tzinfo=TZ)


def pid_is_live(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not cmdline_path.exists():
        return False
    try:
        cmd = cmdline_path.read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return False
    return "scan_runner.py" in cmd


def write_pid() -> None:
    PID_PATH.write_text(str(os.getpid()) + "\n", encoding="utf-8")


def clear_pid_if_ours() -> None:
    try:
        if PID_PATH.exists() and PID_PATH.read_text(encoding="utf-8").strip() == str(os.getpid()):
            PID_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def run_once(dry_run: bool = False, journal_paths: dict | None = None,
             htf_tfs: list[str] | tuple[str, ...] | None = None, skip_15m: bool = False) -> int:
    """One scan cycle. dry_run=True: scan + journal into journal_paths (temp), never POST,
    never touch pushed_state.json. htf_tfs: also run the 1H/4H scan after the 15m part
    (15m alerts are never delayed by it)."""
    lock = acquire_lock()
    if lock is None:
        log("skip: another scan_runner is already running")
        return 0
    try:
        rc = 0 if skip_15m else _cycle_15m(dry_run=dry_run, journal_paths=journal_paths)
        if htf_tfs:
            try:
                rc_h = run_htf(list(htf_tfs), dry_run=dry_run, journal_paths=journal_paths)
            except Exception:
                log("htf runner error:\n" + traceback.format_exc())
                rc_h = 1
            rc = rc or rc_h
        return rc
    finally:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock.close()


def strip_nested(data: dict) -> tuple[dict, list[dict]]:
    """Split nested inner signals out of a scan output; the rest flows exactly as before."""
    sigs = [s for s in (data.get("signals") or []) if isinstance(s, dict)]
    nested = [s for s in sigs if s.get("nested_signal")]
    if not nested:
        return data, []
    return {**data, "signals": [s for s in sigs if not s.get("nested_signal")]}, nested


def _cycle_15m(dry_run: bool = False, journal_paths: dict | None = None) -> int:
    """15m part: regular cycle (unchanged), then the nested inner signals (own book / payload)."""
    box: list = []
    rc = _cycle_15m_regular(dry_run=dry_run, journal_paths=journal_paths, nested_box=box)
    if box:
        try:
            rc_n = run_nested(box[0][1], "15m", box[0][0], dry_run=dry_run, journal_paths=journal_paths)
        except Exception:
            log("nested 15m error:\n" + traceback.format_exc())
            rc_n = 1
        rc = rc or rc_n
    return rc


def _cycle_15m_regular(dry_run: bool = False, journal_paths: dict | None = None,
                       nested_box: list | None = None) -> int:
    """The 15m part of a cycle (unchanged logic). Caller holds the lock."""
    try:
        rc = run_scan()
        if rc != 0:
            log(f"scan exited {rc}; no webhook")
            return rc
        if not SIGNALS_PATH.exists():
            log("missing signals.json; no webhook")
            return 2
        data = json.loads(SIGNALS_PATH.read_text(encoding="utf-8"))
        data, nested_rows = strip_nested(data)
        if nested_box is not None:
            nested_box.append((data, nested_rows))
        rows = [
            s
            for s in (data.get("signals") or [])
            if isinstance(s, dict) and s.get("status") in ACTIONABLE
        ]
        n_entry = sum(1 for s in rows if s.get("status") == "可入场")
        n_retest = sum(1 for s in rows if s.get("status") == STATUS_RETEST)
        n_watch = sum(1 for s in rows if s.get("status") == "观望")
        n_observe = sum(1 for s in rows if s.get("status") == STATUS_OBSERVE)
        scan_time = data.get("asof_shanghai") or now_sh().strftime("%Y-%m-%d %H:%M")
        formal_paths = journal_paths.get("formal") if journal_paths else None
        watch_paths = journal_paths.get("watch") if journal_paths else None
        journal = None
        try:
            journal = paper_journal.update_from_scan(data, paths=formal_paths)
            log(paper_journal.journal_log_line(journal))
        except Exception:
            log("journal update error:\n" + traceback.format_exc())
        wjournal = None
        try:
            wjournal = paper_journal.update_from_scan(
                data, paths=watch_paths or paper_journal.JournalPaths.watch(), book=paper_journal.BOOK_WATCH
            )
            log(paper_journal.journal_log_line(wjournal))
        except Exception:
            log("watch journal update error:\n" + traceback.format_exc())

        state = load_state()
        pushed = state["pushed"]
        fresh = [s for s in rows if pattern_id(s) not in pushed]
        log(
            f"scan ok entries={n_entry} retests={n_retest} watches={n_watch} (15m 观望 kept internal, not pushed) "
            f"observes={n_observe} fresh={len(fresh)} last_bar={data.get('last_bar_shanghai', '')} "
            f"okx_ok={data.get('okx_ok')} okx_scanned={data.get('okx_scanned')} "
            f"scan_s={data.get('scan_seconds_total')} (binance={data.get('scan_seconds_binance')} "
            f"okx={data.get('scan_seconds_okx')})"
        )
        if data.get("okx_ok") is False:
            log(f"OKX skipped/partial this slot: {data.get('okx_error')}")

        plan = paper_journal.notification_plan(journal) if journal is not None else {
            "suppress_trading": False,
            "send_breaker_notice": False,
            "payload": None,
        }
        if dry_run:
            today = now_sh().strftime("%Y-%m-%d")
            f_rows, w_rows, w_over = split_fresh(fresh, pushed, today)
            log(
                f"DRY-RUN no webhook: breaker_suppress={bool(plan.get('suppress_trading'))} "
                f"would_send formal={len(f_rows)} watch={len(w_rows)}"
            )
            return 0
        if plan.get("send_breaker_notice") and plan.get("payload"):
            payload = dict(plan["payload"])
            payload["scan_time"] = scan_time
            style = remembered_auth_style() or "bearer"
            status, style, _ = post_webhook(
                payload, styles=[style] + [s for s in AUTH_STYLES if s != style]
            )
            if 200 <= status < 300:
                paper_journal.mark_notice_sent(journal)
                log(f"circuit breaker notice sent style={style}")
            else:
                log(f"circuit breaker notice failed HTTP {status}; will retry next run")
        if plan.get("suppress_trading"):
            # Formal breaker also silences the 观察 tier (watch book still journals).
            log("breaker tripped; trading webhook suppressed (formal and watch)")
            return 0

        if not fresh:
            return 0
        today = now_sh().strftime("%Y-%m-%d")
        f_rows, w_rows, _w_over = split_fresh(fresh, pushed, today)
        rc = 0
        ts = now_sh().strftime("%Y-%m-%d %H:%M:%S")
        # 1) formal first (priority)
        if f_rows:
            payload = build_payload(f_rows, TIER_FORMAL, scan_time, data.get("last_bar_shanghai"), scan_meta=data)
            style = remembered_auth_style() or "bearer"
            status, style, _ = post_webhook(payload, styles=[style] + [s for s in AUTH_STYLES if s != style])
            if 200 <= status < 300:
                for s in f_rows:
                    pushed[pattern_id(s)] = {
                        "first_pushed": ts, "symbol": s.get("symbol"), "exchange": s.get("exchange") or "binance",
                        "status": s.get("status"), "tier": TIER_FORMAL,
                    }
                save_state(state)
                if journal is not None:
                    paper_journal.mark_alerted(
                        journal,
                        [pattern_id(s) for s in f_rows if s.get("status") == "可入场"],
                        paths=paper_journal.JournalPaths.default(),
                    )
                log(f"pushed {len(f_rows)} new formal signal(s) style={style}")
            else:
                log(f"formal webhook failed HTTP {status}; signals not marked pushed, will retry next run")
                rc = 3
        # 2) watch/observe tier — no daily count limit
        if w_rows:
            n_today = watch_pushed_today(pushed, today) + len(w_rows)
            payload = build_payload(w_rows, TIER_WATCH, scan_time, data.get("last_bar_shanghai"), n_today,
                                    scan_meta=data)
            style = remembered_auth_style() or "bearer"
            status, style, _ = post_webhook(payload, styles=[style] + [s for s in AUTH_STYLES if s != style])
            if 200 <= status < 300:
                for s in w_rows:
                    pushed[pattern_id(s)] = {
                        "first_pushed": ts, "symbol": s.get("symbol"), "exchange": s.get("exchange") or "binance",
                        "status": s.get("status"), "tier": TIER_WATCH,
                    }
                save_state(state)
                if wjournal is not None:
                    paper_journal.mark_alerted(
                        wjournal, [pattern_id(s) for s in w_rows],
                        paths=watch_paths or paper_journal.JournalPaths.watch(),
                    )
                log(f"pushed {len(w_rows)} new watch signal(s) style={style}")
            else:
                log(f"watch webhook failed HTTP {status}; will retry next run")
                rc = rc or 3
        return rc
    except subprocess.TimeoutExpired:
        log("scan timed out; no webhook")
        return 4
    except Exception:
        log("runner error:\n" + traceback.format_exc())
        return 1


# --------------------------------------------------------------------------- 1H / 4H
def htf_due(slot: datetime) -> list[str]:
    """Timeframes whose bar just closed at this Beijing slot: 1H at every :02, 4H at 00/04/../20:02."""
    if slot.minute != HTF_SLOT_MINUTE:
        return []
    tfs = ["1h"]
    if slot.hour in HTF_4H_HOURS:
        tfs.append("4h")
    return tfs


def latest_closed_bar(tf: str, now: datetime | None = None) -> datetime:
    """Open time (Asia/Shanghai) of the latest fully closed bar for tf.

    Exchange candles are labeled by open time: the 4H bar that closes at 16:00 Beijing
    is last_bar_shanghai=12:00 (covers [12:00, 16:00)). At an exact close boundary the
    bar that just closed is the previous one, not the one that is opening now.
    """
    now = now or now_sh()
    if now.tzinfo is None:
        now = now.replace(tzinfo=TZ)
    else:
        now = now.astimezone(TZ)
    if tf == "15m":
        minute = (now.minute // 15) * 15
        floored = now.replace(minute=minute, second=0, microsecond=0)
        return floored - timedelta(minutes=15)
    if tf == "1h":
        floored = now.replace(minute=0, second=0, microsecond=0)
        return floored - timedelta(hours=1)
    if tf == "4h":
        floored = now.replace(minute=0, second=0, microsecond=0)
        current_open = floored.replace(hour=(floored.hour // 4) * 4)
        return current_open - timedelta(hours=4)
    raise ValueError(f"unknown tf {tf}")


def parse_bar_shanghai(text: str | None) -> datetime | None:
    if not text or not str(text).strip():
        return None
    raw = str(text).strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=TZ)
        except ValueError:
            continue
    return None


def signals_path_for(tf: str, signals_dir: Path | None = None) -> Path:
    if tf in HTF_TFS:
        return (signals_dir or HTF_OUT_DIR) / tf / "signals.json"
    return (signals_dir / "signals.json") if signals_dir is not None else SIGNALS_PATH


def read_last_scanned_bar(tf: str, signals_dir: Path | None = None,
                          path: Path | None = None) -> datetime | None:
    """last_bar_shanghai from the tf's signals.json, or None if missing/unreadable."""
    p = path or signals_path_for(tf, signals_dir)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return parse_bar_shanghai(data.get("last_bar_shanghai") or "")


def bar_unscanned(tf: str, now: datetime | None = None, signals_dir: Path | None = None,
                  path: Path | None = None) -> bool:
    """True if signals.json has no last_bar, or last_bar is older than the latest closed bar."""
    last = read_last_scanned_bar(tf, signals_dir=signals_dir, path=path)
    latest = latest_closed_bar(tf, now)
    return last is None or last < latest


def htf_catchup(now: datetime | None = None, signals_dir: Path | None = None) -> list[str]:
    """1H/4H timeframes whose latest closed bar is not yet recorded in signals.json."""
    now = now or now_sh()
    return [tf for tf in HTF_TFS if bar_unscanned(tf, now=now, signals_dir=signals_dir)]


def resolve_htf_tfs(scheduled: list[str] | tuple[str, ...] | None = None,
                    now: datetime | None = None, signals_dir: Path | None = None) -> list[str]:
    """Union of slot-scheduled TFs and catch-up TFs, in HTF_TFS order (1h then 4h)."""
    want = set(scheduled or ())
    want.update(htf_catchup(now=now, signals_dir=signals_dir))
    return [tf for tf in HTF_TFS if tf in want]


def catchup_plan(now: datetime | None = None, signals_dir: Path | None = None,
                 m15_path: Path | None = None) -> dict:
    """What a catch-up pass should run. Pure: no logs, no POST, no file writes."""
    now = now or now_sh()
    tfs = htf_catchup(now=now, signals_dir=signals_dir)
    need_15m = bar_unscanned("15m", now=now, path=m15_path or SIGNALS_PATH)
    return {"htf_tfs": tfs, "need_15m": need_15m}


def run_catchup_if_needed(now: datetime | None = None, signals_dir: Path | None = None,
                          m15_path: Path | None = None, dry_run: bool = False,
                          journal_paths: dict | None = None) -> int:
    """On startup / after a pause: scan lagging 15m and 1H/4H bars. No-op if already current.

    Uses the same run_once path, so post-break-only webhook and freshness windows apply.
    """
    plan = catchup_plan(now=now, signals_dir=signals_dir, m15_path=m15_path)
    tfs, need_15m = plan["htf_tfs"], plan["need_15m"]
    if not tfs and not need_15m:
        return 0
    log(f"catch-up scan tfs={','.join(tfs) or '-'} 15m={need_15m} "
        f"(latest closed bar not yet in signals.json)")
    return run_once(dry_run=dry_run, journal_paths=journal_paths, htf_tfs=tfs, skip_15m=not need_15m)


def run_htf_scan(tfs: list[str]) -> int:
    ensure_core_pool()
    proc = subprocess.run(
        [str(VENV_PY), str(SCAN_HTF_PY), "--tf", ",".join(tfs)],
        cwd=str(HERE),
        timeout=SCAN_TIMEOUT_SEC,
    )
    return proc.returncode


def summarize_htf(sig: dict) -> dict:
    d = summarize(sig)
    d.update({
        "tier": TIER_HTF,
        "timeframe": sig.get("timeframe"),
        "timeframe_cn": sig.get("timeframe_cn") or HTF_TF_CN.get(sig.get("timeframe"), ""),
        "validated": False,
        "suggested_risk_pct": sig.get("suggested_risk_pct", 0.5),
        "official_entry_time": sig.get("official_entry_time"),
        "break_bar_shanghai": sig.get("break_bar_shanghai"),
        "break_level": sig.get("break_level"),
        "head_height": sig.get("head_height"),
        "dist_stop_pct": sig.get("dist_stop_pct"),
        "dist_t1_pct": sig.get("dist_t1_pct"),
        "leverage_cap": sig.get("leverage_cap"),
        "leverage_note": sig.get("leverage_note"),
        "valid_until_shanghai": sig.get("valid_until_shanghai"),
        "htf_trend_bullish": sig.get("htf_trend_bullish"),
        "htf_trend_note": sig.get("htf_trend_note"),
        "confirm_text": sig.get("confirm_text"),
        "exits": sig.get("exits"),
    })
    return d


def build_htf_payload(rows: list[dict], tf: str, scan_time: str, last_bar,
                      scan_meta: dict | None = None, now: datetime | None = None) -> dict:
    scan_meta = scan_meta or {}
    exchanges = sorted({(s.get("exchange") or "binance") for s in rows})
    quiet = is_quiet_hours(now)
    return {
        "test": False,
        "kind": "signals",
        "tier": TIER_HTF,
        "timeframe": tf,
        "timeframe_cn": HTF_TF_CN.get(tf, tf),
        "scan_time": scan_time,
        "last_bar_shanghai": last_bar,
        "quiet": quiet,
        "night": quiet,
        "validated": False,
        "suggested_risk_pct": 0.5,
        "counts_toward_breaker": False,
        "n_entry": sum(1 for s in rows if s.get("status") == "可入场"),
        "n_retest": sum(1 for s in rows if s.get("status") == STATUS_RETEST),
        "n_watch": sum(1 for s in rows if s.get("status") == "观望"),
        "n_observe": 0,
        "signals": [summarize_htf(s) for s in rows],
        "exchanges_in_payload": exchanges,
        "exchanges_in_payload_cn": [EXCHANGE_CN.get(e, e) for e in exchanges],
        "exchanges_scanned": scan_meta.get("exchanges") or ["binance"],
        "okx_ok": scan_meta.get("okx_ok", None),
        "okx_error": scan_meta.get("okx_error") or "",
        "rule": scan_meta.get("rule") or "v3 价格认肩（左肩回调按收盘价）+ 收盘破颈线 + R:R≥1.5",
        "disclaimer": (
            f"{HTF_TF_CN.get(tf, tf)}头肩（未验证）：v3 价格认肩，量/RSI/MACD 只作确认；"
            "1H/4H 历史回测为负、样本太薄，建议风险 0.5%，单独记账，不计入也不受 15m 熔断影响。只通知，不下单。"
        ),
    }


def run_htf(tfs: list[str], dry_run: bool = False, journal_paths: dict | None = None) -> int:
    """1H/4H part of a cycle. Never silenced by the 15m breaker; own books; no daily cap."""
    t0 = time.monotonic()
    try:
        rc = run_htf_scan(tfs)
    except subprocess.TimeoutExpired:
        log(f"htf scan timed out tfs={tfs}; no webhook")
        return 4
    if rc != 0:
        log(f"htf scan exited {rc} tfs={tfs}; no webhook")
        return rc
    out_rc = 0
    pending_nested: list = []
    for tf in tfs:
        path = HTF_OUT_DIR / tf / "signals.json"
        if not path.exists():
            log(f"htf {tf}: missing {path.name}; no webhook")
            out_rc = out_rc or 2
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data, nested_rows = strip_nested(data)
        pending_nested.append((nested_rows, tf, data))
        rows = [s for s in (data.get("signals") or [])
                if isinstance(s, dict) and s.get("status") in HTF_ACTIONABLE]
        for s in rows:
            s.setdefault("timeframe", tf)
        book = paper_journal.HTF_BOOKS[tf]
        jpaths = (journal_paths or {}).get(book) if journal_paths else None
        journal = None
        try:
            journal = paper_journal.update_from_scan(
                data, paths=jpaths or paper_journal.JournalPaths.htf(tf), book=book)
            log(paper_journal.journal_log_line(journal))
        except Exception:
            log(f"htf {tf} journal update error:\n" + traceback.format_exc())
        state = load_state()
        pushed = state["pushed"]
        fresh = [s for s in rows if pattern_id(s) not in pushed]
        log(f"htf {tf} scan ok entries={data.get('n_entry')} retests={data.get('n_retest')} watches={data.get('n_watch')} "
            f"(watches kept internal, not pushed) fresh_entry={len(fresh)} "
            f"last_bar={data.get('last_bar_shanghai', '')} okx_ok={data.get('okx_ok')} "
            f"scan_s={data.get('scan_seconds_total')}")
        if dry_run:
            log(f"DRY-RUN no webhook: htf {tf} would_send={len(fresh)}")
            continue
        if not fresh:
            continue
        scan_time = data.get("asof_shanghai") or now_sh().strftime("%Y-%m-%d %H:%M")
        payload = build_htf_payload(fresh, tf, scan_time, data.get("last_bar_shanghai"), scan_meta=data)
        style = remembered_auth_style() or "bearer"
        status, style, _ = post_webhook(payload, styles=[style] + [s for s in AUTH_STYLES if s != style])
        if 200 <= status < 300:
            ts = now_sh().strftime("%Y-%m-%d %H:%M:%S")
            for s in fresh:
                pushed[pattern_id(s)] = {
                    "first_pushed": ts, "symbol": s.get("symbol"), "exchange": s.get("exchange") or "binance",
                    "status": s.get("status"), "tier": TIER_HTF, "timeframe": tf,
                }
            save_state(state)
            if journal is not None:
                paper_journal.mark_alerted(
                    journal, [pattern_id(s) for s in fresh if s.get("status") == "可入场"],
                    paths=jpaths or paper_journal.JournalPaths.htf(tf),
                )
            log(f"pushed {len(fresh)} new htf {tf} signal(s) style={style}")
        else:
            log(f"htf {tf} webhook failed HTTP {status}; will retry next run")
            out_rc = out_rc or 3
    for nested_rows, tf, data in pending_nested:  # after the regular (outer) alerts of this slot
        try:
            rc_n = run_nested(nested_rows, tf, data, dry_run=dry_run, journal_paths=journal_paths)
        except Exception:
            log(f"nested {tf} error:\n" + traceback.format_exc())
            rc_n = 1
        out_rc = out_rc or rc_n
    log(f"htf cycle done tfs={','.join(tfs)} in {time.monotonic() - t0:.1f}s")
    return out_rc


# --------------------------------------------------------------------------- nested H&S
NESTED_ACTIONABLE = {"可入场", "已破位·等回踩"}  # inner already requires a close below the small neckline to push
NESTED_TF_CN = {"15m": "15分钟", "1h": "1小时", "4h": "4小时"}


def build_nested_payload(rows: list[dict], tf: str, scan_time: str, last_bar,
                         scan_meta: dict | None = None, now: datetime | None = None) -> dict:
    scan_meta = scan_meta or {}
    exchanges = sorted({(s.get("exchange") or "binance") for s in rows})
    quiet = is_quiet_hours(now)
    tier = TIER_WATCH if tf == "15m" else TIER_HTF
    sigs = []
    for s in rows:
        d = summarize_htf(s)
        d["tier"] = tier
        d["timeframe"] = tf
        d["timeframe_cn"] = NESTED_TF_CN.get(tf, tf)
        sigs.append(d)
    return {
        "test": False,
        "kind": "signals",
        "nested": True,
        "tier": tier,
        "timeframe": tf,
        "timeframe_cn": NESTED_TF_CN.get(tf, tf),
        "scan_time": scan_time,
        "last_bar_shanghai": last_bar,
        "quiet": quiet,
        "night": quiet,
        "validated": False,
        "suggested_risk_pct": 0.5,
        "counts_toward_breaker": False,
        "n_entry": sum(1 for s in rows if s.get("status") == "可入场"),
        "n_retest": sum(1 for s in rows if s.get("status") == STATUS_RETEST),
        "n_watch": sum(1 for s in rows if s.get("status") == "观望"),
        "n_observe": 0,
        "signals": sigs,
        "exchanges_in_payload": exchanges,
        "exchanges_in_payload_cn": [EXCHANGE_CN.get(e, e) for e in exchanges],
        "exchanges_scanned": scan_meta.get("exchanges") or ["binance"],
        "okx_ok": scan_meta.get("okx_ok", None),
        "okx_error": scan_meta.get("okx_error") or "",
        "rule": "嵌套头肩：大头肩的右肩作小头，同一套 v3 价格认肩（k=1.0u）找小LS/小RS；小颈线收盘破位入场，"
                "止损小右肩+0.5ATR，TP 按小形态头高，大颈线/大目标为后续目标；量/RSI/MACD 只作确认",
        "disclaimer": (
            f"{NESTED_TF_CN.get(tf, tf)}嵌套小头肩（未验证）：建议风险 0.5%，独立记账（paper_nested_{tf}），"
            "不计入也不受 15m 熔断影响。只通知，不下单。"
        ),
    }


def run_nested(rows: list[dict], tf: str, data: dict, dry_run: bool = False,
               journal_paths: dict | None = None) -> int:
    """Journal + push nested inner signals for one timeframe. Own book; no breaker; no cap."""
    rows = [s for s in (rows or []) if isinstance(s, dict) and s.get("status") in NESTED_ACTIONABLE]
    for s in rows:
        s.setdefault("timeframe", tf)
    book = paper_journal.NESTED_BOOKS[tf]
    jpaths = (journal_paths or {}).get(book) if journal_paths else None
    jpaths = jpaths or paper_journal.JournalPaths.nested(tf)
    journal = None
    try:
        journal = paper_journal.update_from_scan({**data, "signals": rows}, paths=jpaths, book=book)
        log(paper_journal.journal_log_line(journal))
    except Exception:
        log(f"nested {tf} journal update error:\n" + traceback.format_exc())
    state = load_state()
    pushed = state["pushed"]
    fresh = [s for s in rows if pattern_id(s) not in pushed]
    log(f"nested {tf} rows={len(rows)} fresh_entry={len(fresh)} (pre-break 观望 not pushed)")
    if dry_run:
        if fresh:
            log(f"DRY-RUN no webhook: nested {tf} would_send={len(fresh)}")
        return 0
    if not fresh:
        return 0
    scan_time = data.get("asof_shanghai") or now_sh().strftime("%Y-%m-%d %H:%M")
    payload = build_nested_payload(fresh, tf, scan_time, data.get("last_bar_shanghai"), scan_meta=data)
    style = remembered_auth_style() or "bearer"
    status, style, _ = post_webhook(payload, styles=[style] + [s for s in AUTH_STYLES if s != style])
    if 200 <= status < 300:
        ts = now_sh().strftime("%Y-%m-%d %H:%M:%S")
        for s in fresh:
            pushed[pattern_id(s)] = {
                "first_pushed": ts, "symbol": s.get("symbol"), "exchange": s.get("exchange") or "binance",
                "status": s.get("status"), "tier": "nested", "timeframe": tf,
            }
        save_state(state)
        if journal is not None:
            paper_journal.mark_alerted(journal, [pattern_id(s) for s in fresh if s.get("status") == "可入场"],
                                       paths=jpaths)
        log(f"pushed {len(fresh)} new nested {tf} signal(s) style={style}")
        return 0
    log(f"nested {tf} webhook failed HTTP {status}; will retry next run")
    return 3


def test_webhook() -> int:
    payload = {
        "test": True,
        "scan_time": now_sh().strftime("%Y-%m-%d %H:%M"),
        "signals": [],
        "note": "connectivity test only; no trading signals",
    }
    # Prefer trying all styles until one 2xx, so we can report which works.
    status, style, excerpt = post_webhook(payload, styles=list(AUTH_STYLES))
    print(f"TEST_POST status={status} style={style} excerpt={excerpt!r}")
    return 0 if 200 <= status < 300 else 1


def write_heartbeat(state: str, next_slot_dt: datetime | None = None, extra: dict | None = None) -> None:
    """Small JSON file whose mtime/ts proves the daemon loop is alive (ensure_scanner.sh checks it)."""
    now = now_sh()
    data = {
        "pid": os.getpid(),
        "ts_epoch": int(time.time()),
        "ts_shanghai": now.strftime("%Y-%m-%d %H:%M:%S"),
        "state": state,
        "next_slot_shanghai": next_slot_dt.strftime("%Y-%m-%d %H:%M:%S") if next_slot_dt else None,
    }
    if extra:
        data.update(extra)
    tmp = HEARTBEAT_PATH.with_name(HEARTBEAT_PATH.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(HEARTBEAT_PATH)
    except OSError:
        pass


def wait_until(target: datetime, *, chunk: float = SLEEP_CHUNK_SEC, sleep=time.sleep,
               clock=None, mono=time.monotonic, on_tick=None) -> dict:
    """Sleep until wall-clock `target` in short chunks.

    time.sleep() runs on the monotonic clock, which does not advance while the box VM is
    paused/snapshotted. Re-checking the wall clock after every short chunk means that after a
    resume we notice within `chunk` seconds that the slot has passed. Returns
    {"late_sec": wall seconds past target on wake, "jumps": [(wall_gap, mono_gap), ...]}.
    """
    clock = clock or now_sh
    jumps = []
    while True:
        now = clock()
        left = (target - now).total_seconds()
        if left <= 0:
            return {"late_sec": -left, "jumps": jumps}
        step = min(left, chunk)
        w0, m0 = clock(), mono()
        sleep(step)
        w_gap = (clock() - w0).total_seconds()
        m_gap = mono() - m0
        if w_gap - m_gap > 60:  # wall clock moved much more than we slept: VM pause / suspend
            jumps.append((round(w_gap, 1), round(m_gap, 1)))
        if on_tick:
            on_tick()


def daemon_step(now: datetime, target: datetime, late_sec: float) -> str:
    """Decide what to do after waking for `target`. Returns 'run' or 'run_missed'.

    A missed slot is run once immediately (no catch-up burst): the caller recomputes the
    next slot from the current wall clock afterwards. Scans run 24 hours; night posts
    carry quiet=True / night=True in the webhook payload.
    """
    if late_sec > MISSED_SLOT_GRACE_SEC:
        return "run_missed"
    return "run"


def daemon() -> int:
    if PID_PATH.exists():
        try:
            old = int(PID_PATH.read_text(encoding="utf-8").strip())
        except ValueError:
            old = 0
        if old != os.getpid() and pid_is_live(old):
            log(f"daemon already running pid={old}; exiting")
            return 0
    write_pid()

    def _stop(signum, _frame):
        log(f"daemon got signal {signum}, stopping")
        clear_pid_if_ours()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    log(f"daemon started pid={os.getpid()} (sleep chunk {SLEEP_CHUNK_SEC}s, heartbeat {HEARTBEAT_PATH.name})")
    last_hb_log = [time.time()]
    try:
        write_heartbeat("scanning", None, {"catchup": True})
        run_catchup_if_needed()
    except Exception:
        log("startup catch-up error:\n" + traceback.format_exc())
    write_heartbeat("idle", None)
    while True:
        nxt = next_slot()
        delay = max(0.0, (nxt - now_sh()).total_seconds())
        log(f"sleep {int(delay)}s until {nxt.strftime('%Y-%m-%d %H:%M:%S')} Asia/Shanghai")
        write_heartbeat("sleeping", nxt)

        def _tick(nxt=nxt):
            write_heartbeat("sleeping", nxt)
            if time.time() - last_hb_log[0] >= HEARTBEAT_LOG_EVERY_SEC:
                last_hb_log[0] = time.time()
                log(f"heartbeat ok next={nxt.strftime('%H:%M')}")

        res = wait_until(nxt, on_tick=_tick)
        for w_gap, m_gap in res["jumps"]:
            log(f"clock jump detected: wall +{w_gap}s vs monotonic +{m_gap}s (box paused/suspended?)")
        now = now_sh()
        action = daemon_step(now, nxt, res["late_sec"])
        if action == "run_missed":
            log(f"missed slot {nxt.strftime('%H:%M')} by {int(res['late_sec'])}s; running once now (no catch-up burst)")
        scheduled = htf_due(nxt)
        htf = resolve_htf_tfs(scheduled, now=now)
        extra = [tf for tf in htf if tf not in scheduled]
        if extra:
            log(f"htf catch-up added {','.join(extra)} (latest closed bar not yet in signals.json)")
        write_heartbeat("scanning", nxt, {"scan_started_shanghai": now.strftime("%Y-%m-%d %H:%M:%S"),
                                          "htf_tfs": htf, "htf_catchup": extra})
        try:
            run_once(htf_tfs=htf)
        except Exception:
            log("daemon iteration error:\n" + traceback.format_exc())
        write_heartbeat("idle", None)


def main() -> int:
    parser = argparse.ArgumentParser(description="15m H&S scan runner / webhook notifier")
    parser.add_argument("--once", action="store_true", help="run one scan cycle (default)")
    parser.add_argument("--daemon", action="store_true", help="loop on the 15m schedule")
    parser.add_argument("--test-webhook", action="store_true", help="POST a test:true empty payload")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="one scan cycle without any webhook; journals go to a temp dir (live journals untouched)",
    )
    parser.add_argument("--htf", default="", help="also run 1H/4H scan, e.g. --htf 1h,4h")
    parser.add_argument("--htf-only", action="store_true", help="skip the 15m part (with --htf)")
    args = parser.parse_args()
    htf = [x.strip().lower() for x in args.htf.split(",") if x.strip().lower() in HTF_TFS]
    if args.test_webhook:
        return test_webhook()
    if args.daemon:
        return daemon()
    htf = resolve_htf_tfs(htf)
    if args.dry_run:
        import shutil
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="hs_dry_"))
        for name in ("paper_journal.json", "paper_watch_journal.json",
                     "paper_htf_1h_journal.json", "paper_htf_4h_journal.json",
                     "paper_nested_15m_journal.json", "paper_nested_1h_journal.json",
                     "paper_nested_4h_journal.json"):
            src = OUT_DIR / name
            if src.exists():
                shutil.copy2(src, tmp / name)
        paths = {
            "formal": paper_journal.JournalPaths.under(tmp),
            "watch": paper_journal.JournalPaths.under(tmp, book=paper_journal.BOOK_WATCH),
            paper_journal.BOOK_HTF_1H: paper_journal.JournalPaths.under(tmp, book=paper_journal.BOOK_HTF_1H),
            paper_journal.BOOK_HTF_4H: paper_journal.JournalPaths.under(tmp, book=paper_journal.BOOK_HTF_4H),
            **{b: paper_journal.JournalPaths.under(tmp, book=b) for b in paper_journal.NESTED_BOOKS.values()},
        }
        rc = run_once(dry_run=True, journal_paths=paths, htf_tfs=htf, skip_15m=args.htf_only)
        print(f"dry-run rc={rc} temp journals={tmp}")
        return rc
    return run_once(htf_tfs=htf, skip_15m=args.htf_only)


if __name__ == "__main__":
    raise SystemExit(main())
