#!/usr/bin/env python3
"""Local paper-trade journal and circuit breaker for the 15m H&S short scanner.

Does not place orders. Does not change pattern/entry/stop/target rules.
Fill/exit math matches hs_backtest.simulate_trade(..., exit_mode="partial60_1.5H"),
except still-open trades are left open (no end-of-data forced close).

Never logs or prints HS_WEBHOOK_KEY or webhook.env contents.
"""
from __future__ import annotations

import csv
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hs_backtest import TAKER_FEE, slippage_for  # noqa: E402

TZ = ZoneInfo("Asia/Shanghai")
OUT_DIR = HERE / "scan_out"
JOURNAL_JSON = OUT_DIR / "paper_journal.json"
JOURNAL_CSV = OUT_DIR / "paper_trades.csv"
JOURNAL_MD = OUT_DIR / "paper_journal.md"
# 观察档（near-miss，未验证）独立账本：不计入正式统计与正式熔断。
WATCH_JSON = OUT_DIR / "paper_watch_journal.json"
WATCH_CSV = OUT_DIR / "paper_watch_trades.csv"
WATCH_MD = OUT_DIR / "paper_watch_journal.md"
BOOK_FORMAL = "formal"
BOOK_WATCH = "watch"
BOOK_HTF_1H = "htf1h"   # 1H 头肩（未验证，0.5%），独立账本，不触发熔断
BOOK_HTF_4H = "htf4h"   # 4H 头肩（未验证，0.5%），独立账本，不触发熔断
HTF_BOOKS = {"1h": BOOK_HTF_1H, "4h": BOOK_HTF_4H}
BOOK_NESTED_15M = "nested15m"  # 嵌套小头肩（大头肩右肩内的小头肩，未验证 0.5%），独立账本，不触发熔断
BOOK_NESTED_1H = "nested1h"
BOOK_NESTED_4H = "nested4h"
NESTED_BOOKS = {"15m": BOOK_NESTED_15M, "1h": BOOK_NESTED_1H, "4h": BOOK_NESTED_4H}
NESTED_BOOK_TF = {v: k for k, v in NESTED_BOOKS.items()}
# 2026-11-14 三栋: 15m v6 三栋认肩（未验证 0.5%）独立账本；15m 熔断改为只作用在这个账本上。
# 旧 formal / watch 账本不再开新仓（扫描器不再产生这两档），已有持仓继续跟踪到平仓。
BOOK_V6_15M = "v6_15m"
TIER_V6 = "v6"
BREAKER_BOOKS = {BOOK_V6_15M}
BOOK_STATUS = {BOOK_V6_15M: "可入场", BOOK_FORMAL: "可入场", BOOK_WATCH: "观察", BOOK_HTF_1H: "可入场", BOOK_HTF_4H: "可入场",
               BOOK_NESTED_15M: "可入场", BOOK_NESTED_1H: "可入场", BOOK_NESTED_4H: "可入场"}
BOOK_TIMEFRAME = {BOOK_HTF_1H: "1h", BOOK_HTF_4H: "4h"}
WATCH_RISK_PCT = 0.005

FAPI = "https://fapi.binance.com/fapi/v1/klines"
UA = {"User-Agent": "hs-paper-journal/1.0"}
SESSION = requests.Session()
SESSION.headers.update(UA)

RISK_PCT = 0.01
MAX_CONCURRENT = 2
EQUITY_START = 1.0
DAY_R_LIMIT = -3.0
LOSS_STREAK_LIMIT = 3
BREAKER_RESUME_HOUR = 9  # Beijing; overnight 24h scans must not resume a prior day's trip before 09:02
BLOCKED_NOTE = "熔断中，未推送入场提醒。"

FetchFn = Callable[[str, pd.Timestamp | None], pd.DataFrame]


@dataclass(frozen=True)
class JournalPaths:
    out_dir: Path
    json_path: Path
    csv_path: Path
    md_path: Path

    @classmethod
    def default(cls) -> "JournalPaths":
        return cls(OUT_DIR, JOURNAL_JSON, JOURNAL_CSV, JOURNAL_MD)

    @classmethod
    def watch(cls) -> "JournalPaths":
        return cls(OUT_DIR, WATCH_JSON, WATCH_CSV, WATCH_MD)

    @classmethod
    def v6(cls) -> "JournalPaths":
        return cls.under(OUT_DIR, book=BOOK_V6_15M)

    @classmethod
    def htf(cls, tf: str) -> "JournalPaths":
        return cls.under(OUT_DIR, book=HTF_BOOKS[tf])

    @classmethod
    def nested(cls, tf: str) -> "JournalPaths":
        return cls.under(OUT_DIR, book=NESTED_BOOKS[tf])

    @classmethod
    def for_book(cls, book: str) -> "JournalPaths":
        if book == BOOK_V6_15M:
            return cls.under(OUT_DIR, book=book)
        if book in NESTED_BOOK_TF:
            return cls.under(OUT_DIR, book=book)
        if book == BOOK_WATCH:
            return cls.watch()
        if book in BOOK_TIMEFRAME:
            return cls.under(OUT_DIR, book=book)
        return cls.default()

    @classmethod
    def under(cls, out_dir: Path, book: str = BOOK_FORMAL) -> "JournalPaths":
        out_dir = Path(out_dir)
        if book == BOOK_V6_15M:
            return cls(
                out_dir=out_dir,
                json_path=out_dir / "paper_v6_15m_journal.json",
                csv_path=out_dir / "paper_v6_15m_trades.csv",
                md_path=out_dir / "paper_v6_15m_journal.md",
            )
        if book in NESTED_BOOK_TF:
            tf = NESTED_BOOK_TF[book]
            return cls(
                out_dir=out_dir,
                json_path=out_dir / f"paper_nested_{tf}_journal.json",
                csv_path=out_dir / f"paper_nested_{tf}_trades.csv",
                md_path=out_dir / f"paper_nested_{tf}_journal.md",
            )
        if book in BOOK_TIMEFRAME:
            tf = BOOK_TIMEFRAME[book]
            return cls(
                out_dir=out_dir,
                json_path=out_dir / f"paper_htf_{tf}_journal.json",
                csv_path=out_dir / f"paper_htf_{tf}_trades.csv",
                md_path=out_dir / f"paper_htf_{tf}_journal.md",
            )
        if book == BOOK_WATCH:
            return cls(
                out_dir=out_dir,
                json_path=out_dir / "paper_watch_journal.json",
                csv_path=out_dir / "paper_watch_trades.csv",
                md_path=out_dir / "paper_watch_journal.md",
            )
        return cls(
            out_dir=out_dir,
            json_path=out_dir / "paper_journal.json",
            csv_path=out_dir / "paper_trades.csv",
            md_path=out_dir / "paper_journal.md",
        )


def now_sh(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(TZ)
    if now.tzinfo is None:
        return now.replace(tzinfo=TZ)
    return now.astimezone(TZ)


def fmt_sh(dt: datetime | pd.Timestamp | None, with_seconds: bool = False) -> str | None:
    if dt is None or (isinstance(dt, float) and pd.isna(dt)):
        return None
    t = pd.Timestamp(dt)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    t = t.tz_convert(TZ)
    return t.strftime("%Y-%m-%d %H:%M:%S" if with_seconds else "%Y-%m-%d %H:%M")


def _parse_sh(s: str, fmt: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(s, fmt)).tz_localize(TZ)


def parse_shanghai(ts: str | None) -> pd.Timestamp | None:
    if not ts:
        return None
    s = str(ts).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return _parse_sh(s, fmt)
        except ValueError:
            continue
    try:
        t = pd.Timestamp(s)
        if t.tzinfo is None:
            t = t.tz_localize(TZ)
        return t.tz_convert(TZ)
    except (ValueError, TypeError):
        return None


def pattern_id(sig: dict) -> str:
    """Must stay identical to scan_runner.pattern_id.

    1H/4H rows (timeframe set) are keyed per head: tf|symbol|status|H_time, so a right shoulder
    that shifts later never re-alerts the same head, and 15m ids are unchanged.
    Nested inner signals (nested_signal=True, 2026-11-12) get their own namespace
    nested|tf|symbol|status|H_time so they never collide with a regular row."""
    tf = sig.get("timeframe")
    if sig.get("nested_signal"):
        return "|".join(["nested", str(tf or "15m"), str(sig.get("symbol") or ""), str(sig.get("status") or ""),
                         str(sig.get("H_time") or "")])
    if sig.get("tier") == TIER_V6:  # 2026-11-14: 15m v6 rows, one key per head (own namespace)
        return "|".join(["v6", str(tf or "15m"), str(sig.get("symbol") or ""), str(sig.get("status") or ""),
                         str(sig.get("H_time") or "")])
    if tf in ("1h", "4h"):
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


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def empty_journal(now: datetime | None = None, book: str = BOOK_FORMAL) -> dict:
    t = now_sh(now)
    return {
        "version": 1,
        "book": book,
        "updated_shanghai": fmt_sh(t, with_seconds=True),
        "equity": EQUITY_START,
        "equity_start": EQUITY_START,
        "risk_pct": RISK_PCT if book == BOOK_FORMAL else WATCH_RISK_PCT,
        "max_concurrent": MAX_CONCURRENT,
        "breaker": {
            "tripped": False,
            "reason": "",
            "trip_date": None,
            "notice_sent_date": None,
            "day_R": 0.0,
            "loss_streak": 0,
            "resume_note": (
                "次日北京时间第一个扫描（09:02）自动恢复推送；"
                "当日 −3R 计数按日历日清零；"
                "连亏计数保留到之后出现盈利平仓，但次日仍允许重新推送。"
            ),
        },
        "trades": [],
    }


def load_journal(paths: JournalPaths | None = None, book: str = BOOK_FORMAL) -> dict:
    paths = paths or JournalPaths.for_book(book)
    if not paths.json_path.exists():
        return empty_journal(book=book)
    try:
        data = json.loads(paths.json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_journal(book=book)
    if not isinstance(data, dict) or not isinstance(data.get("trades"), list):
        return empty_journal(book=book)
    data.setdefault("book", book)
    data.setdefault("equity", EQUITY_START)
    data.setdefault("equity_start", EQUITY_START)
    data.setdefault("risk_pct", RISK_PCT if book == BOOK_FORMAL else WATCH_RISK_PCT)
    data.setdefault("max_concurrent", MAX_CONCURRENT)
    data.setdefault("breaker", empty_journal()["breaker"])
    data["trades"] = [t for t in data["trades"] if isinstance(t, dict)]
    return data


def exchange_of(symbol_or_sig) -> str:
    """'okx' for OKX instIds (e.g. BTC-USDT-SWAP) or a signal/trade dict tagged okx; else 'binance'."""
    if isinstance(symbol_or_sig, dict):
        ex = symbol_or_sig.get("exchange")
        if ex in ("binance", "okx"):
            return ex
        symbol_or_sig = symbol_or_sig.get("symbol") or ""
    return "okx" if str(symbol_or_sig).upper().endswith("-SWAP") else "binance"


def _cost_symbol(symbol: str) -> str:
    """Cost tables (slippage majors) are keyed by Binance names; map OKX BTC-USDT-SWAP -> BTCUSDT."""
    if exchange_of(symbol) == "okx":
        import okx_market
        return okx_market.binance_equiv(symbol)
    return symbol


def default_fetch_klines(symbol: str, start_utc: pd.Timestamp | None = None) -> pd.DataFrame:
    """Public 15m klines (Binance USDT-M, or OKX for *-SWAP instIds); drops the open bar. No secrets."""
    if exchange_of(symbol) == "okx":
        import okx_market
        return okx_market.fetch_okx_journal_klines(symbol, start_utc)
    start_ms = None
    if start_utc is not None:
        t = pd.Timestamp(start_utc)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        start_ms = int(t.timestamp() * 1000)
    import binance_data
    df = binance_data.fetch_paged_15m(symbol, start_ms)
    if df is None or not len(df):
        return pd.DataFrame()
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    df = df[df["close_time"] < now_ms].copy()
    if "dt" not in df.columns:
        df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.reset_index(drop=True)


def _legacy_fapi_fetch_klines(symbol: str, start_utc: pd.Timestamp | None = None) -> pd.DataFrame:
    """Pre-2026-11-13 direct fapi fetcher (kept for reference; unused)."""
    frames: list[pd.DataFrame] = []
    start_ms = None
    if start_utc is not None:
        t = pd.Timestamp(start_utc)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        start_ms = int(t.timestamp() * 1000)
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    while True:
        params: dict = {"symbol": symbol, "interval": "15m", "limit": 1500}
        if start_ms is not None:
            params["startTime"] = start_ms
        r = SESSION.get(FAPI, params=params, timeout=(8, 30))
        r.raise_for_status()
        raw = r.json()
        if not raw:
            break
        df = pd.DataFrame(
            raw,
            columns=[
                "open_time", "open", "high", "low", "close", "volume", "close_time",
                "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
            ],
        )
        for c in ("open", "high", "low", "close", "volume", "quote_volume"):
            df[c] = df[c].astype("float64")
        df["open_time"] = df["open_time"].astype("int64")
        df["close_time"] = df["close_time"].astype("int64")
        frames.append(df)
        last_open = int(df["open_time"].iloc[-1])
        if len(df) < 1500:
            break
        start_ms = last_open + 15 * 60 * 1000
        if start_ms >= now_ms:
            break
        if start_utc is None:
            break
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    df = df[df["close_time"] < now_ms].copy()
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.reset_index(drop=True)


def simulate_live_short(
    symbol: str,
    official_entry: float,
    stop: float,
    target: float,
    target15: float,
    df15: pd.DataFrame,
    entry_ts: pd.Timestamp,
) -> dict:
    """Walk 15m bars from the official entry. Never force an end-of-data close."""
    slip = slippage_for(_cost_symbol(symbol))
    fee = TAKER_FEE
    fill_entry = float(official_entry) * (1 - slip)
    if stop <= fill_entry or fill_entry <= 0:
        return {
            "status": "missed_invalid",
            "fill_entry": fill_entry,
            "fill_time": None,
            "exit_time": None,
            "exit_px": None,
            "hit": None,
            "pnl_R": None,
            "t1_hit": False,
            "t1_px": None,
            "t1_time": None,
            "remaining": 1.0,
            "bars_used": 0,
            "note": "止损不在官方入场之上，路径无效。",
            "last_px": None,
            "last_time": None,
        }

    if entry_ts.tzinfo is None:
        entry_ts = entry_ts.tz_localize("UTC")
    else:
        entry_ts = entry_ts.tz_convert("UTC")

    if df15 is None or len(df15) == 0:
        return {
            "status": "pending_entry",
            "fill_entry": None,
            "fill_time": None,
            "exit_time": None,
            "exit_px": None,
            "hit": None,
            "pnl_R": None,
            "t1_hit": False,
            "t1_px": None,
            "t1_time": None,
            "remaining": 1.0,
            "bars_used": 0,
            "note": "官方入场K线尚未出现。",
            "last_px": None,
            "last_time": None,
        }

    dts = pd.to_datetime(df15["dt"], utc=True)
    last_ts = pd.Timestamp(dts.iloc[-1])
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    if last_ts < entry_ts:
        return {
            "status": "pending_entry",
            "fill_entry": None,
            "fill_time": None,
            "exit_time": None,
            "exit_px": None,
            "hit": None,
            "pnl_R": None,
            "t1_hit": False,
            "t1_px": None,
            "t1_time": None,
            "remaining": 1.0,
            "bars_used": 0,
            "note": "官方入场K线尚未出现。",
            "last_px": float(df15["close"].iloc[-1]),
            "last_time": fmt_sh(last_ts),
        }

    i0 = int(dts.searchsorted(entry_ts, side="left"))
    if i0 >= len(df15):
        return {
            "status": "pending_entry",
            "fill_entry": None,
            "fill_time": None,
            "exit_time": None,
            "exit_px": None,
            "hit": None,
            "pnl_R": None,
            "t1_hit": False,
            "t1_px": None,
            "t1_time": None,
            "remaining": 1.0,
            "bars_used": 0,
            "note": "官方入场K线尚未出现。",
            "last_px": None,
            "last_time": None,
        }

    h = df15["high"].to_numpy()
    l = df15["low"].to_numpy()
    o = df15["open"].to_numpy()
    c = df15["close"].to_numpy()

    remaining = 1.0
    filled_px_w = 0.0
    filled_w = 0.0
    trail_stop = float(stop)
    t1_hit = False
    t1_px = None
    t1_i = None
    exit_i = i0
    hit = None

    def fill(px: float, w: float, reason: str, i: int) -> None:
        nonlocal remaining, filled_px_w, filled_w, exit_i, hit
        w = min(w, remaining)
        if w <= 0:
            return
        filled_px_w += px * w
        filled_w += w
        remaining -= w
        exit_i = i
        hit = reason

    for i in range(i0, len(df15)):
        hi, lo, op = float(h[i]), float(l[i]), float(o[i])
        hit_stop = hi >= trail_stop
        hit_t1 = lo <= target
        hit_t15 = lo <= target15

        if (not t1_hit) and hit_stop and hit_t1:
            fill(trail_stop * (1 + slip), 1.0, "stop_samebar_priority", i)
            break
        if hit_stop:
            reason = "stop_breakeven" if t1_hit else "stop"
            fill(max(op, trail_stop) * (1 + slip), remaining, reason, i)
            break
        if (not t1_hit) and hit_t1:
            px = min(op, target) * (1 - slip)
            fill(px, 0.60, "target_1.0H", i)
            t1_hit = True
            t1_px = px
            t1_i = i
            trail_stop = fill_entry
            if remaining <= 1e-12:
                break
            if hit_t15:
                fill(min(op, target15) * (1 - slip), remaining, "target_1.5H", i)
                break
            continue
        if t1_hit and hit_t15:
            fill(min(op, target15) * (1 - slip), remaining, "target_1.5H", i)
            break

    fill_time = fmt_sh(dts.iloc[i0])
    last_i = len(df15) - 1
    last_px = float(c[last_i])
    last_time = fmt_sh(dts.iloc[last_i])
    bars_used = int(max(1, (exit_i if remaining <= 1e-12 else last_i) - i0 + 1))
    r_dist = (float(stop) - fill_entry) / fill_entry

    if remaining <= 1e-12 and filled_w > 0:
        exit_px = filled_px_w / filled_w
        raw = (fill_entry - exit_px) / fill_entry
        net = raw - 2 * fee
        pnl_R = net / r_dist if r_dist > 0 else 0.0
        return {
            "status": "closed",
            "fill_entry": fill_entry,
            "fill_time": fill_time,
            "exit_time": fmt_sh(dts.iloc[exit_i]),
            "exit_px": float(exit_px),
            "hit": hit,
            "pnl_R": float(pnl_R),
            "t1_hit": bool(t1_hit),
            "t1_px": t1_px,
            "t1_time": fmt_sh(dts.iloc[t1_i]) if t1_i is not None else None,
            "remaining": 0.0,
            "bars_used": bars_used,
            "note": None,
            "last_px": last_px,
            "last_time": last_time,
        }

    # Still open: mark-to-market remaining at last close (no exit slip);
    # charge open fee on full notional and close fee only on the filled weight.
    mtm_px_w = filled_px_w + last_px * remaining
    mtm_w = filled_w + remaining
    mtm_exit = mtm_px_w / mtm_w if mtm_w else last_px
    raw_closed = ((fill_entry - (filled_px_w / filled_w)) / fill_entry * filled_w) if filled_w else 0.0
    raw_open = (fill_entry - last_px) / fill_entry * remaining
    net_mtm = raw_closed + raw_open - fee - fee * filled_w
    pnl_R = net_mtm / r_dist if r_dist > 0 else 0.0
    return {
        "status": "open",
        "fill_entry": fill_entry,
        "fill_time": fill_time,
        "exit_time": None,
        "exit_px": None,
        "hit": "target_1.0H" if t1_hit else None,
        "pnl_R": float(pnl_R),
        "t1_hit": bool(t1_hit),
        "t1_px": t1_px,
        "t1_time": fmt_sh(dts.iloc[t1_i]) if t1_i is not None else None,
        "remaining": float(remaining),
        "bars_used": bars_used,
        "note": "1.0H 已触，余仓仍持有。" if t1_hit else "纸面单仍持有，尚未触及止损或 1.0H。",
        "last_px": last_px,
        "last_time": last_time,
        "mtm_exit": float(mtm_exit),
    }


def _trade_index(journal: dict) -> dict[str, dict]:
    return {str(t.get("pattern_id") or t.get("id") or ""): t for t in journal["trades"]}


def _open_account_count(journal: dict) -> int:
    n = 0
    for t in journal["trades"]:
        if t.get("account_taken") and t.get("status") in {"pending_entry", "open"}:
            n += 1
    return n


def _append_note(trade: dict, note: str | None) -> None:
    if not note:
        return
    prev = (trade.get("note") or "").strip()
    if note in prev:
        return
    trade["note"] = f"{prev} {note}".strip() if prev else note


def _closed_account_trades(journal: dict) -> list[dict]:
    rows = [
        t
        for t in journal["trades"]
        if t.get("account_taken") and t.get("status") == "closed" and t.get("exit_time")
    ]

    def _key(t: dict):
        ts = parse_shanghai(t.get("exit_time"))
        return ts.to_pydatetime() if ts is not None else datetime.min.replace(tzinfo=TZ)

    rows.sort(key=_key)
    return rows


def loss_streak(journal: dict) -> int:
    closed = _closed_account_trades(journal)
    streak = 0
    for t in reversed(closed):
        pnl = t.get("pnl_R")
        if pnl is None or float(pnl) >= 0:
            break
        streak += 1
    return streak


def day_realized_R(journal: dict, day: str) -> float:
    total = 0.0
    for t in _closed_account_trades(journal):
        et = t.get("exit_time") or ""
        if str(et)[:10] == day:
            pnl = t.get("pnl_R")
            if pnl is not None:
                total += float(pnl)
    return float(total)


def maybe_resume_breaker(journal: dict, now: datetime | None = None) -> bool:
    """Clear a previous calendar day's trip. Returns True if just resumed.

    Resume only at/after 09:00 Beijing, so the first eligible scan is still 09:02
    even though the daemon now also runs overnight.
    """
    t = now_sh(now)
    if t.hour < BREAKER_RESUME_HOUR:
        return False
    today = t.strftime("%Y-%m-%d")
    b = journal.setdefault("breaker", empty_journal()["breaker"])
    trip_date = b.get("trip_date")
    if b.get("tripped") and trip_date and str(trip_date) < today:
        b["tripped"] = False
        b["reason"] = ""
        # notice_sent_date stays as the old day so a new trip today can notify again
        b["day_R"] = day_realized_R(journal, today)
        b["loss_streak"] = loss_streak(journal)
        return True
    return False


def evaluate_breaker(journal: dict, now: datetime | None = None) -> dict:
    """Update breaker flags. An old 3-loss streak does not re-trip until a new close today."""
    t = now_sh(now)
    today = t.strftime("%Y-%m-%d")
    b = journal.setdefault("breaker", empty_journal()["breaker"])
    maybe_resume_breaker(journal, t)
    day_r = day_realized_R(journal, today)
    streak = loss_streak(journal)
    b["day_R"] = float(day_r)
    b["loss_streak"] = int(streak)

    # Stay tripped overnight after a previous calendar day's trip until 09:02.
    if b.get("tripped") and (b.get("trip_date") == today or t.hour < BREAKER_RESUME_HOUR):
        return b

    reasons: list[str] = []
    if day_r <= DAY_R_LIMIT:
        reasons.append(f"当日账户已实现盈亏合计 {day_r:.3f}R，触发熔断（阈值 {DAY_R_LIMIT:.0f}R）")

    closed = _closed_account_trades(journal)
    last_exit_today = False
    if closed:
        last_exit = str(closed[-1].get("exit_time") or "")[:10]
        last_exit_today = last_exit == today
    if streak >= LOSS_STREAK_LIMIT and last_exit_today:
        reasons.append(f"账户内最近 {streak} 笔已平仓均为亏损，触发熔断")

    if reasons:
        b["tripped"] = True
        b["trip_date"] = today
        b["reason"] = "；".join(reasons)
    else:
        b["tripped"] = False
        if b.get("trip_date") != today:
            b["reason"] = ""
    return b


def is_tripped(journal: dict, now: datetime | None = None) -> bool:
    maybe_resume_breaker(journal, now)
    return bool((journal.get("breaker") or {}).get("tripped"))


def breaker_notice_payload(journal: dict, scan_time: str | None = None, now: datetime | None = None) -> dict:
    b = journal.get("breaker") or {}
    t = now_sh(now)
    return {
        "test": False,
        "kind": "circuit_breaker",
        "book": journal.get("book") or BOOK_FORMAL,
        "reason": b.get("reason") or "熔断已触发",
        "day_R": float(b.get("day_R") or 0.0),
        "loss_streak": int(b.get("loss_streak") or 0),
        "scan_time": scan_time or fmt_sh(t),
    }


def notification_plan(journal: dict, now: datetime | None = None) -> dict:
    """Decide whether to suppress the trading webhook and/or send the one-time breaker notice."""
    t = now_sh(now)
    today = t.strftime("%Y-%m-%d")
    maybe_resume_breaker(journal, t)
    b = journal.get("breaker") or {}
    tripped = bool(b.get("tripped"))
    sent = str(b.get("notice_sent_date") or "") == today
    # Do not re-notify a previous day's still-active trip during 00:00-08:59.
    send_notice = tripped and not sent and str(b.get("trip_date") or "") == today
    return {
        "suppress_trading": tripped,
        "send_breaker_notice": send_notice,
        "payload": breaker_notice_payload(journal, now=t) if send_notice else None,
    }


def mark_notice_sent(journal: dict, now: datetime | None = None, paths: JournalPaths | None = None) -> None:
    t = now_sh(now)
    journal.setdefault("breaker", {})["notice_sent_date"] = t.strftime("%Y-%m-%d")
    if paths is not None:
        save_journal(journal, paths=paths, now=t)


def mark_alerted(journal: dict, pattern_ids: list[str], now: datetime | None = None, paths: JournalPaths | None = None) -> None:
    wanted = set(pattern_ids)
    for t in journal["trades"]:
        pid = str(t.get("pattern_id") or t.get("id") or "")
        if pid in wanted:
            t["alerted"] = True
    if paths is not None:
        save_journal(journal, paths=paths, now=now)


def note_alerts_blocked(journal: dict, pattern_ids: list[str]) -> None:
    wanted = set(pattern_ids)
    for t in journal["trades"]:
        pid = str(t.get("pattern_id") or t.get("id") or "")
        if pid in wanted and not t.get("alerted"):
            _append_note(t, BLOCKED_NOTE)


def _apply_sim(trade: dict, sim: dict, *, apply_equity: bool, journal: dict) -> None:
    was_closed = trade.get("status") == "closed"
    trade["status"] = sim["status"]
    if sim.get("fill_entry") is not None:
        trade["fill_entry"] = sim["fill_entry"]
    if sim.get("fill_time"):
        trade["fill_time"] = sim["fill_time"]
    trade["exit_time"] = sim.get("exit_time")
    trade["exit_px"] = sim.get("exit_px")
    trade["hit"] = sim.get("hit")
    trade["pnl_R"] = sim.get("pnl_R")
    trade["t1_hit"] = bool(sim.get("t1_hit"))
    trade["t1_px"] = sim.get("t1_px")
    trade["t1_time"] = sim.get("t1_time")
    trade["remaining"] = sim.get("remaining")
    trade["bars_used"] = sim.get("bars_used")
    trade["last_px"] = sim.get("last_px")
    trade["last_time"] = sim.get("last_time")
    prev = trade.get("note") or ""
    trade["note"] = sim.get("note") or ""
    if BLOCKED_NOTE in prev:
        _append_note(trade, BLOCKED_NOTE)
    if "账户已满 2 笔" in prev:
        _append_note(trade, "账户已满 2 笔并发空单，本笔只记账、不计入账户权益与熔断。")
    if (
        apply_equity
        and not was_closed
        and sim["status"] == "closed"
        and trade.get("account_taken")
        and not trade.get("equity_applied")
        and sim.get("pnl_R") is not None
    ):
        journal["equity"] = float(journal.get("equity") or EQUITY_START) * (
            1.0 + float(sim["pnl_R"]) * float(journal.get("risk_pct") or RISK_PCT)
        )
        trade["equity_applied"] = True


def _new_trade_from_signal(sig: dict, now: datetime, account_taken: bool, reason: str) -> dict:
    pid = pattern_id(sig)
    first = fmt_sh(now, with_seconds=True)
    trade = {
        "id": pid,
        "pattern_id": pid,
        "symbol": sig.get("symbol"),
        "exchange": exchange_of(sig),
        "status": "pending_entry",
        "first_seen": first,
        "alerted": False,
        "account_taken": bool(account_taken),
        "account_reason": reason,
        "official_entry": sig.get("official_entry"),
        "official_entry_time": sig.get("official_entry_time"),
        "fill_entry": None,
        "fill_time": None,
        "stop": sig.get("stop"),
        "target_1h": sig.get("target_1h"),
        "target_15h": sig.get("target_15h"),
        "L_time": sig.get("L_time"),
        "H_time": sig.get("H_time"),
        "R_time": sig.get("R_time"),
        "exit_time": None,
        "exit_px": None,
        "hit": None,
        "pnl_R": None,
        "t1_hit": False,
        "t1_px": None,
        "t1_time": None,
        "remaining": 1.0,
        "bars_used": 0,
        "note": "",
        "equity_applied": False,
        "last_px": None,
        "last_time": None,
        "tier": sig.get("tier") or ("watch" if sig.get("status") == "观察" else "formal"),
        "timeframe": sig.get("timeframe") or "15m",
        "missed_condition": sig.get("missed_condition"),
        "missed_text": sig.get("missed_text"),
    }
    if not account_taken and reason == "capacity":
        trade["note"] = "账户已满 2 笔并发空单，本笔只记账、不计入账户权益与熔断。"
    return trade


def _valid_levels(sig: dict) -> tuple[bool, str]:
    entry = sig.get("official_entry")
    stop = sig.get("stop")
    t1 = sig.get("target_1h")
    t15 = sig.get("target_15h")
    et = sig.get("official_entry_time")
    if entry is None or stop is None or t1 is None or t15 is None or not et:
        return False, "缺少官方入场、止损或目标，无法记账路径。"
    try:
        if float(stop) <= float(entry):
            return False, "止损不在官方入场之上，路径无效。"
    except (TypeError, ValueError):
        return False, "入场或止损不是有效数字。"
    if parse_shanghai(str(et)) is None:
        return False, "官方入场时间无法解析。"
    return True, ""


def _fetch_cached(
    cache: dict[str, pd.DataFrame],
    fetch: FetchFn,
    symbol: str,
    start_sh: pd.Timestamp | None,
    errors: list[str],
) -> pd.DataFrame | None:
    key = symbol
    if key in cache:
        return cache[key]
    start_utc = None
    if start_sh is not None:
        start_utc = start_sh.tz_convert("UTC") - pd.Timedelta(minutes=15)
    try:
        df = fetch(symbol, start_utc)
    except Exception as e:  # noqa: BLE001 — keep scanner alive if a fetch fails
        errors.append(f"{symbol}: {type(e).__name__}")
        cache[key] = pd.DataFrame()
        return None
    if df is None:
        df = pd.DataFrame()
    cache[key] = df
    return df


def _simulate_trade(trade: dict, df: pd.DataFrame | None) -> dict:
    entry = trade.get("official_entry")
    stop = trade.get("stop")
    t1 = trade.get("target_1h")
    t15 = trade.get("target_15h")
    et = parse_shanghai(trade.get("official_entry_time"))
    if entry is None or stop is None or t1 is None or t15 is None or et is None:
        return {
            "status": "missed_invalid",
            "fill_entry": None,
            "fill_time": None,
            "exit_time": None,
            "exit_px": None,
            "hit": None,
            "pnl_R": None,
            "t1_hit": False,
            "t1_px": None,
            "t1_time": None,
            "remaining": 1.0,
            "bars_used": 0,
            "note": why or "缺少官方入场、止损或目标，无法记账路径。",
            "last_px": None,
            "last_time": None,
        }
    entry_ts = et.tz_convert("UTC")
    return simulate_live_short(
        symbol=str(trade.get("symbol") or ""),
        official_entry=float(entry),
        stop=float(stop),
        target=float(t1),
        target15=float(t15),
        df15=df if df is not None else pd.DataFrame(),
        entry_ts=entry_ts,
    )


def update_from_scan(
    data: dict,
    *,
    now: datetime | None = None,
    paths: JournalPaths | None = None,
    fetch_klines_fn: FetchFn | None = None,
    book: str = BOOK_FORMAL,
) -> dict:
    """Ingest new signals and mark open paper trades. 观望 is ignored.

    已破位·等回踩 is also ignored here: the live simulator assumes a market fill at
    official_entry_time, and does not wait for a limit at the break level to be tagged.

    book="formal": ingests 可入场 into paper_journal.json, drives the circuit breaker.
    book="watch":  ingests 观察 into paper_watch_journal.json (0.5% risk). Never evaluates
                   or trips a breaker; never touches the formal journal.
    Always runs even when there are no new signals, so open trades keep being marked.
    """
    if book not in BOOK_STATUS:
        raise ValueError(f"unknown book {book}")
    paths = paths or JournalPaths.for_book(book)
    want_status = BOOK_STATUS[book]
    t = now_sh(now)
    journal = load_journal(paths, book=book)
    maybe_resume_breaker(journal, t)
    fetch = fetch_klines_fn or default_fetch_klines
    cache: dict[str, pd.DataFrame] = {}
    fetch_errors: list[str] = []
    signals = [s for s in (data.get("signals") or []) if isinstance(s, dict)]
    new_ids: list[str] = []

    # 1) Mark existing pending/open trades first (closes may free capacity).
    for trade in journal["trades"]:
        if trade.get("status") not in {"pending_entry", "open"}:
            continue
        et = parse_shanghai(trade.get("official_entry_time"))
        df = _fetch_cached(cache, fetch, str(trade.get("symbol") or ""), et, fetch_errors)
        if df is None and fetch_errors and (trade.get("symbol") in (fetch_errors[-1] if fetch_errors else "")):
            _append_note(trade, "本次更新未取到K线，保持原状态。")
            continue
        if df is None:
            df = pd.DataFrame()
        sim = _simulate_trade(trade, df)
        _apply_sim(trade, sim, apply_equity=True, journal=journal)

    # 2) Ingest new signals of this book's status only.
    known = _trade_index(journal)
    for sig in signals:
        if sig.get("status") != want_status:
            continue
        if bool(sig.get("nested_signal")) != (book in NESTED_BOOK_TF):
            continue  # nested inner signals only go to the paper_nested_* books (and only there)
        if (sig.get("tier") == TIER_V6) != (book == BOOK_V6_15M):
            continue  # v6 rows only go to paper_v6_15m (and only there)
        pid = pattern_id(sig)
        if pid in known:
            continue
        ok, why = _valid_levels(sig)
        if not ok:
            trade = _new_trade_from_signal(sig, t, account_taken=False, reason="invalid")
            trade["status"] = "missed_invalid"
            trade["note"] = why
            journal["trades"].append(trade)
            known[pid] = trade
            new_ids.append(pid)
            continue
        can_take = _open_account_count(journal) < int(journal.get("max_concurrent") or MAX_CONCURRENT)
        reason = "" if can_take else "capacity"
        trade = _new_trade_from_signal(sig, t, account_taken=can_take, reason=reason)
        et = parse_shanghai(sig.get("official_entry_time"))
        df = _fetch_cached(cache, fetch, str(sig.get("symbol") or ""), et, fetch_errors)
        if df is None:
            df = pd.DataFrame()
        sim = _simulate_trade(trade, df)
        _apply_sim(trade, sim, apply_equity=True, journal=journal)
        journal["trades"].append(trade)
        known[pid] = trade
        new_ids.append(pid)

    if book in BREAKER_BOOKS:
        evaluate_breaker(journal, t)
        if (journal.get("breaker") or {}).get("tripped"):
            note_alerts_blocked(journal, new_ids)
    else:
        # Watch / old formal / HTF / nested books: never trip anything (breaker lives on v6_15m).
        b = journal.setdefault("breaker", empty_journal(book=book)["breaker"])
        b["tripped"] = False
        b["reason"] = ""

    journal["updated_shanghai"] = fmt_sh(t, with_seconds=True)
    if fetch_errors:
        journal["last_fetch_errors"] = fetch_errors[:12]
    else:
        journal.pop("last_fetch_errors", None)
    save_journal(journal, paths=paths, now=t)
    return journal


def _fmt_px(x) -> str:
    if x is None:
        return "—"
    try:
        ax = abs(float(x))
    except (TypeError, ValueError):
        return "—"
    if ax >= 1000:
        return f"{float(x):,.2f}"
    if ax >= 1:
        return f"{float(x):.4f}"
    return f"{float(x):.6f}"


def _fmt_r(x) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):+.3f}R"
    except (TypeError, ValueError):
        return "—"


def _status_cn(trade: dict) -> str:
    st = trade.get("status") or ""
    if not trade.get("account_taken") and trade.get("account_reason") == "capacity":
        return {"pending_entry": "未入场（容量外）", "open": "持有中（容量外）", "closed": "已平（容量外）"}.get(st, st)
    return {
        "pending_entry": "等待官方入场K线",
        "open": "持有中",
        "closed": "已平仓",
        "missed_invalid": "无效/跳过",
        "skipped_capacity": "账户未纳入（容量）",
    }.get(st, st)


def _sym_label(tr: dict) -> str:
    return ("欧易 " if exchange_of(tr) == "okx" else "币安 ") + str(tr.get("symbol") or "")


def render_snapshot(journal: dict, now: datetime | None = None) -> str:
    t = now_sh(now)
    b = journal.get("breaker") or {}
    equity = float(journal.get("equity") or EQUITY_START)
    day_r = float(b.get("day_R") or 0.0)
    streak = int(b.get("loss_streak") or 0)
    if (journal.get("book") == BOOK_WATCH or journal.get("book") in BOOK_TIMEFRAME
            or journal.get("book") in NESTED_BOOK_TF):
        return render_watch_snapshot(journal, now=now)
    if b.get("tripped"):
        br = f"已熔断（{b.get('trip_date')}）。{b.get('reason') or ''} 次日 09:02 起恢复推送，开仓纸面单仍继续记账。"
    else:
        br = "未熔断，可推送新信号。"
    lines = [
        ("# 15m·v6 三栋认肩 纸面记账（未验证，0.5% 风险，不下单）" if journal.get("book") == BOOK_V6_15M
         else "# 纸面交易记账（不下单）"),
        "",
        f"更新时间：{fmt_sh(t, with_seconds=True)}（北京时间）",
        f"账户权益：{equity:.4f}（起始 1.0000，每笔账户单按 1% 风险，平仓后权益 × (1 + pnl_R × 0.01)）",
        f"并发上限：账户内最多 {int(journal.get('max_concurrent') or MAX_CONCURRENT)} 笔空单；超出的可入场信号仍记账，但不计入权益与熔断。",
        f"当日已实现（账户单）：{day_r:+.3f}R　连亏：{streak} 笔",
        f"熔断状态：{br}",
        "账户口径：币安与欧易共用一个纸面账户（并发上限、当日 −3R、连亏 3 笔都按两所合计）。",
        "",
        "规则提醒：官方入场为破位后下一根 15m 开盘（含空头滑点）；60% 在 1.0H 离场，余仓看 1.5H，1.0H 之后余仓止损移到入场价。同根既触及止损又触及目标时先记止损。数据尚未走完的单子保持持有，不做收盘强平。",
        "",
    ]
    open_rows = [tr for tr in journal["trades"] if tr.get("status") in {"pending_entry", "open"}]
    closed_rows = [tr for tr in journal["trades"] if tr.get("status") == "closed"]
    closed_rows = list(reversed(_sort_by_exit(closed_rows)))
    invalid_rows = [tr for tr in journal["trades"] if tr.get("status") == "missed_invalid"]

    lines += [f"## 当前未平（{len(open_rows)}）", ""]
    if not open_rows:
        lines += ["暂无持有或等待入场的纸面单。", ""]
    else:
        lines += [
            "| 合约 | 状态 | 账户 | 提醒 | 官方入场 | 成交 | 止损 | 1.0H | 1.5H | 已实现/盯市 | 余仓 | 备注 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for tr in open_rows:
            acc = "计入" if tr.get("account_taken") else "否"
            lines.append(
                "| {sym} | {st} | {acc} | {al} | {oe} | {fe} | {sp} | {t1} | {t15} | {r} | {rem} | {note} |".format(
                    sym=_sym_label(tr),
                    st=_status_cn(tr),
                    acc=acc,
                    al="已推" if tr.get("alerted") else "未推",
                    oe=_fmt_px(tr.get("official_entry")),
                    fe=_fmt_px(tr.get("fill_entry")),
                    sp=_fmt_px(tr.get("stop")),
                    t1=_fmt_px(tr.get("target_1h")),
                    t15=_fmt_px(tr.get("target_15h")),
                    r=_fmt_r(tr.get("pnl_R")),
                    rem=f"{float(tr.get('remaining') or 0):.0%}",
                    note=(tr.get("note") or "").replace("|", "/"),
                )
            )
        lines.append("")

    lines += ["## 最近已平（最多 20 笔）", ""]
    if not closed_rows:
        lines += ["尚无已平仓记录。", ""]
    else:
        lines += [
            "| 合约 | 平仓时间 | 账户 | 触价 | pnl_R | 入场 | 离场 | 备注 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for tr in closed_rows[:20]:
            acc = "计入" if tr.get("account_taken") else "容量外"
            lines.append(
                "| {sym} | {xt} | {acc} | {hit} | {r} | {fe} | {ex} | {note} |".format(
                    sym=_sym_label(tr),
                    xt=tr.get("exit_time") or "—",
                    acc=acc,
                    hit=tr.get("hit") or "—",
                    r=_fmt_r(tr.get("pnl_R")),
                    fe=_fmt_px(tr.get("fill_entry")),
                    ex=_fmt_px(tr.get("exit_px")),
                    note=(tr.get("note") or "").replace("|", "/"),
                )
            )
        lines.append("")

    if invalid_rows:
        lines += [f"## 无效记录（{len(invalid_rows)}）", ""]
        for tr in invalid_rows[-8:]:
            lines.append(f"- {tr.get('symbol')} `{tr.get('pattern_id')}`：{tr.get('note') or '无效'}")
        lines.append("")

    lines += [
        "## 文件",
        "",
        "- 源数据：`scan_out/paper_journal.json`",
        "- 表格：`scan_out/paper_trades.csv`",
        "- 本快照：`scan_out/paper_journal.md`",
        "",
        "本模块只记账、只在熔断时抑制新的入场提醒，绝不自动下单。",
        "",
    ]
    return "\n".join(lines)


def watch_stats(journal: dict) -> dict:
    closed = [t for t in journal.get("trades") or [] if t.get("status") == "closed" and t.get("pnl_R") is not None]
    rs = [float(t["pnl_R"]) for t in closed]
    gp = sum(r for r in rs if r > 0)
    gl = -sum(r for r in rs if r <= 0)
    return {
        "n_closed": len(rs),
        "mean_R": (sum(rs) / len(rs)) if rs else None,
        "pf": (gp / gl) if gl > 0 else None,
        "win_rate": (sum(1 for r in rs if r > 0) / len(rs)) if rs else None,
    }


def render_watch_snapshot(journal: dict, now: datetime | None = None) -> str:
    t = now_sh(now)
    equity = float(journal.get("equity") or EQUITY_START)
    st = watch_stats(journal)
    trades = journal.get("trades") or []
    open_rows = [tr for tr in trades if tr.get("status") in {"pending_entry", "open"}]
    closed_rows = list(reversed(_sort_by_exit([tr for tr in trades if tr.get("status") == "closed"])))
    mean = "—" if st["mean_R"] is None else f"{st['mean_R']:+.3f}R"
    pf = "—" if st["pf"] is None else f"{st['pf']:.2f}"
    tf = BOOK_TIMEFRAME.get(journal.get("book") or "")
    ntf = NESTED_BOOK_TF.get(journal.get("book") or "")
    if ntf:
        head = [
            f"# {ntf.upper()} 嵌套小头肩纸面记账（大头肩右肩内的小头肩，未验证，不下单）",
            "",
            f"更新时间：{fmt_sh(t, with_seconds=True)}（北京时间）",
            f"账本权益：{equity:.4f}（起始 1.0000，每笔按 0.5% 风险）",
            f"已平 {st['n_closed']} 笔，平均 {mean}，PF {pf}。",
            "",
            "本账本独立：不计入 15m 正式/观察或 1H/4H 账本，不触发也不受 15m 熔断影响。",
            "",
        ]
    elif tf:
        head = [
            f"# {tf.upper()} 头肩纸面记账（v3 价格认肩，未验证，不下单）",
            "",
            f"更新时间：{fmt_sh(t, with_seconds=True)}（北京时间）",
            f"{tf.upper()} 账本权益：{equity:.4f}（起始 1.0000，每笔按 0.5% 风险）",
            f"已平 {st['n_closed']} 笔，平均 {mean}，PF {pf}。",
            "",
            "本账本与 15m 正式/观察账本完全分开：不计入 15m 统计、不触发也不受 15m 熔断影响。",
            "",
        ]
    else:
        head = [
            "# 观察档纸面记账（near-miss，未验证，不下单）",
            "",
            f"更新时间：{fmt_sh(t, with_seconds=True)}（北京时间）",
            f"观察档权益：{equity:.4f}（起始 1.0000，每笔按 0.5% 风险）",
            f"已平 {st['n_closed']} 笔，平均 {mean}，PF {pf}。",
            "",
            "本账本与正式账本（paper_journal.json）完全分开：不计入正式统计、不触发正式熔断。",
            "正式熔断期间观察提醒也停发，但观察单仍在这里记账。",
            "",
        ]
    lines = head + [
        f"## 当前未平（{len(open_rows)}）",
        "",
    ]
    if not open_rows:
        lines += ["暂无。", ""]
    else:
        lines += ["| 合约 | 状态 | 提醒 | 差的条件 | 官方入场 | 止损 | 1.0H | 1.5H | 已实现/盯市 |", "|---|---|---|---|---|---|---|---|---|"]
        for tr in open_rows:
            lines.append(
                f"| {_sym_label(tr)} | {_status_cn(tr)} | {'已推' if tr.get('alerted') else '未推'} | "
                f"{(tr.get('missed_text') or '').replace('|', '/')} | {_fmt_px(tr.get('official_entry'))} | "
                f"{_fmt_px(tr.get('stop'))} | {_fmt_px(tr.get('target_1h'))} | {_fmt_px(tr.get('target_15h'))} | "
                f"{_fmt_r(tr.get('pnl_R'))} |"
            )
        lines.append("")
    lines += ["## 最近已平（最多 20 笔）", ""]
    if not closed_rows:
        lines += ["尚无已平仓记录。", ""]
    else:
        lines += ["| 合约 | 平仓时间 | 差的条件 | 触价 | pnl_R |", "|---|---|---|---|---|"]
        for tr in closed_rows[:20]:
            lines.append(
                f"| {_sym_label(tr)} | {tr.get('exit_time') or '—'} | {tr.get('missed_condition') or '—'} | "
                f"{tr.get('hit') or '—'} | {_fmt_r(tr.get('pnl_R'))} |"
            )
        lines.append("")
    lines += [
        "## 文件",
        "",
        "- 源数据：`scan_out/paper_watch_journal.json`",
        "- 表格：`scan_out/paper_watch_trades.csv`",
        "- 本快照：`scan_out/paper_watch_journal.md`",
        "",
        "绝不自动下单。",
        "",
    ]
    return "\n".join(lines)


def _sort_by_exit(rows: list[dict]) -> list[dict]:
    def _key(t: dict):
        ts = parse_shanghai(t.get("exit_time"))
        return ts.to_pydatetime() if ts is not None else datetime.min.replace(tzinfo=TZ)

    return sorted(rows, key=_key)


CSV_FIELDS = [
    "id",
    "exchange",
    "symbol",
    "status",
    "account_taken",
    "account_reason",
    "alerted",
    "first_seen",
    "official_entry_time",
    "official_entry",
    "fill_entry",
    "fill_time",
    "stop",
    "target_1h",
    "target_15h",
    "L_time",
    "H_time",
    "R_time",
    "exit_time",
    "exit_px",
    "hit",
    "pnl_R",
    "t1_hit",
    "remaining",
    "bars_used",
    "tier",
    "missed_condition",
    "missed_text",
    "note",
]


def save_journal(journal: dict, paths: JournalPaths | None = None, now: datetime | None = None) -> None:
    paths = paths or JournalPaths.default()
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    payload = deepcopy(journal)
    payload["updated_shanghai"] = journal.get("updated_shanghai") or fmt_sh(now_sh(now), with_seconds=True)
    _atomic_write(paths.json_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    with paths.csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for t in journal.get("trades") or []:
            row = {k: t.get(k, "") for k in CSV_FIELDS}
            for k, v in list(row.items()):
                if isinstance(v, bool):
                    row[k] = "true" if v else "false"
                elif v is None:
                    row[k] = ""
            w.writerow(row)

    _atomic_write(paths.md_path, render_snapshot(journal, now=now))


def journal_log_line(journal: dict) -> str:
    trades = journal.get("trades") or []
    n_open = sum(1 for t in trades if t.get("status") == "open")
    n_pend = sum(1 for t in trades if t.get("status") == "pending_entry")
    n_closed = sum(1 for t in trades if t.get("status") == "closed")
    n_acc_open = sum(
        1 for t in trades if t.get("account_taken") and t.get("status") in {"pending_entry", "open"}
    )
    b = journal.get("breaker") or {}
    equity = float(journal.get("equity") or EQUITY_START)
    flag = "tripped" if b.get("tripped") else "ok"
    if journal.get("book") != BOOK_FORMAL and journal.get("book"):
        flag = "n/a"
    return (
        f"journal[{journal.get('book') or BOOK_FORMAL}] equity={equity:.4f} open={n_open} pending={n_pend} "
        f"closed={n_closed} account_open={n_acc_open} "
        f"day_R={float(b.get('day_R') or 0):+.3f} streak={int(b.get('loss_streak') or 0)} "
        f"breaker={flag}"
    )
