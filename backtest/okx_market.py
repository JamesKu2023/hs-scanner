#!/usr/bin/env python3
"""OKX public market data for the 15m H&S scanner (USDT-margined perpetual swaps).

Public endpoints only. No API keys, no orders.
  - /api/v5/market/candles          (recent bars, limit<=300, 40 req/2s)
  - /api/v5/market/history-candles  (older bars, limit<=100, 20 req/2s)
  - /api/v5/market/tickers?instType=SWAP  (24h volume ranking)
  - /api/v5/public/instruments?instType=SWAP  (state, listTime)
  - /api/v5/public/funding-rate     (extreme funding filter, pool build only)

Candles come back newest first as
  [ts, o, h, l, c, vol(contracts), volCcy(base coin), volCcyQuote, confirm].
We return the same frame shape the Binance fetcher returns
(open_time, open, high, low, close, volume, close_time, quote_volume, dt),
using volCcy (base-coin volume) as `volume`; the rule only uses volume ratios,
so the unit does not change any decision. Unconfirmed (still-open) bars are dropped.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OKX_BASE = "https://www.okx.com"
OKX_POOL_PATH = ROOT / "okx_pool.json"
OKX_POOL_MAX_AGE_SEC = 24 * 3600
OKX_POOL_TOP_N = 30
MIN_LIST_DAYS = 14
MAX_ABS_FUNDING = 0.001
UA = {"User-Agent": "hs-scan-15m/1.0 (+okx public)"}
BAR_MS = {"15m": 15 * 60 * 1000, "1H": 60 * 60 * 1000, "4H": 4 * 60 * 60 * 1000,
          "1Dutc": 24 * 60 * 60 * 1000}  # 4H/1Dutc: 1H/4H scan (scan_htf.py)

# Same stable / fiat exclusion as the Binance core pool.
try:
    from build_core_pool import STABLE_BASES  # type: ignore
except Exception:  # pragma: no cover
    STABLE_BASES = {"USDC", "USDT", "DAI", "FDUSD", "TUSD", "BUSD", "USDE", "USD1", "PYUSD", "EUR"}


class OkxError(RuntimeError):
    pass


class RateLimiter:
    """Sliding window: at most `n` calls per `per` seconds (thread-safe)."""

    def __init__(self, n: int, per: float):
        self.n, self.per = n, per
        self.calls: deque[float] = deque()
        self.lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.calls and now - self.calls[0] >= self.per:
                    self.calls.popleft()
                if len(self.calls) < self.n:
                    self.calls.append(now)
                    return
                sleep_for = self.per - (now - self.calls[0]) + 0.01
            time.sleep(max(0.01, sleep_for))


# Stay well under OKX limits: candles 40/2s, history-candles 20/2s -> use 15/2s and 8/2s.
LIMIT_CANDLES = RateLimiter(15, 2.0)
LIMIT_HISTORY = RateLimiter(8, 2.0)
LIMIT_PUBLIC = RateLimiter(8, 2.0)

SESSION = requests.Session()
SESSION.headers.update(UA)


def _get(path: str, params: dict, limiter: RateLimiter, retries: int = 2) -> list:
    last: Exception | None = None
    for attempt in range(retries + 1):
        limiter.wait()
        try:
            r = SESSION.get(OKX_BASE + path, params=params, timeout=(8, 20))
            if r.status_code == 429:
                last = OkxError("HTTP 429 rate limited")
                time.sleep(1.0 + attempt)
                continue
            r.raise_for_status()
            body = r.json()
        except (requests.RequestException, ValueError) as e:
            last = e
            time.sleep(0.5 + attempt)
            continue
        code = str(body.get("code"))
        if code == "50011":  # too many requests
            last = OkxError("OKX 50011 rate limited")
            time.sleep(1.0 + attempt)
            continue
        if code != "0":
            raise OkxError(f"OKX code={code} msg={body.get('msg')!r}")
        return body.get("data") or []
    raise OkxError(f"OKX request failed: {type(last).__name__}: {last}")


# ---------------------------------------------------------------------------
# naming helpers


def okx_base(inst_id: str) -> str:
    return inst_id.split("-")[0].upper()


def coin_key(symbol: str, exchange: str) -> str:
    """Normalize to a coin key so the same coin matches across exchanges.

    binance '1000PEPEUSDT' -> 'PEPE', 'BTCUSDT' -> 'BTC'; okx 'PEPE-USDT-SWAP' -> 'PEPE'.
    """
    s = (symbol or "").upper()
    if exchange == "okx":
        b = okx_base(s)
    else:
        b = s[:-4] if s.endswith("USDT") else s
    for pre in ("1000000", "1000", "1M"):
        if b.startswith(pre) and len(b) > len(pre) + 1:
            b = b[len(pre):]
            break
    return b


def binance_equiv(inst_id: str) -> str:
    """'BTC-USDT-SWAP' -> 'BTCUSDT' (used only for cost tables such as slippage majors)."""
    return okx_base(inst_id) + "USDT"


def is_okx_symbol(symbol: str) -> bool:
    return str(symbol or "").upper().endswith("-SWAP")


# ---------------------------------------------------------------------------
# candles


def _frame(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume",
                                     "close_time", "quote_volume", "dt"])
    df = pd.DataFrame([r[:9] for r in rows], columns=[
        "open_time", "open", "high", "low", "close", "vol_contracts", "volume", "quote_volume", "confirm",
    ])
    for c in ("open", "high", "low", "close", "vol_contracts", "volume", "quote_volume"):
        df[c] = df[c].astype("float64")
    df["open_time"] = df["open_time"].astype("int64")
    return df


def fetch_okx_klines(inst_id: str, bar: str, limit: int, start_ms: int | None = None) -> pd.DataFrame:
    """Most recent `limit` closed bars (or all closed bars since start_ms), oldest first."""
    if bar not in BAR_MS:
        raise ValueError(f"unsupported bar {bar}")
    step = BAR_MS[bar]
    frames: list[pd.DataFrame] = []
    got = 0
    after: int | None = None
    use_history = False
    for _ in range(200):  # hard page cap
        if use_history:
            params = {"instId": inst_id, "bar": bar, "limit": 100}
            path, lim = "/api/v5/market/history-candles", LIMIT_HISTORY
        else:
            params = {"instId": inst_id, "bar": bar, "limit": 300}
            path, lim = "/api/v5/market/candles", LIMIT_CANDLES
        if after is not None:
            params["after"] = str(after)
        rows = _get(path, params, lim)
        if not rows:
            if not use_history:
                use_history = True  # candles only covers recent bars; continue with history
                continue
            break
        df = _frame(rows)
        frames.append(df)
        got += len(df)
        oldest = int(df["open_time"].min())
        after = oldest
        if start_ms is not None:
            if oldest <= start_ms:
                break
        elif got >= limit + 2:
            break
    if not frames:
        return _frame([])
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    df = df[df["confirm"].astype(str) == "1"].copy()
    df["close_time"] = df["open_time"] + step - 1
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    df = df[df["close_time"] < now_ms]
    if start_ms is not None:
        df = df[df["open_time"] >= start_ms]
    else:
        df = df.tail(limit)
    df = df.drop(columns=["confirm", "vol_contracts"]).copy()
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.reset_index(drop=True)


def fetch_okx_journal_klines(inst_id: str, start_utc: pd.Timestamp | None = None) -> pd.DataFrame:
    """Paper-journal fetcher: closed 15m bars since start_utc (or the latest 300)."""
    if start_utc is None:
        return fetch_okx_klines(inst_id, "15m", 300)
    t = pd.Timestamp(start_utc)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return fetch_okx_klines(inst_id, "15m", 0, start_ms=int(t.timestamp() * 1000))


# ---------------------------------------------------------------------------
# pool


def build_okx_pool(top_n: int = OKX_POOL_TOP_N, check_funding: bool = True) -> dict:
    tickers = _get("/api/v5/market/tickers", {"instType": "SWAP"}, LIMIT_PUBLIC)
    insts = _get("/api/v5/public/instruments", {"instType": "SWAP"}, LIMIT_PUBLIC)
    meta = {i["instId"]: i for i in insts}
    now_ms = int(time.time() * 1000)
    excluded = {"not_usdt_swap": 0, "stable": 0, "not_crypto": 0, "not_live": 0,
                "short_history": 0, "extreme_funding": 0}
    cands = []
    for t in tickers:
        iid = str(t.get("instId") or "")
        if not iid.endswith("-USDT-SWAP"):
            excluded["not_usdt_swap"] += 1
            continue
        base = okx_base(iid)
        if base in STABLE_BASES:
            excluded["stable"] += 1
            continue
        m = meta.get(iid) or {}
        # instCategory 1 = crypto; 3 = tokenized stocks/ETFs, 4 = commodities (XAU, CL...).
        # The rule was researched on crypto perps only, so keep crypto only.
        if str(m.get("instCategory") or "1") != "1":
            excluded["not_crypto"] += 1
            continue
        if m.get("state") != "live" or m.get("settleCcy") != "USDT":
            excluded["not_live"] += 1
            continue
        try:
            lt = int(m.get("listTime") or 0)
        except ValueError:
            lt = 0
        if lt and now_ms - lt < MIN_LIST_DAYS * 86400 * 1000:
            excluded["short_history"] += 1
            continue
        try:
            qv = float(t.get("volCcy24h") or 0) * float(t.get("last") or 0)
        except ValueError:
            qv = 0.0
        cands.append({"instId": iid, "base": base, "quote_volume_24h": qv, "last": float(t.get("last") or 0)})
    cands.sort(key=lambda x: x["quote_volume_24h"], reverse=True)
    chosen = []
    for c in cands:
        if len(chosen) >= top_n:
            break
        if check_funding:
            try:
                fr = _get("/api/v5/public/funding-rate", {"instId": c["instId"]}, LIMIT_PUBLIC)
                rate = float((fr[0] if fr else {}).get("fundingRate") or 0)
                c["funding_rate"] = rate
                if abs(rate) > MAX_ABS_FUNDING:
                    excluded["extreme_funding"] += 1
                    continue
            except Exception:  # funding is a soft filter; never block the pool on it
                c["funding_rate"] = None
        chosen.append(c)
    now = pd.Timestamp.now(tz="UTC")
    return {
        "asof_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "asof_shanghai": now.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M"),
        "rules": {
            "market": "OKX USDT-margined perpetual swap (instType=SWAP, *-USDT-SWAP)",
            "top_n_by_24h_quote_volume": top_n,
            "quote_volume": "volCcy24h * last",
            "exclude_stable_bases": True,
            "crypto_only": "instCategory == 1 (drops tokenized stocks/ETFs and commodities)",
            "min_list_days": MIN_LIST_DAYS,
            "max_abs_funding": MAX_ABS_FUNDING,
        },
        "excluded_counts": excluded,
        "symbols": chosen,
    }


def _pool_age(path: Path) -> float:
    """Age from asof_utc (git checkouts reset mtimes), else mtime."""
    try:
        asof = json.loads(path.read_text(encoding="utf-8")).get("asof_utc")
        if asof:
            t = pd.Timestamp(str(asof)[:19]).tz_localize("UTC")
            return max(0.0, time.time() - t.timestamp())
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return time.time() - path.stat().st_mtime


def load_okx_pool(path: Path = OKX_POOL_PATH, max_age: float = OKX_POOL_MAX_AGE_SEC,
                  log=print) -> list[str]:
    """Return OKX instIds. Rebuild daily; on failure fall back to a stale cache; else []."""
    fresh = path.exists() and _pool_age(path) <= max_age
    if not fresh:
        try:
            pool = build_okx_pool()
            if pool["symbols"]:
                path.write_text(json.dumps(pool, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                log(f"okx pool rebuilt: {len(pool['symbols'])} symbols")
        except Exception as e:
            log(f"okx pool rebuild failed: {type(e).__name__}: {e}")
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [s["instId"] if isinstance(s, dict) else str(s) for s in data.get("symbols") or []]
    except (OSError, ValueError, KeyError):
        return []
