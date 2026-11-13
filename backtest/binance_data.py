"""Binance kline source with a fallback for networks where fapi.binance.com is blocked.

GitHub-hosted runners sit in the US, and Binance USDT-M futures (fapi) answers HTTP 451 there.
Order per symbol (HS_BINANCE_SOURCE=auto, the default):
  1. fapi.binance.com USDT-M perpetual klines (unchanged behaviour on the box);
  2. data-api.binance.vision spot klines for the same coin (1000PEPEUSDT -> PEPEUSDT with
     price x1000 and volume /1000);
  3. OKX USDT perpetual (<COIN>-USDT-SWAP) as a proxy, for perp-only coins that have no spot pair.
After the first 451/403 from fapi the module stops asking fapi for the rest of the process.
HS_BINANCE_SOURCE=fapi forces 1 only (raise on failure); =spot skips fapi.

Every successful fetch records which source was used in SOURCE_BY_SYMBOL so the scanner can
mark rows/payloads with data_source. Public market data only; no keys, no orders.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FAPI_BASE = "https://fapi.binance.com"
SPOT_BASE = "https://data-api.binance.vision"
PERP_LIST_PATH = ROOT / "perp_symbols.json"
SPOT_MAX_LIMIT = 1000
FAPI_MAX_LIMIT = 1500

SRC_FAPI = "binance_fapi"
SRC_SPOT = "binance_spot"
SRC_OKX_PROXY = "okx_swap_proxy"
SOURCE_CN = {
    SRC_FAPI: "币安合约K线",
    SRC_SPOT: "币安现货K线（GitHub 无法访问合约接口）",
    SRC_OKX_PROXY: "欧易永续K线代替（币安现货无此币，GitHub 无法访问币安合约接口）",
}
COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
OKX_BAR = {"15m": "15m", "1h": "1H", "4h": "4H", "1d": "1Dutc"}
INTERVAL_MS = {"15m": 15 * 60_000, "1h": 3_600_000, "4h": 4 * 3_600_000, "1d": 86_400_000}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "hs-scan-15m/1.0"})

STATE = {"fapi_blocked": False, "fapi_block_reason": ""}
SOURCE_BY_SYMBOL: dict[str, str] = {}
NO_SPOT: set[str] = set()


class SourceUnavailable(RuntimeError):
    pass


def mode() -> str:
    m = os.environ.get("HS_BINANCE_SOURCE", "auto").strip().lower()
    return m if m in ("auto", "fapi", "spot") else "auto"


def spot_symbol(perp: str) -> tuple[str, float]:
    """Spot pair and price multiplier for a Binance perp symbol."""
    s = perp.upper()
    for pre, mult in (("1000000", 1_000_000.0), ("1000", 1000.0), ("1M", 1_000_000.0)):
        if s.startswith(pre) and len(s) > len(pre) + 4:
            return s[len(pre):], mult
    return s, 1.0


def okx_inst(perp: str) -> str:
    base, _ = spot_symbol(perp)
    coin = base[:-4] if base.endswith("USDT") else base
    return f"{coin}-USDT-SWAP"


def _frame(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(raw, columns=COLS)
    for c in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[c] = df[c].astype("float64")
    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    return df


def _fapi(symbol: str, params: dict) -> pd.DataFrame:
    p = {**params, "symbol": symbol, "limit": min(int(params.get("limit") or 500), FAPI_MAX_LIMIT)}
    if os.environ.get("HS_SIM_FAPI_451") == "1":  # tests / local simulation of a US runner
        STATE["fapi_blocked"] = True
        STATE["fapi_block_reason"] = "HTTP 451 (simulated)"
        raise SourceUnavailable("fapi HTTP 451 (simulated)")
    r = SESSION.get(f"{FAPI_BASE}/fapi/v1/klines", params=p, timeout=(8, 30))
    if r.status_code in (451, 403):
        STATE["fapi_blocked"] = True
        STATE["fapi_block_reason"] = f"HTTP {r.status_code}"
        raise SourceUnavailable(f"fapi HTTP {r.status_code}")
    r.raise_for_status()
    return _frame(r.json())


def _spot(symbol: str, params: dict) -> pd.DataFrame:
    pair, mult = spot_symbol(symbol)
    if pair in NO_SPOT:
        raise SourceUnavailable(f"no spot pair {pair}")
    p = {**params, "symbol": pair, "limit": min(int(params.get("limit") or 500), SPOT_MAX_LIMIT)}
    r = SESSION.get(f"{SPOT_BASE}/api/v3/klines", params=p, timeout=(8, 30))
    if r.status_code == 400:
        NO_SPOT.add(pair)
        raise SourceUnavailable(f"no spot pair {pair}")
    r.raise_for_status()
    df = _frame(r.json())
    if mult != 1.0 and len(df):
        for c in ("open", "high", "low", "close"):
            df[c] = df[c] * mult
        for c in ("volume",):
            df[c] = df[c] / mult
    return df


_OKX_POOL_CACHE: dict = {}


def okx_pool_set() -> set[str]:
    if "set" not in _OKX_POOL_CACHE:
        try:
            data = json.loads((ROOT / "okx_pool.json").read_text(encoding="utf-8"))
            _OKX_POOL_CACHE["set"] = {s["instId"] if isinstance(s, dict) else str(s)
                                      for s in data.get("symbols") or []}
        except (OSError, ValueError, KeyError):
            _OKX_POOL_CACHE["set"] = set()
    return _OKX_POOL_CACHE["set"]


def _okx_proxy(symbol: str, interval: str, limit: int, start_ms: int | None) -> pd.DataFrame:
    import okx_market
    bar = OKX_BAR.get(interval)
    if okx_inst(symbol) in okx_pool_set():
        raise SourceUnavailable(f"okx proxy: {okx_inst(symbol)} already scanned in the OKX pool")
    if bar is None:
        raise SourceUnavailable(f"okx proxy: unsupported interval {interval}")
    df = okx_market.fetch_okx_klines(okx_inst(symbol), bar, limit, start_ms=start_ms)
    if df is None or not len(df):
        raise SourceUnavailable(f"okx proxy: no data for {okx_inst(symbol)}")
    return df


def fetch(symbol: str, interval: str, limit: int, start_ms: int | None = None) -> pd.DataFrame:
    """Raw klines (open bar included for Binance sources) as a DataFrame in the fapi column layout."""
    m = mode()
    params: dict = {"interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    errors = []
    if m == "fapi" or (m == "auto" and not STATE["fapi_blocked"]):
        try:
            df = _fapi(symbol, params)
            SOURCE_BY_SYMBOL[symbol] = SRC_FAPI
            return df
        except (SourceUnavailable, requests.RequestException) as e:
            if m == "fapi":
                raise
            errors.append(f"fapi: {type(e).__name__} {e}")
    try:
        df = _spot(symbol, params)
        SOURCE_BY_SYMBOL[symbol] = SRC_SPOT
        return df
    except (SourceUnavailable, requests.RequestException) as e:
        errors.append(f"spot: {type(e).__name__} {e}")
    try:
        df = _okx_proxy(symbol, interval, limit, start_ms)
        SOURCE_BY_SYMBOL[symbol] = SRC_OKX_PROXY
        return df
    except Exception as e:  # noqa: BLE001
        errors.append(f"okx: {type(e).__name__} {e}")
    raise SourceUnavailable(f"{symbol} {interval}: no source ({'; '.join(errors)})")


def fetch_paged_15m(symbol: str, start_ms: int | None) -> pd.DataFrame:
    """15m klines since start_ms (or the latest batch), paging across the source's max limit."""
    frames = []
    now_ms = int(time.time() * 1000)
    step = INTERVAL_MS["15m"]
    cur = start_ms
    for _ in range(50):
        df = fetch(symbol, "15m", FAPI_MAX_LIMIT, start_ms=cur)
        if df is None or not len(df):
            break
        frames.append(df)
        if cur is None or SOURCE_BY_SYMBOL.get(symbol) == SRC_OKX_PROXY:
            break  # okx proxy already pages internally from start_ms
        page_cap = FAPI_MAX_LIMIT if SOURCE_BY_SYMBOL.get(symbol) == SRC_FAPI else SPOT_MAX_LIMIT
        last_open = int(df["open_time"].iloc[-1])
        if len(df) < page_cap:
            break
        cur = last_open + step
        if cur >= now_ms:
            break
    if not frames:
        return pd.DataFrame(columns=COLS)
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def source_of(symbol: str, exchange: str = "binance") -> str:
    if exchange == "okx":
        return "okx_swap"
    return SOURCE_BY_SYMBOL.get(symbol, SRC_FAPI)


def summary() -> dict:
    srcs: dict[str, list[str]] = {}
    for s, src in sorted(SOURCE_BY_SYMBOL.items()):
        srcs.setdefault(src, []).append(s)
    return {"fapi_blocked": STATE["fapi_blocked"], "fapi_block_reason": STATE["fapi_block_reason"],
            "binance_sources": srcs, "mode": mode()}


def load_perp_list(path: Path = PERP_LIST_PATH) -> set[str]:
    try:
        return set(json.loads(path.read_text(encoding="utf-8")).get("symbols") or [])
    except (OSError, ValueError):
        return set()


def build_pool_from_spot(top_n: int = 30, perps: set[str] | None = None, stable: set[str] | None = None) -> dict:
    """Fallback core pool: spot 24h quote volume, restricted to coins with a USDT perp.

    Perp-only coins (no spot pair) cannot be ranked here and are dropped. Funding and listing
    age filters of build_core_pool.py need fapi and are skipped.
    """
    perps = perps if perps is not None else load_perp_list()
    if not perps:
        raise SourceUnavailable("perp_symbols.json missing/empty")
    if stable is None:
        try:
            from build_core_pool import STABLE_BASES as stable  # type: ignore
        except Exception:  # noqa: BLE001
            stable = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDE", "USD1", "PYUSD", "EUR"}
    spot_to_perp: dict[str, str] = {}
    for p in perps:
        pair, _ = spot_symbol(p)
        if pair not in spot_to_perp or p == pair:
            spot_to_perp[pair] = p
    r = SESSION.get(f"{SPOT_BASE}/api/v3/ticker/24hr", timeout=(8, 30))
    r.raise_for_status()
    rows = []
    for t in r.json():
        pair = t.get("symbol", "")
        if pair not in spot_to_perp:
            continue
        base = pair[:-4]
        if base in stable:
            continue
        try:
            qv = float(t.get("quoteVolume") or 0)
        except ValueError:
            continue
        rows.append((qv, spot_to_perp[pair], base))
    rows.sort(reverse=True)
    now = pd.Timestamp.now(tz="UTC")
    syms = [{"symbol": p, "base_asset": b, "quote_volume_24h_spot": qv, "rank": i + 1,
             "slippage_bps": 2 if p in ("BTCUSDT", "ETHUSDT") else 5}
            for i, (qv, p, b) in enumerate(rows[:top_n])]
    return {
        "asof_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "asof_shanghai": now.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M"),
        "rules": {"market": "Binance USDT-M perpetual (ranked by SPOT 24h quote volume; fapi unreachable)",
                  "top_n_by_24h_quote_volume": top_n, "source": "spot_fallback"},
        "n": len(syms),
        "symbols": syms,
    }
