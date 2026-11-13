#!/usr/bin/env python3
"""
Head-and-shoulders short backtest (三栋 v1.0 rules, retest entry).

Reproducible: python hs_backtest.py
Requires parquet files in ./data/{SYMBOL}_{15m,1h,4h}.parquet
from download_data.py
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

# ---- costs ----
TAKER_FEE = 0.0005  # 0.05% per side (user specified)
SLIP_MAJOR = 0.0002  # BTC/ETH
SLIP_ALT = 0.0005
MAJORS = {"BTCUSDT", "ETHUSDT"}

# ---- pattern / entry params (documented in results.md) ----
PEAK_DISTANCE = 5
MIN_PATTERN_BARS = 20
MAX_PATTERN_BARS = 80
MIN_TROUGH_GAP = 3
BREAK_LOOKAHEAD_1H = 24          # max 1H bars after right shoulder to find break
RETEST_MAX_15M = 32              # 8 hours = 32 * 15m  (N)
RR_MIN = 1.5
MAX_CONCURRENT = 2
START_EQUITY = 10_000.0
VOL_LAMBDAS = (1.3, 1.5, 2.0)
EXIT_MODES = ("full_1.0H", "partial60_trail", "partial60_1.5H")

MMR = {"BTCUSDT": 0.004, "ETHUSDT": 0.005}  # others 0.01
MMR_DEFAULT = 0.01


def slippage_for(symbol: str) -> float:
    return SLIP_MAJOR if symbol in MAJORS else SLIP_ALT


def mmr_for(symbol: str) -> float:
    return MMR.get(symbol, MMR_DEFAULT)


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def ema(arr: np.ndarray, span: int) -> np.ndarray:
    s = pd.Series(arr, dtype="float64")
    return s.ewm(span=span, adjust=False).mean().to_numpy()


def atr(high, low, close, period=14) -> np.ndarray:
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return pd.Series(tr).rolling(period, min_periods=period).mean().to_numpy()


def rsi(close, period=14) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au = pd.Series(up).ewm(alpha=1 / period, adjust=False).mean()
    ad = pd.Series(dn).ewm(alpha=1 / period, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.to_numpy()


def macd(close, fast=12, slow=26, signal=9):
    f = ema(close, fast)
    s = ema(close, slow)
    line = f - s
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def vol_ma20(volume: np.ndarray) -> np.ndarray:
    # mean of previous 20 bars, excluding current
    out = pd.Series(volume).shift(1).rolling(20, min_periods=20).mean().to_numpy()
    return out


def _resample_from_15m(df15: pd.DataFrame, rule: str) -> pd.DataFrame:
    x = df15.set_index("dt").sort_index()
    agg = x.resample(rule, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "quote_volume": "sum",
    }).dropna(subset=["open", "close"])
    agg = agg.reset_index()
    agg["open_time"] = (agg["dt"].astype("int64") // 10**6)
    # pandas 3 may store datetime as us; normalize via timestamp
    agg["open_time"] = (agg["dt"].map(lambda t: int(pd.Timestamp(t).timestamp() * 1000))).astype("int64")
    return agg


def load_ohlcv(symbol: str, interval: str) -> pd.DataFrame | None:
    p15 = DATA / f"{symbol}_15m.parquet"
    if not p15.exists():
        return None
    df = pd.read_parquet(p15)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    # Vision files are milliseconds; guard if values look like seconds
    ot = df["open_time"].astype("int64")
    if ot.median() < 10**12:
        df["dt"] = pd.to_datetime(ot, unit="s", utc=True)
    else:
        df["dt"] = pd.to_datetime(ot, unit="ms", utc=True)
    if interval != "15m":
        df = _resample_from_15m(df, interval)
    return df.reset_index(drop=True)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), df["volume"].to_numpy()
    df = df.copy()
    df["ema20"] = ema(c, 20)
    df["ema60"] = ema(c, 60)
    df["atr"] = atr(h, l, c, 14)
    df["rsi"] = rsi(c, 14)
    ml, ms, mh = macd(c)
    df["macd"], df["macd_sig"], df["macd_hist"] = ml, ms, mh
    df["vol_ma20"] = vol_ma20(v)
    return df


def neck_at(x1, y1, x2, y2, x: float) -> float:
    if x2 == x1:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def vol_part(volume: np.ndarray, x: int) -> float:
    a = max(0, x - 1)
    b = min(len(volume), x + 2)
    return float(np.mean(volume[a:b]))


def macd_ok_right_shoulder(ml, mh, high, L, H, R) -> bool:
    """Death cross OR shrinking hist OR bearish divergence (head vs left)."""
    # death cross in [R-8, R]
    a = max(1, R - 8)
    for i in range(a, R + 1):
        if ml[i - 1] >= (ml[i] * 0 + ml[i]) and False:
            pass
        if ml[i - 1] >= ml[i] and ml[i] < (ml[i] - ml[i]):  # noop guard
            pass
    sig_ok = False
    # need macd_sig too — caller passes ml as line; we compare with provided arrays
    return True  # replaced below


def macd_rs_ok(ml, ms, mh, high, L, H, R) -> bool:
    a = max(1, R - 8)
    for i in range(a, R + 1):
        if ml[i - 1] >= ms[i - 1] and ml[i] < ms[i]:
            return True
    if R >= 2 and mh[R] < mh[R - 1] < mh[R - 2]:
        return True
    if high[H] > high[L] and ml[H] < ml[L]:
        return True
    return False


def macd_break_ok(ml, ms, mh, t: int) -> bool:
    if t < 1:
        return False
    if ml[t] < ms[t]:
        return True
    if mh[t] < mh[t - 1] and mh[t] < 0:
        return True
    return False


def is_4h_bullish(row) -> bool:
    """No new shorts when 4H is bullish: EMA20 > EMA60 AND close > EMA20."""
    if not np.isfinite(row["ema20"]) or not np.isfinite(row["ema60"]):
        return True  # treat unknown as not tradable (conservative: skip)
    return bool(row["ema20"] > row["ema60"] and row["close"] > row["ema20"])


# ---------------------------------------------------------------------------
# Pattern detection + retest entries (signal generation, no exits yet)
# ---------------------------------------------------------------------------
@dataclass
class Signal:
    symbol: str
    lam: float
    L: int
    H: int
    R: int
    break_i: int
    entry_time: pd.Timestamp  # 15m open, UTC
    entry_px: float
    stop: float
    target: float
    target15: float
    neck: float
    head_high: float
    rs_high: float
    atr_1h: float
    height: float
    left_high: float
    vol_ratio: float
    rr: float


def detect_signals(symbol: str, df1: pd.DataFrame, df4: pd.DataFrame, df15: pd.DataFrame) -> list[Signal]:
    n = len(df1)
    high = df1["high"].to_numpy()
    low = df1["low"].to_numpy()
    opn = df1["open"].to_numpy()
    close = df1["close"].to_numpy()
    vol = df1["volume"].to_numpy()
    atrv = df1["atr"].to_numpy()
    rsi_ = df1["rsi"].to_numpy()
    ml = df1["macd"].to_numpy()
    ms = df1["macd_sig"].to_numpy()
    mh = df1["macd_hist"].to_numpy()
    vma = df1["vol_ma20"].to_numpy()
    times = df1["dt"].to_numpy()

    # 4H lookup: last closed 4H bar strictly before timestamp ts
    t4 = df4["dt"].to_numpy()
    bull4 = np.array([is_4h_bullish(df4.iloc[i]) for i in range(len(df4))], dtype=bool)

    def fourh_bullish_at(ts) -> bool:
        # last 4H with open_time + 4h <= ts  i.e. bar that has closed
        # 4H open_time is bar start; closed at start+4h
        idx = np.searchsorted(t4, ts, side="right") - 1
        # require that 4H bar has closed: t4[idx] + 4h <= ts
        while idx >= 0:
            if t4[idx] + np.timedelta64(4, "h") <= ts:
                return bool(bull4[idx])
            idx -= 1
        return True

    t15 = df15["dt"].to_numpy()
    o15 = df15["open"].to_numpy()
    h15 = df15["high"].to_numpy()
    l15 = df15["low"].to_numpy()
    c15 = df15["close"].to_numpy()
    v15 = df15["volume"].to_numpy()

    # peak prominence array
    prom = np.where(np.isfinite(atrv), np.maximum(0.5 * atrv, 0.003 * close), 0.003 * close)
    prom_s = float(np.nanmedian(prom[np.isfinite(prom)])) if np.isfinite(prom).any() else 0.0
    if prom_s <= 0:
        return []
    peaks, _ = find_peaks(high, distance=PEAK_DISTANCE, prominence=prom_s)
    if len(peaks) < 3:
        return []

    signals: list[Signal] = []
    used_break = set()

    for pi in range(len(peaks) - 2):
        L, Hh, R = int(peaks[pi]), int(peaks[pi + 1]), int(peaks[pi + 2])
        span = R - L
        if not (MIN_PATTERN_BARS <= span <= MAX_PATTERN_BARS):
            continue
        if not (high[L] < high[Hh] and high[R] < high[Hh]):
            continue
        if not np.isfinite(atrv[Hh]) or not np.isfinite(atrv[R]):
            continue
        min_excess = max(0.5 * atrv[Hh], 0.003 * close[Hh])
        if high[Hh] - max(high[L], high[R]) < min_excess:
            continue
        if high[R] - high[L] > 1.5 * atrv[R]:
            continue
        lt = L + int(np.argmin(low[L : Hh + 1]))
        rt = Hh + int(np.argmin(low[Hh : R + 1]))
        if rt - lt < MIN_TROUGH_GAP:
            continue
        y1, y2 = float(low[lt]), float(low[rt])
        # volume structure 左 >= 头 >= 右
        vl, vh, vr = vol_part(vol, L), vol_part(vol, Hh), vol_part(vol, R)
        if not (vl >= vh >= vr > 0):
            continue
        if not macd_rs_ok(ml, ms, mh, high, L, Hh, R):
            continue

        # search break
        max_t = min(n - 1, R + BREAK_LOOKAHEAD_1H)
        for t in range(R + 1, max_t + 1):
            if t in used_break:
                continue
            if not np.isfinite(vma[t]) or not np.isfinite(rsi_[t]) or not np.isfinite(atrv[t]):
                continue
            if high[t] > high[Hh]:
                break  # pattern invalidated
            nk = neck_at(lt, y1, rt, y2, t)
            # close below neck, bearish body
            if not (close[t] < nk and close[t] < opn[t]):
                continue
            if rsi_[t] >= 50:
                continue
            if not macd_break_ok(ml, ms, mh, t):
                continue
            vol_ratio = vol[t] / vma[t] if vma[t] > 0 else 0.0
            if vol_ratio <= min(VOL_LAMBDAS):
                continue  # must at least beat 1.3; finer filter later

            # 4H filter at break close time (bar t closes at t+1h)
            break_close_ts = times[t] + np.timedelta64(1, "h")
            if fourh_bullish_at(break_close_ts):
                continue

            height = high[Hh] - neck_at(lt, y1, rt, y2, Hh)
            if height <= 0:
                continue
            neck_b = nk
            target = neck_b - height
            target15 = neck_b - 1.5 * height
            rs_high = float(high[R])
            # stop at later of right shoulder and any swing high between R and break
            swing = float(np.max(high[R : t + 1]))
            stop_raw = max(rs_high, swing) + 0.5 * float(atrv[t])

            # 15m retest after 1H bar close
            # first 15m index strictly after break close
            start15 = int(np.searchsorted(t15, break_close_ts, side="left"))
            end15 = min(len(df15) - 2, start15 + RETEST_MAX_15M)  # leave 1 bar for next open entry
            if start15 >= end15:
                continue

            band = max(0.15 * float(atrv[t]), 0.001 * neck_b)
            break_vol = float(vol[t])
            found = None
            rebound_vol_fail = False
            for j in range(start15, end15 + 1):
                # invalidate if 15m already ran to target or through stop before entry
                if h15[j] >= stop_raw:
                    found = None
                    break
                if l15[j] <= target:
                    found = None
                    break
                # failed break: 15m close back above neckline band upper
                if c15[j] > neck_b + band:
                    found = None
                    break
                # bounce volume vs breakdown (use 15m bars that close above prior 15m / rise toward neck)
                # Approximate: if a 15m up-bar (close>open) that reaches the band has volume
                # mapped... we check 1H bars after break that bounce: handled via 15m
                # If this 15m is a bounce bar into the band and its volume * (1h/15m scale)
                # is messy. Use: if close>open and high reaches band and v15 > break_vol
                # (break_vol is 1H volume, not comparable). Instead check subsequent 1H
                # bounce later. Skip incompatible units.
                touched = h15[j] >= neck_b - band
                closed_below = c15[j] < neck_b
                # rejection candle: touched band from below, closed back below neck
                rng = h15[j] - l15[j]
                upper_wick = h15[j] - max(c15[j], o15[j])
                wick_ok = rng > 0 and upper_wick / rng >= 0.30
                bearish = c15[j] < o15[j]
                if touched and closed_below and (bearish or wick_ok):
                    # entry next 15m open
                    if j + 1 >= len(df15):
                        break
                    found = j
                    break

            if found is None:
                continue

            # check 1H bounce volume between break and entry: any 1H up bar with vol > break_vol
            entry_ts = t15[found + 1]
            # 1H bars strictly after break bar and with close_time <= entry
            bounce_fail = False
            for u in range(t + 1, n):
                if times[u] >= entry_ts:
                    break
                if close[u] > opn[u] and vol[u] > break_vol:
                    bounce_fail = True
                    break
                if high[u] > rs_high:
                    bounce_fail = True
                    break
            if bounce_fail:
                continue

            entry_px = float(o15[found + 1])
            # apply later slippage at simulation; RR on raw prices
            if not (stop_raw > entry_px > target):
                continue
            rr = (entry_px - target) / (stop_raw - entry_px)
            if rr < RR_MIN:
                continue

            # store one signal per break, tagged with max lambda it satisfies;
            # actual lambda filter applied when collecting
            sig = Signal(
                symbol=symbol,
                lam=vol_ratio,  # store actual ratio; filter by ratio > lambda
                L=L, H=Hh, R=R, break_i=t,
                entry_time=pd.Timestamp(entry_ts).tz_convert("UTC"),
                entry_px=entry_px,
                stop=stop_raw,
                target=target,
                target15=target15,
                neck=neck_b,
                head_high=float(high[Hh]),
                rs_high=rs_high,
                atr_1h=float(atrv[t]),
                height=float(height),
                left_high=float(high[L]),
                vol_ratio=float(vol_ratio),
                rr=float(rr),
            )
            signals.append(sig)
            used_break.add(t)
            break  # next pattern

    return signals


# ---------------------------------------------------------------------------
# Trade simulation on 15m path
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    symbol: str
    lam: float
    exit_mode: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    stop: float
    target: float
    exit_px: float  # volume-weighted average exit
    pnl_R: float    # in R multiples, after fees+slip
    pnl_pct_on_notional: float
    hit: str
    bars_held_15m: int
    max_lev: float
    stop_pct: float
    rr_planned: float
    vol_ratio: float


def max_safe_leverage(symbol: str, stop_pct: float) -> float:
    """Isolated USDT-M short: LP ≈ entry * (1 + 1/L - MMR).
    Require (LP-entry)/entry >= 2 * stop_pct, plus fee buffer 0.10%.
    """
    mmr = mmr_for(symbol)
    fee_buf = 0.001
    denom = 2.0 * stop_pct + mmr + fee_buf
    if denom <= 0:
        return 1.0
    lev = 1.0 / denom
    return float(max(1.0, min(lev, 75.0)))


def simulate_trade(sig: Signal, df15: pd.DataFrame, exit_mode: str) -> Trade | None:
    entry_ts = pd.Timestamp(sig.entry_time)
    if entry_ts.tzinfo is None:
        entry_ts = entry_ts.tz_localize("UTC")
    else:
        entry_ts = entry_ts.tz_convert("UTC")
    i0 = int(df15["dt"].searchsorted(entry_ts, side="left"))
    if i0 >= len(df15) - 1:
        return None

    slip = slippage_for(sig.symbol)
    fee = TAKER_FEE
    entry = sig.entry_px * (1 - slip)  # short: slightly worse (lower sell)
    stop = sig.stop
    target = sig.target
    target15 = sig.target15
    stop_pct = (stop - entry) / entry
    if stop_pct <= 0:
        return None

    h = df15["high"].to_numpy()
    l = df15["low"].to_numpy()
    o = df15["open"].to_numpy()
    dt = df15["dt"].to_numpy()
    atr_trail = sig.atr_1h

    remaining = 1.0
    realized = 0.0  # notional pnl fraction * weight, before round-trip cost accounting
    # We compute VWAP exit of filled fractions; fees on each fill.
    filled_px_w = 0.0
    filled_w = 0.0
    trail_stop = stop
    t1_hit = False
    end_i = len(df15) - 1
    exit_i = end_i
    hit = "eod"

    def fill(px: float, w: float, reason: str, i: int):
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
        # conservative: if both stop and target in same bar, assume stop first (adverse)
        hit_stop = hi >= trail_stop
        hit_t1 = lo <= target
        hit_t15 = lo <= target15

        if exit_mode == "full_1.0H":
            if hit_stop and hit_t1:
                fill(trail_stop * (1 + slip), 1.0, "stop_samebar_priority", i)
                break
            if hit_stop:
                fill(max(op, trail_stop) * (1 + slip), 1.0, "stop", i)
                break
            if hit_t1:
                fill(min(op, target) * (1 - slip), 1.0, "target_1.0H", i)
                break

        elif exit_mode == "partial60_1.5H":
            if (not t1_hit) and hit_stop and hit_t1:
                fill(trail_stop * (1 + slip), 1.0, "stop_samebar_priority", i)
                break
            if hit_stop:
                fill(max(op, trail_stop) * (1 + slip), remaining, "stop", i)
                break
            if (not t1_hit) and hit_t1:
                fill(min(op, target) * (1 - slip), 0.60, "target_1.0H", i)
                t1_hit = True
                trail_stop = entry  # move remainder to breakeven
                if remaining <= 1e-12:
                    break
                # if also hit 1.5H same bar after T1
                if hit_t15:
                    fill(min(op, target15) * (1 - slip), remaining, "target_1.5H", i)
                    break
                continue
            if t1_hit and hit_t15:
                fill(min(op, target15) * (1 - slip), remaining, "target_1.5H", i)
                break

        elif exit_mode == "partial60_trail":
            if (not t1_hit) and hit_stop and hit_t1:
                fill(trail_stop * (1 + slip), 1.0, "stop_samebar_priority", i)
                break
            if hit_stop:
                fill(max(op, trail_stop) * (1 + slip), remaining, "stop", i)
                break
            if (not t1_hit) and hit_t1:
                fill(min(op, target) * (1 - slip), 0.60, "target_1.0H", i)
                t1_hit = True
                trail_stop = entry
                if remaining <= 1e-12:
                    break
                continue
            if t1_hit:
                # trail: lowest low since T1 + 1*ATR ; stop only moves down (favorable for short)
                # running min updated including this bar's low after checking stop at old trail
                new_trail = lo + 1.0 * atr_trail
                if new_trail < trail_stop:
                    trail_stop = new_trail

        if remaining <= 1e-12:
            break
    else:
        # end of data
        last = float(df15["close"].iloc[-1])
        fill(last * (1 - slip), remaining, "eod", len(df15) - 1)

    if filled_w <= 0:
        return None
    exit_px = filled_px_w / filled_w
    # short pnl on notional
    raw = (entry - exit_px) / entry
    # fees: open always; close always (each side 0.05%). Partial exits still one open + weighted closes ≈ 2 sides.
    # Use 2 * fee on full notional (open+close).
    net = raw - 2 * fee
    R = (stop - entry) / entry
    pnl_R = net / R if R > 0 else 0.0

    return Trade(
        symbol=sig.symbol,
        lam=sig.lam,
        exit_mode=exit_mode,
        entry_time=sig.entry_time,
        exit_time=pd.Timestamp(df15["dt"].iloc[min(exit_i, len(df15) - 1)]),
        entry=entry,
        stop=stop,
        target=target,
        exit_px=exit_px,
        pnl_R=float(pnl_R),
        pnl_pct_on_notional=float(net),
        hit=hit,
        bars_held_15m=int(max(1, exit_i - i0 + 1)),
        max_lev=max_safe_leverage(sig.symbol, stop_pct),
        stop_pct=float(stop_pct),
        rr_planned=float(sig.rr),
        vol_ratio=float(sig.vol_ratio),
    )


# ---------------------------------------------------------------------------
# Portfolio equity
# ---------------------------------------------------------------------------
def run_portfolio(trades: list[Trade], risk_pct: float) -> dict:
    """Event-based equity with up to MAX_CONCURRENT overlapping trades.
    PnL applied at each trade's exit. Position notional = equity_at_entry * risk / stop_pct.
    If max concurrent exceeded at entry, skip (already filtered if we pre-select).
    Here we assume `trades` are already concurrency-filtered or we filter now.
    """
    if not trades:
        return {
            "equity_curve": pd.Series(dtype=float),
            "final": START_EQUITY,
            "max_dd": 0.0,
            "max_lose_streak": 0,
            "n_taken": 0,
            "n_skipped_concurrent": 0,
        }
    tr = sorted(trades, key=lambda t: t.entry_time)
    equity = START_EQUITY
    curve_times = [tr[0].entry_time]
    curve_eq = [equity]
    open_pos = []  # list of (exit_time, entry_equity_used_for_risk, trade)
    skipped = 0
    taken = []
    peak = equity
    max_dd = 0.0

    def close_due(ts):
        nonlocal equity, peak, max_dd
        still = []
        closed = []
        for item in open_pos:
            if item[0] <= ts:
                closed.append(item)
            else:
                still.append(item)
        # chronological close
        closed.sort(key=lambda x: x[0])
        for exit_t, eq_at_entry, td in closed:
            notional = eq_at_entry * risk_pct / td.stop_pct
            equity += notional * td.pnl_pct_on_notional
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            curve_times.append(exit_t)
            curve_eq.append(equity)
        return still

    for td in tr:
        open_pos = close_due(td.entry_time)
        if len(open_pos) >= MAX_CONCURRENT:
            skipped += 1
            continue
        taken.append(td)
        open_pos.append((td.exit_time, equity, td))

    open_pos = close_due(pd.Timestamp("2262-01-01", tz="UTC"))

    # losing streak on taken trades in time order
    streak = max_streak = 0
    for td in taken:
        if td.pnl_R < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    s = pd.Series(curve_eq, index=pd.DatetimeIndex(curve_times)).sort_index()
    # collapse duplicate timestamps by last
    s = s[~s.index.duplicated(keep="last")]
    return {
        "equity_curve": s,
        "final": equity,
        "max_dd": max_dd,
        "max_lose_streak": max_streak,
        "n_taken": len(taken),
        "n_skipped_concurrent": skipped,
        "taken": taken,
    }


def trade_stats(trades: list[Trade]) -> dict:
    if not trades:
        return {
            "n": 0, "win_rate": np.nan, "avg_R": np.nan, "expectancy_R": np.nan,
            "profit_factor": np.nan, "med_R": np.nan, "avg_hold_h": np.nan,
        }
    r = np.array([t.pnl_R for t in trades], dtype=float)
    wins = r[r > 0]
    losses = r[r <= 0]
    gp = wins.sum() if len(wins) else 0.0
    gl = -losses.sum() if len(losses) else 0.0
    pf = gp / gl if gl > 0 else (np.inf if gp > 0 else np.nan)
    return {
        "n": int(len(trades)),
        "win_rate": float((r > 0).mean()),
        "avg_R": float(r.mean()),
        "expectancy_R": float(r.mean()),
        "profit_factor": float(pf) if np.isfinite(pf) else None,
        "med_R": float(np.median(r)),
        "avg_hold_h": float(np.mean([t.bars_held_15m * 0.25 for t in trades])),
    }


def to_shanghai(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M")


def main():
    uni_path = DATA / "universe.json"
    if uni_path.exists():
        symbols = [s for s, _ in json.loads(uni_path.read_text())["symbols"]]
    else:
        symbols = sorted({p.name.split("_")[0] for p in DATA.glob("*_1h.parquet")})

    coverage = []
    all_signals: list[Signal] = []
    print(f"Symbols: {len(symbols)}")
    for sym in symbols:
        df1 = load_ohlcv(sym, "1h")
        df4 = load_ohlcv(sym, "4h")
        df15 = load_ohlcv(sym, "15m")
        if df1 is None or df4 is None or df15 is None:
            coverage.append({"symbol": sym, "status": "missing_tf"})
            print(f"  skip {sym}: missing timeframe")
            continue
        if len(df1) < 200 or len(df15) < 200:
            coverage.append({"symbol": sym, "status": "too_short", "n1h": len(df1)})
            print(f"  skip {sym}: too short ({len(df1)} 1h)")
            continue
        df1 = add_indicators(df1)
        df4 = add_indicators(df4)
        cov = {
            "symbol": sym,
            "status": "ok",
            "start": str(df1["dt"].iloc[0]),
            "end": str(df1["dt"].iloc[-1]),
            "n_1h": int(len(df1)),
            "n_15m": int(len(df15)),
        }
        coverage.append(cov)
        sigs = detect_signals(sym, df1, df4, df15)
        print(f"  {sym:16s} {df1['dt'].iloc[0].date()} -> {df1['dt'].iloc[-1].date()}  signals={len(sigs)}")
        all_signals.extend(sigs)
        # cache 15m for later sim
        (OUT / "_cache").mkdir(exist_ok=True)

    (OUT / "coverage.json").write_text(json.dumps(coverage, indent=2, default=str))
    print(f"Total raw signals (vol_ratio>1.3 & filters): {len(all_signals)}")

    # preload 15m frames
    frames15 = {}
    for sym in {s.symbol for s in all_signals}:
        frames15[sym] = load_ohlcv(sym, "15m")

    rows = []
    trades_by_key: dict[tuple, list[Trade]] = {}
    for lam in VOL_LAMBDAS:
        sigs = [s for s in all_signals if s.vol_ratio > lam]
        print(f"lambda={lam}: {len(sigs)} signals")
        for mode in EXIT_MODES:
            trades = []
            for s in sigs:
                td = simulate_trade(s, frames15[s.symbol], mode)
                if td is None:
                    continue
                td.lam = lam  # tag filter used
                trades.append(td)
            trades_by_key[(lam, mode)] = trades
            print(f"  mode={mode}: {len(trades)} trades")

    # persist trades
    recs = []
    for (lam, mode), trades in trades_by_key.items():
        for t in trades:
            recs.append({
                "lambda": lam,
                "exit_mode": mode,
                "symbol": t.symbol,
                "entry_utc": str(t.entry_time),
                "entry_shanghai": to_shanghai(t.entry_time),
                "exit_shanghai": to_shanghai(t.exit_time),
                "entry": t.entry,
                "stop": t.stop,
                "target": t.target,
                "exit_px": t.exit_px,
                "pnl_R": t.pnl_R,
                "hit": t.hit,
                "max_lev": t.max_lev,
                "stop_pct": t.stop_pct,
                "rr_planned": t.rr_planned,
                "vol_ratio": t.vol_ratio,
                "hold_h": t.bars_held_15m * 0.25,
            })
    tdf = pd.DataFrame(recs)
    tdf.to_csv(OUT / "trades.csv", index=False)

    # equity + stats
    summary = []
    curves = {}
    for (lam, mode), trades in trades_by_key.items():
        st = trade_stats(trades)
        port1 = run_portfolio(trades, 0.01)
        port2 = run_portfolio(trades, 0.02)
        lev = np.array([t.max_lev for t in trades]) if trades else np.array([])
        summary.append({
            "lambda": lam,
            "exit_mode": mode,
            **st,
            "max_dd_1pct": port1["max_dd"],
            "max_dd_2pct": port2["max_dd"],
            "final_1pct": port1["final"],
            "final_2pct": port2["final"],
            "ret_1pct": port1["final"] / START_EQUITY - 1,
            "ret_2pct": port2["final"] / START_EQUITY - 1,
            "max_lose_streak": port1["max_lose_streak"],
            "n_portfolio_1pct": port1["n_taken"],
            "skipped_concurrent": port1["n_skipped_concurrent"],
            "lev_p25": float(np.percentile(lev, 25)) if len(lev) else None,
            "lev_p50": float(np.percentile(lev, 50)) if len(lev) else None,
            "lev_p75": float(np.percentile(lev, 75)) if len(lev) else None,
            "lev_min": float(lev.min()) if len(lev) else None,
            "lev_max": float(lev.max()) if len(lev) else None,
        })
        curves[(lam, mode)] = port1["equity_curve"]

    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT / "summary.csv", index=False)
    print(sdf.to_string(index=False))

    # pick best: among variants with n>=10, highest expectancy_R then profit_factor;
    # if none have n>=10, use all.
    cand = sdf.dropna(subset=["expectancy_R"])
    if (cand["n"] >= 10).any():
        pool = cand[cand["n"] >= 10]
    else:
        pool = cand
    if len(pool):
        pool = pool.sort_values(["expectancy_R", "profit_factor", "n"], ascending=False)
        best = pool.iloc[0]
        best_key = (float(best["lambda"]), str(best["exit_mode"]))
    else:
        best_key = (1.5, "full_1.0H")
        best = None

    # plot equity
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    for fp in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            name = font_manager.FontProperties(fname=fp).get_name()
            rcParams["font.family"] = name
            rcParams["axes.unicode_minus"] = False
            break

    fig, ax = plt.subplots(figsize=(11, 5.5))
    try:
        from matplotlib import font_manager
        cjk = [f.fname for f in font_manager.fontManager.ttflist
               if any(k in f.name.lower() for k in ("noto sans cjk", "source han", "wqy", "droid sans fallback", "nimbus"))]
        # prefer noto/wqy
        picks = [f.fname for f in font_manager.fontManager.ttflist if "NotoSansCJK" in f.fname or "Noto Sans CJK" in f.name or "wqy" in f.fname.lower() or "DroidSansFallback" in f.fname]
        if picks:
            from matplotlib import rcParams
            fp = font_manager.FontProperties(fname=picks[0])
            rcParams["font.family"] = fp.get_name()
            font_manager.fontManager.addfont(picks[0])
            rcParams["font.sans-serif"] = [fp.get_name()]
            rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    colors = {
        "full_1.0H": "C0",
        "partial60_trail": "C1",
        "partial60_1.5H": "C2",
    }
    styles = {1.3: "-", 1.5: "--", 2.0: ":"}
    for (lam, mode), curve in curves.items():
        if curve is None or len(curve) < 2:
            continue
        c = curve.copy()
        c.index = c.index.tz_convert("Asia/Shanghai")
        lw = 2.4 if (lam, mode) == best_key else 1.1
        alpha = 1.0 if (lam, mode) == best_key else 0.55
        label = f"λ={lam} {mode}"
        if (lam, mode) == best_key:
            label += " [best]"
        ax.plot(c.index, c.values, linestyle=styles.get(lam, "-"),
                color=colors.get(mode, "C7"), lw=lw, alpha=alpha, label=label)
    ax.axhline(START_EQUITY, color="gray", lw=0.8, ls="--")
    ax.set_title("头肩顶回踩做空 — 权益曲线（单笔风险 1%，起始 10,000 USDT）")
    ax.set_ylabel("权益 (USDT)")
    ax.set_xlabel("时间（Asia/Shanghai）")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "equity.png", dpi=140)
    fig.savefig(ROOT / "equity.png", dpi=140)
    print("wrote", ROOT / "equity.png")

    # example trades from best variant
    best_trades = trades_by_key.get(best_key, [])
    best_trades_sorted = sorted(best_trades, key=lambda t: t.entry_time)

    meta = {
        "best_key": {"lambda": best_key[0], "exit_mode": best_key[1]},
        "n_signals_raw": len(all_signals),
        "start_equity": START_EQUITY,
        "rr_min": RR_MIN,
        "retest_max_15m": RETEST_MAX_15M,
        "fee": TAKER_FEE,
        "universe": symbols,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    # pickle-free dump of best examples
    examples = []
    # pick mixed wins/losses
    wins = [t for t in best_trades_sorted if t.pnl_R > 0][:3]
    losses = [t for t in best_trades_sorted if t.pnl_R <= 0][:3]
    mid = best_trades_sorted[len(best_trades_sorted)//2:len(best_trades_sorted)//2+2] if best_trades_sorted else []
    seen = set()
    for t in wins + losses + mid:
        key = (t.symbol, str(t.entry_time))
        if key in seen:
            continue
        seen.add(key)
        examples.append(t)
    (OUT / "examples.json").write_text(json.dumps([
        {
            "symbol": t.symbol,
            "entry_shanghai": to_shanghai(t.entry_time),
            "exit_shanghai": to_shanghai(t.exit_time),
            "entry": t.entry, "stop": t.stop, "target": t.target,
            "exit_px": t.exit_px, "pnl_R": t.pnl_R, "hit": t.hit,
            "max_lev": t.max_lev, "vol_ratio": t.vol_ratio,
        } for t in examples
    ], indent=2))
    print("DONE best", best_key, "n", len(best_trades))


if __name__ == "__main__":
    main()
