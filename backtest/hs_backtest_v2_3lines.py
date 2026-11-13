#!/usr/bin/env python3
"""v2.0 H&S short: three independent lines (1D / 1H / 15m), retest entry."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import hs_backtest as hb
from hs_backtest import (
    DATA, OUT, RR_MIN, START_EQUITY, add_indicators, is_4h_bullish,
    load_ohlcv, macd_break_ok, macd_rs_ok, neck_at, run_portfolio,
    simulate_trade, to_shanghai, trade_stats, vol_part, Signal,
)

LAM = 1.5
EXIT_MODE = "partial60_1.5H"
PEAK_DISTANCE = 5
MIN_SPAN = 12
MAX_SPAN = 80
MIN_TROUGH_GAP = 3
BREAK_LOOKAHEAD = 24
RETEST_BARS = 12  # upper bound of 8–12 pattern bars
RISK_PCTS = (0.01, 0.02)

FONT = None
for p in (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
):
    if Path(p).exists():
        FONT = font_manager.FontProperties(fname=p)
        break


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def resample_weekly(df_d: pd.DataFrame) -> pd.DataFrame:
    x = df_d.set_index("dt").sort_index()
    agg = x.resample("W-MON", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
        **({"quote_volume": "sum"} if "quote_volume" in x.columns else {}),
    }).dropna(subset=["open", "close"])
    return add_indicators(agg.reset_index())


def _searchsorted_right(t: pd.Series, ts: pd.Timestamp) -> int:
    """searchsorted that survives pandas unit mismatches (ms bars vs us/ns now())."""
    ts = _ts(ts)
    unit = getattr(getattr(t, "dtype", None), "unit", None)
    if unit:
        try:
            return int(t.searchsorted(ts.as_unit(unit, round_ok=True), side="right"))
        except (ValueError, TypeError, AttributeError):
            pass
    s = pd.to_datetime(t, utc=True)
    if getattr(s.dtype, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    arr = s.to_numpy(dtype="datetime64[ns]")
    naive = ts.tz_convert("UTC").tz_localize(None).as_unit("ns")
    return int(np.searchsorted(arr, naive.to_datetime64(), side="right"))


def htf_bullish_at(df_htf: pd.DataFrame | None, ts, bar_delta: pd.Timedelta) -> bool:
    """True = clearly bullish or unknown → forbid short."""
    if df_htf is None or len(df_htf) == 0:
        return True
    ts = _ts(ts)
    t = df_htf["dt"]
    idx = _searchsorted_right(t, ts) - 1
    while idx >= 0:
        if _ts(t.iloc[idx]) + bar_delta <= ts:
            return is_4h_bullish(df_htf.iloc[idx])
        idx -= 1
    return True


def band_width(atr_v: float, neck: float) -> float:
    return max(0.15 * float(atr_v), 0.001 * float(neck))


def levels_at(lt, y1, rt, y2, t, atr_v):
    nk = float(neck_at(lt, y1, rt, y2, t))
    plat_lo, plat_hi = (float(y1), float(y2)) if y1 <= y2 else (float(y2), float(y1))
    bw = band_width(atr_v, nk)
    break_lvl = max(nk - bw, plat_lo - bw)
    resist_hi = max(nk + bw, plat_hi + bw)
    mid = max(nk, plat_lo)
    return nk, plat_lo, plat_hi, bw, break_lvl, resist_hi, mid


def scan_retest_entries(
    symbol: str,
    df: pd.DataFrame,
    bar_delta: pd.Timedelta,
    df_htf: pd.DataFrame | None,
    htf_delta: pd.Timedelta,
    funnel: dict,
) -> list[Signal]:
    n = len(df)
    if n < 100:
        return []
    high, low, opn, close = (df[c].to_numpy() for c in ("high", "low", "open", "close"))
    vol = df["volume"].to_numpy()
    atrv, rsi_ = df["atr"].to_numpy(), df["rsi"].to_numpy()
    ml, ms, mh = df["macd"].to_numpy(), df["macd_sig"].to_numpy(), df["macd_hist"].to_numpy()
    vma = df["vol_ma20"].to_numpy()
    times = df["dt"]

    prom = np.where(np.isfinite(atrv), np.maximum(0.5 * atrv, 0.003 * close), 0.003 * close)
    prom_s = float(np.nanmedian(prom[np.isfinite(prom)])) if np.isfinite(prom).any() else 0.0
    if prom_s <= 0:
        return []
    peaks, _ = find_peaks(high, distance=PEAK_DISTANCE, prominence=prom_s)
    if len(peaks) < 3:
        return []

    signals: list[Signal] = []
    used_break: set[int] = set()
    used_head: set[int] = set()

    # Iterate potential heads chronologically
    for ih, Hh in enumerate(peaks):
        Hh = int(Hh)
        if Hh in used_head:
            continue
        lefts = [int(p) for p in peaks[:ih] if MIN_SPAN // 4 <= (Hh - int(p)) <= MAX_SPAN]
        rights = [int(p) for p in peaks[ih + 1:] if (int(p) - Hh) <= MAX_SPAN]
        if not lefts or not rights:
            continue
        found_for_head = False
        for L in lefts:
            if found_for_head:
                break
            for R in rights:
                if found_for_head:
                    break
                span = R - L
                if not (MIN_SPAN <= span <= MAX_SPAN):
                    continue
                if not (high[L] < high[Hh] and high[R] < high[Hh]):
                    continue
                if not np.isfinite(atrv[Hh]) or not np.isfinite(atrv[R]):
                    continue
                # H must be unique max in [L, R]
                if float(np.max(high[L:R + 1])) > high[Hh] + 1e-15:
                    continue
                excess = high[Hh] - max(high[L], high[R])
                need = max(0.5 * atrv[Hh], 0.003 * close[Hh])
                if excess < need:
                    continue
                if high[R] - high[L] > 1.5 * atrv[R]:
                    continue
                lt = L + int(np.argmin(low[L:Hh + 1]))
                rt = Hh + int(np.argmin(low[Hh:R + 1]))
                if rt - lt < MIN_TROUGH_GAP:
                    continue
                funnel["shape"] += 1
                y1, y2 = float(low[lt]), float(low[rt])
                vl, vh, vr = vol_part(vol, L), vol_part(vol, Hh), vol_part(vol, R)
                if not (vr > 0 and vr <= vh and vr <= vl):
                    continue
                if not macd_rs_ok(ml, ms, mh, high, L, Hh, R):
                    continue
                funnel["patterns"] += 1

                max_t = min(n - 2, R + BREAK_LOOKAHEAD)  # leave a bar for entry
                for t in range(R + 1, max_t + 1):
                    if t in used_break:
                        continue
                    if not np.isfinite(vma[t]) or not np.isfinite(rsi_[t]) or not np.isfinite(atrv[t]):
                        continue
                    if high[t] > high[Hh]:
                        break
                    nk, plat_lo, plat_hi, bw, brk_lvl, resist_hi, mid = levels_at(
                        lt, y1, rt, y2, t, atrv[t]
                    )
                    if not (close[t] < opn[t] and close[t] < brk_lvl):
                        continue
                    if rsi_[t] >= 50:
                        continue
                    if not macd_break_ok(ml, ms, mh, t):
                        continue
                    vol_ratio = float(vol[t] / vma[t]) if vma[t] > 0 else 0.0
                    if vol_ratio <= LAM:
                        continue
                    height = high[Hh] - neck_at(lt, y1, rt, y2, Hh)
                    if height <= 0:
                        continue
                    break_close = _ts(times.iloc[t]) + bar_delta
                    if htf_bullish_at(df_htf, break_close, htf_delta):
                        funnel["htf_block"] += 1
                        continue
                    funnel["breaks"] += 1
                    target = float(nk - height)
                    target15 = float(nk - 1.5 * height)
                    rs_high = float(high[R])
                    break_vol = float(vol[t])

                    # retest on the SAME timeframe
                    end_r = min(n - 2, t + RETEST_BARS)
                    chosen = None
                    for j in range(t + 1, end_r + 1):
                        # invalidate
                        swing_hi = float(np.max(high[R:j + 1]))
                        near = min(rs_high, swing_hi)
                        stop_j = near + 0.5 * float(atrv[t])
                        cap = rs_high + 0.5 * float(atrv[t])
                        stop_j = min(stop_j, cap)
                        if high[j] >= stop_j or low[j] <= target:
                            chosen = None
                            break
                        nk_j, plat_lo_j, plat_hi_j, bw_j, brk_j, resist_j, mid_j = levels_at(
                            lt, y1, rt, y2, j, atrv[t]
                        )
                        if close[j] > resist_j:
                            chosen = None
                            break
                        if close[j] > opn[j] and vol[j] > break_vol:
                            chosen = None
                            break
                        if high[j] > rs_high:
                            chosen = None
                            break
                        touched = high[j] >= brk_j  # reached band/platform from below
                        closed_below = close[j] < mid_j
                        rng = high[j] - low[j]
                        upper_wick = high[j] - max(close[j], opn[j])
                        wick_ok = rng > 0 and upper_wick / rng >= 0.30
                        bearish = close[j] < opn[j]
                        if touched and closed_below and (bearish or wick_ok):
                            chosen = j
                            break
                    if chosen is None:
                        continue

                    entry_i = chosen + 1
                    entry_ts = _ts(times.iloc[entry_i])
                    entry_px = float(opn[entry_i])
                    swing_hi = float(np.max(high[R:chosen + 1]))
                    near = min(rs_high, swing_hi)
                    stop = min(near + 0.5 * float(atrv[t]), rs_high + 0.5 * float(atrv[t]))
                    if not (stop > entry_px > target):
                        funnel["geom_fail"] += 1
                        continue
                    rr = (entry_px - target) / (stop - entry_px)
                    if rr < RR_MIN:
                        funnel["rr_fail"] += 1
                        continue
                    sig = Signal(
                        symbol=symbol, lam=vol_ratio, L=L, H=Hh, R=R, break_i=t,
                        entry_time=entry_ts, entry_px=entry_px, stop=float(stop),
                        target=target, target15=target15, neck=float(nk),
                        head_high=float(high[Hh]), rs_high=rs_high,
                        atr_1h=float(atrv[t]), height=float(height),
                        left_high=float(high[L]), vol_ratio=vol_ratio, rr=float(rr),
                    )
                    signals.append(sig)
                    used_break.add(t)
                    used_head.add(Hh)
                    found_for_head = True
                    funnel["signals"] += 1
                    break
    return signals


def empty_funnel():
    return defaultdict(int, {k: 0 for k in (
        "shape", "patterns", "breaks", "htf_block", "geom_fail", "rr_fail", "signals"
    )})


def load_frames(sym: str):
    df15 = load_ohlcv(sym, "15m")
    if df15 is None or len(df15) < 400:
        return None
    df15i = add_indicators(df15)
    df1 = add_indicators(load_ohlcv(sym, "1h"))
    df4 = add_indicators(load_ohlcv(sym, "4h"))
    dfD = add_indicators(load_ohlcv(sym, "1D"))
    dfW = resample_weekly(dfD) if dfD is not None and len(dfD) > 80 else None
    return df15i, df1, df4, dfD, dfW


def main():
    uni = DATA / "universe.json"
    symbols = [s[0] if isinstance(s, list) else s for s in json.loads(uni.read_text())["symbols"]]
    print("symbols", len(symbols), flush=True)

    lines = {
        "1D": dict(bar=pd.Timedelta(days=1), htf_delta=pd.Timedelta(days=7), pat="D", htf="W"),
        "1H": dict(bar=pd.Timedelta(hours=1), htf_delta=pd.Timedelta(hours=4), pat="1", htf="4"),
        "15m": dict(bar=pd.Timedelta(minutes=15), htf_delta=pd.Timedelta(hours=1), pat="15", htf="1"),
    }
    sigs = {k: [] for k in lines}
    funnels = {k: empty_funnel() for k in lines}
    frames15 = {}

    for sym in symbols:
        pack = load_frames(sym)
        if pack is None:
            print(f"  skip {sym}", flush=True)
            continue
        df15, df1, df4, dfD, dfW = pack
        frames15[sym] = df15
        mapping = {
            "1D": (dfD, dfW),
            "1H": (df1, df4),
            "15m": (df15, df1),
        }
        msg = [f"{sym:16s}"]
        for name, (df_pat, df_htf) in mapping.items():
            if df_pat is None or len(df_pat) < 100:
                msg.append(f"{name} n/a")
                continue
            cfg = lines[name]
            fn = empty_funnel()
            found = scan_retest_entries(
                sym, df_pat, cfg["bar"], df_htf, cfg["htf_delta"], fn
            )
            for k, v in fn.items():
                funnels[name][k] += v
            sigs[name].extend(found)
            msg.append(f"{name} pat={fn['patterns']:4d} brk={fn['breaks']:3d} sig={fn['signals']:3d}")
        print("  " + " | ".join(msg), flush=True)

    # simulate
    trades = {}
    port = {}
    rows = []
    all_trade_rows = []
    for name in lines:
        raw = []
        for sig in sigs[name]:
            df15 = frames15.get(sig.symbol)
            if df15 is None:
                continue
            tr = simulate_trade(sig, df15, EXIT_MODE)
            if tr is not None:
                raw.append(tr)
        trades[name] = raw
        p1 = run_portfolio(raw, 0.01)
        p2 = run_portfolio(raw, 0.02)
        port[name] = {0.01: p1, 0.02: p2}
        st = trade_stats(raw)
        wr = st["win_rate"]
        print(
            f"{name:4s} n={st['n']:4d} wr={wr} avgR={st['avg_R']} pf={st['profit_factor']} "
            f"dd1={p1['max_dd']} funnel {dict(funnels[name])}",
            flush=True,
        )
        rows.append({
            "line": name,
            "n_signals": len(sigs[name]),
            **st,
            "max_dd_1pct": p1["max_dd"],
            "max_dd_2pct": p2["max_dd"],
            "final_1pct": p1["final"],
            "ret_1pct": p1["final"] / START_EQUITY - 1,
            "max_lose_streak": p1["max_lose_streak"],
            "n_taken": p1["n_taken"],
            "skipped_concurrent": p1["n_skipped_concurrent"],
            **{f"fn_{k}": int(funnels[name][k]) for k in funnels[name]},
        })
        for tr in raw:
            all_trade_rows.append({
                "line": name,
                "symbol": tr.symbol,
                "entry_shanghai": to_shanghai(tr.entry_time),
                "exit_shanghai": to_shanghai(tr.exit_time),
                "entry": tr.entry, "stop": tr.stop, "target": tr.target,
                "pnl_R": tr.pnl_R, "hit": tr.hit,
                "rr_planned": tr.rr_planned, "vol_ratio": tr.vol_ratio,
                "hold_h": tr.bars_held_15m * 0.25,
                "exit_px": tr.exit_px, "stop_pct": tr.stop_pct,
            })

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "summary_v2_3lines.csv", index=False)
    pd.DataFrame(all_trade_rows).to_csv(OUT / "trades_v2_3lines.csv", index=False)
    Path(OUT / "funnel_v2_3lines.json").write_text(json.dumps(
        {k: dict(v) for k, v in funnels.items()}, indent=2
    ))

    # equity figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False, dpi=120)
    titles = {"1D": "日线主线（周线过滤）", "1H": "1 小时主线（4H 过滤）", "15m": "15 分钟主线（1H 过滤）"}
    colors = {"1D": "#1d3557", "1H": "#e76f51", "15m": "#2a9d8f"}
    for ax, name in zip(axes, ("1D", "1H", "15m")):
        eq = port[name][0.01]["equity_curve"]
        if eq is not None and len(eq):
            ax.plot(eq.values, color=colors[name], lw=1.4)
        ax.set_title(titles[name], fontproperties=FONT, fontsize=12)
        ax.axhline(START_EQUITY, color="#999", lw=0.8, ls="--")
        ax.set_ylabel("权益 (1% 风险)", fontproperties=FONT)
        ax.grid(True, alpha=0.25)
        n = rows[[r["line"] for r in rows].index(name)]
        ax.text(0.01, 0.05, f"n={n['n']}  avgR={n['avg_R']:.3f}" if n["n"] else "无成交",
                transform=ax.transAxes, fontproperties=FONT, fontsize=9)
    axes[-1].set_xlabel("成交序号（组合内时间序）", fontproperties=FONT)
    fig.suptitle("头肩顶做空 v2.0 三条独立主线 · 回踩 · 1% 风险", fontproperties=FONT, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = Path("/workspace/trading/backtest/equity_v2_3lines.png")
    fig.savefig(out_png)
    print("wrote", out_png, out_png.stat().st_size, flush=True)
    return rows, all_trade_rows, funnels, port


if __name__ == "__main__":
    main()
