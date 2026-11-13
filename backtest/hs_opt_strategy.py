#!/usr/bin/env python3
"""v2.0 三周期独立主线：预先锁定的入场 / 止盈 / 形态质量对照（含两两组合）。

看结果之前写死的设定，禁止看完再加维度：
A 入场：破位即入 vs 回踩（基线）
B 止盈：量度 60%@1.0H+余仓1.5H（基线） vs 纯 ATR 追踪 1.5 / 2.5 倍
C 形态减法（同时满足）：
   1) 颈线更平：|右谷-左谷| / ATR[R] ≤ 0.75
   2) 右肩明显低于头：(头高-右肩高) / ATR[H] ≥ 1.0
   3) 两肩对称：|右肩高-左肩高| / ATR[R] ≤ 0.75
组合（预先登记，不是看完单改进再挑）：
   A+B1, A+B2, A+C, C+B1, C+B2
推广门槛（每周期独立）：
   OOS 笔数 ≥ {1D:8, 1H:40, 15m:40}
   OOS 平均 R > 0
   OOS 平均 R ≥ 该周期基线 OOS 平均 R + 0.05
"""
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

from hs_backtest import (
    DATA, OUT, RR_MIN, START_EQUITY, TAKER_FEE, Trade,
    add_indicators, load_ohlcv, macd_break_ok, macd_rs_ok, neck_at,
    run_portfolio, simulate_trade, slippage_for, trade_stats, vol_part,
    Signal, max_safe_leverage,
)
from hs_backtest_v2_3lines import (
    MIN_SPAN, MAX_SPAN, MIN_TROUGH_GAP, PEAK_DISTANCE, BREAK_LOOKAHEAD,
    RETEST_BARS, _ts, levels_at, htf_bullish_at, resample_weekly, empty_funnel,
)

LAM = 1.5
MEASURED_EXIT = "partial60_1.5H"

# ---- locked quality thresholds ----
NECK_FLAT_ATR = 0.75
HEAD_RS_ATR = 1.0
SHOULDER_SYM_ATR = 0.75

# ---- locked trail multiples (pattern-period ATR stored on Signal.atr_1h) ----
TRAIL_KS = (1.5, 2.5)

OOS_MIN_N = {"1D": 8, "1H": 40, "15m": 40}
OOS_MIN_EDGE = 0.05

# Pre-registered variants: (vid, label, entry, exit, quality)
# entry: retest | breakout
# exit: measured | trail_1.5 | trail_2.5
VARIANTS = [
    ("baseline", "基线：回踩+量度离场", "retest", "measured", False),
    ("A", "A 破位即入", "breakout", "measured", False),
    ("B1", "B 追踪 1.5×ATR", "retest", "trail_1.5", False),
    ("B2", "B 追踪 2.5×ATR", "retest", "trail_2.5", False),
    ("C", "C 高质量形态", "retest", "measured", True),
    ("A+B1", "A+B 破位+追踪1.5", "breakout", "trail_1.5", False),
    ("A+B2", "A+B 破位+追踪2.5", "breakout", "trail_2.5", False),
    ("A+C", "A+C 破位+高质量", "breakout", "measured", True),
    ("C+B1", "C+B 高质量+追踪1.5", "retest", "trail_1.5", True),
    ("C+B2", "C+B 高质量+追踪2.5", "retest", "trail_2.5", True),
]

FONT = None
for p in (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
):
    if Path(p).exists():
        FONT = font_manager.FontProperties(fname=p)
        break


def quality_ok(high, atrv, L, Hh, R, y1, y2) -> tuple[bool, dict]:
    atr_r = float(atrv[R])
    atr_h = float(atrv[Hh])
    if not (np.isfinite(atr_r) and atr_r > 0 and np.isfinite(atr_h) and atr_h > 0):
        return False, {}
    neck_abs = abs(float(y2) - float(y1)) / atr_r
    head_rs = (float(high[Hh]) - float(high[R])) / atr_h
    sym = abs(float(high[R]) - float(high[L])) / atr_r
    ok = (neck_abs <= NECK_FLAT_ATR) and (head_rs >= HEAD_RS_ATR) and (sym <= SHOULDER_SYM_ATR)
    return ok, {"neck_abs_atr": neck_abs, "head_rs_atr": head_rs, "shoulder_sym_atr": sym}


def make_signal(symbol, df, L, Hh, R, t, entry_i, lt, y1, rt, y2, vol_ratio) -> Signal | None:
    high, opn = df["high"].to_numpy(), df["open"].to_numpy()
    atrv = df["atr"].to_numpy()
    times = df["dt"]
    height = float(high[Hh] - neck_at(lt, y1, rt, y2, Hh))
    if height <= 0:
        return None
    nk = float(neck_at(lt, y1, rt, y2, t))
    target = float(nk - height)
    target15 = float(nk - 1.5 * height)
    rs_high = float(high[R])
    entry_px = float(opn[entry_i])
    swing_hi = float(np.max(high[R:entry_i]))  # highs up to bar before entry
    near = min(rs_high, swing_hi)
    stop = min(near + 0.5 * float(atrv[t]), rs_high + 0.5 * float(atrv[t]))
    if not (stop > entry_px):
        return None
    rr = (entry_px - target) / (stop - entry_px) if (stop > entry_px) else 0.0
    return Signal(
        symbol=symbol, lam=vol_ratio, L=L, H=Hh, R=R, break_i=t,
        entry_time=_ts(times.iloc[entry_i]), entry_px=entry_px, stop=float(stop),
        target=target, target15=target15, neck=float(nk),
        head_high=float(high[Hh]), rs_high=rs_high, atr_1h=float(atrv[t]),
        height=float(height), left_high=float(high[L]), vol_ratio=vol_ratio, rr=float(rr),
    )


def try_retest_idx(df, L, Hh, R, t, lt, y1, rt, y2):
    """Return confirmation bar index j, or None. Same rules as v2_3lines."""
    high, low, opn, close = (df[c].to_numpy() for c in ("high", "low", "open", "close"))
    vol, atrv = df["volume"].to_numpy(), df["atr"].to_numpy()
    n = len(df)
    height = high[Hh] - neck_at(lt, y1, rt, y2, Hh)
    if height <= 0:
        return None
    nk0 = neck_at(lt, y1, rt, y2, t)
    target = float(nk0 - height)
    rs_high = float(high[R])
    break_vol = float(vol[t])
    end_r = min(n - 2, t + RETEST_BARS)
    for j in range(t + 1, end_r + 1):
        swing_hi = float(np.max(high[R:j + 1]))
        stop_j = min(min(rs_high, swing_hi) + 0.5 * float(atrv[t]), rs_high + 0.5 * float(atrv[t]))
        if high[j] >= stop_j or low[j] <= target:
            return None
        _, _, _, _, brk_j, resist_j, mid_j = levels_at(lt, y1, rt, y2, j, atrv[t])
        if close[j] > resist_j:
            return None
        if close[j] > opn[j] and vol[j] > break_vol:
            return None
        if high[j] > rs_high:
            return None
        rng = high[j] - low[j]
        upper_wick = high[j] - max(close[j], opn[j])
        wick_ok = rng > 0 and upper_wick / rng >= 0.30
        if (high[j] >= brk_j) and (close[j] < mid_j) and (close[j] < opn[j] or wick_ok):
            return j
    return None


def measured_ok(sig: Signal | None) -> bool:
    """Match v2.0 exactly: allow theoretically invalid (e.g. negative) targets
    if stop > entry > target and R:R >= 1.5. Do not silently drop them from baseline.
    """
    if sig is None:
        return False
    if not (sig.stop > sig.entry_px > sig.target):
        return False
    if sig.rr < RR_MIN:
        return False
    return True


def trail_ok(sig: Signal | None) -> bool:
    return sig is not None and sig.stop > sig.entry_px


def scan_line(symbol, df, bar_delta, df_htf, htf_delta):
    """One record per accepted head.

    - sig_breakout: first valid breakdown (rule A).
    - sig_retest_meas: first retest that also passes v2.0 geom + R:R>=1.5
      (locks the head, identical to hs_backtest_v2_3lines).
    - sig_retest_trail: first retest whose structure stop is above entry
      (does not require measured R:R; used only by trailing-exit variants).
    """
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

    recs = []
    used_break: set[int] = set()
    used_head: set[int] = set()

    for ih, Hh in enumerate(peaks):
        Hh = int(Hh)
        if Hh in used_head:
            continue
        lefts = [int(p) for p in peaks[:ih] if MIN_SPAN // 4 <= (Hh - int(p)) <= MAX_SPAN]
        rights = [int(p) for p in peaks[ih + 1:] if (int(p) - Hh) <= MAX_SPAN]
        if not lefts or not rights:
            continue
        found_for_head = False
        first_bo = None
        q_bo = False
        trail_rt = None
        q_trail = False
        meas_rt = None
        q_meas = False
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
                if float(np.max(high[L:R + 1])) > high[Hh] + 1e-15:
                    continue
                excess = high[Hh] - max(high[L], high[R])
                if excess < max(0.5 * atrv[Hh], 0.003 * close[Hh]):
                    continue
                if high[R] - high[L] > 1.5 * atrv[R]:
                    continue
                lt = L + int(np.argmin(low[L:Hh + 1]))
                rt = Hh + int(np.argmin(low[Hh:R + 1]))
                if rt - lt < MIN_TROUGH_GAP:
                    continue
                y1, y2 = float(low[lt]), float(low[rt])
                vl, vh, vr = vol_part(vol, L), vol_part(vol, Hh), vol_part(vol, R)
                if not (vr > 0 and vr <= vh and vr <= vl):
                    continue
                if not macd_rs_ok(ml, ms, mh, high, L, Hh, R):
                    continue
                q_ok, qmet = quality_ok(high, atrv, L, Hh, R, y1, y2)

                max_t = min(n - 2, R + BREAK_LOOKAHEAD)
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
                        continue

                    sig_bo = make_signal(symbol, df, L, Hh, R, t, t + 1, lt, y1, rt, y2, vol_ratio)
                    if first_bo is None and sig_bo is not None:
                        first_bo = sig_bo
                        q_bo = q_ok
                    j = try_retest_idx(df, L, Hh, R, t, lt, y1, rt, y2)
                    if j is None:
                        continue
                    sig_rt = make_signal(symbol, df, L, Hh, R, t, j + 1, lt, y1, rt, y2, vol_ratio)
                    if sig_rt is None:
                        continue
                    if trail_rt is None and trail_ok(sig_rt):
                        trail_rt = sig_rt
                        q_trail = q_ok
                    if measured_ok(sig_rt):
                        meas_rt = sig_rt
                        q_meas = q_ok
                        used_break.add(t)
                        used_head.add(Hh)
                        found_for_head = True
                        break
                    # retest printed but failed R:R / geom — like v2, keep searching later breaks
        if first_bo is not None or trail_rt is not None or meas_rt is not None:
            recs.append({
                "symbol": symbol,
                "sig_breakout": first_bo,
                "q_breakout": q_bo,
                "sig_retest_trail": trail_rt,
                "q_retest_trail": q_trail,
                "sig_retest_meas": meas_rt,
                "q_retest_meas": q_meas,
            })
    return recs


def simulate_trail(sig: Signal, df15: pd.DataFrame, k: float) -> Trade | None:
    """Pure ATR trailing stop for a short. No measured targets.
    Trail uses prior bars' running low only (no intra-bar tighten-then-hit).
    """
    entry_ts = _ts(sig.entry_time)
    i0 = int(df15["dt"].searchsorted(entry_ts, side="left"))
    if i0 >= len(df15) - 1:
        return None
    slip = slippage_for(sig.symbol)
    fee = TAKER_FEE
    entry = sig.entry_px * (1 - slip)
    stop = float(sig.stop)
    stop_pct = (stop - entry) / entry
    if stop_pct <= 0:
        return None
    h = df15["high"].to_numpy()
    l = df15["low"].to_numpy()
    o = df15["open"].to_numpy()
    atr_v = float(sig.atr_1h)
    n = len(df15)
    hi = h[i0:n]
    lo = l[i0:n]
    op = o[i0:n]
    if len(hi) == 0:
        return None
    cummin = np.minimum.accumulate(lo)
    prev = np.empty_like(cummin)
    prev[0] = np.inf
    prev[1:] = cummin[:-1]
    raw_trail = prev + k * atr_v
    trail = np.where(np.isfinite(prev), np.minimum(stop, raw_trail), stop)
    hit = hi >= trail
    if hit.any():
        j = int(np.argmax(hit))  # first True
        reason = "trail_stop" if trail[j] < stop - 1e-12 else "stop"
        fill_px = max(float(op[j]), float(trail[j])) * (1 + slip)
        exit_i = i0 + j
    else:
        j = len(hi) - 1
        reason = "eod"
        fill_px = float(df15["close"].iloc[-1]) * (1 - slip)
        exit_i = n - 1
    raw = (entry - fill_px) / entry
    net = raw - 2 * fee
    R = stop_pct
    pnl_R = net / R if R > 0 else 0.0
    dummy_tgt = float(sig.target) if np.isfinite(sig.target) else entry
    return Trade(
        symbol=sig.symbol, lam=sig.lam, exit_mode=f"trail_atr_{k}",
        entry_time=sig.entry_time,
        exit_time=pd.Timestamp(df15["dt"].iloc[min(exit_i, n - 1)]),
        entry=entry, stop=stop, target=dummy_tgt, exit_px=fill_px,
        pnl_R=float(pnl_R), pnl_pct_on_notional=float(net), hit=reason,
        bars_held_15m=int(max(1, exit_i - i0 + 1)),
        max_lev=max_safe_leverage(sig.symbol, stop_pct),
        stop_pct=float(stop_pct), rr_planned=float(sig.rr), vol_ratio=float(sig.vol_ratio),
    )


def max_lose_streak(trades) -> int:
    s = ms = 0
    for td in sorted(trades, key=lambda x: _ts(x.entry_time)):
        if td.pnl_R < 0:
            s += 1
            ms = max(ms, s)
        else:
            s = 0
    return ms


def pack_metrics(trades, prefix: str) -> dict:
    st = trade_stats(trades)
    p = run_portfolio(trades, 0.01)
    return {
        f"{prefix}n": st["n"],
        f"{prefix}win_rate": st["win_rate"],
        f"{prefix}avg_R": st["avg_R"],
        f"{prefix}med_R": st["med_R"],
        f"{prefix}pf": st["profit_factor"],
        f"{prefix}avg_hold_h": st["avg_hold_h"],
        f"{prefix}max_lose_streak": max_lose_streak(trades),
        f"{prefix}n_taken": p["n_taken"],
        f"{prefix}skipped": p["n_skipped_concurrent"],
        f"{prefix}final": p["final"],
        f"{prefix}dd": p["max_dd"],
        f"{prefix}ret": p["final"] / START_EQUITY - 1,
    }


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


def fmt_pct(x):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.1f}%"


def fmt_r(x):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.3f}"


def fmt_pf(x):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"


def fmt_n(x):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{int(x)}"


def write_report(df, splits, meta, png_path, out_md):
    lines = []
    a = lines.append
    a("# 头肩顶做空 v2.0：三周期 × 三条改进（及预先登记的两两组合）")
    a("")
    a("日期：2026-11-10（Asia/Shanghai）")
    a("脚本：`hs_opt_strategy.py`")
    a("规则底本：`hs_short_model.md` v2.0。日线 / 1H / 15m **各自画形态、各自回踩或破位、各自看上一级「明显偏多」**，不把别的周期混进来过滤。")
    a("数据：与三线回测相同的本地 15m（重采样 1H / 4H / 日 / 周），未重下。费用开平各 0.05%；滑点 BTC/ETH 2bps、其他 5bps。")
    a("")
    a("---")
    a("")
    a("## 0. 事先锁定（看结果之前写死）")
    a("")
    a("### 基线（每个周期一套对照）")
    a("")
    a("- 回踩颈线入场（最多等 12 根形态周期）")
    a("- λ = 1.5")
    a("- 离场：60% @ 1.0H，余仓 1.5H")
    a("- 止损：更近摆动高 + 0.5ATR，右肩 + 0.5ATR 为上限")
    a("- 上一级明显偏多禁空（日线看周线，1H 看 4H，15m 看 1H）：EMA20>EMA60 且收盘>EMA20；数据不足视为禁空")
    a("- 量度离场仍要求 `止损 > 入场 > 目标`、目标 > 0、相对 1.0H 的 R:R ≥ 1.5（与 v2.0 一致）")
    a("")
    a("### 改进定义")
    a("")
    a("| 代号 | 改什么 | 锁定细则 |")
    a("|---|---|---|")
    a("| A | 入场改为破位即入 | 有效跌破确认棒收盘后，下一根开盘做空，不等回踩。量度离场时仍走 R:R≥1.5 与目标>0；追踪离场时只要求结构止损高于入场价 |")
    a("| B1 / B2 | 止盈改为纯 ATR 追踪 | 初始结构止损不变；之后追踪价 = 已走完棒的累计最低价 + k×形态周期 ATR，只向下移。k∈{1.5, 2.5}。不再使用 1.0H/1.5H 量度目标（从而绕开日线目标算到负值）。**不**用量度 R:R 过滤入场 |")
    a("| C | 只做高质量形态 | 同时满足：(1) 颈线更平 `|右谷−左谷| / ATR[R] ≤ 0.75`；(2) 右肩明显低于头 `(头高−右肩高) / ATR[H] ≥ 1.0`；(3) 两肩对称 `|右肩高−左肩高| / ATR[R] ≤ 0.75`。不叠其它指标 |")
    a("")
    a("单改进：A、B1、B2、C 各自相对基线单独切换。")
    a("两两组合（预先全部登记，不是看完单改进再挑）：A+B1、A+B2、A+C、C+B1、C+B2。不做三改进叠满。")
    a("")
    a("### 时间分割与推广门槛")
    a("")
    a("每个周期用该周期公共 K 线的日历跨度，前 70% 为 IS、后 30% 为 OOS；一笔交易看出场无所谓，只看入场时间。")
    a("")
    for tf, sp in splits.items():
        a(f"- **{tf}** 区间 {sp['tmin']} → {sp['tmax']} UTC；分割 {sp['split']} UTC（北京 {sp['split_sh']}）")
    a("")
    a("只有同时满足才称该改进在该周期有效：")
    a("")
    a("1. OOS 笔数 ≥ 8（日线）/ 40（1H、15m）")
    a("2. OOS 平均 R > 0")
    a("3. OOS 平均 R ≥ 同一周期基线 OOS 平均 R + 0.05")
    a("")
    a("选参叙事只对照 IS 排名与 OOS 是否跟上。不按品种、不按年份切片。")
    a("")
    a("两类数字分开：表中平均 R / 胜率 / PF / 中位 R / 最长连亏是**全部成交信号**；组合期末、回撤是 1% 风险、同时最多 2 笔（IS 与 OOS 各自从 10,000 起算）。")
    a("")

    tf_names = {"1D": "日线主线（周线过滤）", "1H": "1 小时主线（4H 过滤）", "15m": "15 分钟主线（1H 过滤）"}
    promoted_all = []

    def row_md(r):
        return (
            f"| {r.label} | {fmt_n(r.is_n)} | {fmt_pct(r.is_win_rate)} | {fmt_r(r.is_avg_R)} | "
            f"{r.is_med_R:.3f} | {fmt_pf(r.is_pf)} | {fmt_n(r.is_max_lose_streak)} | "
            f"{100 * r.is_dd:.1f}% | {r.is_final:.0f} | "
            f"{fmt_n(r.oos_n)} | {fmt_pct(r.oos_win_rate)} | {fmt_r(r.oos_avg_R)} | "
            f"{r.oos_med_R:.3f} | {fmt_pf(r.oos_pf)} | {fmt_n(r.oos_max_lose_streak)} | "
            f"{100 * r.oos_dd:.1f}% | {r.oos_final:.0f} |"
        )

    header = (
        "| 组合 | IS笔 | IS胜率 | IS平均R | IS中位R | IS PF | IS连亏 | IS回撤 | IS期末 | "
        "OOS笔 | OOS胜率 | OOS平均R | OOS中位R | OOS PF | OOS连亏 | OOS回撤 | OOS期末 |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"

    sec = 1
    for tf in ("1D", "1H", "15m"):
        sub = df[df.tf == tf].copy()
        base = sub[sub.vid == "baseline"].iloc[0]
        min_n = OOS_MIN_N[tf]
        a(f"## {sec}. {tf_names[tf]}")
        a("")
        a(
            f"基线全样本：n={int(base.n)}，胜率 {fmt_pct(base.win_rate)}，平均 R {fmt_r(base.avg_R)}，"
            f"中位 R {base.med_R:.3f}，PF {fmt_pf(base.pf)}，最长连亏 {int(base.max_lose_streak)}，"
            f"组合期末 {base.final:.0f}（{100 * base.ret:.1f}%），回撤 {100 * base.dd:.1f}%。"
        )
        a("")
        a(header)
        a(sep)
        # stable order = VARIANTS order
        order = {v[0]: i for i, v in enumerate(VARIANTS)}
        sub = sub.assign(_o=sub.vid.map(order)).sort_values("_o")
        for _, r in sub.iterrows():
            mark = " ←基线" if r.vid == "baseline" else ""
            a(row_md(r)[:-1] + mark + " |")
        a("")

        # promotion
        oos_thr = base.oos_avg_R + OOS_MIN_EDGE
        a(
            f"该周期门槛：OOS n≥{min_n}，OOS 平均 R>0，且 ≥ 基线 OOS {fmt_r(base.oos_avg_R)} + 0.05 = {fmt_r(oos_thr)}。"
        )
        a("")
        hits = []
        notes = []
        for _, r in sub.iterrows():
            if r.vid == "baseline":
                continue
            reasons = []
            n_ok = (r.oos_n >= min_n) if np.isfinite(r.oos_n) else False
            pos = (r.oos_avg_R > 0) if np.isfinite(r.oos_avg_R) else False
            edge = (r.oos_avg_R >= oos_thr) if np.isfinite(r.oos_avg_R) else False
            if n_ok and pos and edge:
                hits.append(r)
                promoted_all.append(r)
                notes.append(
                    f"- **{r.label}**：跨过门槛。OOS n={int(r.oos_n)}，平均 R {fmt_r(r.oos_avg_R)}，"
                    f"PF {fmt_pf(r.oos_pf)}，回撤 {100 * r.oos_dd:.1f}%。"
                    + (
                        " IS 也同步为正。"
                        if np.isfinite(r.is_avg_R) and r.is_avg_R > 0
                        else " 注意：样本内平均 R 并未同步更好，推广仍以 OOS 为准。"
                    )
                )
            else:
                why = []
                if not n_ok:
                    why.append(f"OOS 笔数 {fmt_n(r.oos_n)}<{min_n}")
                if np.isfinite(r.oos_avg_R):
                    if not pos:
                        why.append(f"OOS 平均 R {fmt_r(r.oos_avg_R)}≤0")
                    elif not edge:
                        why.append(f"相对基线优势只有 {r.oos_avg_R - base.oos_avg_R:+.3f}R，不足 +0.05")
                else:
                    why.append("OOS 无成交")
                ofit = ""
                if np.isfinite(r.is_avg_R) and np.isfinite(r.oos_avg_R):
                    if r.is_avg_R > (base.is_avg_R if np.isfinite(base.is_avg_R) else 0) and r.oos_avg_R < base.oos_avg_R:
                        ofit = " **IS 变好、OOS 变差，判为过拟合倾向，不能推广。**"
                    elif r.is_avg_R > 0 and r.oos_avg_R <= 0:
                        ofit = " IS 为正而 OOS 不稳/为负，不能把 IS 当成能赚钱。"
                notes.append(f"- {r.label}：未达标（{'；'.join(why)}）。{ofit}")
        if hits:
            a("跨过门槛的组合：")
        else:
            a("**本周期没有组合跨过推广门槛。**")
        a("")
        for x in notes:
            a(x)
        a("")
        sec += 1

    a(f"## {sec}. 跨周期总表")
    a("")
    a("下表统计每个非基线组合在几个周期上：OOS 平均 R>0、以及是否跨过该周期完整门槛。")
    a("")
    a("| 组合 | 日线 OOS 平均R (n) | 1H OOS 平均R (n) | 15m OOS 平均R (n) | OOS 为正的周期数 | 跨过完整门槛的周期数 |")
    a("|---|---:|---:|---:|---:|---:|")
    for vid, label, *_ in VARIANTS:
        if vid == "baseline":
            continue
        cells = []
        n_pos = 0
        n_pass = 0
        for tf in ("1D", "1H", "15m"):
            r = df[(df.tf == tf) & (df.vid == vid)].iloc[0]
            base = df[(df.tf == tf) & (df.vid == "baseline")].iloc[0]
            cells.append(f"{fmt_r(r.oos_avg_R)} ({fmt_n(r.oos_n)})")
            if np.isfinite(r.oos_avg_R) and r.oos_avg_R > 0:
                n_pos += 1
            if (
                np.isfinite(r.oos_n) and r.oos_n >= OOS_MIN_N[tf]
                and np.isfinite(r.oos_avg_R) and r.oos_avg_R > 0
                and r.oos_avg_R >= base.oos_avg_R + OOS_MIN_EDGE
            ):
                n_pass += 1
        a(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {n_pos} | {n_pass} |")
    a("")

    dual = [vid for vid, *_ in VARIANTS if vid != "baseline"]
    dual_pass = []
    for vid, label, *_ in VARIANTS:
        if vid == "baseline":
            continue
        n_pass = 0
        for tf in ("1D", "1H", "15m"):
            r = df[(df.tf == tf) & (df.vid == vid)].iloc[0]
            base = df[(df.tf == tf) & (df.vid == "baseline")].iloc[0]
            if (
                np.isfinite(r.oos_n) and r.oos_n >= OOS_MIN_N[tf]
                and np.isfinite(r.oos_avg_R) and r.oos_avg_R > 0
                and r.oos_avg_R >= base.oos_avg_R + OOS_MIN_EDGE
            ):
                n_pass += 1
        if n_pass >= 2:
            dual_pass.append((label, n_pass))
    if dual_pass:
        a("在**至少两个周期**上跨过完整门槛的组合：")
        for label, n_pass in dual_pass:
            a(f"- {label}（{n_pass} 个周期）")
    else:
        a("**没有任何一种入场+止盈+形态组合在至少两个周期上同时跨过 OOS 稳健门槛。** 因此目前找不到「三个周期都能用、且样本外稳健为正」的统一做空框架。")
    a("")
    a(f"权益图：`{png_path}`（每个周期一个面板：基线 vs 该周期最佳候选；竖线为该周期 IS/OOS 分割。若无人达标，最佳候选按 IS 平均 R 最高的非基线组合绘制，**不等于可推广**。）")
    a("")
    a(f"## {sec + 1}. 结论")
    a("")
    a("逐周期看「有没有人跨过门槛」，而不是为了统一去编一个三周期通用赢家：")
    a("")
    for tf in ("1D", "1H", "15m"):
        sub = df[df.tf == tf]
        names = [r.label for _, r in sub.iterrows() if r.vid != "baseline" and any(
            (p.tf == tf and p.vid == r.vid) for p in promoted_all
        )]
        base = sub[sub.vid == "baseline"].iloc[0]
        if names:
            a(f"- **{tf_names[tf]}**：{ '、'.join(names) } 跨过门槛。基线 OOS {fmt_r(base.oos_avg_R)}（n={fmt_n(base.oos_n)}）。")
        else:
            extra = ""
            if tf == "1D" and (not np.isfinite(base.oos_n) or base.oos_n < OOS_MIN_N["1D"]):
                extra = " 日线基线本身成交过少，样本外几乎没有评价效力。"
            a(f"- **{tf_names[tf]}**：无人达标。基线 OOS {fmt_r(base.oos_avg_R)}（n={fmt_n(base.oos_n)}）。{extra}")
    a("")
    a("建议：")
    a("")
    a("- 不要把三条独立主线硬捏成同一套参数，除非出现至少两个周期同时达标的组合。")
    a("- 不要把 IS 第一名写进主模型。")
    a("- 主模型默认仍维持 v2.0：回踩、λ=1.5、量度 60%/1.5H、上一级明显偏多禁空。若某周期有达标组合，只作为该周期的待验证开关，另开一段更新样本外再确认。")
    a("- 日线量度目标为负的问题：追踪止损在规则上绕开了它，但其 OOS 是否真的更好必须以表中数字为准，不能先验当成已修好。")
    a("")
    a("附件：")
    a("")
    a("- 完整数值：`/workspace/trading/backtest/output/grid_v2_strategy_opt.csv`")
    a("- 分割元数据：`/workspace/trading/backtest/output/opt_v2_strategy_meta.json`")
    a(f"- 权益图：`{png_path}`")
    a("")
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def main():
    uni = json.loads((DATA / "universe.json").read_text())
    symbols = [s[0] if isinstance(s, list) else s for s in uni["symbols"]]
    print("symbols", len(symbols), flush=True)

    lines_cfg = {
        "1D": dict(bar=pd.Timedelta(days=1), htf_delta=pd.Timedelta(days=7)),
        "1H": dict(bar=pd.Timedelta(hours=1), htf_delta=pd.Timedelta(hours=4)),
        "15m": dict(bar=pd.Timedelta(minutes=15), htf_delta=pd.Timedelta(hours=1)),
    }

    # cache[sym] = frames; recs[tf] = list of candidate recs
    frames15 = {}
    recs_by_tf = {k: [] for k in lines_cfg}
    span = {k: [None, None] for k in lines_cfg}

    for sym in symbols:
        pack = load_frames(sym)
        if pack is None:
            print(f"  skip {sym}", flush=True)
            continue
        df15, df1, df4, dfD, dfW = pack
        frames15[sym] = df15
        mapping = {"1D": (dfD, dfW), "1H": (df1, df4), "15m": (df15, df1)}
        msg = [f"{sym:16s}"]
        for name, (df_pat, df_htf) in mapping.items():
            if df_pat is None or len(df_pat) < 100:
                msg.append(f"{name} n/a")
                continue
            d0, d1 = _ts(df_pat["dt"].iloc[0]), _ts(df_pat["dt"].iloc[-1])
            if span[name][0] is None or d0 < span[name][0]:
                span[name][0] = d0
            if span[name][1] is None or d1 > span[name][1]:
                span[name][1] = d1
            cfg = lines_cfg[name]
            found = scan_line(sym, df_pat, cfg["bar"], df_htf, cfg["htf_delta"])
            recs_by_tf[name].extend(found)
            n_rt = sum(1 for r in found if r["sig_retest_meas"] is not None)
            n_bo = sum(1 for r in found if r["sig_breakout"] is not None)
            n_q = sum(1 for r in found if r["q_retest_meas"])
            msg.append(f"{name} rec={len(found):4d} rt={n_rt:3d} bo1={n_bo:3d} q={n_q:3d}")
        print("  " + " | ".join(msg), flush=True)

    splits = {}
    for name in lines_cfg:
        tmin, tmax = span[name]
        split = tmin + 0.7 * (tmax - tmin)
        splits[name] = {
            "tmin": str(tmin), "tmax": str(tmax), "split": str(split),
            "split_ts": split,
            "split_sh": _ts(split).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M"),
        }
        print(f"split {name}: {tmin} -> {tmax}  70% {split}  SH {splits[name]['split_sh']}", flush=True)

    # Simulate unique (tf, entry, exit) trade lists, tagged with quality
    # keys: (tf, entry_mode, exit_mode) -> list[(Trade, quality)]
    sim_cache = {}

    def collect_sigs(tf, entry_mode, exit_mode):
        out = []
        for rec in recs_by_tf[tf]:
            if entry_mode == "retest":
                if exit_mode == "measured":
                    sig, q = rec["sig_retest_meas"], rec["q_retest_meas"]
                else:
                    sig, q = rec["sig_retest_trail"], rec["q_retest_trail"]
            else:
                sig, q = rec["sig_breakout"], rec["q_breakout"]
            if sig is None:
                continue
            out.append((sig, q))
        return out

    needed = set((e, x) for _, _, e, x, _ in VARIANTS)
    for tf in lines_cfg:
        for entry_mode, exit_mode in needed:
            sigs = collect_sigs(tf, entry_mode, exit_mode)
            tagged = []
            for sig, q in sigs:
                if exit_mode == "measured":
                    if not measured_ok(sig):
                        continue
                    tr = simulate_trade(sig, frames15[sig.symbol], MEASURED_EXIT)
                else:
                    if not trail_ok(sig):
                        continue
                    k = 1.5 if exit_mode == "trail_1.5" else 2.5
                    tr = simulate_trail(sig, frames15[sig.symbol], k)
                if tr is not None:
                    tagged.append((tr, q))
            sim_cache[(tf, entry_mode, exit_mode)] = tagged
            print(
                f"  sim {tf:4s} {entry_mode:8s} {exit_mode:10s} "
                f"sigs={len(sigs):5d} trades={len(tagged):5d} q={sum(q for _, q in tagged):5d}",
                flush=True,
            )

    # Assemble variants
    raw = {}  # (tf, vid) -> list[Trade]
    rows = []
    for tf in lines_cfg:
        split = splits[tf]["split_ts"]
        for vid, label, entry_mode, exit_mode, want_q in VARIANTS:
            tagged = sim_cache[(tf, entry_mode, exit_mode)]
            trs = [t for t, q in tagged if (q if want_q else True)]
            raw[(tf, vid)] = trs
            is_tr = [t for t in trs if _ts(t.entry_time) < split]
            oos_tr = [t for t in trs if _ts(t.entry_time) >= split]
            rec = {"tf": tf, "vid": vid, "label": label,
                   "entry": entry_mode, "exit": exit_mode, "quality": int(want_q)}
            rec.update(pack_metrics(trs, ""))
            rec.update(pack_metrics(is_tr, "is_"))
            rec.update(pack_metrics(oos_tr, "oos_"))
            rows.append(rec)
            print(
                f"  VAR {tf:4s} {vid:6s} n={rec['n']:4d} is={rec['is_n']:4d}/{rec['is_avg_R']} "
                f"oos={rec['oos_n']:4d}/{rec['oos_avg_R']}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "grid_v2_strategy_opt.csv"
    df.to_csv(csv_path, index=False)
    print("wrote", csv_path, flush=True)

    meta = {
        "quality": {
            "neck_flat_atr": NECK_FLAT_ATR,
            "head_rs_atr": HEAD_RS_ATR,
            "shoulder_sym_atr": SHOULDER_SYM_ATR,
        },
        "trail_ks": list(TRAIL_KS),
        "oos_min_n": OOS_MIN_N,
        "oos_min_edge": OOS_MIN_EDGE,
        "variants": [v[0] for v in VARIANTS],
        "splits": {k: {kk: vv for kk, vv in val.items() if kk != "split_ts"} for k, val in splits.items()},
        "retest_bars": RETEST_BARS,
        "lam": LAM,
        "note": "Thresholds and pairwise combos locked before viewing results.",
    }
    (OUT / "opt_v2_strategy_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # Equity figure: 3 panels, baseline vs best candidate
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False, dpi=120)
    titles = {"1D": "日线", "1H": "1 小时", "15m": "15 分钟"}
    png = Path("/workspace/trading/backtest/equity_v2_strategy_opt.png")
    for ax, tf in zip(axes, ("1D", "1H", "15m")):
        sub = df[df.tf == tf]
        base_row = sub[sub.vid == "baseline"].iloc[0]
        min_n = OOS_MIN_N[tf]
        oos_thr = base_row.oos_avg_R + OOS_MIN_EDGE
        promo = sub[
            (sub.vid != "baseline")
            & (sub.oos_n >= min_n)
            & (sub.oos_avg_R > 0)
            & (sub.oos_avg_R >= oos_thr)
        ]
        if len(promo):
            best = promo.sort_values("oos_avg_R", ascending=False).iloc[0]
            best_tag = "OOS达标"
        else:
            cand = sub[sub.vid != "baseline"].sort_values("is_avg_R", ascending=False)
            best = cand.iloc[0]
            best_tag = "IS最优（未推广）"
        split = splits[tf]["split_ts"]
        plot_items = [
            (base_row.label, raw[(tf, "baseline")], "#1d3557"),
            (f"{best_tag}：{best.label}", raw[(tf, best.vid)], "#e76f51"),
        ]
        for label, trs, color in plot_items:
            p = run_portfolio(trs, 0.01)
            eq = p["equity_curve"]
            if eq is None or len(eq) == 0:
                continue
            ax.plot(pd.DatetimeIndex(eq.index), eq.values, label=label, color=color, lw=1.5)
        ax.axvline(split, color="#333", ls="--", lw=1.0,
                   label=f"分割 {splits[tf]['split_sh']}")
        ax.axhline(START_EQUITY, color="#999", lw=0.8, ls=":")
        ax.set_title(titles[tf], fontproperties=FONT, fontsize=12)
        ax.set_ylabel("权益（1%风险）", fontproperties=FONT)
        ax.legend(prop=FONT, fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("时间（UTC）", fontproperties=FONT)
    fig.suptitle("头肩顶做空：三周期基线 vs 最佳改进候选（竖线=IS/OOS）", fontproperties=FONT, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(png)
    print("wrote", png, png.stat().st_size, flush=True)

    md = Path("/workspace/trading/backtest/results_v2_strategy_opt.md")
    write_report(df, splits, meta, str(png), md)
    print("wrote", md, md.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
