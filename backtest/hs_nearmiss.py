#!/usr/bin/env python3
"""15m H&S『观察』档（near-miss）检测器。定义锁定于
/workspace/trading/research/hs_nearmiss_meta.json（看结果之前写死）。

硬条件（几何、破位棒、λ=1.5、1H 偏多禁空、止损、R:R≥1.5）与正式规则完全相同；
5 个软条件中恰好差 1 条、且差得不多（每条一个容差）。不改正式规则，不下单。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from hs_backtest import macd_break_ok, macd_rs_ok, neck_at, vol_part
from hs_backtest_v2_3lines import (
    BREAK_LOOKAHEAD, MAX_SPAN, MIN_SPAN, MIN_TROUGH_GAP, PEAK_DISTANCE,
    _ts, htf_bullish_at, levels_at,
)
from hs_opt_strategy import (
    HEAD_RS_ATR, LAM, NECK_FLAT_ATR, SHOULDER_SYM_ATR,
    make_signal, measured_ok, scan_line,
)

# ---- locked tolerances (hs_nearmiss_meta.json) ----
NM_NECK_FLAT_MAX = 1.00     # formal 0.75
NM_HEAD_RS_MIN = 0.75       # formal 1.00
NM_SYM_MAX = 1.00           # formal 0.75
NM_RS_VOL_RATIO_MAX = 1.20  # formal vol_part(R)/min(H,L) <= 1.00
SUGGESTED_RISK_PCT = 0.5

COND_CN = {
    "neck_flat": "颈线平坦度",
    "head_rs": "头比右肩高度",
    "shoulder_sym": "两肩对称",
    "rs_volume": "右肩缩量",
    "rs_macd": "右肩 MACD 转弱",
}


def soft_eval(high, vol, ml, ms, mh, atrv, L, Hh, R, y1, y2) -> dict:
    """Return per-condition status: 'pass' | 'near' | 'fail' plus measured values."""
    atr_r, atr_h = float(atrv[R]), float(atrv[Hh])
    neck = abs(float(y2) - float(y1)) / atr_r
    head = (float(high[Hh]) - float(high[R])) / atr_h
    sym = abs(float(high[R]) - float(high[L])) / atr_r
    vl, vh, vr = vol_part(vol, L), vol_part(vol, Hh), vol_part(vol, R)
    vmin = min(vh, vl)
    vratio = (vr / vmin) if vmin > 0 else float("inf")
    macd_ok = bool(macd_rs_ok(ml, ms, mh, high, L, Hh, R))

    def band(ok, near):
        return "pass" if ok else ("near" if near else "fail")

    st = {
        "neck_flat": band(neck <= NECK_FLAT_ATR, neck <= NM_NECK_FLAT_MAX),
        "head_rs": band(head >= HEAD_RS_ATR, head >= NM_HEAD_RS_MIN),
        "shoulder_sym": band(sym <= SHOULDER_SYM_ATR, sym <= NM_SYM_MAX),
        "rs_volume": band(vr > 0 and vratio <= 1.0, vr > 0 and vratio <= NM_RS_VOL_RATIO_MAX),
        "rs_macd": band(macd_ok, True),
    }
    vals = {"neck_flat": neck, "head_rs": head, "shoulder_sym": sym,
            "rs_volume": vratio, "rs_macd": float(macd_ok)}
    return {"status": st, "values": vals}


def miss_text(cond: str, value: float) -> str:
    if cond == "neck_flat":
        return f"颈线两谷差 {value:.2f} ATR（正式要求 ≤ {NECK_FLAT_ATR:.2f}，超出 {value - NECK_FLAT_ATR:.2f} ATR）"
    if cond == "head_rs":
        return f"头比右肩高 {value:.2f} ATR（正式要求 ≥ {HEAD_RS_ATR:.2f}，差 {HEAD_RS_ATR - value:.2f} ATR）"
    if cond == "shoulder_sym":
        return f"两肩高度差 {value:.2f} ATR（正式要求 ≤ {SHOULDER_SYM_ATR:.2f}，超出 {value - SHOULDER_SYM_ATR:.2f} ATR）"
    if cond == "rs_volume":
        return f"右肩均量是头/左肩较小者的 {value:.2f} 倍（正式要求 ≤ 1.00，多 {100 * (value - 1):.0f}%）"
    if cond == "rs_macd":
        return "右肩没有 MACD 转弱（无死叉、无柱体连缩、无头对左肩背离）"
    return cond


def miss_amount(cond: str, value: float) -> float | None:
    if cond == "neck_flat":
        return value - NECK_FLAT_ATR
    if cond == "head_rs":
        return HEAD_RS_ATR - value
    if cond == "shoulder_sym":
        return value - SHOULDER_SYM_ATR
    if cond == "rs_volume":
        return value - 1.0
    return None


def formal_heads(recs) -> tuple[set[int], set]:
    heads, entries = set(), set()
    for rec in recs:
        s = rec.get("sig_breakout")
        if s is not None and rec.get("q_breakout") and measured_ok(s):
            heads.add(int(s.H))
            entries.add(_ts(s.entry_time))
    return heads, entries


def scan_nearmiss(symbol, df, bar_delta, df_htf, htf_delta, recs=None, rr_mode: str = "ok") -> list[dict]:
    """Return list of {'sig': Signal, 'miss': cond, 'value': float, 'miss_amount', 'miss_text'}.

    rr_mode: "ok" (default) keeps only measured_ok signals (live 观察);
             "low" keeps only breaks whose measured R:R at the next-bar open fails
             (live 已破位·等回踩 for the observe tier); "any" keeps both.
    """
    n = len(df)
    if n < 100:
        return []
    if recs is None:
        recs = scan_line(symbol, df, bar_delta, df_htf, htf_delta)
    f_heads, f_entries = formal_heads(recs)

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

    out = []
    for ih, Hh in enumerate(peaks):
        Hh = int(Hh)
        if Hh in f_heads:
            continue
        lefts = [int(p) for p in peaks[:ih] if MIN_SPAN // 4 <= (Hh - int(p)) <= MAX_SPAN]
        rights = [int(p) for p in peaks[ih + 1:] if (int(p) - Hh) <= MAX_SPAN]
        if not lefts or not rights:
            continue
        cand = None
        for L in lefts:
            if cand is not None:
                break
            for R in rights:
                if cand is not None:
                    break
                span = R - L
                if not (MIN_SPAN <= span <= MAX_SPAN):
                    continue
                if not (high[L] < high[Hh] and high[R] < high[Hh]):
                    continue
                if not np.isfinite(atrv[Hh]) or not np.isfinite(atrv[R]) or atrv[Hh] <= 0 or atrv[R] <= 0:
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
                ev = soft_eval(high, vol, ml, ms, mh, atrv, L, Hh, R, y1, y2)
                st = ev["status"]
                if any(v == "fail" for v in st.values()):
                    continue
                misses = [k for k, v in st.items() if v == "near"]
                if len(misses) > 1:
                    continue
                max_t = min(n - 2, R + BREAK_LOOKAHEAD)
                for t in range(R + 1, max_t + 1):
                    if not np.isfinite(vma[t]) or not np.isfinite(rsi_[t]) or not np.isfinite(atrv[t]):
                        continue
                    if high[t] > high[Hh]:
                        break
                    _, _, _, _, brk_lvl, _, _ = levels_at(lt, y1, rt, y2, t, atrv[t])
                    if not (close[t] < opn[t] and close[t] < brk_lvl):
                        continue
                    if rsi_[t] >= 50:
                        continue
                    if not macd_break_ok(ml, ms, mh, t):
                        continue
                    vol_ratio = float(vol[t] / vma[t]) if vma[t] > 0 else 0.0
                    if vol_ratio <= LAM:
                        continue
                    if high[Hh] - neck_at(lt, y1, rt, y2, Hh) <= 0:
                        continue
                    if htf_bullish_at(df_htf, _ts(times.iloc[t]) + bar_delta, htf_delta):
                        continue
                    sig = make_signal(symbol, df, L, Hh, R, t, t + 1, lt, y1, rt, y2, vol_ratio)
                    if sig is None:
                        continue
                    cand = (sig, misses, ev["values"], lt, y1, rt, y2)
                    break
        if cand is None:
            continue
        sig, misses, vals, lt, y1, rt, y2 = cand
        if len(misses) != 1:
            continue
        ok = measured_ok(sig)
        if rr_mode == "ok" and not ok:
            continue
        if rr_mode == "low" and ok:
            continue
        if _ts(sig.entry_time) in f_entries:
            continue
        c = misses[0]
        out.append({
            "sig": sig, "miss": c, "miss_cn": COND_CN[c], "value": float(vals[c]),
            "miss_amount": miss_amount(c, float(vals[c])), "miss_text": miss_text(c, float(vals[c])),
            "lt": lt, "y1": y1, "rt": rt, "y2": y2,
        })
    return out
