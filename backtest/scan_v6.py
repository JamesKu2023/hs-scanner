"""v6 三栋认肩 (2026-11-14) — live detector + display scores. Notify only, never places orders.

Pivots (price only, research/hs_v6_core.py, locked research/hs_v6_meta.json):
  LS = most prominent swing high (deepest independent pullback) on the rising leg into the head;
  RS = highest bar of the post-head rebound segment (from the first post-head pullback low to the
       bar before the break); neckline = the two pullback lows (lt in [LS,H], rt in [H,RS]).
Break: a bearish candle that CLOSES below the sloped v6 neckline (no band).
       Retest limit = the neckline value at the break bar.
15m hard conditions = research V6B: span 12–80, trough gap >= 3, |RS-LS| <= 0.5*hgt,
       head excess >= 0.2*hgt, RSI<50 + MACD break + volume > 1.5x MA20 on the break bar,
       1H not clearly bullish, R:R >= 1.5 (else 已破位·等回踩 when R:R at the break level >= 1.5).
Proportion score and confirmation x/3 are display only (never filters).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / "research")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hs_backtest import macd_break_ok, neck_at  # noqa: E402
from hs_backtest_v2_3lines import MAX_SPAN, MIN_SPAN, MIN_TROUGH_GAP, _ts, band_width, htf_bullish_at  # noqa: E402
from hs_opt_strategy import LAM, make_signal, measured_ok  # noqa: E402
import hs_shoulder_v3_core as V  # noqa: E402
import hs_v6_core as C6  # noqa: E402

TIER_V6 = "v6"
V6_RISK_PCT = 0.5
V6_TIER_CN = {"15m": "15m·v6 三栋认肩·未验证", "1h": "1H·v6 三栋认肩·未验证", "4h": "4H·v6 三栋认肩·未验证"}
K = C6.K


def next_greater(high: np.ndarray) -> np.ndarray:
    n = len(high); out = np.full(n, n - 1, dtype=int); st: list[int] = []
    for i in range(n):
        while st and high[i] > high[st[-1]]:
            out[st.pop()] = i
        st.append(i)
    return out


def v6_break_level(lt, y1, rt, y2, t, atr_t) -> tuple[float, float]:
    """(sloped v6 neck at t, break level). 三栋 2026-11-14: a break is a CLOSE below the v6
    neckline itself (the line through the two pullback lows) -> break level = neck, no band;
    the retest limit is the neckline value at the break bar."""
    nk = float(neck_at(lt, y1, rt, y2, t))
    return nk, nk


def geo_head(high, low, L, H, R):
    """V6B price geometry. Returns (lt, y1, rt, y2, hgt) or None."""
    if not (MIN_SPAN <= R - L <= MAX_SPAN):
        return None
    if float(np.max(high[L:R + 1])) > high[H] + 1e-15:
        return None
    lt = L + int(np.argmin(low[L:H + 1])); rt = H + int(np.argmin(low[H:R + 1]))
    if rt - lt < MIN_TROUGH_GAP:
        return None
    y1, y2 = float(low[lt]), float(low[rt]); hgt = float(high[H] - neck_at(lt, y1, rt, y2, H))
    if hgt <= 0 or abs(high[R] - high[L]) > 0.5 * hgt or high[H] - max(high[L], high[R]) < 0.2 * hgt:
        return None
    return lt, y1, rt, y2, hgt


def _closeness(a, b) -> float:
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
        return 0.0
    return min(a, b) / max(a, b)


def prop_score(high, low, L, H, R, t, A=None) -> dict:
    """v6 proportion score 0–100 (display only). Spans/amplitudes leg to leg (hs_v6_core.props).
    4 equal parts: LS span vs RS span, mean shoulder span vs head span, LS amp vs RS amp,
    mean shoulder amplitude vs head (1.0 inside 0.5–0.9×head, linear falloff outside)."""
    try:
        if A is None:
            A = V.leg_origin(high, low, H)
        lt = L + int(np.argmin(low[L:H + 1]))
        p0 = C6.ls_start(high, low, L, A, 0.0)
        m = C6.props(high, low, dict(L=L, p0=p0, lt=lt), H, R, t)
        s1 = _closeness(m["span_L"], m["span_R"])
        s2 = _closeness((m["span_L"] + m["span_R"]) / 2, m["span_H"])
        s3 = _closeness(m["aL"], m["aR"])
        a = (m["aL"] + m["aR"]) / 2
        s4 = 1.0 if 0.5 <= a <= 0.9 else max(0.0, 1.0 - (0.5 - a) / 0.5 if a < 0.5 else 1.0 - (a - 0.9) / 0.9)
        score = int(round(100 * (s1 + s2 + s3 + s4) / 4))
        txt = (f"比例分 {score}/100（跨度 左{m['span_L']}·头{m['span_H']}·右{m['span_R']} 根，"
               f"左/头 {m['tL']:.2f}、右/头 {m['tR']:.2f}；幅度 左/头 {m['aL']:.2f}、右/头 {m['aR']:.2f}）")
        return dict(prop_score=score, prop_score_text=txt, v6_ls_start=int(p0),
                    v6_spans={"ls": int(m["span_L"]), "head": int(m["span_H"]), "rs": int(m["span_R"])},
                    v6_ratios={k: round(float(m[k]), 3) for k in ("tL", "tR", "aL", "aR")})
    except Exception:
        return dict(prop_score=None, prop_score_text="比例分 —", v6_spans=None, v6_ratios=None)


def confirm_count(conf: dict) -> dict:
    """确认项 x/3: RSI head divergence, MACD-hist head divergence, volume step-down LS>H>RS."""
    d = (conf or {}).get("div") or {}
    items = [("RSI背离", d.get("rsi")), ("MACD柱背离", d.get("hist")), ("量能递减", (conf or {}).get("vol_stepdown"))]
    n = sum(1 for _, v in items if v)
    txt = f"确认 {n}/3：" + " ".join(f"{k}{'√' if v else ('?' if v is None else '×')}" for k, v in items)
    return dict(confirm_n=n, confirm_total=3, confirm_x3_text=txt)


def decorate(row: dict, df: pd.DataFrame, t: int, tf: str, pivot_confirmation) -> dict:
    """Add v6 tier labels + proportion score + confirmation x/3. Never changes trade fields."""
    high = df["high"].to_numpy(dtype=float); low = df["low"].to_numpy(dtype=float)
    row["v6"] = True
    row["tier_cn"] = V6_TIER_CN.get(tf, f"{tf}·v6 三栋认肩·未验证")
    row["validated"] = False
    row.update(prop_score(high, low, int(row["L"]), int(row["H"]), int(row["R"]), int(t)))
    row.update(confirm_count(pivot_confirmation(df, int(row["L"]), int(row["H"]), int(row["R"]))))
    return row


def detect_15m(symbol: str, df: pd.DataFrame, df_htf: pd.DataFrame | None, bar=pd.Timedelta(minutes=15),
               htf_delta=pd.Timedelta(hours=1)) -> list[dict]:
    """Causal V6B scan. Per head: first qualifying break with R:R>=1.5 -> 'entry'; the first
    qualifying low-R:R break before it -> 'lowrr' (retest candidate). Break bar <= n-2 (next bar
    open = official entry, same convention as the old 15m rule)."""
    n = len(df)
    if n < 100:
        return []
    high, low, opn, close = (df[c].to_numpy(dtype=float) for c in ("high", "low", "open", "close"))
    vol = df["volume"].to_numpy(dtype=float); atrv = df["atr"].to_numpy(dtype=float)
    rsi_ = df["rsi"].to_numpy(dtype=float); vma = df["vol_ma20"].to_numpy(dtype=float)
    ml, ms, mh = (df[c].to_numpy(dtype=float) for c in ("macd", "macd_sig", "macd_hist"))
    times = df["dt"]
    sw = V.swing_highs(high); uu = V.unit_series(atrv); nge = next_greater(high)
    out: list[dict] = []
    for H in np.nonzero(sw)[0]:
        H = int(H); u = uu[H]
        if H < 20 or not (np.isfinite(u) and u > 0 and np.isfinite(atrv[H]) and atrv[H] > 0):
            continue
        if H + 4 > n - 2:
            continue
        P = C6.pick_ls(high, low, sw, nge, H, u, K)
        if P is None:
            continue
        L = P["L"]; W = C6.RightWalker(high, low, H, u, K); combo: dict = {}; lowrr = None
        for t in range(H + 1, min(n - 2, H + V.MAX_SIDE + C6.MAX_RS_TO_BREAK) + 1):
            if high[t] > high[H]:
                break
            R = W.rs(); W.update(t)
            if R is None or R >= t or t - R > C6.MAX_RS_TO_BREAK or R - H > V.MAX_SIDE:
                continue
            if not (V.SYM_LO <= (R - H) / (H - L) <= V.SYM_HI):
                continue
            if R not in combo:
                combo[R] = geo_head(high, low, L, H, R)
            g = combo[R]
            if g is None:
                continue
            lt, y1, rt, y2, hgt = g
            if not (np.isfinite(vma[t]) and np.isfinite(rsi_[t]) and np.isfinite(atrv[t])):
                continue
            nk, brk = v6_break_level(lt, y1, rt, y2, t, atrv[t])
            if not (close[t] < opn[t] and close[t] < brk):
                continue
            if rsi_[t] >= 50 or not macd_break_ok(ml, ms, mh, t):
                continue
            vr = float(vol[t] / vma[t]) if vma[t] > 0 else 0.0
            if vr <= LAM:
                continue
            if htf_bullish_at(df_htf, _ts(times.iloc[t]) + bar, htf_delta):
                continue
            sig = make_signal(symbol, df, L, H, R, t, t + 1, lt, y1, rt, y2, vr)
            if sig is None:
                continue
            ev = dict(sig=sig, L=L, H=H, R=R, t=t, lt=lt, y1=y1, rt=rt, y2=y2, neck=nk, brk=brk, hgt=hgt)
            if measured_ok(sig):
                out.append(dict(ev, kind="entry"))
                break
            if lowrr is None:
                lowrr = dict(ev, kind="lowrr")
                out.append(lowrr)
    return out
