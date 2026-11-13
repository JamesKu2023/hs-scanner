"""Shoulder picker v3 (research only): PRICE-ONLY pivot selection.

Indicators are never used to choose LS/H/RS.
- swing high p: high[p] >= max(high[p-3:p]) and high[p] > max(high[p+1:p+3]) (confirmed 2 bars later).
- unit u = median ATR14 over the 96 bars ending at H (head-bar ATR is inflated by the spike).
- LS depth d(p): high[p] - min(low[p..e]) where e = first bar after p with high > high[p] (<= H).
  i.e. the independent pullback after the LS (its low is the left neckline point).
- RS rebound r(p): high[p] - min(low[H..p]) (rally from the right trough into the RS).
- LS = argmax depth among left swings with depth >= k*u, 4 <= H-p <= 80, high < head, time symmetry
  (RS-H)/(H-LS) in [1/3, 3]; RS = argmax rebound among confirmed right swings with rebound >= k*u.
  Prominence, not 'side absolute highest'.
- leg / cobbled filter (三栋 ARIA case): LS must sit on the rising leg into the head. Leg origin
  A = lowest low in the 80 bars before H; LS candidates only after A; rise A->LS >= k*u and no
  high >= LS between A and LS.
"""
from __future__ import annotations
import numpy as np, pandas as pd

SW_LEFT, SW_RIGHT = 3, 2
MIN_SEP = 4
MAX_SIDE = 80
SYM_LO, SYM_HI = 1 / 3, 3.0


def swing_highs(high: np.ndarray) -> np.ndarray:
    n = len(high); out = np.zeros(n, bool)
    for p in range(SW_LEFT, n - SW_RIGHT):
        if high[p] >= high[p - SW_LEFT:p].max() and high[p] > high[p + 1:p + 1 + SW_RIGHT].max():
            out[p] = True
    return out


def unit_series(atr: np.ndarray) -> np.ndarray:
    return pd.Series(atr).rolling(96, min_periods=20).median().to_numpy()


def ls_depth(high, low, p, H):
    hp = high[p]; e = H
    for j in range(p + 1, H + 1):
        if high[j] > hp:
            e = j; break
    return float(hp - low[p:e + 1].min())


def rs_rebound(high, low, H, p):
    return float(high[p] - low[H:p + 1].min())


def cobbled_ok(high, low, L, H, u, k):
    """LS must sit on a rising leg: leg origin A before LS, rise A->LS >= k*u, no high >= LS in [A, LS)."""
    A = leg_origin(high, low, H)
    if not (A < L):
        return False
    if L > A + 1 and high[A + 1:L].max() >= high[L]:
        return False
    return (high[L] - low[A]) >= k * u


def pick_rs(high, low, sw, H, t, u, k, confirm=SW_RIGHT):
    best = None
    for p in range(H + 1, t - confirm + 1):
        if not sw[p] or high[p] >= high[H] or p - H > MAX_SIDE:
            continue
        r = rs_rebound(high, low, H, p)
        if r < k * u or high[H] - low[H:p + 1].min() < k * u:
            continue
        if best is None or r > best[1] + 1e-15 or (abs(r - best[1]) <= 1e-15 and p > best[0]):
            best = (p, r)
    return best


def leg_origin(high, low, H):
    """Start of the rising leg into the head: lowest low in the MAX_SIDE bars before H,
    but only after the last bar whose high exceeded the head (the leg cannot cross a higher high)."""
    a = max(0, H - MAX_SIDE)
    above = np.nonzero(high[a:H] > high[H])[0]
    if len(above):
        a = a + int(above[-1]) + 1
    return a + int(np.argmin(low[a:H + 1]))


def pick_ls(high, low, sw, H, R, u, k, leg=True):
    """leg=True: candidates only after the leg origin (LS must belong to the rising leg into H)."""
    best = None
    lo_p = max(SW_LEFT, H - MAX_SIDE)
    if leg:
        lo_p = max(lo_p, leg_origin(high, low, H) + 1)
    for p in range(lo_p, H - MIN_SEP + 1):
        if not sw[p] or high[p] >= high[H]:
            continue
        if high[p + 1:H].max(initial=-np.inf) > high[H]:
            continue
        if R is not None:
            s = (R - H) / (H - p)
            if not (SYM_LO <= s <= SYM_HI):
                continue
        d = ls_depth(high, low, p, H)
        if d < k * u:
            continue
        if best is None or d > best[1] + 1e-15 or (abs(d - best[1]) <= 1e-15 and p > best[0]):
            best = (p, d)
    return best
