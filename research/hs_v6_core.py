"""Shoulder picker v6 (research; locked hs_v6_meta.json). Price-only pivots, causal RS.
LS = most prominent (deepest independent pullback) swing on the rising leg (as v3).
RS = highest bar of the rebound segment that starts at the first post-head pullback low.
Spans/amplitudes measured leg-to-leg (三栋 2026-11-14 LINK annotation)."""
from __future__ import annotations
import numpy as np
import hs_shoulder_v3_core as V

K = 1.5
MAX_RS_TO_BREAK = 24


def ls_start(high, low, L, A, thr):
    """Walk left from LS tracking the min low; stop at the first bar whose high is >=
    min + max(thr, 0.5*(high[L]-min)) (a counter-rally of half the LS rally). Bounded by A."""
    m = low[L]; mp = L
    for i in range(L - 1, max(A, 0) - 1, -1):
        if low[i] < m:
            m, mp = low[i], i
        elif high[i] - m >= max(thr, 0.5 * (high[L] - m)):
            break
    return mp


def pick_ls(high, low, sw, nge, H, u, k=K):
    A = V.leg_origin(high, low, H)
    best = None
    for p in range(max(V.SW_LEFT, A + 1), H - V.MIN_SEP + 1):
        if not sw[p] or high[p] >= high[H]:
            continue
        e = min(int(nge[p]), H)
        d = float(high[p] - low[p:e + 1].min())
        if d < k * u:
            continue
        if best is None or d > best[1] + 1e-15 or (abs(d - best[1]) <= 1e-15 and p > best[0]):
            best = (p, d)
    if best is None:
        return None
    L = best[0]
    if not V.cobbled_ok(high, low, L, H, u, k):
        return None
    return dict(L=L, A=A, p0=ls_start(high, low, L, A, k * u), lt=L + int(np.argmin(low[L:H + 1])))


class RightWalker:
    """Feed bars t = H+1, H+2, ...; rs() before update(t) gives the RS using bars < t only."""
    def __init__(self, high, low, H, u, k=K):
        self.h, self.l, self.H, self.thr = high, low, H, k * u
        self.m = np.inf; self.mp = None; self.rt0 = None; self.RS = None

    def rs(self):
        return self.RS

    def update(self, t):
        h, l = self.h, self.l
        if self.rt0 is None:
            # rebound is measured from the min of EARLIER bars only (a bar cannot be its own trough)
            if self.mp is not None and h[t] - self.m >= self.thr and h[self.H] - self.m >= self.thr:
                self.rt0 = self.mp; self.RS = t
            elif l[t] < self.m:
                self.m, self.mp = l[t], t
        elif h[t] >= h[self.RS]:
            self.RS = t


def props(high, low, P, H, RS, t):
    L, p0, lt = P["L"], P["p0"], P["lt"]
    rt = H + int(np.argmin(low[H:RS + 1]))
    spL, spH, spR = max(1, lt - p0), max(1, rt - lt), max(1, t - rt)
    aL = (2 * high[L] - low[p0] - low[lt]) / 2
    aH = (2 * high[H] - low[lt] - low[rt]) / 2
    aR = (2 * high[RS] - low[rt] - low[t]) / 2
    return dict(rt=rt, span_L=spL, span_H=spH, span_R=spR, tL=spL / spH, tR=spR / spH,
                aL=aL / aH if aH > 0 else np.nan, aR=aR / aH if aH > 0 else np.nan)


def prop_ok(m):
    return (0.25 <= m["tL"] <= 1.5 and 0.25 <= m["tR"] <= 1.5 and 0.30 <= m["aL"] <= 1.0 and 0.30 <= m["aR"] <= 1.0)
