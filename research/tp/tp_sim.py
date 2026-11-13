"""Exit-management simulator for locked variants (research/hs_tp_mgmt_meta.json)."""
import sys
_here = __import__("pathlib").Path(__file__).resolve().parents[2]; sys.path.insert(0, str(_here / "backtest")); sys.path.insert(0, str(_here / "research"))
import numpy as np, pandas as pd
from hs_backtest import slippage_for, TAKER_FEE

FIBK = {"f1.0": 1.0, "f1.272": 1.272, "f1.618": 1.618, "f2.0": 2.0}
PLANS = {  # (fixed legs [(level, frac)], trail)  remaining after fixed legs goes to trail (if trail) 
    "BASE": ([("1.0H", .60), ("1.5H", .40)], False),
    "T1": ([("1.0H", .50)], True),
    "T2": ([("1.0H", .40), ("f1.618", .30)], True),
    "T3": ([("f1.272", .50), ("f1.618", .25)], True),
    "T4": ([("1.0H", .60)], True),
    "T5": ([("f1.0", .50), ("f1.618", .25), ("f2.0", .25)], False),
    "T6": ([("1.0H", .60), ("f1.618", .20)], True),
}


def level_px(name, sig, y2):
    if name == "1.0H": return sig.target
    if name == "1.5H": return sig.target15
    return sig.rs_high - FIBK[name] * (sig.head_high - y2)


def build_legs(plan, sig, y2, entry):
    fixed, trail = PLANS[plan]
    legs = [[level_px(n, sig, y2), f, n] for n, f in fixed]
    out, carry = [], 0.0
    for px, f, n in legs:  # invalid (>= entry) -> weight to next leg
        if px >= entry: carry += f; continue
        out.append([px, f + carry, n]); carry = 0.0
    if carry > 0:
        if trail: pass  # goes to runner automatically
        elif out: out[-1][1] += carry
    out.sort(key=lambda x: -x[0])
    return out, trail


def sim(sig, df, y2, plan, mid=None, i_end=None):
    t = pd.Timestamp(sig.entry_time)
    i0 = int(df["dt"].searchsorted(t, side="left"))
    if i0 >= len(df) - 1: return None
    slip, fee = slippage_for(sig.symbol), TAKER_FEE
    entry = sig.entry_px * (1 - slip); stop = sig.stop
    if (stop - entry) / entry <= 0: return None
    legs, trail = build_legs(plan, sig, y2, entry)
    if not legs and not trail: return None
    h, l, o, c = (df[k].to_numpy() for k in ("high", "low", "open", "close"))
    if mid is None: mid = df["close"].rolling(20).mean().to_numpy()
    n = len(df) if i_end is None else min(len(df), i_end + 1)
    rem, fpw, fw = 1.0, 0.0, 0.0; cur = stop; t1 = False; t1_i = None; hit = "eod"; exit_i = n - 1; fills = []
    def fill(px, w, why, i):
        nonlocal rem, fpw, fw, hit, exit_i
        w = min(w, rem)
        if w <= 1e-12: return
        fpw += px * w; fw += w; rem -= w; hit = why; exit_i = i; fills.append((i, px, w, why))
    k = 0
    for i in range(i0, n):
        hi, lo, op, cl = float(h[i]), float(l[i]), float(o[i]), float(c[i])
        if (not t1) and hi >= cur and legs and lo <= legs[0][0]:
            fill(cur * (1 + slip), 1.0, "stop_samebar_priority", i); break
        if hi >= cur:
            fill(max(op, cur) * (1 + slip), rem, "stop" if not t1 else "be", i); break
        while k < len(legs) and lo <= legs[k][0]:
            px, f, nm = legs[k]
            w = rem if (k == len(legs) - 1 and not trail) else f
            fill(min(op, px) * (1 - slip), w, "tp_" + nm, i); k += 1
            if not t1: t1 = True; t1_i = i; cur = entry
        if rem <= 1e-12: break
        if t1 and trail and i > t1_i and k >= 0 and np.isfinite(mid[i]) and cl > mid[i]:
            # runner: only the part beyond fixed legs? -> exit everything still open (incl. pending fixed legs)
            fill(cl * (1 + slip), rem, "bbmid", i); break
    else:
        fill(float(c[n - 1]) * (1 - slip), rem, "eod", n - 1)
    xp = fpw / fw; net = (entry - xp) / entry - 2 * fee; pnl = net / ((stop - entry) / entry)
    return dict(symbol=sig.symbol, entry_time=sig.entry_time, exit_time=df["dt"].iloc[exit_i], pnl_R=float(pnl), hit=hit,
                bars=exit_i - i0 + 1, stop_pct=(stop - entry) / entry, pnl_pct=float(net), fills=fills, legs=legs, entry=entry, stop=stop, i0=i0)


class T:
    def __init__(self, d):
        self.pnl_R = d["pnl_R"]; self.entry_time = d["entry_time"]; self.exit_time = d["exit_time"]
        self.stop_pct = d["stop_pct"]; self.pnl_pct_on_notional = d["pnl_pct"]; self.bars_held_15m = d["bars"]
