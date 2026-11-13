#!/workspace/venv/bin/python
"""Nested / complex head-and-shoulders (嵌套头肩, 2026-11-12, 三栋 UNI 4H annotation). Notify only.

三栋: "整个右肩又是一个头肩" -- the outer pattern's RIGHT SHOULDER is itself the HEAD of a smaller
H&S. Same price-only picker as scan_htf (v3 + close-based LS pullback); volume / RSI / MACD are
never gates. Rules (locked in research/hs_nested_meta.json before any result was looked at):

- Outer: any price-valid outer pattern from scan_htf.detect (trace: identifiable at bar t0,
  regardless of R:R), or (15m live) any BASE row's L/H/R.
- Inner head Hi = outer RS.  Unit u = 96-bar median ATR at Hi;  k_in = 1.0 (inner is one degree
  smaller than the outer; outer k = 1.5).
- Inner LS: confirmed swing high q with outerH < q <= Hi-4, high[q] < high[Hi]; rally into q from
  the lowest low in [outerH, q] >= k_in*u; close-based pullback after q (to the first bar above
  high[q], capped at Hi) >= k_in*u. Deepest pullback wins (time symmetry with the inner RS).
- Inner RS (at bar t, no look-ahead): confirmed swing high p (p+2 <= t), Hi < p <= Hi+80,
  high[p] < high[Hi], not exceeded since; rebound from the lowest low in [Hi, p] >= k_in*u and
  head-to-that-low >= k_in*u. Candidates by rebound size (largest first); the first one whose
  inner LS + geometry passes wins.
- Geometry: scan_htf.price_geometry (head highest, |RS-LS| <= 0.5 Hgt, head excess >= 0.2 Hgt,
  troughs >= 3 bars apart, L->R span 12-80 bars); time symmetry (R-H)/(H-L) in [1/3, 3].
- Signal: identical to scan_htf -- bearish close below the inner break level within 24 bars after
  the inner RS (and at/after outer t0); entry = that close; stop = inner RS high + 0.5 ATR;
  TP1 = inner neck - 1.0 inner Hgt (close 60%, then BE), TP2 = 1.5 Hgt; R:R >= 1.5; 观望 = formed,
  not broken, R:R at the break level >= 1.5 (kept in signals.json only; the runner webhooks an
  inner signal only after a close below the small neckline). Outer neck / outer targets shown as
  follow-up targets. Nested outer notes only ride on post-break parent alerts (the runner no
  longer POSTs pre-break parent 观望).
- If the regular detector already has a pattern whose head is (+-1 bar) an outer RS, that row is
  tagged nested_role="inner" and the nested duplicate is dropped (reverse direction).
- Outer rows whose RS contains an inner structure get nested_role="outer" + child info.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import scan_htf as HT  # noqa: E402  (also puts research/ on sys.path)
import hs_shoulder_v3_core as V  # noqa: E402
import scan_15m as SC
from hs_backtest import neck_at, to_shanghai
from hs_backtest_v2_3lines import BREAK_LOOKAHEAD, _ts, levels_at

K_IN = 1.0
NEST_TOL = 1
RR_MIN = HT.RR_MIN
STOP_ATR = HT.STOP_ATR
NESTED_RISK_PCT = 0.5
COLOR_INNER = "#7b2cbf"

CFG_15M = dict(bar=pd.Timedelta(minutes=15), label="15m", cn="15分钟", up_cn="1H",
               up_delta=pd.Timedelta(hours=1), fresh_bars=2, valid_bars=4)


def _arrays(df):
    high, low, opn, close = (df[c].to_numpy(dtype=float) for c in ("high", "low", "open", "close"))
    atrv = df["atr"].to_numpy(dtype=float)
    return dict(high=high, low=low, open=opn, close=close, atr=atrv, sw=V.swing_highs(high),
                u=V.unit_series(atrv), nge=HT.next_greater(high), n=len(df))


def outer_patterns(df, k=HT.K_U):
    """(regular detect events, list of price-valid outer patterns with t0)."""
    trace: dict = {}
    events = HT.detect(df, k, trace=trace)
    return events, sorted(trace.values(), key=lambda o: (o["t0"], o["H"]))


def inner_ls_candidates(A, Ho, Hi, u, k=K_IN):
    high, low, close, sw, nge = A["high"], A["low"], A["close"], A["sw"], A["nge"]
    out = []
    for q in range(max(Ho + 1, V.SW_LEFT), Hi - V.MIN_SEP + 1):
        if not sw[q] or high[q] >= high[Hi]:
            continue
        if high[q] - float(np.min(low[Ho:q + 1])) < k * u:
            continue
        d = HT.ls_pullback_close(high, close, nge, q, Hi)
        if d >= k * u:
            out.append((q, float(d)))
    return out


def inner_geometry_at(A, Ho, Hi, t, u, lcs, k=K_IN):
    """Best inner (L, R, geometry) using bars <= t only, or None."""
    high, low = A["high"], A["low"]
    opts = []
    for p in range(Hi + 1, min(t - V.SW_RIGHT, Hi + V.MAX_SIDE) + 1):
        if not A["sw"][p] or high[p] >= high[Hi]:
            continue
        if p < t and high[p + 1:t + 1].max() > high[p]:
            continue
        m = float(np.min(low[Hi:p + 1])); r = float(high[p] - m)
        if r >= k * u and high[Hi] - m >= k * u:
            opts.append((r, p))
    for _, p in sorted(opts, reverse=True):
        ok = [(q, d) for q, d in lcs if V.SYM_LO <= (p - Hi) / (Hi - q) <= V.SYM_HI]
        if not ok:
            continue
        L = max(ok, key=lambda x: (x[1], x[0]))[0]
        g = HT.price_geometry(high, low, A["atr"], L, Hi, p)
        if g is None:
            continue
        return L, p, g
    return None


def nested_scan(df, outers, k=K_IN, A=None):
    """Walk every outer bar by bar. Returns (events, structures).

    events: {"kind": "entry"|"forming"|"break_lowrr"|"retest", inner L/H/R/lt/y1/rt/y2/hgt/neck/brk/stop/t1/t2,
             "t", "outer": outer dict}
    structures: {(outerH, outerR): [(t, inner dict), ...]} every bar the inner structure is valid."""
    A = A or _arrays(df)
    n = A["n"]; high, opn, close, atrv = A["high"], A["open"], A["close"], A["atr"]
    events: list[dict] = []; structures: dict = {}; seen_inner: set = set()
    for o in outers:
        Ho, Hi = int(o["H"]), int(o["R"])
        u = A["u"][Hi]
        if not (np.isfinite(u) and u > 0):
            continue
        lcs = inner_ls_candidates(A, Ho, Hi, u, k)
        if not lcs:
            continue
        hist = structures.setdefault((Ho, Hi), [])
        forming_done = False; broken = False; done = False; lowrr_done = False; retest_done = False
        for t in range(Hi + V.SW_RIGHT + 1, min(n - 1, Hi + V.MAX_SIDE + BREAK_LOOKAHEAD) + 1):
            if high[t] > high[Hi]:
                break
            if t < int(o["t0"]):
                continue  # the outer pattern is not identifiable yet
            res = inner_geometry_at(A, Ho, Hi, t, u, lcs, k)
            if res is None:
                continue
            L, R, (lt, y1, rt, y2, hgt) = res
            if not np.isfinite(atrv[t]):
                continue
            nk, _, _, _, brk, _, _ = levels_at(lt, y1, rt, y2, t, atrv[t])
            stop = float(high[R] + STOP_ATR * atrv[t])
            t1, t2 = float(nk - hgt), float(nk - 1.5 * hgt)
            base = dict(L=L, H=Hi, R=R, lt=lt, y1=y1, rt=rt, y2=y2, hgt=hgt, neck=float(nk), t=t,
                        stop=stop, t1=t1, t2=t2, brk=float(brk), outer=o)
            hist.append((t, base))
            if done or not (R + 1 <= t <= R + BREAK_LOOKAHEAD):
                continue
            key = (Hi, L, R)
            if close[t] < opn[t] and close[t] < brk:
                broken = True
                entry = float(close[t])
                rr = (entry - t1) / (stop - entry) if stop > entry else 0.0
                if stop > entry > t1 and t2 > 0 and rr >= RR_MIN:
                    if key not in seen_inner:
                        events.append(dict(base, kind="entry", entry=entry, rr=float(rr)))
                        seen_inner.add(key)
                    done = True
                else:
                    if not lowrr_done:
                        events.append(dict(base, kind="break_lowrr", entry=entry, rr=float(rr)))
                        lowrr_done = True
                    if (not retest_done) and stop > brk > t1 and t2 > 0:
                        rr_brk = (brk - t1) / (stop - brk)
                        if rr_brk >= RR_MIN:
                            events.append(dict(base, kind="retest", entry=float(brk), rr=float(rr_brk)))
                            retest_done = True
                continue
            if broken or (forming_done and t != n - 1):
                continue
            if close[t] >= brk and stop > brk > t1 and t2 > 0:
                rr_est = (brk - t1) / (stop - brk)
                if rr_est >= RR_MIN:
                    events.append(dict(base, kind="forming", entry=float(brk), rr=float(rr_est),
                                       first=not forming_done))
                    forming_done = True
    return events, structures


def structure_at(structures, Ho, Hi, t):
    """Latest inner structure for outer (Ho, Hi) using bars <= t."""
    best = None
    for tt, g in structures.get((Ho, Hi), []):
        if tt <= t:
            best = g
    return best


# ---------------------------------------------------------------- row decoration
def _px(x):
    return SC.fmt_px(x)


def _geom(d):
    return {k: d[k] for k in ("L", "H", "R", "lt", "y1", "rt", "y2")}


def _pattern_info(df, g, t, atr_t=None):
    times = df["dt"]; high = df["high"].to_numpy(dtype=float)
    atrv = df["atr"].to_numpy(dtype=float)
    a = atr_t if atr_t is not None else float(atrv[min(t, len(df) - 1)])
    hgt = g.get("hgt")
    if hgt is None:
        hgt = float(high[g["H"]] - neck_at(g["lt"], g["y1"], g["rt"], g["y2"], g["H"]))
    nk, _, _, _, brk, _, _ = levels_at(g["lt"], g["y1"], g["rt"], g["y2"], t, a)
    return {
        "L_time": to_shanghai(times.iloc[g["L"]]), "H_time": to_shanghai(times.iloc[g["H"]]),
        "R_time": to_shanghai(times.iloc[g["R"]]),
        "L_px": float(high[g["L"]]), "H_px": float(high[g["H"]]), "R_px": float(high[g["R"]]),
        "neck_now": float(nk), "break_level": float(brk), "head_height": float(hgt),
        "target_1h": float(nk - hgt), "target_15h": float(nk - 1.5 * hgt),
        "neck_s2": SC.neck_s2(g["lt"], g["y1"], g["rt"], g["y2"], g["H"], float(high[g["H"]])),
    }


def parent_targets_text(parent: dict, ref: float | None) -> str:
    """Outer levels that are still BELOW the inner entry/break (the follow-up targets)."""
    lv = [("大颈线", parent["neck_now"]), ("大形态 1.0H", parent["target_1h"]), ("大形态 1.5H", parent["target_15h"])]
    below = [(n, v) for n, v in lv if ref is None or v < ref]
    if not below:
        return "大形态目标已在入场位上方，无后续目标"
    s = "后续看" + " → ".join(f"{n} {_px(v)}" for n, v in below)
    if ref is not None and parent["neck_now"] >= ref:
        s = f"价格已在大颈线 {_px(parent['neck_now'])} 下方，" + s
    return s


def inner_text(parent: dict, child: dict, status: str, ref: float | None = None) -> str:
    head = (f"大头肩的右肩内又成一个小头肩（小LS {child['L_time'][5:]} {_px(child['L_px'])} / "
            f"小H=大RS {child['H_time'][5:]} {_px(child['H_px'])} / 小RS {child['R_time'][5:]} {_px(child['R_px'])}）")
    tgt = parent_targets_text(parent, ref)
    if status == HT.STATUS_ENTRY:
        return f"{head}，小颈线破位先入场；{tgt}。"
    if status == HT.STATUS_RETEST:
        return f"{head}，小颈线已收盘跌破但按收盘价追盈亏比不足 1.5，在小颈线 {_px(child['break_level'])} 挂限价等回踩；{tgt}。"
    return f"{head}，等小颈线 {_px(child['break_level'])} 收盘跌破再进；{tgt}。"


def outer_text(child: dict, lowrr: dict | None = None) -> str:
    s = (f"嵌套头肩：右肩内又成一个小头肩（小LS {child['L_time'][5:]} {_px(child['L_px'])} / "
         f"小H=右肩 {child['H_time'][5:]} {_px(child['H_px'])} / 小RS {child['R_time'][5:]} {_px(child['R_px'])}，"
         f"小颈线 {_px(child['neck_now'])}）")
    if lowrr is not None:
        s += f"；小颈线已破但按小形态止损 R:R {lowrr['rr']:.1f}<1.5，未单独发小形态入场"
    return s + "。"


def decorate_inner(row: dict, df, ev: dict, tf: str) -> dict:
    o = ev["outer"]; t = ev["t"]
    parent = _pattern_info(df, o, t)
    child = _pattern_info(df, ev, t)
    row.update({
        "nested": True, "nested_signal": True, "nested_role": "inner",
        "tier": "watch" if tf == "15m" else HT.TIER_HTF,
        "validated": False, "suggested_risk_pct": NESTED_RISK_PCT, "counts_toward_breaker": False,
        "parent": parent, "child": child,
        "parent_neck_now": parent["neck_now"],
        "parent_target_1h": parent["target_1h"], "parent_target_15h": parent["target_15h"],
        "nested_text": inner_text(parent, child, row["status"],
                                  row.get("official_entry") or row.get("break_level")),
        "_nested_other": _geom(o),
    })
    row["note"] = (row.get("note") or "").split(" 未验证")[0] + (
        " 嵌套小头肩（未验证），建议风险 0.5%，独立记账（paper_nested），不计入也不受15m熔断影响。")
    return row


def inner_break_rr(df, struct: dict, t: int) -> dict | None:
    """If bar t closed bearish below the inner break level: the inner-pattern R:R at that close."""
    opn, close, high = (df[c].to_numpy(dtype=float) for c in ("open", "close", "high"))
    atrv = df["atr"].to_numpy(dtype=float)
    if t >= len(df) or not np.isfinite(atrv[t]):
        return None
    nk, _, _, _, brk, _, _ = levels_at(struct["lt"], struct["y1"], struct["rt"], struct["y2"], t, atrv[t])
    if not (close[t] < opn[t] and close[t] < brk):
        return None
    stop = float(high[struct["R"]] + STOP_ATR * atrv[t]); t1 = float(nk - struct["hgt"]); e = float(close[t])
    return {"rr": (e - t1) / (stop - e) if stop > e else 0.0, "entry": e, "stop": stop, "t1": t1}


def annotate_outer(row: dict, df, struct: dict, t: int, lowrr: dict | None = None) -> dict:
    child = _pattern_info(df, struct, t)
    if lowrr is None:
        b = inner_break_rr(df, struct, t)
        lowrr = b if (b is not None and b["rr"] < RR_MIN) else None
    row.update({"nested": True, "nested_role": "outer", "child": child,
                "nested_text": outer_text(child, lowrr), "_nested_other": _geom(struct)})
    return row


def annotate_reverse_inner(row: dict, df, o: dict, t: int) -> dict:
    parent = _pattern_info(df, o, t)
    child = _pattern_info(df, row, t)
    row.update({"nested": True, "nested_role": "inner", "parent": parent,
                "parent_neck_now": parent["neck_now"], "parent_target_1h": parent["target_1h"],
                "parent_target_15h": parent["target_15h"],
                "nested_text": inner_text(parent, child, row["status"],
                                          row.get("official_entry") or row.get("break_level"))
                + "（本形态本身即按常规规则成立）",
                "_nested_other": _geom(o)})
    return row


def find_parent(outers, H, t):
    for o in outers:
        if abs(int(o["R"]) - int(H)) <= NEST_TOL and int(o["t0"]) <= t and int(o["H"]) < int(H) - V.MIN_SEP:
            return o
    return None


def live_nested(symbol: str, tf: str, df, df_up, rows: list[dict], events: list[dict], outers: list[dict],
                extra_outers: list[dict] | None = None, cfg: dict | None = None) -> list[dict]:
    """Annotate regular rows (outer/inner) in place and return the nested inner signal rows that are
    actionable at the last closed bar."""
    n = len(df)
    if n < 60:
        return []
    A = _arrays(df)
    all_outers = list(outers) + list(extra_outers or [])
    nev, structs = nested_scan(df, all_outers, A=A)
    last = n - 1
    lowrr = {}
    for e in nev:
        if e["kind"] == "break_lowrr":
            lowrr[(e["outer"]["H"], e["outer"]["R"])] = e
    # regular rows: outer role / reverse inner role
    reg_heads = set()
    for r in rows:
        t = int(r.get("_ev_t", last))
        st = structure_at(structs, r["H"], r["R"], t)
        if st is not None:
            annotate_outer(r, df, st, t, lowrr.get((r["H"], r["R"])))
            continue
        par = find_parent(all_outers, r["H"], t)
        if par is not None:
            annotate_reverse_inner(r, df, par, t)
            reg_heads.add(int(r["H"]))
    for e in events:
        reg_heads.add(int(e["H"])) if find_parent(all_outers, e["H"], e["t"]) else None
    # inner signals
    high, low, close = A["high"], A["low"], A["close"]
    cur = float(close[-1]); cfg = cfg or HT.TF_CFG[tf]; fresh = cfg["fresh_bars"]
    out = []; done = set()
    for e in nev:
        if any(abs(e["H"] - h) <= NEST_TOL for h in reg_heads):
            continue
        key = (e["H"], e["L"], e["R"], e["kind"])
        if key in done:
            continue
        if e["kind"] == "entry":
            if e["t"] < n - 1 - fresh or not (e["stop"] > cur > e["t1"]):
                continue
            if e["t"] + 1 < n and (high[e["t"] + 1:].max() >= e["stop"] or low[e["t"] + 1:].min() <= e["t1"]):
                continue
            if (cur - e["t1"]) / (e["stop"] - cur) < RR_MIN:
                continue
        elif e["kind"] == "retest":
            if e["t"] < n - 1 - fresh:
                continue
            if high[e["t"]:].max() >= e["stop"] or low[e["t"]:].min() <= e["t1"]:
                continue
        elif not (e["kind"] == "forming" and e["t"] == last):
            continue
        row = HT.make_row(symbol, tf, df, e, df_up, cfg=cfg)
        decorate_inner(row, df, e, tf)
        out.append(row); done.add(key)
    heads_entry = {r["H"] for r in out if r["status"] == HT.STATUS_ENTRY}
    heads_retest = {r["H"] for r in out if r["status"] == HT.STATUS_RETEST}
    kept = []
    for r in out:
        if r["status"] == HT.STATUS_ENTRY:
            kept.append(r)
        elif r["status"] == HT.STATUS_RETEST and r["H"] not in heads_entry:
            kept.append(r)
        elif r["status"] not in (HT.STATUS_ENTRY, HT.STATUS_RETEST) and r["H"] not in heads_entry and r["H"] not in heads_retest:
            kept.append(r)
    return kept


def outers_from_rows(rows: list[dict], df) -> list[dict]:
    """15m BASE rows (formal/watch/observe) as outer patterns (known at the last bar)."""
    out = []
    high = df["high"].to_numpy(dtype=float)
    for r in rows:
        try:
            L, H, R = int(r["L"]), int(r["H"]), int(r["R"])
            lt, y1, rt, y2 = int(r["lt"]), float(r["y1"]), int(r["rt"]), float(r["y2"])
        except (KeyError, TypeError, ValueError):
            continue
        hgt = float(high[H] - neck_at(lt, y1, rt, y2, H))
        if hgt > 0:
            out.append(dict(L=L, H=H, R=R, lt=lt, y1=y1, rt=rt, y2=y2, hgt=hgt, t0=min(len(df) - 1, R + V.SW_RIGHT)))
    return out


def scan_15m_nested(symbol: str, df15i, df1i, rows: list[dict]) -> list[dict]:
    """15m: annotate BASE rows + v3 outers, return nested inner rows (tier watch, 0.5%, unvalidated)."""
    events, outers = outer_patterns(df15i)
    extra = outers_from_rows(rows, df15i)
    known = {(o["H"], o["R"]) for o in outers}
    extra = [o for o in extra if (o["H"], o["R"]) not in known]
    out = live_nested(symbol, "15m", df15i, df1i, rows, [], outers, extra, cfg=CFG_15M)
    for r in out:
        r["note"] = ("15m 嵌套小头肩：" + (r.get("note") or "")).replace("15m 收盘", "收盘")
    return out
