#!/workspace/venv/bin/python
"""1H / 4H head-and-shoulders top scan (notify only, never places orders).

Pivot picker: 三栋-approved v3 PRICE-ONLY shoulder picker (research/hs_shoulder_v3_core.py),
with the 2026-11-12 fix: the pullback after the left shoulder is measured on CLOSES, not on
long lower wicks (XRP 11-12 case: a wick-only dip made a swing 3 bars before the head look like
an LS). Volume / RSI / MACD are NEVER hard gates here; they are reported as confirmation info.

Rules (unvalidated; 1H/4H backtests were negative -> tier "htf", suggested risk 0.5%):
- LS: swing high on the rising leg into H, 4..80 bars before H, close-based pullback >= 1.5u
  (u = 96-bar median ATR14), the deepest such pullback wins; leg / cobbled filter as v3.
- RS: confirmed swing high after H (<= 80 bars), rebound >= 1.5u, head-to-right-trough >= 1.5u;
  time symmetry (R-H)/(H-L) in [1/3, 3]; L->R span 12..80 bars.
- Price geometry (head-height based): head highest in [L,R]; |RS-LS| <= 0.5*Hgt; head excess
  over the higher shoulder >= 0.2*Hgt; troughs >= 3 bars apart. Neckline slope is NOT a filter
  (neck_s2 / angle are shown only).
- Break: bearish bar that CLOSES below the neckline break level within 24 bars after RS.
  Entry = that close (next bar open); stop = RS high + 0.5 ATR; TP1 1.0H (close 60%, then
  break-even), TP2 1.5H; R:R >= 1.5 at entry. Higher-TF trend (4H for 1H, 1D for 4H) is info only.
- 已破位·等回踩: a bearish close below the break level exists, but R:R at that close is < 1.5
  (or entry <= TP1). If R:R computed at the break level (limit entry = brk) is still >= 1.5
  and stop > brk > t1, emit once per head on the first such bar. Pushed via webhook; hang a
  limit short at brk, same stop/TP/validity as entry. Not a market chase.
- 观望: L/H/R valid at the last closed bar, not yet broken, estimated R:R at the break level >= 1.5.
  Kept in signals.json / logs for tracking; the runner does not webhook 观望 (2026-11-12, 三栋:
  only push after a closed bearish bar has already broken the neckline). An expired break
  (outside fresh_bars / validity window) is not emitted as 可入场 or 已破位·等回踩.

Writes scan_out/htf/<tf>/signals.json and charts. Usage: scan_htf.py --tf 1h,4h
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT / "research")):
    if p not in sys.path:
        sys.path.insert(0, p)

from hs_backtest import add_indicators, neck_at, to_shanghai  # noqa: E402
from hs_backtest_v2_3lines import (  # noqa: E402
    BREAK_LOOKAHEAD, MAX_SPAN, MIN_SPAN, MIN_TROUGH_GAP, _ts, htf_bullish_at, levels_at,
)
import hs_shoulder_v3_core as V  # noqa: E402
import okx_market  # noqa: E402
import scan_15m as SC  # noqa: E402
import binance_data  # noqa: E402

TIER_HTF = "htf"
STATUS_ENTRY = "可入场"
STATUS_WATCH = "观望"
STATUS_RETEST = "已破位·等回踩"
HTF_RISK_PCT = 0.5
RR_MIN = 1.5
K_U = 1.5
STOP_ATR = 0.5
MMR = 0.005            # maintenance-margin allowance in the leverage cap
LEV_HARD_MAX = 50
OUT_ROOT = HERE / "scan_out" / "htf"

TF_CFG = {
    "1h": dict(bar=pd.Timedelta(hours=1), label="1H", cn="1小时", bin="1h", okx="1H", limit=600,
               up_bin="4h", up_okx="4H", up_limit=300, up_delta=pd.Timedelta(hours=4), up_cn="4H",
               fresh_bars=2, valid_bars=3),
    "4h": dict(bar=pd.Timedelta(hours=4), label="4H", cn="4小时", bin="4h", okx="4H", limit=500,
               up_bin="1d", up_okx="1Dutc", up_limit=200, up_delta=pd.Timedelta(days=1), up_cn="日线",
               fresh_bars=1, valid_bars=2),
}


# ---------------------------------------------------------------- pivots (price only)
def next_greater(high: np.ndarray) -> np.ndarray:
    n = len(high); out = np.full(n, n - 1, dtype=int); st: list[int] = []
    for i in range(n):
        while st and high[i] > high[st[-1]]:
            out[st.pop()] = i
        st.append(i)
    return out


def ls_pullback_close(high, close, nge, p, H) -> float:
    """Independent pullback after a left-shoulder candidate, on CLOSES (not wicks).

    From LS bar p to the first bar that trades above high[p] (capped at H)."""
    e = min(int(nge[p]), H)
    return float(high[p] - np.min(close[p:e + 1]))


def ls_candidates(high, low, close, sw, nge, H, u, k=K_U):
    a = max(0, H - V.MAX_SIDE)
    above = np.nonzero(high[a:H] > high[H])[0]
    a2 = a + int(above[-1]) + 1 if len(above) else a
    A = a2 + int(np.argmin(low[a2:H + 1]))
    out = []
    for p in range(max(V.SW_LEFT, A + 1), H - V.MIN_SEP + 1):
        if not sw[p] or high[p] >= high[H]:
            continue
        d = ls_pullback_close(high, close, nge, p, H)
        if d >= k * u:
            out.append((p, d))
    return out


def price_geometry(high, low, atrv, L, H, R):
    if not (MIN_SPAN <= R - L <= MAX_SPAN) or not np.isfinite(atrv[R]) or atrv[R] <= 0:
        return None
    if float(np.max(high[L:R + 1])) > high[H] + 1e-15:
        return None
    lt = L + int(np.argmin(low[L:H + 1])); rt = H + int(np.argmin(low[H:R + 1]))
    if rt - lt < MIN_TROUGH_GAP:
        return None
    y1, y2 = float(low[lt]), float(low[rt])
    hgt = float(high[H] - neck_at(lt, y1, rt, y2, H))
    if hgt <= 0:
        return None
    if abs(high[R] - high[L]) > 0.5 * hgt:
        return None
    if high[H] - max(high[L], high[R]) < 0.2 * hgt:
        return None
    return lt, y1, rt, y2, hgt


def detect(df: pd.DataFrame, k: float = K_U, trace: dict | None = None) -> list[dict]:
    """Walk every head bar by bar (no look-ahead). Returns events:
    {"kind": "entry"|"forming"|"retest", "t", "L", "H", "R", "lt", "y1", "rt", "y2", "hgt", ...}.
    One entry per head (first qualifying break); one retest per head (first low-RR break whose
    R:R at the break level is still >= 1.5); 'forming' = first bar a head is ready (dedup by H).
    trace (optional dict, 2026-11-12 nested H&S): filled with every price-valid outer pattern
    {(H, R): {L, H, R, lt, y1, rt, y2, hgt, t0}} where t0 = first bar it was identifiable
    (regardless of R:R). Does not change the returned events."""
    n = len(df)
    if n < 60:
        return []
    high, low, opn, close = (df[c].to_numpy(dtype=float) for c in ("high", "low", "open", "close"))
    atrv = df["atr"].to_numpy(dtype=float)
    sw = V.swing_highs(high); uu = V.unit_series(atrv); nge = next_greater(high)
    events: list[dict] = []
    for H in np.nonzero(sw)[0]:
        H = int(H); u = uu[H]
        if H < 20 or not (np.isfinite(u) and u > 0):
            continue
        cands = ls_candidates(high, low, close, sw, nge, H, u, k)
        if not cands:
            continue
        best = None; ls_cache: dict[int, int | None] = {}; run_min = np.inf; pref = {}
        forming_done = False; broken = False; retest_done = False
        for t in range(H + 1, min(n - 1, H + V.MAX_SIDE + BREAK_LOOKAHEAD) + 1):
            if high[t] > high[H]:
                break
            run_min = min(run_min, low[t]); pref[t] = run_min
            p = t - V.SW_RIGHT
            if p > H and sw[p] and p - H <= V.MAX_SIDE and high[p] < high[H]:
                m = min(pref[p], low[H]); r = float(high[p] - m)
                if r >= k * u and high[H] - m >= k * u and (best is None or r >= best[1] - 1e-15):
                    best = (p, r)
            if best is None:
                continue
            R = best[0]
            if not (R + 1 <= t <= R + BREAK_LOOKAHEAD) or high[R:t + 1].max() > high[R]:
                continue
            if R not in ls_cache:
                ok = [(q, d) for q, d in cands if V.SYM_LO <= (R - H) / (H - q) <= V.SYM_HI]
                L = max(ok, key=lambda x: (x[1], x[0]))[0] if ok else None
                if L is not None and not V.cobbled_ok(high, low, L, H, u, k):
                    L = None
                ls_cache[R] = L
            L = ls_cache[R]
            if L is None:
                continue
            g = price_geometry(high, low, atrv, L, H, R)
            if g is None or not np.isfinite(atrv[t]):
                continue
            lt, y1, rt, y2, hgt = g
            if trace is not None and (H, R) not in trace:
                trace[(H, R)] = dict(L=L, H=H, R=R, lt=lt, y1=y1, rt=rt, y2=y2, hgt=hgt, t0=t)
            nk, _, _, _, brk, _, _ = levels_at(lt, y1, rt, y2, t, atrv[t])
            stop = float(high[R] + STOP_ATR * atrv[t])
            t1, t2 = float(nk - hgt), float(nk - 1.5 * hgt)
            base = dict(L=L, H=H, R=R, lt=lt, y1=y1, rt=rt, y2=y2, hgt=hgt, neck=float(nk), t=t,
                        stop=stop, t1=t1, t2=t2, brk=float(brk))
            if close[t] < opn[t] and close[t] < brk:
                broken = True
                entry = float(close[t])
                rr = (entry - t1) / (stop - entry) if stop > entry else 0.0
                if stop > entry > t1 and t2 > 0 and rr >= RR_MIN:  # targets must be real prices
                    events.append(dict(base, kind="entry", entry=entry, rr=float(rr)))
                    break
                # Deep/low-RR close: do not drop silently. If a limit at the break level
                # still has R:R >= 1.5, emit one retest event and keep looking for a later entry.
                if (not retest_done) and stop > brk > t1 and t2 > 0:
                    rr_brk = (brk - t1) / (stop - brk)
                    if rr_brk >= RR_MIN:
                        events.append(dict(base, kind="retest", entry=float(brk), rr=float(rr_brk)))
                        retest_done = True
                continue  # A-style: keep looking at later bars of the window
            if broken or (forming_done and t != n - 1):
                continue  # keep the first ready bar (history) and the last closed bar (live)
            if close[t] >= brk and stop > brk > t1 and t2 > 0:
                rr_est = (brk - t1) / (stop - brk)
                if rr_est >= RR_MIN:
                    events.append(dict(base, kind="forming", entry=float(brk), rr=float(rr_est),
                                       first=not forming_done))
                    forming_done = True
    return events


# ---------------------------------------------------------------- rows
def leverage_cap(entry: float, stop: float) -> int | None:
    """Max integer leverage so liquidation distance (≈1/lev − MMR) >= 2 × stop distance."""
    if not (entry and stop and stop > entry):
        return None
    d = (stop - entry) / entry
    lev = int(math.floor(1.0 / (2.0 * d + MMR)))
    return max(1, min(LEV_HARD_MAX, lev))


def make_row(symbol: str, tf: str, df: pd.DataFrame, ev: dict, df_up: pd.DataFrame | None,
             cfg: dict | None = None) -> dict:
    cfg = cfg or TF_CFG[tf]; bar = cfg["bar"]
    t = ev["t"]; times = df["dt"]; close = df["close"].to_numpy(dtype=float)
    cur = float(close[-1]); last_close_ts = _ts(times.iloc[-1]) + bar
    is_entry = ev["kind"] == "entry"
    is_retest = ev["kind"] == "retest"
    is_broken = is_entry or is_retest
    entry = float(ev["entry"]) if is_broken else None
    entry_time = _ts(times.iloc[t]) + bar
    ref = entry if is_broken else float(ev["brk"])
    stop, t1, t2 = ev["stop"], ev["t1"], ev["t2"]
    rr_now = (cur - t1) / (stop - cur) if (is_entry and stop > cur > t1) else ev["rr"]
    valid_until = _ts(times.iloc[t]) + bar * (1 + cfg["valid_bars"])
    try:
        up_bull = bool(htf_bullish_at(df_up, last_close_ts, cfg["up_delta"])) if df_up is not None else None
    except Exception:
        up_bull = None
    conf = SC.pivot_confirmation(df, ev["L"], ev["H"], ev["R"])
    lev = leverage_cap(entry if is_broken else ref, stop)
    if is_entry:
        status = STATUS_ENTRY
    elif is_retest:
        status = STATUS_RETEST
    else:
        status = STATUS_WATCH
    row = {
        "symbol": symbol, "status": status,
        "tier": TIER_HTF, "timeframe": tf, "timeframe_cn": cfg["cn"], "validated": False,
        "suggested_risk_pct": HTF_RISK_PCT,
        "L": ev["L"], "H": ev["H"], "R": ev["R"],
        "lt": ev["lt"], "y1": ev["y1"], "rt": ev["rt"], "y2": ev["y2"],
        "L_time": to_shanghai(times.iloc[ev["L"]]), "H_time": to_shanghai(times.iloc[ev["H"]]),
        "R_time": to_shanghai(times.iloc[ev["R"]]),
        "break_bar_shanghai": to_shanghai(times.iloc[t]) if is_broken else None,
        "neck": ev["neck"], "break_level": ev["brk"], "head_height": ev["hgt"],
        "suggest_entry": entry if is_broken else None,
        "official_entry": entry, "official_entry_time": to_shanghai(entry_time) if is_broken else None,
        "stop": stop, "target_1h": t1, "target_15h": t2, "current": cur, "rr": float(rr_now),
        "dist_stop_pct": 100.0 * (stop - (entry or ref)) / (entry or ref),
        "dist_t1_pct": 100.0 * ((entry or ref) - t1) / (entry or ref),
        "leverage_cap": lev,
        "leverage_note": (f"杠杆≤{lev}x（强平距离≥2×止损距离）" if lev else None),
        "valid_until_shanghai": to_shanghai(valid_until) if is_broken else None,
        "htf_trend_bullish": up_bull,
        "htf_trend_note": (None if up_bull is None else
                           (f"{cfg['up_cn']}明显偏多（逆势，仅提示，不过滤）" if up_bull else f"{cfg['up_cn']}未明显偏多")),
        "confirm_text": SC.confirmation_text(conf),
        "exits": "TP1=1.0H 平60%后止损移到保本；TP2=1.5H 平余仓",
        "note": (
            f"{cfg['label']} 收盘跌破颈线，按收盘价/下一根开盘入场；有效至 {to_shanghai(valid_until)}（北京时间），过期不追。"
            if is_entry else
            f"{cfg['label']} 已收盘跌破颈线，但按收盘价追的盈亏比不足 1.5；在 {SC.fmt_px(ev['brk'])} 附近挂限价空单等回踩，有效至 {to_shanghai(valid_until)}，过期撤单。"
            if is_retest else
            f"{cfg['label']} 头肩已成型、尚未破位：等 {cfg['label']} 阴线收盘跌破 {SC.fmt_px(ev['brk'])} 再进，不要提前空。"
        ) + " 未验证（1H/4H 回测为负），建议风险 0.5%，单独记账，不计入15m熔断。",
    }
    SC.attach_neck_slope(row, df)
    SC.attach_shoulder_prop(row, df)
    return row


def live_rows(symbol: str, tf: str, df: pd.DataFrame, df_up: pd.DataFrame | None,
              events: list[dict] | None = None, nested: bool = True) -> list[dict]:
    """Rows actionable at the last closed bar."""
    n = len(df)
    if n < 60:
        return []
    trace: dict = {}
    if events is None:
        events = detect(df, trace=trace)
    high, low, close = (df[c].to_numpy(dtype=float) for c in ("high", "low", "close"))
    cur = float(close[-1]); fresh = TF_CFG[tf]["fresh_bars"]
    out = []
    for ev in events:
        if ev["kind"] in ("entry", "retest"):
            if ev["t"] < n - 1 - fresh:
                continue
            if ev["kind"] == "entry":
                if not (ev["stop"] > cur > ev["t1"]):
                    continue
                if ev["t"] + 1 < n and (high[ev["t"] + 1:].max() >= ev["stop"] or low[ev["t"] + 1:].min() <= ev["t1"]):
                    continue
                if (cur - ev["t1"]) / (ev["stop"] - cur) < RR_MIN:
                    continue
            else:
                # Retest: drop only if stop or TP1 traded at/after the break bar.
                # Do not drop just because price already tagged the break level (limit would have filled)
                # or because the current-price R:R is below 1.5 (R:R is measured at brk).
                if high[ev["t"]:].max() >= ev["stop"] or low[ev["t"]:].min() <= ev["t1"]:
                    continue
            out.append(dict(make_row(symbol, tf, df, ev, df_up), _ev_t=int(ev["t"])))
        elif ev["kind"] == "forming" and ev["t"] == n - 1:
            out.append(dict(make_row(symbol, tf, df, ev, df_up), _ev_t=int(ev["t"])))
    # Prefer a market entry over a retest over a still-forming watch, same head.
    entry_heads = {r["H"] for r in out if r["status"] == STATUS_ENTRY}
    retest_heads = {r["H"] for r in out if r["status"] == STATUS_RETEST}
    kept = []
    for r in out:
        if r["status"] == STATUS_ENTRY:
            kept.append(r)
        elif r["status"] == STATUS_RETEST and r["H"] not in entry_heads:
            kept.append(r)
        elif r["status"] == STATUS_WATCH and r["H"] not in entry_heads and r["H"] not in retest_heads:
            kept.append(r)
        elif r["status"] not in (STATUS_ENTRY, STATUS_RETEST, STATUS_WATCH):
            kept.append(r)
    out = kept
    if nested:
        # 2026-11-12 嵌套头肩: annotate outer/inner roles + add nested inner signals (never breaks the scan)
        try:
            import nested_hs
            outers = sorted(trace.values(), key=lambda o: (o["t0"], o["H"])) if trace else \
                nested_hs.outer_patterns(df)[1]
            out += nested_hs.live_nested(symbol, tf, df, df_up, out, events, outers)
        except Exception as e:  # noqa: BLE001
            print(f"  nested skipped {tf} {symbol}: {type(e).__name__} {e}", flush=True)
    return out


# ---------------------------------------------------------------- IO
def fetch_frames(exchange: str, sym: str, tfs: list[str], cache: dict) -> dict:
    def get(interval_bin, interval_okx, limit):
        key = (exchange, sym, interval_bin)
        if key not in cache:
            if exchange == SC.EXCHANGE_BINANCE:
                cache[key] = SC.fetch_klines(sym, interval_bin, limit)
            else:
                cache[key] = okx_market.fetch_okx_klines(sym, interval_okx, limit)
        return cache[key]
    out = {}
    for tf in tfs:
        c = TF_CFG[tf]
        out[tf] = (get(c["bin"], c["okx"], c["limit"]), get(c["up_bin"], c["up_okx"], c["up_limit"]))
    return out


def chart_path(r: dict) -> Path:
    prefix = "okx_" if r.get("exchange") == SC.EXCHANGE_OKX else ""
    if r["status"] == STATUS_ENTRY:
        suffix = ""
    elif r["status"] == STATUS_RETEST:
        suffix = "_retest"
    else:
        suffix = "_watch"
    if r.get("nested_signal"):
        suffix = "_nested" + suffix
    return OUT_ROOT / r["timeframe"] / f"{prefix}{r['symbol']}{suffix}.png"


def title_extra(r: dict) -> str:
    if r.get("nested_signal"):
        return "嵌套小头肩·未验证·风险0.5%"
    if r.get("nested_role") == "outer":
        return "含嵌套小头肩·未验证·风险0.5%"
    return "未验证·风险0.5%"


def scan(tfs: list[str]) -> int:
    t0 = time.monotonic()
    pool = json.loads((ROOT / "core_pool.json").read_text())
    bsyms = [s["symbol"] if isinstance(s, dict) else s for s in pool["symbols"]]
    try:
        osyms = okx_market.load_okx_pool(log=lambda m: print(m, flush=True)) or []
        okx_err = ""
    except Exception as e:
        osyms, okx_err = [], f"{type(e).__name__}: {e}"
    for tf in tfs:
        d = OUT_ROOT / tf; d.mkdir(parents=True, exist_ok=True)
        for p in d.glob("*.png"):
            p.unlink(missing_ok=True)
    rows = {tf: [] for tf in tfs}; newest = {tf: None for tf in tfs}
    stats = {"binance_ok": 0, "okx_ok": 0, "fail": 0}
    okx_consec = 0
    for exchange, syms in ((SC.EXCHANGE_BINANCE, bsyms), (SC.EXCHANGE_OKX, osyms)):
        for sym in syms:
            if exchange == SC.EXCHANGE_OKX and okx_consec >= SC.OKX_MAX_CONSEC_FAIL:
                okx_err = okx_err or "too many consecutive OKX failures"
                break
            cache: dict = {}
            try:
                fr = fetch_frames(exchange, sym, tfs, cache)
                if exchange == SC.EXCHANGE_OKX:
                    okx_consec = 0
            except Exception as e:
                stats["fail"] += 1
                if exchange == SC.EXCHANGE_OKX:
                    okx_consec += 1
                print(f"  FAIL {exchange}:{sym} fetch: {type(e).__name__} {e}", flush=True)
                continue
            stats["binance_ok" if exchange == SC.EXCHANGE_BINANCE else "okx_ok"] += 1
            for tf in tfs:
                df, dup = fr[tf]
                try:
                    if df is None or len(df) < 120:
                        continue
                    dfi = add_indicators(df)
                    dupi = add_indicators(dup) if dup is not None and len(dup) else None
                    last = _ts(dfi["dt"].iloc[-1])
                    newest[tf] = last if newest[tf] is None else max(newest[tf], last)
                    found = live_rows(sym, tf, dfi, dupi)
                    for r in found:
                        r["exchange"] = exchange
                        r["exchange_cn"] = SC.EXCHANGE_CN[exchange]
                        r["coin"] = okx_market.coin_key(sym, exchange)
                        cp = chart_path(r)
                        SC.plot_setup(dfi, r, cp, title_extra=title_extra(r), tf_label=TF_CFG[tf]["label"])
                        r["chart"] = str(cp)
                        r["data_source"] = binance_data.source_of(sym, exchange)
                        try:
                            r["chart_spec"] = SC.chart_spec(dfi, r, tf, title_extra=title_extra(r),
                                                            tf_label=TF_CFG[tf]["label"])
                        except Exception as e:  # noqa: BLE001
                            print(f"  chart_spec {tf} {exchange}:{sym}: {type(e).__name__} {e}", flush=True)
                        rows[tf].append(r)
                    if found:
                        print(f"  {tf} {exchange[:3]} {sym}: " + ", ".join(r["status"] for r in found), flush=True)
                except Exception as e:
                    stats["fail"] += 1
                    print(f"  FAIL {tf} {exchange}:{sym} scan: {type(e).__name__} {e}", flush=True)
    now = pd.Timestamp.now(tz="UTC")
    for tf in tfs:
        rs = rows[tf]
        SC.tag_same_coin(rs)
        rank = {STATUS_ENTRY: 0, STATUS_RETEST: 1, STATUS_WATCH: 2}
        rs.sort(key=lambda r: (rank.get(r["status"], 9), 0 if r["exchange"] == SC.EXCHANGE_BINANCE else 1))
        last_bar = newest[tf]
        meta = {
            "asof_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "asof_shanghai": now.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M"),
            "timeframe": tf,
            "last_bar_shanghai": last_bar.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M") if last_bar is not None else "",
            "exchanges": [SC.EXCHANGE_BINANCE] + ([SC.EXCHANGE_OKX] if osyms else []),
            "okx_ok": bool(osyms) and not okx_err, "okx_error": okx_err,
            "binance_scanned": stats["binance_ok"], "okx_scanned": stats["okx_ok"],
            "n_entry": sum(1 for r in rs if r["status"] == STATUS_ENTRY and not r.get("nested_signal")),
            "n_retest": sum(1 for r in rs if r["status"] == STATUS_RETEST and not r.get("nested_signal")),
            "n_watch": sum(1 for r in rs if r["status"] == STATUS_WATCH and not r.get("nested_signal")),
            "n_nested": sum(1 for r in rs if r.get("nested_signal")),
            "rule": "scan_htf.py: v3 价格认肩（左肩回调按收盘价）+ 收盘破颈线 + R:R≥1.5；量/RSI/MACD 只做确认信息",
            "scan_seconds_total": round(time.monotonic() - t0, 1),
            "binance_data": binance_data.summary(),
        }
        ser = [{k: v for k, v in r.items() if k not in ("L", "H", "R", "lt", "rt", "y1", "y2")
                and not k.startswith("_")} for r in rs]
        out = OUT_ROOT / tf / "signals.json"
        out.write_text(json.dumps({**meta, "n": len(ser), "signals": ser}, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
        print(f"wrote {out} tf={tf} entries={meta['n_entry']} retests={meta['n_retest']} watches={meta['n_watch']} "
              f"scan_s={meta['scan_seconds_total']}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1h", help="comma list: 1h,4h")
    a = ap.parse_args()
    tfs = [x.strip().lower() for x in a.tf.split(",") if x.strip()]
    bad = [x for x in tfs if x not in TF_CFG]
    if bad:
        print(f"unknown tf {bad}", file=sys.stderr)
        return 2
    return scan(tfs)


if __name__ == "__main__":
    raise SystemExit(main())
