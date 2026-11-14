#!/usr/bin/env python3
"""Scan the core pool for 15m H&S short setups (A+C rules). Draw annotated charts.

Dependencies: requests, pandas, numpy, scipy, matplotlib
Reads: /workspace/trading/core_pool.json (Binance USDT-M)
       /workspace/trading/okx_pool.json  (OKX USDT-SWAP top 30, rebuilt daily by okx_market.py)
Rules: /workspace/trading/hs_15m_breakout_quality.md (locked A+C parameters)
Writes: /workspace/trading/backtest/scan_out/signals.json
        /workspace/trading/backtest/scan_out/scan.md
        /workspace/trading/backtest/scan_out/<SYMBOL>.png             (Binance)
        /workspace/trading/backtest/scan_out/okx_<instId>.png         (OKX)
        /workspace/trading/backtest/scan_out/_demo_NEARUSDT.png  (historical annotation check)

Idempotent: each run wipes previous signal charts in scan_out/ and rewrites JSON/MD.
Does not place orders.

Usage:
    python3 scan_15m.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hs_backtest import add_indicators, macd_break_ok, neck_at, to_shanghai  # noqa: E402
from hs_backtest_v2_3lines import (  # noqa: E402
    BREAK_LOOKAHEAD, _ts, htf_bullish_at, levels_at,
)
from hs_opt_strategy import LAM, measured_ok, scan_line  # noqa: E402
from hs_15m_ac_forward import find_watch  # noqa: E402
from hs_nearmiss import SUGGESTED_RISK_PCT, scan_nearmiss  # noqa: E402
import okx_market  # noqa: E402
import binance_data  # noqa: E402

# 档位：正式（已过样本外的 A+C，含「可入场」「观望」）与观察（near-miss，未验证）。
TIER_FORMAL = "formal"
TIER_WATCH = "watch"
STATUS_OBSERVE = "观察"
STATUS_RETEST = "已破位·等回踩"
FORMAL_RISK_PCT = 1.0

EXCHANGE_BINANCE = "binance"
EXCHANGE_OKX = "okx"
EXCHANGE_CN = {EXCHANGE_BINANCE: "币安", EXCHANGE_OKX: "欧易"}

POOL_PATH = ROOT / "core_pool.json"
OUT_DIR = HERE / "scan_out"
FAPI = "https://fapi.binance.com/fapi/v1/klines"
UA = {"User-Agent": "hs-scan-15m/1.0"}
BAR_15 = pd.Timedelta(minutes=15)
BAR_1H = pd.Timedelta(hours=1)
RR_MIN = 1.5
# "刚破位"：官方入场之后最多再看 24 根已收盘 15m（6 小时）
ENTRY_FRESH_BARS = 24
KLINE_LIMIT_15M = 800
KLINE_LIMIT_1H = 300
NESTED_15M = True  # 2026-11-12: also look for nested (inner) H&S inside every right shoulder

FONT = None
for fp in (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
):
    if Path(fp).exists():
        FONT = font_manager.FontProperties(fname=fp)
        break

SESSION = requests.Session()
SESSION.headers.update(UA)


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Binance klines: fapi first, spot / OKX-perp fallback where fapi is blocked (binance_data.py)."""
    df = binance_data.fetch(symbol, interval, limit)
    if df is None or not len(df):
        return pd.DataFrame()
    df = df.copy()
    now_ms = int(now_utc().timestamp() * 1000)
    df = df[df["close_time"] < now_ms].copy()
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.reset_index(drop=True)


def fmt_px(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    ax = abs(float(x))
    if ax >= 1000:
        return f"{float(x):,.2f}"
    if ax >= 1:
        return f"{float(x):.4f}"
    return f"{float(x):.6f}"


def fmt_rr(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{float(x):.2f}"


def troughs(df: pd.DataFrame, L: int, Hh: int, R: int):
    low = df["low"].to_numpy()
    lt = L + int(np.argmin(low[L:Hh + 1]))
    rt = Hh + int(np.argmin(low[Hh:R + 1]))
    return lt, float(low[lt]), rt, float(low[rt])


# Neckline slope is display-only (not a filter, not used for sizing or exits).
# s2 = (RS trough − LS trough) / (head high − neckline at the head).
# Class band: 下斜 s2 < -0.10, 平 −0.10..+0.10 inclusive, 上翘 s2 > +0.10.
NECK_S2_FLAT = 0.10


def neck_class_cn(s2) -> str | None:
    if s2 is None:
        return None
    try:
        v = float(s2)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    if v < -NECK_S2_FLAT:
        return "下斜"
    if v > NECK_S2_FLAT:
        return "上翘"
    return "平"


def neck_s2(lt, y1, rt, y2, head_idx, head_high) -> float | None:
    """(RS trough − LS trough) / (head − neckline at head). None if height is not positive."""
    try:
        hgt = float(head_high) - float(neck_at(lt, y1, rt, y2, int(head_idx)))
        if not np.isfinite(hgt) or hgt <= 0:
            return None
        s2 = (float(y2) - float(y1)) / hgt
        return float(s2) if np.isfinite(s2) else None
    except Exception:
        return None


def attach_neck_slope(row: dict, df15: pd.DataFrame | None = None, head_high=None) -> dict:
    """Fill neck_s2 / neck_class_cn on the row. Never changes prices, stops, or targets."""
    s2 = None
    try:
        Hh = int(row["H"])
        if head_high is None and df15 is not None and 0 <= Hh < len(df15):
            head_high = float(df15["high"].iloc[Hh])
        s2 = neck_s2(row["lt"], row["y1"], row["rt"], row["y2"], Hh, head_high)
    except Exception:
        s2 = None
    row["neck_s2"] = None if s2 is None else round(float(s2), 4)
    row["neck_class_cn"] = neck_class_cn(s2)
    return row


def neck_info_text(s2, cls, ang) -> str:
    s2s = "—" if s2 is None or not np.isfinite(s2) else f"{float(s2):+.2f}"
    clss = cls or "—"
    if ang is None or not (isinstance(ang, (int, float)) and np.isfinite(ang)):
        angs = "—"
    else:
        angs = f"{float(ang):+.0f}°"
    return f"颈线 s2={s2s}（{clss}）｜图上约 {angs}"


def neck_visual_angle_deg(ax, lt, y1, rt, y2) -> float | None:
    """Visual neckline angle in degrees as drawn on this axes (gallery transData method)."""
    try:
        p1 = ax.transData.transform((float(lt), float(y1)))
        p2 = ax.transData.transform((float(rt), float(y2)))
        ang = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        return float(ang) if np.isfinite(ang) else None
    except Exception:
        return None



# Shoulder/head proportions are display-only (not a filter, not used for sizing or exits).
# Time spans reuse hs_v4_prop_core.prop_metrics, in bar counts:
#   head span = LS trough → RS trough (rt - lt)
#   LS span   = pre-LS trough p0 (lowest in [L-(H-L), L], the rise start / previous trough) → LS trough
#   RS span   = RS trough → first post-RS bar at/below the neckline (the following trough),
#               or R+1 when that trough is not yet in the window (same fallback as v4_check t=R+1).
# Height is the peak minus the neckline at that bar (相对颈线), then divided by head height.
# Bands: time 0.25–1.50 (BANDS s_lo/s_hi); height 0.30–1.00 (三栋: 0.3–1.0 左右).
PROP_T_LO, PROP_T_HI = 0.25, 1.50
PROP_H_LO, PROP_H_HI = 0.30, 1.00


def _finite_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _prop_pre_trough(low, L, H, high=None) -> int:
    """LS rally origin. 2026-11-14 (三栋 LINK 批注, v6 量法): walk left from LS tracking the
    lowest low and stop at the first counter-rally larger than half of the LS rally; window
    is at most H-L bars before LS. Without highs, falls back to the window's lowest low."""
    L, H = int(L), int(H)
    a0 = max(0, L - (H - L))
    if high is None:
        return a0 + int(np.argmin(low[a0:L + 1]))
    m, mp = float(low[L]), L
    for i in range(L - 1, a0 - 1, -1):
        if float(low[i]) < m:
            m, mp = float(low[i]), i
        elif float(high[i]) - m >= 0.5 * (float(high[L]) - m):
            break
    return mp


def _prop_post_trough(low, lt, y1, rt, y2, R, H, L) -> int:
    """Index of the RS following trough, or R+1 if it is not yet visible."""
    n = len(low)
    fallback = min(n - 1, int(R) + 1) if n else int(R)
    width = max(1, int(H) - int(L))
    end = min(n - 1, int(R) + width)
    for i in range(int(R) + 1, end + 1):
        try:
            nk = float(neck_at(lt, y1, rt, y2, i))
        except Exception:
            continue
        if not np.isfinite(nk):
            continue
        if float(low[i]) <= nk:
            return i
    return fallback


def shoulder_prop_metrics(high, low, L, H, R, lt, rt, y1, y2) -> dict:
    """Return time/height ratios. Missing pieces become None. Never raises."""
    out = dict(prop_ls_time=None, prop_rs_time=None, prop_ls_height=None, prop_rs_height=None)
    try:
        L, H, R, lt, rt = int(L), int(H), int(R), int(lt), int(rt)
        n = min(len(high), len(low))
        if not (0 <= min(L, H, R, lt, rt) and max(L, H, R, lt, rt) < n):
            return out
        # 2026-11-14 (三栋 LINK 批注, v6 量法, display only): each wave is measured leg to leg.
        # span: LS = rally origin p0 -> left trough lt; head = lt -> rt; RS = rt -> the fall
        # back to the neckline p3. Height = wave amplitude (rise+fall)/2, not height above
        # the neckline (which made shoulders look tiny when the neck sits near shoulder level).
        span_h = max(1, rt - lt)
        p0 = _prop_pre_trough(low, L, H, high)
        p3 = _prop_post_trough(low, lt, y1, rt, y2, R, H, L)
        span_l = max(1, lt - p0)
        span_r = max(1, p3 - rt)
        sl, sr = float(span_l / span_h), float(span_r / span_h)
        if np.isfinite(sl):
            out["prop_ls_time"] = sl
        if np.isfinite(sr):
            out["prop_rs_time"] = sr
        p3c = min(int(p3), n - 1)
        r_end = float(np.min(low[R:p3c + 1])) if p3c > R else float(low[rt])
        a_h = (2 * float(high[H]) - float(low[lt]) - float(low[rt])) / 2
        if np.isfinite(a_h) and a_h > 0:
            a_l = (2 * float(high[L]) - float(low[p0]) - float(low[lt])) / 2
            a_r = (2 * float(high[R]) - float(low[rt]) - r_end) / 2
            if np.isfinite(a_l) and a_l > 0:
                out["prop_ls_height"] = float(a_l / a_h)
            if np.isfinite(a_r) and a_r > 0:
                out["prop_rs_height"] = float(a_r / a_h)
    except Exception:
        pass
    return out


def _prop_issues(ls_t, rs_t, ls_h, rs_h):
    issues = []
    for label, val, lo, hi in (
        ("左肩跨度", ls_t, PROP_T_LO, PROP_T_HI),
        ("右肩跨度", rs_t, PROP_T_LO, PROP_T_HI),
        ("左肩高度", ls_h, PROP_H_LO, PROP_H_HI),
        ("右肩高度", rs_h, PROP_H_LO, PROP_H_HI),
    ):
        v = _finite_float(val)
        if v is None:
            continue
        if v < lo:
            issues.append((label, "lo"))
        elif v > hi:
            issues.append((label, "hi"))
    return issues


def prop_class_cn(ls_t, rs_t, ls_h, rs_h) -> str | None:
    issues = _prop_issues(ls_t, rs_t, ls_h, rs_h)
    measured = [v for v in (ls_t, rs_t, ls_h, rs_h) if _finite_float(v) is not None]
    if not measured:
        return None
    if not issues:
        return "比例协调"
    if all(k == "lo" for _, k in issues):
        return "比例偏小"
    return "比例失调"


def _fmt_mult(x) -> str:
    v = _finite_float(x)
    if v is None:
        return "—"
    return f"{v:.1f}×头"


def prop_info_text(ls_t, rs_t, ls_h, rs_h, cls=None) -> str:
    issues = _prop_issues(ls_t, rs_t, ls_h, rs_h)
    body = (
        f"左肩跨度{_fmt_mult(ls_t)}、右肩{_fmt_mult(rs_t)}；"
        f"左肩高度{_fmt_mult(ls_h)}、右肩{_fmt_mult(rs_h)}"
    )
    if not measured_any(ls_t, rs_t, ls_h, rs_h):
        return "肩头比例 —"
    if not issues:
        tail = "比例协调"
    elif all(k == "lo" for _, k in issues):
        tail = "、".join(n for n, _ in issues) + "偏小"
    else:
        tail = "、".join(n for n, _ in issues) + "失调"
    return f"{body}，{tail}"


def measured_any(*vals) -> bool:
    return any(_finite_float(v) is not None for v in vals)


def v6_pivot_note(row: dict, df15, high, low) -> str | None:
    """Display only (2026-11-14, 三栋 LINK 批注): where the research v6 picker
    (research/hs_v6_core.py) would put LS/RS for this head. Never changes the signal,
    entry, stop or targets. Returns None if v6 agrees or cannot be computed."""
    try:
        if df15 is None or "atr" not in df15.columns:
            return None
        rp = str(ROOT / "research")
        if rp not in sys.path:
            sys.path.append(rp)
        import hs_shoulder_v3_core as _V
        import hs_v6_core as _C6
        H, L0, R0 = int(row["H"]), int(row["L"]), int(row["R"])
        n = min(len(high), len(low))
        t = n - 1
        et = row.get("official_entry_time")
        if et:
            ts = pd.Timestamp(et)
            ts = ts.tz_localize("Asia/Shanghai") if ts.tzinfo is None else ts
            i0 = int(df15["dt"].searchsorted(ts.tz_convert("UTC"), side="left"))
            if 0 < i0 <= n:
                t = i0 - 1
        u = float(_V.unit_series(df15["atr"].to_numpy(dtype=float))[H])
        if not (np.isfinite(u) and u > 0):
            return None
        hi = np.asarray(high, dtype=float); lo = np.asarray(low, dtype=float)
        sw = _V.swing_highs(hi)
        nge = np.full(len(hi), len(hi) - 1, dtype=int); st = []
        for i in range(len(hi)):
            while st and hi[i] > hi[st[-1]]:
                nge[st.pop()] = i
            st.append(i)
        P = _C6.pick_ls(hi, lo, sw, nge, H, u)
        W = _C6.RightWalker(hi, lo, H, u)
        for j in range(H + 1, t):
            if hi[j] > hi[H]:
                break
            W.update(j)
        R6 = W.rs()
        tt = df15["dt"]
        parts = []
        if P is not None and P["L"] != L0:
            parts.append(f"左肩 {to_shanghai(tt.iloc[P['L']])[5:]} {fmt_px(hi[P['L']])}")
        if R6 is not None and R6 != R0 and R6 < t:
            parts.append(f"右肩 {to_shanghai(tt.iloc[R6])[5:]} {fmt_px(hi[R6])}")
        if not parts:
            return None
        return "v6认肩参考：" + "、".join(parts) + "（与规则选点不同，只作参考，入场止损按规则）"
    except Exception:
        return None


def attach_shoulder_prop(row: dict, df15: pd.DataFrame | None = None,
                         high=None, low=None) -> dict:
    """Fill prop_* fields on the row. Never changes prices, stops, or targets."""
    empty = dict(prop_ls_time=None, prop_rs_time=None, prop_ls_height=None,
                 prop_rs_height=None, prop_class_cn=None, prop_text="肩头比例 —")
    try:
        if high is None and df15 is not None and "high" in df15.columns:
            high = df15["high"].to_numpy()
        if low is None and df15 is not None and "low" in df15.columns:
            low = df15["low"].to_numpy()
        if high is None or low is None:
            row.update(empty)
            attach_chart_refs(row, df15, high=high, low=low)
            return row
        m = shoulder_prop_metrics(high, low, row["L"], row["H"], row["R"],
                                  row["lt"], row["rt"], row["y1"], row["y2"])
        def rnd(x):
            return None if x is None else round(float(x), 4)
        ls_t, rs_t = rnd(m["prop_ls_time"]), rnd(m["prop_rs_time"])
        ls_h, rs_h = rnd(m["prop_ls_height"]), rnd(m["prop_rs_height"])
        cls = prop_class_cn(ls_t, rs_t, ls_h, rs_h)
        row["prop_ls_time"] = ls_t
        row["prop_rs_time"] = rs_t
        row["prop_ls_height"] = ls_h
        row["prop_rs_height"] = rs_h
        row["prop_class_cn"] = cls
        row["prop_text"] = prop_info_text(ls_t, rs_t, ls_h, rs_h, cls)
        note = v6_pivot_note(row, df15, high, low)
        row["prop_v6_note"] = note
        if note:
            row["prop_text"] = row["prop_text"] + "；" + note
    except Exception:
        row.update(empty)
    attach_chart_refs(row, df15, high=high, low=low)
    return row


# Fibonacci trend-extension + dual neckline are display-only (not a filter, not used
# for sizing or exits). Definition matches research/tp/tp_sim.py / results_tp_mgmt.md:
#   A=head high, B=right-trough low (lowest low between H and RS), C=RS high;
#   fib_k = C - k*(A-B), k in {1.0, 1.272, 1.618, 2.0}.
FIB_EXT_RATIOS = (("1.0", 1.0), ("1.272", 1.272), ("1.618", 1.618), ("2.0", 2.0))


def _finite_px(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def fib_ext_prices(head_high, trough_low, rs_high) -> dict | None:
    """Wu-Lang fib extension: RS_high - k*(H_high - right_trough_low). Display only."""
    try:
        a, b, c = float(head_high), float(trough_low), float(rs_high)
        span = a - b
        if not (np.isfinite(span) and np.isfinite(c) and span > 0):
            return None
        out = {}
        for name, k in FIB_EXT_RATIOS:
            px = c - k * span
            if not np.isfinite(px):
                return None
            out[name] = float(px)
        return out
    except (TypeError, ValueError):
        return None


def fib_info_text(ext: dict | None) -> str:
    if not ext:
        return "斐波那契扩展（只作参考）：—"
    parts = [f"{k}={fmt_px(ext[k])}" for k, _ in FIB_EXT_RATIOS if k in ext]
    return "斐波那契扩展（只作参考）：" + "｜".join(parts)


def attach_fib_ext(row: dict, df15: pd.DataFrame | None = None,
                   high=None, low=None) -> dict:
    """Fill fib_ext / fib_text. Never changes prices, stops, targets, or status."""
    ext = None
    try:
        if high is None and df15 is not None and "high" in getattr(df15, "columns", []):
            high = df15["high"].to_numpy()
        if low is None and df15 is not None and "low" in getattr(df15, "columns", []):
            low = df15["low"].to_numpy()
        Hh, R = int(row["H"]), int(row["R"])
        head_high = float(high[Hh]) if high is not None else None
        rs_high = float(high[R]) if high is not None else None
        y2 = row.get("y2")
        if y2 is None and low is not None:
            rt = row.get("rt")
            y2 = float(low[int(rt)]) if rt is not None else float(np.min(low[Hh:R + 1]))
        ext = fib_ext_prices(head_high, y2, rs_high)
    except Exception:
        ext = None
    row["fib_ext"] = ext
    row["fib_text"] = fib_info_text(ext)
    return row


def sloped_neck_now(row: dict) -> float | None:
    """Sloped neckline level already on the row (neck, else break_level, else neck at rt)."""
    for key in ("neck", "break_level"):
        v = _finite_px(row.get(key))
        if v is not None:
            return v
    try:
        return float(neck_at(row["lt"], row["y1"], row["rt"], row["y2"], int(row["rt"])))
    except Exception:
        return None


def attach_dual_neck(row: dict) -> dict:
    """Fill dual_neck_* fields. Meaningful only when neck_class_cn is 上翘/下斜.

    Horizontal line = the trough price will hit first on the way down:
      up-slope (y2>y1): horiz=y1 (lower); sloped breaks first, so tier1=sloped, tier2=y1.
      down-slope (y2<y1): horiz=y2 (higher) breaks first, so tier1=y2, tier2=sloped.
    Near-flat (the NECK_S2_FLAT band / class 平): tiers coincide, text says so.
    """
    try:
        y1, y2 = _finite_px(row.get("y1")), _finite_px(row.get("y2"))
        sloped = sloped_neck_now(row)
        cls = row.get("neck_class_cn")
        if cls is None and row.get("neck_s2") is not None:
            cls = neck_class_cn(row.get("neck_s2"))
        if y1 is None or y2 is None or sloped is None:
            row["dual_neck_tier1"] = None
            row["dual_neck_tier2"] = None
            row["dual_neck_text"] = "双颈线（只作参考）：—"
            return row
        if cls not in ("上翘", "下斜"):
            row["dual_neck_tier1"] = sloped
            row["dual_neck_tier2"] = sloped
            row["dual_neck_text"] = "双颈线（只作参考）：颈线近乎水平，双颈线基本重合"
            return row
        if cls == "上翘":
            t1, t2 = sloped, y1
            t1n, t2n = "斜颈线", "水平颈线"
            slope_word = "上翘"
        else:
            t1, t2 = y2, sloped
            t1n, t2n = "水平颈线", "斜颈线"
            slope_word = "下翘"
        ang = row.get("neck_angle_deg")
        if isinstance(ang, (int, float)) and np.isfinite(ang):
            slope_part = f"颈线{slope_word}约{float(ang):+.0f}°"
        else:
            slope_part = f"颈线{slope_word}"
        row["dual_neck_tier1"] = float(t1)
        row["dual_neck_tier2"] = float(t2)
        row["dual_neck_text"] = (
            f"双颈线（只作参考）：{slope_part}，"
            f"第一档 {t1n} {fmt_px(t1)} 先轻仓，"
            f"第二档 {t2n} {fmt_px(t2)} 破了再加仓，止损不变"
        )
        return row
    except Exception:
        row["dual_neck_tier1"] = None
        row["dual_neck_tier2"] = None
        row["dual_neck_text"] = "双颈线（只作参考）：—"
        return row


def attach_chart_refs(row: dict, df15: pd.DataFrame | None = None,
                      high=None, low=None) -> dict:
    """Fibonacci extensions + dual neckline. Display only; never changes trade fields."""
    attach_fib_ext(row, df15, high=high, low=low)
    attach_dual_neck(row)
    return row


def collect_ac_signals(symbol, df15, df1, recs=None):
    if recs is None:
        recs = scan_line(symbol, df15, BAR_15, df1, BAR_1H)
    out = []
    for rec in recs:
        sig = rec.get("sig_breakout")
        if sig is None or not rec.get("q_breakout"):
            continue
        if not measured_ok(sig):
            continue
        out.append(sig)
    return out


def live_entries(symbol, df15, df1, sigs, now: pd.Timestamp,
                 status: str = "可入场", extras: list[dict] | None = None) -> list[dict]:
    """Live gate shared by formal (可入场) and near-miss (观察) signals.

    Same freshness / price-between-stop-and-T1 / R:R>=1.5 / 1H veto for both tiers.
    """
    if not len(df15):
        return []
    h15 = df15["high"].to_numpy()
    l15 = df15["low"].to_numpy()
    c15 = df15["close"].to_numpy()
    t15 = df15["dt"]
    last_ts = _ts(t15.iloc[-1])
    cur = float(c15[-1])
    found = []
    for k, sig in enumerate(sigs):
        et = _ts(sig.entry_time)
        # freshness: official entry not older than ENTRY_FRESH_BARS, or still the last bar
        age_bars = (last_ts - et) / BAR_15
        if age_bars > ENTRY_FRESH_BARS:
            continue
        if not (sig.stop > cur > sig.target):
            continue
        i0 = int(df15["dt"].searchsorted(et, side="left"))
        if i0 < len(df15):
            if float(np.max(h15[i0:])) >= sig.stop:
                continue
            if float(np.min(l15[i0:])) <= sig.target:
                continue
        rr_now = (cur - sig.target) / (sig.stop - cur) if sig.stop > cur else 0.0
        if rr_now < RR_MIN:
            continue
        if htf_bullish_at(df1, now, BAR_1H):
            continue
        lt, y1, rt, y2 = troughs(df15, sig.L, sig.H, sig.R)
        suggest = float(sig.entry_px) if age_bars <= 0 else cur
        row = {
            "symbol": symbol,
            "status": status,
            "tier": TIER_WATCH if status == STATUS_OBSERVE else TIER_FORMAL,
            "suggested_risk_pct": SUGGESTED_RISK_PCT if status == STATUS_OBSERVE else FORMAL_RISK_PCT,
            "L": int(sig.L), "H": int(sig.H), "R": int(sig.R),
            "lt": lt, "y1": y1, "rt": rt, "y2": y2,
            "L_time": to_shanghai(df15["dt"].iloc[sig.L]),
            "H_time": to_shanghai(df15["dt"].iloc[sig.H]),
            "R_time": to_shanghai(df15["dt"].iloc[sig.R]),
            "neck": float(sig.neck),
            "suggest_entry": suggest,
            "official_entry": float(sig.entry_px),
            "official_entry_time": to_shanghai(et),
            "stop": float(sig.stop),
            "target_1h": float(sig.target),
            "target_15h": float(sig.target15),
            "current": cur,
            "rr": float(rr_now),
            "dist_stop_pct": 100.0 * (sig.stop - cur) / cur,
            "dist_t1_pct": 100.0 * (cur - sig.target) / cur,
            "note": (
                "破位下一根已开盘/刚收盘，官方入场价可用。"
                if age_bars <= 1 else
                "官方入场已过若干根，价格仍夹在止损与 1.0H 之间，结构未死。"
            ),
        }
        if extras is not None:
            row.update(extras[k])
            if status == STATUS_OBSERVE:
                row["note"] = (
                    f"观察档（未验证）：差一条——{extras[k].get('missed_text', '')}。建议风险 0.5%，自行判断。"
                    + row["note"]
                )
        attach_neck_slope(row, df15)
        attach_shoulder_prop(row, df15)
        found.append(row)
    return found


def live_observe(symbol, df15, df1, recs, now: pd.Timestamp) -> list[dict]:
    """Near-miss『观察』rows (hs_nearmiss_meta.json). Not validated; suggested risk 0.5%."""
    nm = scan_nearmiss(symbol, df15, BAR_15, df1, BAR_1H, recs=recs)
    if not nm:
        return []
    sigs = [w["sig"] for w in nm]
    extras = [{
        "missed_condition": w["miss"],
        "missed_condition_cn": w["miss_cn"],
        "missed_value": round(float(w["value"]), 4),
        "missed_by": None if w["miss_amount"] is None else round(float(w["miss_amount"]), 4),
        "missed_text": w["miss_text"],
    } for w in nm]
    return live_entries(symbol, df15, df1, sigs, now, status=STATUS_OBSERVE, extras=extras)


def _break_level_of(sig, df15) -> float | None:
    """Neckline break level at the official break bar (limit price for a retest)."""
    try:
        lt, y1, rt, y2 = troughs(df15, sig.L, sig.H, sig.R)
        atrv = df15["atr"].to_numpy(dtype=float)
        t = int(sig.break_i)
        if t < 0 or t >= len(atrv) or not np.isfinite(atrv[t]):
            return None
        _, _, _, _, brk, _, _ = levels_at(lt, y1, rt, y2, t, atrv[t])
        return float(brk)
    except Exception:
        return None


def collect_lowrr_breakouts(recs) -> list:
    """Formal A+C breaks that exist but fail measured R:R at the next-bar open (silent-drop case)."""
    out = []
    for rec in recs or []:
        sig = rec.get("sig_breakout")
        if sig is None or not rec.get("q_breakout"):
            continue
        if measured_ok(sig):
            continue
        out.append(sig)
    return out


def live_retest(symbol, df15, df1, sigs, now: pd.Timestamp,
                extras: list[dict] | None = None, *,
                tier: str = TIER_FORMAL, suggested_risk_pct: float = FORMAL_RISK_PCT) -> list[dict]:
    """Post-break row when close/next-open R:R < 1.5 but R:R at the break level is still >= 1.5.

    Limit entry = break level. Same freshness window and stop/T1 invalidation as live_entries.
    Does not require current-price R:R >= 1.5, and does not drop if price already tagged brk
    (that would have filled the limit). Tier/risk stay those of the underlying pattern.
    """
    if not len(df15):
        return []
    h15 = df15["high"].to_numpy()
    l15 = df15["low"].to_numpy()
    c15 = df15["close"].to_numpy()
    t15 = df15["dt"]
    last_ts = _ts(t15.iloc[-1])
    cur = float(c15[-1])
    found = []
    for k, sig in enumerate(sigs):
        et = _ts(sig.entry_time)
        age_bars = (last_ts - et) / BAR_15
        if age_bars > ENTRY_FRESH_BARS:
            continue
        brk = _break_level_of(sig, df15)
        if brk is None:
            continue
        stop, t1, t2 = float(sig.stop), float(sig.target), float(sig.target15)
        if not (stop > brk > t1 and t2 > 0):
            continue
        rr_brk = (brk - t1) / (stop - brk)
        if rr_brk < RR_MIN:
            continue
        i0 = int(getattr(sig, "break_i", -1))
        if i0 < 0:
            i0 = int(df15["dt"].searchsorted(et, side="left"))
        if i0 < len(df15):
            if float(np.max(h15[i0:])) >= stop or float(np.min(l15[i0:])) <= t1:
                continue
        if htf_bullish_at(df1, now, BAR_1H):
            continue
        lt, y1, rt, y2 = troughs(df15, sig.L, sig.H, sig.R)
        valid_until = et + ENTRY_FRESH_BARS * BAR_15
        try:
            break_sh = to_shanghai(_ts(t15.iloc[int(sig.break_i)]))
        except Exception:
            break_sh = None
        row = {
            "symbol": symbol,
            "status": STATUS_RETEST,
            "tier": tier,
            "suggested_risk_pct": suggested_risk_pct,
            "L": int(sig.L), "H": int(sig.H), "R": int(sig.R),
            "lt": lt, "y1": y1, "rt": rt, "y2": y2,
            "L_time": to_shanghai(df15["dt"].iloc[sig.L]),
            "H_time": to_shanghai(df15["dt"].iloc[sig.H]),
            "R_time": to_shanghai(df15["dt"].iloc[sig.R]),
            "neck": float(sig.neck),
            "break_level": float(brk),
            "break_bar_shanghai": break_sh,
            "suggest_entry": float(brk),
            "official_entry": float(brk),
            "official_entry_time": to_shanghai(et),
            "stop": stop,
            "target_1h": t1,
            "target_15h": t2,
            "current": cur,
            "rr": float(rr_brk),
            "dist_stop_pct": 100.0 * (stop - brk) / brk,
            "dist_t1_pct": 100.0 * (brk - t1) / brk,
            "valid_until_shanghai": to_shanghai(valid_until),
            "note": (
                f"15m 已收盘跌破颈线，但按收盘价追的盈亏比不足 1.5；在 {fmt_px(brk)} 附近挂限价空单等回踩，"
                f"有效至 {to_shanghai(valid_until)}，过期撤单。"
            ),
        }
        if extras is not None:
            row.update(extras[k])
            if tier == TIER_WATCH:
                row["note"] = (
                    f"观察档（未验证）：差一条——{extras[k].get('missed_text', '')}。"
                    f"建议风险 0.5%，自行判断。" + row["note"]
                )
        attach_neck_slope(row, df15)
        attach_shoulder_prop(row, df15)
        found.append(row)
    return found


def live_observe_retest(symbol, df15, df1, recs, now: pd.Timestamp) -> list[dict]:
    """Near-miss breaks that fail measured R:R: same retest row, observe tier / 0.5% risk."""
    nm = scan_nearmiss(symbol, df15, BAR_15, df1, BAR_1H, recs=recs, rr_mode="low")
    if not nm:
        return []
    sigs = [w["sig"] for w in nm]
    extras = [{
        "missed_condition": w["miss"],
        "missed_condition_cn": w["miss_cn"],
        "missed_value": round(float(w["value"]), 4),
        "missed_by": None if w["miss_amount"] is None else round(float(w["miss_amount"]), 4),
        "missed_text": w["miss_text"],
    } for w in nm]
    return live_retest(symbol, df15, df1, sigs, now, extras=extras,
                       tier=TIER_WATCH, suggested_risk_pct=SUGGESTED_RISK_PCT)


def merge_tiers(entries: list[dict], observes: list[dict]) -> list[dict]:
    """Formal keeps priority: drop an 观察/观察回踩 row if the same symbol already has a formal row."""
    formal_syms = {r["symbol"] for r in entries}
    return [r for r in observes if r["symbol"] not in formal_syms]


def watch_to_row(w: dict, df15: pd.DataFrame) -> dict:
    lt, y1, rt, y2 = troughs(df15, w["L"], w["H"], w["R"])
    atr_r = float(df15["atr"].to_numpy()[w["R"]])
    stop_est = float(w["rs_high"]) + 0.5 * atr_r if np.isfinite(atr_r) else None
    cur = float(w["current"])
    rr = None
    if stop_est is not None and stop_est > cur > w["target"]:
        rr = (cur - w["target"]) / (stop_est - cur)
    row = {
        "symbol": w["symbol"],
        "status": "观望",
        "tier": TIER_FORMAL,
        "suggested_risk_pct": FORMAL_RISK_PCT,
        "L": int(w["L"]), "H": int(w["H"]), "R": int(w["R"]),
        "lt": lt, "y1": y1, "rt": rt, "y2": y2,
        "L_time": w["L_time"], "H_time": w["H_time"], "R_time": w["R_time"],
        "neck": float(w["neck"]),
        "suggest_entry": None,
        "official_entry": None,
        "official_entry_time": None,
        "stop": stop_est,
        "target_1h": float(w["target"]),
        "target_15h": float(w["target15"]),
        "current": cur,
        "rr": rr,
        "dist_stop_pct": None if stop_est is None else 100.0 * (stop_est - cur) / cur,
        "dist_t1_pct": 100.0 * (cur - w["target"]) / cur if w["target"] else None,
        "note": f"右肩已确认，距破位等待窗结束还剩 {w['bars_left']} 根 15m。等待有效跌破后下一根开盘，不要提前空。",
    }
    attach_neck_slope(row, df15)
    return attach_shoulder_prop(row, df15)


# ---------------------------------------------------------------------------
# Chart-only indicator confirmation (cosmetic). Does NOT affect signal selection,
# the webhook payload, or the journal. 2026-11-12, per 三栋: shoulders/head are
# defined by PRICE only (the scanner's own LS/H/RS); RSI / MACD hist / volume are
# read AT those pivots to confirm or show divergence -- never to find pivots.
MOM_WIN = 2          # indicator value = max within +/-2 bars of the price pivot (indicators lag a little)


def pivot_confirmation(df15: pd.DataFrame, L: int, Hh: int, R: int) -> dict:
    """Indicator readings at the scanner's own LS / H / RS pivots.

    rsi / hist: max of RSI14 / MACD hist(12,26,9) within +/-MOM_WIN bars of the pivot.
    vol: 3-bar mean volume around the pivot (same vol_part the scanner uses).
    Divergence at head vs LS (head price is higher by construction): RSI / hist / vol lower than at LS.
    vol_stepdown: LS > H > RS.  Values may be None when data is missing. Never raises.
    """
    n = len(df15)
    out = {"pivots": {}, "div": {"rsi": None, "hist": None, "vol": None}, "vol_stepdown": None}
    try:
        def col(name):
            return df15[name].to_numpy(dtype=float) if name in df15 else np.full(n, np.nan)
        rsi, mh, vol, hi = col("rsi"), col("macd_hist"), col("volume"), col("high")

        def pk(arr, p):
            seg = arr[max(0, p - MOM_WIN): min(n, p + MOM_WIN + 1)]
            seg = seg[np.isfinite(seg)]
            return float(seg.max()) if len(seg) else None

        def vp(p):
            seg = vol[max(0, p - 1): min(n, p + 2)]
            seg = seg[np.isfinite(seg)]
            return float(seg.mean()) if len(seg) else None

        for k, p in (("LS", L), ("H", Hh), ("RS", R)):
            p = int(p)
            ok = 0 <= p < n
            out["pivots"][k] = {"i": p, "price": float(hi[p]) if ok else None,
                                "rsi": pk(rsi, p) if ok else None, "hist": pk(mh, p) if ok else None,
                                "vol": vp(p) if ok else None}
        P = out["pivots"]
        for key in ("rsi", "hist", "vol"):
            a, b = P["LS"][key], P["H"][key]
            if a is not None and b is not None:
                out["div"][key] = bool(b < a)
        vl, vh, vr = P["LS"]["vol"], P["H"]["vol"], P["RS"]["vol"]
        if None not in (vl, vh, vr):
            out["vol_stepdown"] = bool(vl > vh > vr)
    except Exception as e:  # cosmetic only
        print(f"  pivot confirmation skipped: {type(e).__name__} {e}", flush=True)
    return out


def confirmation_text(conf: dict) -> str:
    """One-line Chinese summary, e.g. '头部背离(对左肩): RSI是 柱是 量否 | 量能递减 LS>H>RS: 否'."""
    def yn(v):
        return "?" if v is None else ("是" if v else "否")
    d = conf.get("div", {})
    return (f"头部背离(对左肩): RSI{yn(d.get('rsi'))} 柱{yn(d.get('hist'))} 量{yn(d.get('vol'))}"
            f" | 量能递减 LS>H>RS: {yn(conf.get('vol_stepdown'))}")


def _draw_momentum(ax, ax_rsi, ax_mh, df15, idx, row, h):
    """Draw RSI / MACD-hist panels; annotate values only at the scanner's LS/H/RS. Never raises."""
    try:
        n = len(df15)
        rsi = df15["rsi"].to_numpy(dtype=float) if "rsi" in df15 else np.full(n, np.nan)
        mh = df15["macd_hist"].to_numpy(dtype=float) if "macd_hist" in df15 else np.full(n, np.nan)
        ax_rsi.plot(idx, rsi[idx], color="#6a4c93", lw=1.0)
        for lv, ls in ((70, ":"), (50, "-"), (30, ":")):
            ax_rsi.axhline(lv, color="grey", lw=0.6, ls=ls)
        ax_rsi.set_ylabel("RSI14", fontsize=8)
        ax_rsi.grid(True, alpha=0.2)
        mhi = np.nan_to_num(mh[idx])
        ax_mh.bar(idx, mhi, color=np.where(mhi >= 0, "#2a9d8f", "#c1121f"), width=0.8)
        ax_mh.axhline(0, color="grey", lw=0.6)
        if "macd" in df15.columns:
            ax_mh.plot(idx, df15["macd"].to_numpy(dtype=float)[idx],
                       color="#1d3557", lw=1.0, label="DIF")
        if "macd_sig" in df15.columns:
            ax_mh.plot(idx, df15["macd_sig"].to_numpy(dtype=float)[idx],
                       color="#e76f51", lw=1.0, label="DEA")
        ax_mh.set_ylabel("MACD", fontproperties=FONT, fontsize=8)
        ax_mh.grid(True, alpha=0.2)
        if "macd" in df15.columns or "macd_sig" in df15.columns:
            ax_mh.legend(fontsize=7, loc="upper left")
        conf = pivot_confirmation(df15, int(row["L"]), int(row["H"]), int(row["R"]))
        cols = {"LS": "#457b9d", "H": "#e9c46a", "RS": "#e76f51"}
        for k, pv in conf["pivots"].items():
            p = pv["i"]
            if not (idx[0] <= p <= idx[-1]):
                continue
            if pv["rsi"] is not None:
                ax_rsi.scatter([p], [pv["rsi"]], s=22, color=cols[k], zorder=5, edgecolors="k", linewidths=0.3)
                ax_rsi.annotate(f"{k} {pv['rsi']:.1f}", (p, pv["rsi"]), xytext=(0, 5), textcoords="offset points",
                                ha="center", fontsize=7, color="k")
            if pv["hist"] is not None:
                ax_mh.scatter([p], [pv["hist"]], s=22, color=cols[k], zorder=5, edgecolors="k", linewidths=0.3)
                ax_mh.annotate(f"{k} {pv['hist']:.3g}", (p, pv["hist"]), xytext=(0, 5), textcoords="offset points",
                               ha="center", fontsize=7, color="k")
        # divergence lines LS->H on indicator panels (red = divergence, grey = none)
        P = conf["pivots"]
        for axx, key in ((ax_rsi, "rsi"), (ax_mh, "hist")):
            a, b = P["LS"][key], P["H"][key]
            if a is not None and b is not None and idx[0] <= P["LS"]["i"] and P["H"]["i"] <= idx[-1]:
                axx.plot([P["LS"]["i"], P["H"]["i"]], [a, b], color="#c1121f" if b < a else "grey", lw=0.9, ls="--")
        ax.text(0.01, 0.02, "确认(只读指标,不定肩): " + confirmation_text(conf), transform=ax.transAxes,
                fontproperties=FONT, fontsize=8.5, color="#333",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="#bbb", lw=0.5), zorder=7)
        return conf
    except Exception as e:  # cosmetic only: never block an alert
        print(f"  momentum panel skipped: {type(e).__name__} {e}", flush=True)
        return None


BB_N, BB_K = 20, 2.0   # Bollinger(20,2) on close -- chart reference only (三栋: 中轨是参考)
VOL_MA_N = 20


def bollinger(df15: pd.DataFrame, n: int = BB_N, k: float = BB_K):
    """(mid, upper, lower) arrays; NaN where not enough data. Never raises."""
    c = pd.Series(df15["close"].to_numpy(dtype=float)) if "close" in df15 else pd.Series(np.full(len(df15), np.nan))
    mid = c.rolling(n, min_periods=n).mean()
    sd = c.rolling(n, min_periods=n).std(ddof=0)
    return mid.to_numpy(), (mid + k * sd).to_numpy(), (mid - k * sd).to_numpy()


def _draw_bollinger(ax, df15, idx):
    """Bollinger mid (solid) + bands (light) on the price panel. Drawing only; never raises."""
    try:
        mid, up, lo = bollinger(df15)
        ax.fill_between(idx, lo[idx], up[idx], color="#8d99ae", alpha=0.08, lw=0, zorder=0)
        ax.plot(idx, up[idx], color="#8d99ae", lw=0.7, ls="--", alpha=0.7, zorder=1)
        ax.plot(idx, lo[idx], color="#8d99ae", lw=0.7, ls="--", alpha=0.7, zorder=1)
        ax.plot(idx, mid[idx], color="#f4a261", lw=1.1, zorder=1, label="布林中轨 BB20")
    except Exception as e:  # cosmetic only
        print(f"  bollinger skipped: {type(e).__name__} {e}", flush=True)


def _draw_volume(ax_vol, df15, idx, row):
    """Volume bars (green up / red down) + MA20; marks LS/H/RS 3-bar mean volume. Drawing only; never raises."""
    try:
        n = len(df15)
        v = df15["volume"].to_numpy(dtype=float) if "volume" in df15 else np.full(n, np.nan)
        o = df15["open"].to_numpy(dtype=float); c = df15["close"].to_numpy(dtype=float)
        vi = np.nan_to_num(v[idx])
        ax_vol.bar(idx, vi, color=np.where(c[idx] >= o[idx], "#2a9d8f", "#c1121f"), width=0.8, alpha=0.75)
        vma = pd.Series(v).rolling(VOL_MA_N, min_periods=VOL_MA_N).mean().to_numpy()
        ax_vol.plot(idx, vma[idx], color="#1d3557", lw=1.0, label=f"MA{VOL_MA_N}")
        cols = {"LS": "#457b9d", "H": "#e9c46a", "RS": "#e76f51"}
        for k, key in (("LS", "L"), ("H", "H"), ("RS", "R")):
            p = int(row[key])
            if idx[0] <= p <= idx[-1]:
                seg = v[max(0, p - 1): min(n, p + 2)]
                seg = seg[np.isfinite(seg)]
                if len(seg):
                    ax_vol.scatter([p], [float(seg.mean())], s=22, color=cols[k], zorder=5, edgecolors="k", linewidths=0.3)
                    ax_vol.annotate(k, (p, float(seg.mean())), xytext=(0, 4), textcoords="offset points",
                                    ha="center", fontsize=7)
        ax_vol.set_ylabel("成交量", fontproperties=FONT, fontsize=8)
        ax_vol.grid(True, alpha=0.2)
        ax_vol.legend(fontsize=7, loc="upper left")
    except Exception as e:  # cosmetic only
        print(f"  volume panel skipped: {type(e).__name__} {e}", flush=True)


def _draw_horizontal_neck(ax, row, right):
    """Dashed horizontal neckline at the first-hit trough. Drawing only; never raises."""
    try:
        y1, y2 = _finite_px(row.get("y1")), _finite_px(row.get("y2"))
        if y1 is None or y2 is None:
            return
        if y2 > y1:
            horiz, x0 = y1, int(row["lt"])
        else:
            horiz, x0 = y2, int(row.get("rt", row["lt"]))
        ax.plot([x0, right], [horiz, horiz], color="#3a86ff", ls="--", lw=1.15,
                zorder=4, label="水平颈线")
    except Exception as e:
        print(f"  horizontal neck skipped: {type(e).__name__} {e}", flush=True)


def _draw_fib_ext(ax, row, left, right, R):
    """Thin dashed fib extension lines on the right of the price panel. Drawing only."""
    try:
        ext = row.get("fib_ext")
        if not ext:
            return
        x0 = int(max(int(R), left + 0.62 * (right - left)))
        y0, y1_ = ax.get_ylim()
        lo_lim = y0 - 0.8 * (y1_ - y0)
        color = "#6d597a"
        for name, _k in FIB_EXT_RATIOS:
            px = ext.get(name)
            if px is None or not np.isfinite(float(px)) or float(px) < lo_lim:
                continue
            px = float(px)
            ax.plot([x0, right], [px, px], color=color, ls="--", lw=0.8, zorder=1)
            ax.annotate(
                f"Fib {name} {fmt_px(px)}", (right, px),
                xytext=(-2, 2), textcoords="offset points",
                ha="right", va="bottom", fontsize=7, color=color,
                fontproperties=FONT, zorder=6,
            )
    except Exception as e:
        print(f"  fib ext skipped: {type(e).__name__} {e}", flush=True)



CHART_IDX_KEYS = ("L", "H", "R", "lt", "rt")


def _json_safe(v):
    return json.loads(json.dumps(v, ensure_ascii=False, default=str))


def chart_spec(df: pd.DataFrame, row: dict, tf: str = "15m", title_extra: str = "", tf_label: str = "15m") -> dict:
    """Everything render_from_payload.py needs to redraw this chart on another host.

    Bar indices are stored as bar open times (UTC ms) so a re-fetched frame can be re-aligned.
    """
    if "open_time" in df.columns:
        ot = df["open_time"].to_numpy(dtype="int64")
    else:
        ot = (pd.to_datetime(df["dt"], utc=True).astype("int64") // 1_000_000).to_numpy()

    def t(i):
        return int(ot[int(i)])

    spec = {
        "v": 1, "tf": tf, "tf_label": tf_label, "title_extra": title_extra,
        "n_bars": int(len(df)), "first_open_ms": t(0), "last_open_ms": t(len(df) - 1),
        "idx_ms": {k: t(row[k]) for k in CHART_IDX_KEYS},
        "y1": float(row["y1"]), "y2": float(row["y2"]),
    }
    other = row.get("_nested_other")
    if other:
        spec["other_idx_ms"] = {k: t(other[k]) for k in CHART_IDX_KEYS}
        spec["other_y"] = [float(other["y1"]), float(other["y2"])]
    spec["row"] = _json_safe({k: v for k, v in row.items()
                              if k not in CHART_IDX_KEYS + ("y1", "y2", "chart", "chart_spec")
                              and not k.startswith("_")})
    return spec


def plot_setup(df15: pd.DataFrame, row: dict, out_path: Path, title_extra: str = "", tf_label: str = "15m"):
    n = len(df15)
    L, Hh, R = row["L"], row["H"], row["R"]
    other = row.get("_nested_other")  # 2026-11-12 嵌套头肩: the other (outer/inner) pattern
    role = row.get("nested_role") if other else None
    left = max(0, min(L, other["L"]) - 12) if other else max(0, L - 12)
    right = min(n - 1, max(R + 16, n - 1))
    # keep window readable
    if right - left > (260 if other else 160):
        left = max(0, (min(L, other["L"]) - 4) if other else R - 120)
        if right - left > 300:
            left = max(0, R - 200)
    sl = slice(left, right + 1)
    idx = np.arange(left, right + 1)
    o = df15["open"].to_numpy()
    h = df15["high"].to_numpy()
    l = df15["low"].to_numpy()
    c = df15["close"].to_numpy()
    times = df15["dt"]

    fig, (ax, ax_vol, ax_rsi, ax_mh) = plt.subplots(
        4, 1, figsize=(13, 10.6), dpi=130, sharex=True,
        gridspec_kw={"height_ratios": [3.4, 0.95, 1.0, 1.0]},
    )
    _draw_bollinger(ax, df15, idx)
    for i in idx:
        up = c[i] >= o[i]
        col = "#2a9d8f" if up else "#c1121f"
        ax.vlines(i, l[i], h[i], color=col, lw=0.9, zorder=2)
        body_lo, body_hi = min(o[i], c[i]), max(o[i], c[i])
        height = max(body_hi - body_lo, (h[i] - l[i]) * 0.002 + 1e-12)
        ax.add_patch(Rectangle(
            (i - 0.32, body_lo), 0.64, height,
            facecolor=col, edgecolor=col, lw=0.4, zorder=3,
        ))

    # neckline through troughs, extended
    lt, y1, rt, y2 = row["lt"], row["y1"], row["rt"], row["y2"]
    xs = np.array([lt, rt, right], dtype=float)
    ys = np.array([
        neck_at(lt, y1, rt, y2, lt),
        neck_at(lt, y1, rt, y2, rt),
        neck_at(lt, y1, rt, y2, right),
    ])
    small = role == "inner"
    ax.plot(xs, ys, color=(NESTED_COLOR if small else "#1d3557"), lw=1.6, ls="-", zorder=4,
            label=("小颈线 (inner)" if small else "Neckline"))

    if small:
        peaks = [(L, float(h[L]), "小LS", NESTED_COLOR), (Hh, float(h[Hh]), "小H", NESTED_COLOR),
                 (R, float(h[R]), "小RS", NESTED_COLOR)]
    else:
        peaks = [
            (L, float(h[L]), "LS 左肩", "#457b9d"),
            (Hh, float(h[Hh]), "H 头部", "#e9c46a"),
            (R, float(h[R]), "RS 右肩", "#e76f51"),
        ]
    if other:
        ol, oh, orr = int(other["L"]), int(other["H"]), int(other["R"])
        olt, oy1, ort, oy2 = other["lt"], other["y1"], other["rt"], other["y2"]
        oxs = np.array([olt, ort, right], dtype=float)
        oys = np.array([neck_at(olt, oy1, ort, oy2, x) for x in oxs])
        if small:  # other = outer (big) pattern in the original colours
            ax.plot(oxs, oys, color="#1d3557", lw=1.4, ls="-", zorder=4, label="大颈线 (outer)")
            peaks += [(ol, float(h[ol]), "LS 左肩", "#457b9d"), (oh, float(h[oh]), "H 头部", "#e9c46a")]
            if orr != Hh:
                peaks.append((orr, float(h[orr]), "RS 右肩", "#e76f51"))
            else:
                peaks[1] = (Hh, float(h[Hh]), "小H = RS 右肩", NESTED_COLOR)
        else:      # other = inner (small) pattern in the nested colour
            ax.plot(oxs, oys, color=NESTED_COLOR, lw=1.3, ls="-", zorder=4, label="小颈线 (inner)")
            peaks.append((ol, float(h[ol]), "小LS", NESTED_COLOR))
            peaks.append((orr, float(h[orr]), "小RS", NESTED_COLOR))
            if oh == R:
                peaks[2] = (R, float(h[R]), "RS 右肩 = 小H", "#e76f51")
            else:
                peaks.append((oh, float(h[oh]), "小H", NESTED_COLOR))
    for x, y, lab, col in peaks:
        ax.scatter([x], [y], s=70, color=col, zorder=5, edgecolors="k", linewidths=0.4)
        ax.annotate(
            lab, (x, y), xytext=(0, 10), textcoords="offset points",
            ha="center", color=col, fontsize=9,
            fontproperties=FONT, zorder=6,
        )

    def hline(val, color, ls, label):
        if val is None or not np.isfinite(val):
            return
        ax.axhline(val, color=color, ls=ls, lw=1.15, label=label, zorder=1)

    hline(row.get("suggest_entry") or row.get("official_entry"), "#1d3557", "--", "Entry")
    hline(row.get("stop"), "#c1121f", "--", "Stop")
    hline(row.get("target_1h"), "#2a9d8f", "--", "Target 1.0H")
    hline(row.get("target_15h"), "#264653", ":", "Target 1.5H")
    if small:
        ref = row.get("official_entry") or row.get("break_level")
        for key, lab in (("parent_neck_now", "大颈线(后续目标)"), ("parent_target_1h", "大形态 1.0H")):
            v = row.get(key)
            if v is not None and (ref is None or v < ref):
                hline(v, "#9d4edd", "-.", lab)
                break

    if row.get("neck_s2") is None and row.get("neck_class_cn") is None:
        attach_neck_slope(row, df15)
    attach_shoulder_prop(row, df15)
    _draw_horizontal_neck(ax, row, right)
    _draw_fib_ext(ax, row, left, right, R)

    _draw_volume(ax_vol, df15, idx, row)
    _draw_momentum(ax, ax_rsi, ax_mh, df15, idx, row, h)
    try:  # headroom so the H label is not clipped (cosmetic)
        y0, y1_ = ax.get_ylim(); ax.set_ylim(y0, y1_ + 0.05 * (y1_ - y0))
    except Exception:
        pass
    ticks = np.linspace(left, right, num=min(7, right - left + 1), dtype=int)
    labels = [to_shanghai(times.iloc[i])[5:] for i in ticks]  # MM-DD HH:MM
    ax_mh.set_xticks(ticks)
    ax_mh.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlim(left - 1, right + 1)
    ax.grid(True, alpha=0.25)
    exch = EXCHANGE_CN.get(row.get("exchange") or "", "")
    title = f"{exch + ' ' if exch else ''}{row['symbol']}  {tf_label}  {row.get('status', '')}  {title_extra}".strip()
    ax.set_title(title, fontproperties=FONT, fontsize=13)
    ax.set_ylabel("Price", fontproperties=FONT)
    ax.legend(prop=FONT, fontsize=8, loc="best")
    fig.tight_layout()
    try:
        fig.canvas.draw()
        ang = neck_visual_angle_deg(ax, lt, y1, rt, y2)
    except Exception:
        ang = None
    if row.get("neck_s2") is None and row.get("neck_class_cn") is None:
        attach_neck_slope(row, df15)
    attach_shoulder_prop(row, df15)
    row["neck_angle_deg"] = None if ang is None else round(float(ang), 1)
    attach_dual_neck(row)
    neck_line = neck_info_text(row.get("neck_s2"), row.get("neck_class_cn"), row.get("neck_angle_deg"))
    prop_line = row.get("prop_text") or ""
    info_line = neck_line if not prop_line else f"{neck_line}\n{prop_line}"
    if row.get("nested_text"):
        info_line += "\n" + _wrap_cn(str(row["nested_text"]), 56)
    for extra in (row.get("fib_text") or "", row.get("dual_neck_text") or ""):
        if extra:
            info_line += "\n" + _wrap_cn(str(extra), 56)
    try:
        updated = False
        for txt in ax.texts:
            if str(txt.get_text()).startswith("确认(只读指标"):
                txt.set_text(str(txt.get_text()) + "\n" + info_line)
                updated = True
                break
        if not updated:
            ax.text(0.01, 0.02, info_line, transform=ax.transAxes,
                    fontproperties=FONT, fontsize=8.5, color="#333",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="#bbb", lw=0.5), zorder=7)
    except Exception as e:
        print(f"  neck angle label skipped: {type(e).__name__} {e}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


NESTED_COLOR = "#7b2cbf"


def _wrap_cn(text: str, width: int) -> str:
    """Wrap a Chinese info line at punctuation (，；。（ / space) so prices are never split."""
    lines, cur = [], ""
    tokens, tok = [], ""
    for ch in text:
        tok += ch
        if ch in "，；。）/":
            tokens.append(tok); tok = ""
    if tok:
        tokens.append(tok)
    for tk in tokens:
        if cur and len(cur) + len(tk) > width:
            lines.append(cur); cur = ""
        cur += tk
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def exch_label(r: dict) -> str:
    return EXCHANGE_CN.get(r.get("exchange") or EXCHANGE_BINANCE, str(r.get("exchange")))


def chart_path(r: dict) -> Path:
    """Binance keeps the historical names; OKX gets an okx_ prefix so nothing is overwritten."""
    prefix = "okx_" if r.get("exchange") == EXCHANGE_OKX else ""
    base = f"{prefix}{r['symbol']}"
    if r.get("nested_signal"):
        if r["status"] == "可入场":
            nest_sfx = ""
        elif r["status"] == STATUS_RETEST:
            nest_sfx = "_retest"
        else:
            nest_sfx = "_watch"
        return OUT_DIR / f"{base}_nested{nest_sfx}.png"
    if r["status"] == STATUS_OBSERVE:
        return OUT_DIR / f"{base}_observe.png"
    if r["status"] == STATUS_RETEST:
        sfx = "_observe_retest" if r.get("tier") == TIER_WATCH else "_retest"
        return OUT_DIR / f"{base}{sfx}.png"
    p = OUT_DIR / f"{base}.png"
    if p.exists() and r["status"] == "观望":
        return OUT_DIR / f"{base}_watch.png"
    return p


def tag_same_coin(rows: list[dict]) -> int:
    """Mark rows whose coin also fired on the other exchange in this slot. Returns #rows marked."""
    by_coin: dict[str, list[dict]] = {}
    for r in rows:
        by_coin.setdefault(r.get("coin") or "", []).append(r)
    n = 0
    for coin, rs in by_coin.items():
        if not coin:
            continue
        for r in rs:
            others = [o for o in rs if o.get("exchange") != r.get("exchange")]
            if others:
                o = others[0]
                r["same_coin_on_other_exchange"] = {
                    "exchange": o.get("exchange"),
                    "exchange_cn": EXCHANGE_CN.get(o.get("exchange"), ""),
                    "symbol": o.get("symbol"),
                    "status": o.get("status"),
                    "tier": o.get("tier"),
                }
                n += 1
            else:
                r["same_coin_on_other_exchange"] = None
    return n


def write_outputs(meta: dict, rows: list[dict]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    serializable = []
    for r in rows:
        item = {k: v for k, v in r.items() if k not in ("L", "H", "R", "lt", "rt", "y1", "y2")
                and not k.startswith("_")}
        serializable.append(item)
    payload = {**meta, "n": len(rows), "signals": serializable}
    (OUT_DIR / "signals.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# 15m 头肩顶做空扫描（A+C，不自动下单）",
        "",
        f"扫描时间：{meta['asof_shanghai']}（北京时间）",
        f"最后一根已收盘 15m：{meta['last_bar_shanghai']}（北京时间）",
        f"核心池：币安 {meta.get('pool_size_binance', meta['pool_size'])} 个（`core_pool.json`）"
        f" + 欧易 {meta.get('pool_size_okx', 0)} 个（`okx_pool.json`）"
        + ("" if meta.get("okx_ok", True) else f"；欧易本轮跳过：{meta.get('okx_error', '')}"),
        "欧易与币安用同一套规则；规则只在币安数据上回测过，欧易未单独验证。",
        "规则：`hs_15m_breakout_quality.md`（破位即入 + 高质量三门槛 + λ=1.5 + 1H 明显偏多禁空 + 量度 60%@1.0H / 余仓 1.5H）。",
        "",
        f"可入场 {meta['n_entry']} 个，观望 {meta['n_watch']} 个，观察（未验证 near-miss）{meta.get('n_observe', 0)} 个。",
        "",
    ]
    nested_rows = [r for r in rows if r.get("nested_signal")]
    rows = [r for r in rows if not r.get("nested_signal")]
    entries = [r for r in rows if r["status"] == "可入场"]
    watches = [r for r in rows if r["status"] == "观望"]
    if entries:
        lines += [
            "## 可入场",
            "",
            "| 交易所 | 品种 | 左/头/右（北京） | 颈线 | 建议入场 | 止损 | 1.0H | 1.5H | 当前价 | 风报 | 距止损 | 图 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in entries:
            lines.append(
                f"| {exch_label(r)} | {r['symbol']} | {r['L_time']} / {r['H_time']} / {r['R_time']} | "
                f"{fmt_px(r['neck'])} | {fmt_px(r['suggest_entry'])} | {fmt_px(r['stop'])} | "
                f"{fmt_px(r['target_1h'])} | {fmt_px(r['target_15h'])} | {fmt_px(r['current'])} | "
                f"{fmt_rr(r['rr'])} | {r['dist_stop_pct']:+.2f}% | `{Path(r['chart']).name}` |"
            )
            if r.get("note"):
                lines.append(f"|  |  | {r['note']} | | | | | | | | | |")
        lines.append("")
    else:
        lines += ["## 可入场", "", "当前没有可按规则考虑入场的破位单。", ""]

    if watches:
        lines += [
            "## 观望（形态已凝固、尚未有效跌破）",
            "",
            "| 交易所 | 品种 | 左/头/右（北京） | 颈线 | 预估止损 | 预估 1.0H / 1.5H | 当前价 | 状态 | 图 |",
            "|---|---|---|---:|---:|---:|---:|---|---|",
        ]
        for r in watches:
            lines.append(
                f"| {exch_label(r)} | {r['symbol']} | {r['L_time']} / {r['H_time']} / {r['R_time']} | "
                f"{fmt_px(r['neck'])} | {fmt_px(r['stop'])} | "
                f"{fmt_px(r['target_1h'])} / {fmt_px(r['target_15h'])} | "
                f"{fmt_px(r['current'])} | 观望 | `{Path(r['chart']).name}` |"
            )
            if r.get("note"):
                lines.append(f"|  |  | {r['note']} | | | | | | |")
        lines.append("")
    else:
        lines += ["## 观望", "", "当前没有高质量、右肩已确认且仍在 24 根破位窗内的观望形态。", ""]

    observes = [r for r in rows if r["status"] == STATUS_OBSERVE]
    if observes:
        lines += [
            "## 观察（near-miss，未验证，建议风险 0.5%）",
            "",
            "| 交易所 | 品种 | 左/头/右（北京） | 差的条件 | 建议入场 | 止损 | 1.0H | 1.5H | 当前价 | 风报 | 图 |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in observes:
            lines.append(
                f"| {exch_label(r)} | {r['symbol']} | {r['L_time']} / {r['H_time']} / {r['R_time']} | "
                f"{r.get('missed_text', '')} | {fmt_px(r['suggest_entry'])} | {fmt_px(r['stop'])} | "
                f"{fmt_px(r['target_1h'])} | {fmt_px(r['target_15h'])} | {fmt_px(r['current'])} | "
                f"{fmt_rr(r['rr'])} | `{Path(r['chart']).name}` |"
            )
        lines.append("")
    else:
        lines += ["## 观察", "", "当前没有差一条的 near-miss 形态。", ""]

    lines += [
        "建议入场对「可入场」是现价或官方下一根开盘；对「观望」必须等有效跌破，不要提前空。",
        "风报 = (当前价 − 1.0H) / (止损 − 当前价)，必须 ≥ 1.5 才允许标成可入场。",
        "「同币另一所」列出同一轮里另一家交易所也出信号的同一个币，可合并看。",
        "本扫描不自动下单。",
        "",
    ]
    if nested_rows:
        lines += ["## 嵌套小头肩（大头肩右肩内的小头肩，未验证，建议风险 0.5%，独立记账）", ""]
        for r in nested_rows:
            lines.append(f"- {exch_label(r)} {r['symbol']} {r['status']}：{r.get('nested_text', '')} "
                         f"入场 {fmt_px(r.get('suggest_entry') or r.get('break_level'))} 止损 {fmt_px(r.get('stop'))} "
                         f"1.0H {fmt_px(r.get('target_1h'))} 1.5H {fmt_px(r.get('target_15h'))} `{Path(r['chart']).name}`")
        lines.append("")
    same = [r for r in rows if r.get("same_coin_on_other_exchange")]
    if same:
        lines += ["## 同一个币两所同时出信号", ""]
        for r in same:
            o = r["same_coin_on_other_exchange"]
            lines.append(
                f"- {exch_label(r)} {r['symbol']}（{r['status']}） ↔ "
                f"{EXCHANGE_CN.get(o.get('exchange'), o.get('exchange'))} {o.get('symbol')}（{o.get('status')}）"
            )
        lines.append("")
    (OUT_DIR / "scan.md").write_text("\n".join(lines), encoding="utf-8")


def make_demo_chart() -> str:
    """Reproduce the 2026-11-07 NEARUSDT A+C winner to prove annotations render."""
    from hs_backtest import load_ohlcv
    try:
        raw15 = load_ohlcv("NEARUSDT", "15m")
        raw1 = load_ohlcv("NEARUSDT", "1h")
    except Exception:
        raw15 = raw1 = None
    if raw15 is None or len(raw15) < 200:
        raw15 = fetch_klines("NEARUSDT", "15m", 1500)
        raw1 = fetch_klines("NEARUSDT", "1h", 500)
    if raw15 is None or len(raw15) < 200:
        return "demo skipped (no NEARUSDT data)"
    df15 = add_indicators(raw15)
    df1 = add_indicators(raw1)
    sigs = collect_ac_signals("NEARUSDT", df15, df1)
    target = pd.Timestamp("2026-11-07 10:00:00", tz="UTC")  # 18:00 Beijing
    chosen = None
    for sig in sigs:
        if abs((_ts(sig.entry_time) - target) / pd.Timedelta(minutes=15)) <= 4:
            chosen = sig
            break
    if chosen is None and sigs:
        # fall back to the signal whose entry is closest to that stamp
        chosen = min(sigs, key=lambda s: abs((_ts(s.entry_time) - target).total_seconds()))
    if chosen is None:
        return "demo skipped (no A+C signal on NEARUSDT)"
    lt, y1, rt, y2 = troughs(df15, chosen.L, chosen.H, chosen.R)
    row = {
        "symbol": "NEARUSDT",
        "status": "示范（历史成交）",
        "L": int(chosen.L), "H": int(chosen.H), "R": int(chosen.R),
        "lt": lt, "y1": y1, "rt": rt, "y2": y2,
        "suggest_entry": float(chosen.entry_px),
        "official_entry": float(chosen.entry_px),
        "stop": float(chosen.stop),
        "target_1h": float(chosen.target),
        "target_15h": float(chosen.target15),
    }
    attach_neck_slope(row, df15)
    attach_shoulder_prop(row, df15)
    out = OUT_DIR / "_demo_NEARUSDT.png"
    plot_setup(df15, row, out, title_extra="demo NEAR 2026-11-07")
    return f"wrote {out} ({out.stat().st_size} bytes)"


def clear_old_charts():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUT_DIR.glob("*.png"):
        if p.name.startswith("_demo_"):
            continue
        p.unlink(missing_ok=True)


def scan_symbol(exchange: str, sym: str, df15: pd.DataFrame, df1: pd.DataFrame,
                now: pd.Timestamp) -> tuple[list[dict], pd.DataFrame | None]:
    """Same A+C rule + 观望 + 观察 for one symbol on one exchange. Rule logic unchanged."""
    df15i = add_indicators(df15)
    df1i = add_indicators(df1)
    recs = scan_line(sym, df15i, BAR_15, df1i, BAR_1H)
    sigs = collect_ac_signals(sym, df15i, df1i, recs=recs)
    ents = live_entries(sym, df15i, df1i, sigs, now)
    retests = live_retest(sym, df15i, df1i, collect_lowrr_breakouts(recs), now)
    wts = [watch_to_row(w, df15i) for w in find_watch(sym, df15i, df1i, now)]
    try:
        obs = live_observe(sym, df15i, df1i, recs, now)
        obs += live_observe_retest(sym, df15i, df1i, recs, now)
        obs = merge_tiers(ents + retests, obs)
    except Exception as e:  # near-miss tier must never break formal scanning
        print(f"  FAIL {exchange}:{sym} observe: {type(e).__name__} {e}", flush=True)
        obs = []
    print(f"  {exchange[:3]} {sym:18s} ac={len(sigs):3d} entry={len(ents)} retest={len(retests)} "
          f"watch={len(wts)} observe={len(obs)}",
          flush=True)
    rows = ents + retests + wts + obs
    if NESTED_15M:
        # 2026-11-12 嵌套头肩 (unvalidated, own book, never gates or changes the BASE rows' levels)
        try:
            import nested_hs
            nest = nested_hs.scan_15m_nested(sym, df15i, df1i, rows)
            if nest:
                print(f"  {exchange[:3]} {sym:18s} nested={len(nest)}", flush=True)
            rows = rows + nest
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {exchange}:{sym} nested: {type(e).__name__} {e}", flush=True)
    for r in rows:
        r["exchange"] = exchange
        r["exchange_cn"] = EXCHANGE_CN[exchange]
        r["coin"] = okx_market.coin_key(sym, exchange)
    return rows, df15i


def run_exchange(exchange: str, symbols: list[str], fetch, now: pd.Timestamp, rows: list[dict]):
    """Scan one exchange's pool. Returns (newest_bar_ts, n_ok, n_fail). Never raises."""
    newest = None
    n_ok = n_fail = 0
    for sym in symbols:
        try:
            df15 = fetch(sym, "15m", KLINE_LIMIT_15M)
            df1 = fetch(sym, "1h", KLINE_LIMIT_1H)
        except Exception as e:
            n_fail += 1
            print(f"  FAIL {exchange}:{sym} fetch: {type(e).__name__} {e}", flush=True)
            continue
        if df15 is None or len(df15) < 120 or df1 is None or len(df1) < 80:
            print(f"  skip {exchange}:{sym} short history 15m={0 if df15 is None else len(df15)}", flush=True)
            continue
        try:
            found, df15i = scan_symbol(exchange, sym, df15, df1, now)
            last = _ts(df15i["dt"].iloc[-1])
            newest = last if newest is None else max(newest, last)
            n_ok += 1
            for r in found:
                chart = chart_path(r)
                plot_setup(df15i, r, chart)
                r["chart"] = str(chart)
                r["data_source"] = binance_data.source_of(sym, exchange)
                try:
                    r["chart_spec"] = chart_spec(df15i, r, "15m")
                except Exception as e:  # noqa: BLE001  (chart spec is optional)
                    print(f"  chart_spec {exchange}:{sym}: {type(e).__name__} {e}", flush=True)
                rows.append(r)
        except Exception as e:
            # One symbol must not abort the whole scan (and miss alerts / journal updates).
            n_fail += 1
            print(f"  FAIL {exchange}:{sym} scan: {type(e).__name__} {e}", flush=True)
            continue
    return newest, n_ok, n_fail


def fetch_okx(sym: str, interval: str, limit: int) -> pd.DataFrame:
    return okx_market.fetch_okx_klines(sym, "1H" if interval == "1h" else interval, limit)


OKX_MAX_CONSEC_FAIL = 5  # e.g. OKX down: stop hammering it this slot


def main() -> int:
    t_start = time.monotonic()
    if not POOL_PATH.exists():
        print(f"missing {POOL_PATH}; run build_core_pool.py first", file=sys.stderr)
        return 2
    pool = json.loads(POOL_PATH.read_text())
    symbols = [s["symbol"] if isinstance(s, dict) else s for s in pool["symbols"]]
    print(f"pool binance {len(symbols)} symbols", flush=True)
    clear_old_charts()

    now = now_utc()
    rows: list[dict] = []
    newest, b_ok, b_fail = run_exchange(EXCHANGE_BINANCE, symbols, fetch_klines, now, rows)
    t_binance = time.monotonic() - t_start

    # ---- OKX: any failure skips OKX for this slot, never breaks Binance results ----
    okx_ok, okx_error, okx_syms = True, "", []
    t_okx0 = time.monotonic()
    o_ok = o_fail = 0
    try:
        okx_syms = okx_market.load_okx_pool(log=lambda m: print(m, flush=True))
        if not okx_syms:
            raise okx_market.OkxError("okx pool unavailable")
        print(f"pool okx {len(okx_syms)} symbols", flush=True)
        consec = {"n": 0}

        def guarded_fetch(sym, interval, limit):
            if consec["n"] >= OKX_MAX_CONSEC_FAIL:
                raise okx_market.OkxError("too many consecutive OKX failures; skipping rest")
            try:
                df = fetch_okx(sym, interval, limit)
            except Exception:
                consec["n"] += 1
                raise
            consec["n"] = 0
            return df

        okx_rows: list[dict] = []
        o_newest, o_ok, o_fail = run_exchange(EXCHANGE_OKX, okx_syms, guarded_fetch, now, okx_rows)
        if o_ok == 0:
            raise okx_market.OkxError(f"no OKX symbol scanned ({o_fail} failures)")
        rows.extend(okx_rows)
        if consec["n"] >= OKX_MAX_CONSEC_FAIL:
            okx_ok, okx_error = False, f"partial: stopped after {OKX_MAX_CONSEC_FAIL} consecutive failures"
    except Exception as e:
        okx_ok, okx_error = False, f"{type(e).__name__}: {e}"
        print(f"OKX skipped this slot: {okx_error}", flush=True)
    t_okx = time.monotonic() - t_okx0

    n_same = tag_same_coin(rows)

    demo_msg = make_demo_chart()
    print(demo_msg, flush=True)

    meta = {
        "asof_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "asof_shanghai": now.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M"),
        "last_bar_shanghai": (
            newest.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M") if newest is not None else ""
        ),
        "pool_size": len(symbols) + len(okx_syms),
        "pool_size_binance": len(symbols),
        "pool_size_okx": len(okx_syms),
        "exchanges": [EXCHANGE_BINANCE] + ([EXCHANGE_OKX] if okx_syms else []),
        "okx_ok": okx_ok,
        "okx_error": okx_error,
        "okx_scanned": o_ok,
        "binance_scanned": b_ok,
        "n_same_coin_rows": n_same,
        "n_entry": sum(1 for r in rows if r["status"] == "可入场" and not r.get("nested_signal")),
        "n_retest": sum(1 for r in rows if r["status"] == STATUS_RETEST and not r.get("nested_signal")),
        "n_watch": sum(1 for r in rows if r["status"] == "观望" and not r.get("nested_signal")),
        "n_nested": sum(1 for r in rows if r.get("nested_signal")),
        "n_observe": sum(1 for r in rows if r["status"] == STATUS_OBSERVE),
        "rule": "hs_15m_breakout_quality.md A+C",
        "okx_validation": "OKX 用同一规则扫描；规则只在币安数据上回测，欧易未单独验证。",
        "demo_chart": demo_msg,
        "scan_seconds_binance": round(t_binance, 1),
        "scan_seconds_okx": round(t_okx, 1),
        "scan_seconds_total": round(time.monotonic() - t_start, 1),
        "binance_data": binance_data.summary(),
    }
    # formal tier first (priority), observe last; Binance before OKX within a status
    order = {"可入场": 0, STATUS_RETEST: 1, "观望": 2, STATUS_OBSERVE: 3}
    rows.sort(key=lambda r: (order.get(r["status"], 9), 0 if r.get("exchange") == EXCHANGE_BINANCE else 1))
    write_outputs(meta, rows)
    print(f"wrote {OUT_DIR / 'signals.json'} entries={meta['n_entry']} retests={meta['n_retest']} "
          f"watches={meta['n_watch']} observes={meta['n_observe']} okx_ok={okx_ok} same_coin_rows={n_same}")
    print(f"scan duration total={meta['scan_seconds_total']}s binance={meta['scan_seconds_binance']}s "
          f"okx={meta['scan_seconds_okx']}s")
    print(f"wrote {OUT_DIR / 'scan.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
