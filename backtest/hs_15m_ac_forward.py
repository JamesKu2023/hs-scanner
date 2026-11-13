#!/usr/bin/env python3
"""Update 15m data to today and forward-test locked 15m A+C. Scan only if verdict passes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.signal import find_peaks

from hs_backtest import (
    DATA, OUT, RR_MIN, START_EQUITY, add_indicators, load_ohlcv,
    macd_break_ok, macd_rs_ok, neck_at, run_portfolio, simulate_trade,
    trade_stats, vol_part, to_shanghai,
)
from hs_backtest_v2_3lines import (
    MIN_SPAN, MAX_SPAN, MIN_TROUGH_GAP, PEAK_DISTANCE, BREAK_LOOKAHEAD,
    _ts, levels_at, htf_bullish_at,
)
from hs_opt_strategy import (
    LAM, MEASURED_EXIT, quality_ok, make_signal, measured_ok, max_lose_streak,
    pack_metrics, scan_line,
)

PREV_LAST_BAR = pd.Timestamp("2026-10-31 23:45:00", tz="UTC")
FORWARD_START = pd.Timestamp("2026-11-01 00:00:00", tz="UTC")
# Official entry of last historical bar would be at PREV_LAST_BAR itself if it were an entry.
# Forward = entries at or after the first unseen bar.
FAPI = "https://fapi.binance.com/fapi/v1/klines"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "hs-15m-ac-forward/1.0"
BAR_15 = pd.Timedelta(minutes=15)
BAR_1H = pd.Timedelta(hours=1)


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def fetch_15m_since(symbol: str, start_ms: int) -> pd.DataFrame:
    rows = []
    cursor = start_ms
    end_ms = int(now_utc().timestamp() * 1000)
    while cursor < end_ms:
        r = SESSION.get(
            FAPI,
            params={"symbol": symbol, "interval": "15m", "startTime": cursor, "limit": 1500},
            timeout=(8, 30),
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        nxt = last_open + int(BAR_15.total_seconds() * 1000)
        if len(batch) < 1500 or nxt <= cursor:
            break
        cursor = nxt
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ])
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = df[c].astype("float64")
    df["open_time"] = df["open_time"].astype("int64")
    # drop unclosed bar
    now_ms = int(now_utc().timestamp() * 1000)
    df = df[df["close_time"].astype("int64") < now_ms].copy()
    return df


def update_symbol_15m(symbol: str) -> str:
    p = DATA / f"{symbol}_15m.parquet"
    if not p.exists():
        return f"missing {symbol}"
    old = pd.read_parquet(p)
    last = int(old["open_time"].astype("int64").max())
    # if values look like seconds
    if last < 10**12:
        last_ms = last * 1000
    else:
        last_ms = last
    start_ms = last_ms + int(BAR_15.total_seconds() * 1000)
    new = fetch_15m_since(symbol, start_ms)
    if new is None or len(new) == 0:
        return f"{symbol:16s} +0 (already fresh? last={pd.to_datetime(last_ms, unit='ms', utc=True)})"
    # Existing vision files store several leftover columns as strings.
    for col in ("close_time", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"):
        if col in new.columns:
            new[col] = new[col].astype(str)
        if col in old.columns:
            old[col] = old[col].astype(str)
    keep_cols = [c for c in old.columns if c in new.columns]
    merged = pd.concat([old, new[keep_cols]], ignore_index=True)
    merged = merged.drop_duplicates("open_time").sort_values("open_time")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in merged.columns:
            merged[c] = merged[c].astype("float64")
    merged["open_time"] = merged["open_time"].astype("int64")
    merged.to_parquet(p, index=False)
    t1 = pd.to_datetime(int(merged["open_time"].max()), unit="ms", utc=True)
    return f"{symbol:16s} +{len(new):4d} rows -> {t1} SH {t1.tz_convert('Asia/Shanghai')}"


def collect_ac_signals(symbol, df15, df1):
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


def find_watch(symbol, df15, df1, now_ts):
    """Quality H&S with no valid first break yet, still inside 24-bar window."""
    n = len(df15)
    if n < 100:
        return []
    high, low, opn, close = (df15[c].to_numpy() for c in ("high", "low", "open", "close"))
    vol = df15["volume"].to_numpy()
    atrv, rsi_ = df15["atr"].to_numpy(), df15["rsi"].to_numpy()
    ml, ms, mh = df15["macd"].to_numpy(), df15["macd_sig"].to_numpy(), df15["macd_hist"].to_numpy()
    vma = df15["vol_ma20"].to_numpy()
    times = df15["dt"]

    prom = np.where(np.isfinite(atrv), np.maximum(0.5 * atrv, 0.003 * close), 0.003 * close)
    prom_s = float(np.nanmedian(prom[np.isfinite(prom)])) if np.isfinite(prom).any() else 0.0
    if prom_s <= 0:
        return []
    peaks, _ = find_peaks(high, distance=PEAK_DISTANCE, prominence=prom_s)
    if len(peaks) < 3:
        return []

    watches = []
    used_head = set()
    last_closed = n - 1
    for ih, Hh in enumerate(peaks):
        Hh = int(Hh)
        if Hh in used_head:
            continue
        lefts = [int(p) for p in peaks[:ih] if MIN_SPAN // 4 <= (Hh - int(p)) <= MAX_SPAN]
        rights = [int(p) for p in peaks[ih + 1:] if (int(p) - Hh) <= MAX_SPAN]
        if not lefts or not rights:
            continue
        for L in lefts:
            for R in rights:
                if not (MIN_SPAN <= (R - L) <= MAX_SPAN):
                    continue
                # peak should be confirmed: at least PEAK_DISTANCE closed bars after R
                if last_closed - R < PEAK_DISTANCE:
                    continue
                if not (high[L] < high[Hh] and high[R] < high[Hh]):
                    continue
                if not np.isfinite(atrv[Hh]) or not np.isfinite(atrv[R]):
                    continue
                if float(np.max(high[L:R + 1])) > high[Hh] + 1e-15:
                    continue
                if high[Hh] - max(high[L], high[R]) < max(0.5 * atrv[Hh], 0.003 * close[Hh]):
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
                if not q_ok:
                    continue
                # still in break window?
                elapsed = last_closed - R
                if elapsed > BREAK_LOOKAHEAD:
                    continue
                if elapsed < 1:
                    continue
                # invalidated by new high?
                if float(np.max(high[R + 1:last_closed + 1])) > high[Hh]:
                    continue
                # already has a valid break? then this is not a watch
                had_break = False
                max_t = min(last_closed, R + BREAK_LOOKAHEAD)
                for t in range(R + 1, max_t + 1):
                    if not np.isfinite(vma[t]) or not np.isfinite(rsi_[t]) or not np.isfinite(atrv[t]):
                        continue
                    if high[t] > high[Hh]:
                        break
                    _, _, _, _, brk_lvl, _, _ = levels_at(lt, y1, rt, y2, t, atrv[t])
                    if not (close[t] < opn[t] and close[t] < brk_lvl):
                        continue
                    if rsi_[t] >= 50 or not macd_break_ok(ml, ms, mh, t):
                        continue
                    vol_ratio = float(vol[t] / vma[t]) if vma[t] > 0 else 0.0
                    if vol_ratio <= LAM:
                        continue
                    height = high[Hh] - neck_at(lt, y1, rt, y2, Hh)
                    if height <= 0:
                        continue
                    break_close = _ts(times.iloc[t]) + BAR_15
                    if htf_bullish_at(df1, break_close, BAR_1H):
                        continue
                    had_break = True
                    break
                if had_break:
                    continue
                # current 1H filter at now
                if htf_bullish_at(df1, now_ts, BAR_1H):
                    continue
                height = float(high[Hh] - neck_at(lt, y1, rt, y2, Hh))
                nk = float(neck_at(lt, y1, rt, y2, last_closed))
                _, _, _, _, brk_lvl, _, _ = levels_at(lt, y1, rt, y2, last_closed, atrv[last_closed])
                watches.append({
                    "symbol": symbol,
                    "L": L, "H": Hh, "R": R,
                    "L_time": to_shanghai(times.iloc[L]),
                    "H_time": to_shanghai(times.iloc[Hh]),
                    "R_time": to_shanghai(times.iloc[R]),
                    "neck": nk,
                    "break_lvl": float(brk_lvl),
                    "head_high": float(high[Hh]),
                    "rs_high": float(high[R]),
                    "ls_high": float(high[L]),
                    "height": height,
                    "target": nk - height,
                    "target15": nk - 1.5 * height,
                    "current": float(close[last_closed]),
                    "bars_since_R": int(elapsed),
                    "bars_left": int(BREAK_LOOKAHEAD - elapsed),
                    **qmet,
                })
                used_head.add(Hh)
                break
            else:
                continue
            break
    return watches


def fmt_px(x):
    ax = abs(float(x))
    if ax >= 1000:
        return f"{x:,.2f}"
    if ax >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"


def verdict_from(st, port):
    n = int(st["n"])
    avg = st["avg_R"]
    pf = st["profit_factor"]
    if n == 0 or avg is None or (isinstance(avg, float) and not np.isfinite(avg)):
        return "FAIL_NO_TRADES", "前向段没有成交，无法核实，不算通过。"
    pf_ok = pf is None or (isinstance(pf, float) and not np.isfinite(pf) and avg > 0) or (pf is not None and pf > 1)
    if avg > 0 and pf_ok:
        if n >= 15:
            return "PASS", "前向段平均 R 为正且 PF>1，笔数达到 15。"
        return "PASS_THIN", "前向段平均 R 为正且 PF>1，但笔数不足 15，标为「方向一致但样本薄」。"
    return "FAIL", "前向段平均 R 转负或利润因子未大于 1，核实失败。"


def write_forward_md(path, meta, st, port, trades, extra_note, verdict, reason):
    lines = []
    a = lines.append
    a("# 15m A+C 前向核实")
    a("")
    a("日期：2026-11-10（Asia/Shanghai）")
    a("规则：`/workspace/trading/hs_15m_breakout_quality.md`（锁定阈值，核实过程未改参）。")
    a("v2.0 主模型未改。本段 **不是** 原先用来选出 A+C 的 OOS（该 OOS 为 2025-09-06 23:49:30 UTC → 2026-10-31 23:45 UTC）。")
    a("")
    a("## 数据")
    a("")
    a(f"- 上次回测 15m 最后一根开盘：{meta['prev_last_bar_utc']} UTC（北京 {meta['prev_last_bar_sh']}）")
    a(f"- 前向段定义：入场时间 ≥ {meta['forward_start_utc']} UTC（北京 {meta['forward_start_sh']}）")
    a(f"- 本次数据更新后，样本最晚一根已收盘 15m：{meta['new_last_bar_utc']} UTC（北京 {meta['new_last_bar_sh']}）")
    a(f"- 前向日历长度约 {meta['forward_days']:.2f} 天")
    a("- 更新方式：在原有本地 15m parquet 之后，用币安 USDT-M `/fapi/v1/klines` 续写到当前已收盘棒，未重下 2023–2026-10 历史。")
    a("")
    a("## 成功标准（事先写明）")
    a("")
    a("- 理想：前向段笔数 ≥ 15，平均 R > 0 且 PF > 1。")
    a("- 笔数少但平均 R 仍为正且 PF>1：记为「方向一致但样本薄」，允许随后扫描。")
    a("- 前向段转负或崩盘：核实失败，不扫描、不改规则。")
    a("- 前向段 0 笔：无法核实，不算通过，不扫描。")
    a("")
    a("## 结果")
    a("")
    n = st["n"]
    if n == 0:
        a("前向段 **0 笔**成交。")
    else:
        a("| 项目 | 全部前向成交 |")
        a("|---|---:|")
        a(f"| 笔数 | {n} |")
        a(f"| 胜率 | {100 * st['win_rate']:.1f}% |")
        a(f"| 平均 R | {st['avg_R']:+.3f} |")
        a(f"| 中位 R | {st['med_R']:+.3f} |")
        a(f"| 利润因子 | {st['profit_factor'] if st['profit_factor'] is not None else '—'} |")
        a(f"| 最长连亏 | {max_lose_streak(trades)} |")
        a(f"| 组合吃进 / 跳过 | {port['n_taken']} / {port['n_skipped_concurrent']} |")
        a(f"| 组合期末（1% 风险，从 10,000 起） | {port['final']:.0f} |")
        a(f"| 组合收益 | {100 * (port['final'] / START_EQUITY - 1):+.1f}% |")
        a(f"| 组合最大回撤 | {100 * port['max_dd']:.1f}% |")
        a("")
        a("逐笔（北京时间）：")
        a("")
        a("| 品种 | 入场 | 离场 | 入场价 | 止损 | 目标1.0H | R | 结果 |")
        a("|---|---|---|---:|---:|---:|---:|---|")
        for tr in sorted(trades, key=lambda x: _ts(x.entry_time)):
            a(
                f"| {tr.symbol} | {to_shanghai(tr.entry_time)} | {to_shanghai(tr.exit_time)} | "
                f"{fmt_px(tr.entry)} | {fmt_px(tr.stop)} | {fmt_px(tr.target)} | "
                f"{tr.pnl_R:+.3f} | {tr.hit} |"
            )
    a("")
    a(f"**结论：{verdict}** — {reason}")
    if extra_note:
        a("")
        a(extra_note)
    a("")
    if verdict.startswith("PASS"):
        a("按规则允许进入第三步：扫描当前潜在机会（不自动下单）。")
    else:
        a("按规则 **不扫描** 当前入场机会，也不改阈值挽救。")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_scan_md(path, entries, watches, now_sh, last_bar_sh):
    lines = []
    a = lines.append
    a("# 15m A+C 当前扫描（不自动下单）")
    a("")
    a(f"扫描时刻（机器时钟，北京）：{now_sh}")
    a(f"使用的最后一根已收盘 15m：{last_bar_sh}（北京）")
    a("规则：`/workspace/trading/hs_15m_breakout_quality.md`。前向核实已通过或方向一致但样本薄，才执行本次扫描。")
    a("")
    a("## 1. 已破位、当前仍可考虑入场")
    a("")
    if not entries:
        a("没有。没有品种同时满足：高质量形态、第一根有效跌破已出现、价格仍在止损与 1.0H 目标之间、R:R 按现价仍 ≥ 1.5、破位后尚未打到止损或目标 1.0H。")
    else:
        a("| 品种 | 左/头/右（北京） | 颈线 | 建议入场 | 止损 | 1.0H | 1.5H | 当前价 | 距止损 | 距1.0H | 现价R:R | 状态 |")
        a("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for e in entries:
            a(
                f"| {e['symbol']} | {e['L_time']} / {e['H_time']} / {e['R_time']} | "
                f"{fmt_px(e['neck'])} | {fmt_px(e['suggest_entry'])} | {fmt_px(e['stop'])} | "
                f"{fmt_px(e['target'])} | {fmt_px(e['target15'])} | {fmt_px(e['current'])} | "
                f"{e['dist_stop_pct']:+.2f}% | {e['dist_t1_pct']:+.2f}% | {e['rr_now']:.2f} | 可考虑入场 |"
            )
            a("")
            a(f"- {e['symbol']}：官方入场为破位下一根开盘 {e['official_entry_sh']} @ {fmt_px(e['official_entry'])}。"
              f"{e['note']}")
    a("")
    a("## 2. 高质量形态已凝固、尚未破颈线（观望）")
    a("")
    if not watches:
        a("没有。当前没有「高质量 + 右肩已确认 + 仍在 24 根等待窗内 + 尚未有效跌破 + 1H 未明显偏多」的形态。")
    else:
        a("| 品种 | 左/头/右（北京） | 颈线 / 破位水位 | 当前价 | 预估 1.0H / 1.5H | 头/右肩 | 窗内剩余 | 状态 |")
        a("|---|---|---:|---:|---:|---:|---:|---|")
        for w in watches:
            a(
                f"| {w['symbol']} | {w['L_time']} / {w['H_time']} / {w['R_time']} | "
                f"{fmt_px(w['neck'])} / {fmt_px(w['break_lvl'])} | {fmt_px(w['current'])} | "
                f"{fmt_px(w['target'])} / {fmt_px(w['target15'])} | "
                f"{fmt_px(w['head_high'])} / {fmt_px(w['rs_high'])} | "
                f"{w['bars_left']} 根 | 观望 |"
            )
    a("")
    a("以上只是扫描，不构成下单指令。入场仍须等第 3–8 节清单逐条勾完。")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    uni = json.loads((DATA / "universe.json").read_text())
    symbols = [s[0] if isinstance(s, list) else s for s in uni["symbols"]]
    print("updating", len(symbols), "symbols", flush=True)
    last_bars = []
    for sym in symbols:
        try:
            msg = update_symbol_15m(sym)
        except Exception as e:
            msg = f"FAIL {sym}: {type(e).__name__} {e}"
        print("  ", msg, flush=True)

    now = now_utc()
    frames = {}
    newest = None
    oldest_new = None
    for sym in symbols:
        df15 = load_ohlcv(sym, "15m")
        if df15 is None or len(df15) < 400:
            print("  skip", sym, flush=True)
            continue
        df15i = add_indicators(df15)
        df1 = add_indicators(load_ohlcv(sym, "1h"))
        frames[sym] = (df15i, df1)
        t1 = _ts(df15i["dt"].iloc[-1])
        newest = t1 if newest is None else max(newest, t1)
        last_bars.append(t1)
    print("newest closed 15m", newest, "SH", newest.tz_convert("Asia/Shanghai"), flush=True)

    # A+C trades with entry in forward window
    fwd_trades = []
    live_entries = []
    for sym, (df15i, df1) in frames.items():
        sigs = collect_ac_signals(sym, df15i, df1)
        print(f"  {sym:16s} A+C signals={len(sigs)}", flush=True)
        h15 = df15i["high"].to_numpy()
        l15 = df15i["low"].to_numpy()
        c15 = df15i["close"].to_numpy()
        t15 = df15i["dt"]
        for sig in sigs:
            et = _ts(sig.entry_time)
            if et >= FORWARD_START:
                tr = simulate_trade(sig, df15i, MEASURED_EXIT)
                if tr is not None:
                    fwd_trades.append(tr)
            # live: break already happened, price still in range
            # official entry bar must exist
            i0 = int(df15i["dt"].searchsorted(et, side="left"))
            if i0 >= len(df15i):
                continue
            # path since official entry (or since entry bar) must not have hit stop or target
            # if entry is in the future (next bar after last close) — that's a fresh official entry
            cur = float(c15[-1])
            if not (sig.stop > cur > sig.target):
                continue
            # check highs/lows from entry bar onward (if entry bar already exists)
            start_chk = i0 if i0 < len(df15i) else len(df15i)
            if start_chk < len(df15i):
                if float(np.max(h15[start_chk:])) >= sig.stop:
                    continue
                if float(np.min(l15[start_chk:])) <= sig.target:
                    continue
            rr_now = (cur - sig.target) / (sig.stop - cur) if sig.stop > cur else 0.0
            if rr_now < RR_MIN:
                continue
            # 1H still not bullish now
            if htf_bullish_at(df1, now, BAR_1H):
                continue
            note = ""
            if et > _ts(t15.iloc[-1]):
                note = "破位棒刚收盘，下一根开盘即官方入场点。"
            elif et >= FORWARD_START:
                note = "官方入场已过，但价格仍夹在止损与目标之间，结构尚未死亡。"
            else:
                # historical leftover still in range — unusual if old; skip leftovers from before forward
                # Only show setups whose breakdown/entry is recent enough: entry within last 3 days or still pending
                if (now - et) > pd.Timedelta(days=3):
                    continue
                note = "近期官方入场已过，价格仍在区间内。"
            live_entries.append({
                "symbol": sym,
                "L_time": to_shanghai(df15i["dt"].iloc[sig.L]),
                "H_time": to_shanghai(df15i["dt"].iloc[sig.H]),
                "R_time": to_shanghai(df15i["dt"].iloc[sig.R]),
                "neck": sig.neck,
                "suggest_entry": cur if et <= _ts(t15.iloc[-1]) else sig.entry_px,
                "official_entry": sig.entry_px,
                "official_entry_sh": to_shanghai(et),
                "stop": sig.stop,
                "target": sig.target,
                "target15": sig.target15,
                "current": cur,
                "dist_stop_pct": 100 * (sig.stop - cur) / cur,
                "dist_t1_pct": 100 * (cur - sig.target) / cur,
                "rr_now": rr_now,
                "note": note,
            })

    st = trade_stats(fwd_trades)
    port = run_portfolio(fwd_trades, 0.01)
    verdict, reason = verdict_from(st, port)
    print("FORWARD", verdict, "n=", st["n"], "avgR=", st.get("avg_R"), "pf=", st.get("profit_factor"), flush=True)

    extra = (
        f"前向窗口只有约 {(newest - FORWARD_START) / pd.Timedelta(days=1):.1f} 天，"
        "远短于原先约 14 个月的 OOS；笔数偏少是日历长度限制，不是改规则造成的。"
    )
    meta = {
        "prev_last_bar_utc": str(PREV_LAST_BAR),
        "prev_last_bar_sh": PREV_LAST_BAR.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M"),
        "forward_start_utc": str(FORWARD_START),
        "forward_start_sh": FORWARD_START.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M"),
        "new_last_bar_utc": str(newest),
        "new_last_bar_sh": newest.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M"),
        "forward_days": float((newest - FORWARD_START) / pd.Timedelta(days=1)),
        "verdict": verdict,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "opt_v2_15m_forward_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    write_forward_md(
        "/workspace/trading/backtest/results_v2_15m_forward.md",
        meta, st, port, fwd_trades, extra, verdict, reason,
    )
    print("wrote results_v2_15m_forward.md", flush=True)

    scan_path = Path("/workspace/trading/backtest/scan_15m_now.md")
    if verdict.startswith("PASS"):
        watches = []
        for sym, (df15i, df1) in frames.items():
            watches.extend(find_watch(sym, df15i, df1, now))
        write_scan_md(
            scan_path, live_entries, watches,
            now.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M"),
            newest.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M"),
        )
        print("wrote scan", len(live_entries), "entries", len(watches), "watches", flush=True)
    else:
        if scan_path.exists():
            scan_path.unlink()
        print("no scan (verdict", verdict, ")", flush=True)
    return verdict, st


if __name__ == "__main__":
    main()
