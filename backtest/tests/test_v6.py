"""15m / HTF v6 三栋认肩 (2026-11-14). Real Binance data to 11-14 09:00 Beijing. Never POSTs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import scan_15m as SC  # noqa: E402
import scan_v6 as S6  # noqa: E402
from hs_backtest import add_indicators, neck_at, to_shanghai  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"


def load(sym):
    out = []
    for iv in ("15m", "1h"):
        d = pd.read_csv(FIX / f"{sym.lower()}_{iv}_20261114.csv")
        d["dt"] = pd.to_datetime(d["open_time"], unit="ms", utc=True)
        out.append(add_indicators(d))
    return out


def T(d, i):
    return to_shanghai(d["dt"].iloc[i])


def head_events(d15, d1, sym, head="2026-11-13 21:30"):
    return [e for e in S6.detect_15m(sym, d15, d1) if T(d15, e["H"]) == head]


def test_link_v6_pivots_match_annotation_and_no_break():
    d15, d1 = load("LINKUSDT")
    h, l = d15["high"].to_numpy(), d15["low"].to_numpy()
    import hs_shoulder_v3_core as V, hs_v6_core as C6
    H = [i for i in range(len(d15)) if T(d15, i) == "2026-11-13 21:30"][0]
    u = V.unit_series(d15["atr"].to_numpy())[H]
    P = C6.pick_ls(h, l, V.swing_highs(h), S6.next_greater(h), H, u)
    assert T(d15, P["L"]) == "2026-11-13 15:00" and T(d15, P["p0"]) == "2026-11-13 11:45"
    W = C6.RightWalker(h, l, H, u)
    for t in range(H + 1, len(d15)):
        W.update(t)
    assert T(d15, W.rs()) == "2026-11-14 03:15"
    lt, rt = P["lt"], H + int(np.argmin(l[H:W.rs() + 1]))
    assert T(d15, lt) == "2026-11-13 16:15" and T(d15, rt) == "2026-11-13 23:30"
    c = d15["close"].to_numpy()
    gap = min(c[t] - neck_at(lt, l[lt], rt, l[rt], t) for t in range(W.rs() + 1, len(d15)))
    assert gap > 0  # 三栋: LINK never closed below the v6 neckline -> no entry
    assert head_events(d15, d1, "LINKUSDT") == []
    now = d15["dt"].iloc[-1] + pd.Timedelta(minutes=15)
    assert SC.live_v6("LINKUSDT", d15, d1, now) == []


def test_near_v6_breaks_at_0700_but_rr_too_low_so_not_pushed():
    d15, d1 = load("NEARUSDT")
    evs = head_events(d15, d1, "NEARUSDT")
    assert len(evs) == 1 and evs[0]["kind"] == "lowrr"
    e = evs[0]
    assert (T(d15, e["L"]), T(d15, e["R"]), T(d15, e["t"])) == ("2026-11-13 15:15", "2026-11-14 04:00", "2026-11-14 07:00")
    assert e["sig"].rr < 1.5
    now = d15["dt"].iloc[-1] + pd.Timedelta(minutes=15)
    assert SC.live_v6("NEARUSDT", d15, d1, now) == []


def test_scan_symbol_emits_only_v6_or_nested_rows():
    d15, d1 = load("LINKUSDT")
    raw15 = d15[["open_time", "open", "high", "low", "close", "volume", "dt"]].copy()
    raw1 = d1[["open_time", "open", "high", "low", "close", "volume", "dt"]].copy()
    rows, _ = SC.scan_symbol("binance", "LINKUSDT", raw15, raw1, raw15["dt"].iloc[-1] + pd.Timedelta(minutes=15))
    assert all(r.get("tier") == "v6" or r.get("nested_signal") or r.get("nested") for r in rows)
    assert not any(r.get("tier") in ("formal", "watch") and not r.get("nested_signal") for r in rows)


def test_prop_score_and_confirm_on_link_v6_pivots():
    d15, _ = load("LINKUSDT")
    h, l = d15["high"].to_numpy(), d15["low"].to_numpy()
    ix = {T(d15, i): i for i in range(len(d15))}
    m = S6.prop_score(h, l, ix["2026-11-13 15:00"], ix["2026-11-13 21:30"], ix["2026-11-14 03:15"], ix["2026-11-14 06:00"])
    assert m["v6_spans"] == {"ls": 18, "head": 29, "rs": 26}
    assert 70 <= m["prop_score"] <= 100
    c = S6.confirm_count({"div": {"rsi": True, "hist": False}, "vol_stepdown": None})
    assert c["confirm_n"] == 1 and c["confirm_x3_text"] == "确认 1/3：RSI背离√ MACD柱背离× 量能递减?"


def test_v6_break_level_is_the_neckline_itself():
    nk, brk = S6.v6_break_level(0, 10.0, 10, 9.0, 20, 0.5)
    assert nk == pytest.approx(8.0) and brk == pytest.approx(8.0)
