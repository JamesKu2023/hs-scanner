"""Display-only Fibonacci extensions, MACD DIF/DEA lines, and dual neckline.

Must not change signal selection, entry, stop, targets, status, or R:R.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import scan_15m  # noqa: E402
import scan_htf  # noqa: E402
import scan_runner  # noqa: E402
from hs_backtest import add_indicators  # noqa: E402

# Keep the research/tp definition in the test so a later drift is caught.
sys.path.insert(0, str(HERE.parent / "research" / "tp"))
from tp_sim import FIBK, level_px  # noqa: E402

from test_momentum_panel import _hs_frame  # noqa: E402
from test_htf import synth_hs  # noqa: E402


TRADE_KEYS = ("status", "suggest_entry", "official_entry", "stop", "target_1h",
              "target_15h", "rr", "L", "H", "R", "lt", "rt", "y1", "y2", "neck")


def _row(df, **kw):
    base = {"symbol": "TESTUSDT", "status": "观望", "L": 120, "H": 140, "R": 160,
            "lt": 130, "y1": float(df["low"][130]), "rt": 150, "y2": float(df["low"][150]),
            "stop": 104.0, "target_1h": 98.0, "target_15h": 96.0, "exchange": "binance"}
    base.update(kw)
    return base


def test_fib_ext_matches_tp_sim_definition():
    # A=head 10, B=right trough 8, C=RS high 9 -> 9 - k*2
    got = scan_15m.fib_ext_prices(10.0, 8.0, 9.0)
    class Sig:
        rs_high, head_high = 9.0, 10.0
        target = target15 = 0.0
    for name, k in (("f1.0", 1.0), ("f1.272", 1.272), ("f1.618", 1.618), ("f2.0", 2.0)):
        key = name[1:]
        assert got[key] == pytest.approx(level_px(name, Sig(), 8.0))
        assert got[key] == pytest.approx(9.0 - FIBK[name] * (10.0 - 8.0))
    assert list(got) == ["1.0", "1.272", "1.618", "2.0"]
    assert scan_15m.fib_ext_prices(8.0, 10.0, 9.0) is None  # non-positive span
    assert scan_15m.fib_ext_prices(None, 8.0, 9.0) is None


def test_attach_fib_ext_text_and_does_not_touch_trade_fields():
    df = _hs_frame()
    row = _row(df, official_entry=100.0, rr=1.7, neck=100.2)
    before = {k: row.get(k) for k in TRADE_KEYS}
    scan_15m.attach_fib_ext(row, df)
    assert {k: row.get(k) for k in TRADE_KEYS} == before
    Hh, R, y2 = row["H"], row["R"], row["y2"]
    expect = scan_15m.fib_ext_prices(float(df["high"].iloc[Hh]), y2, float(df["high"].iloc[R]))
    assert row["fib_ext"] == expect
    assert row["fib_text"].startswith("斐波那契扩展（只作参考）：")
    assert "1.0=" in row["fib_text"] and "1.272=" in row["fib_text"]
    assert "1.618=" in row["fib_text"] and "2.0=" in row["fib_text"]
    bare = {"H": 0, "R": 1, "y2": 1.0, "stop": 2.0, "status": "可入场"}
    scan_15m.attach_fib_ext(bare)
    assert bare["stop"] == 2.0 and bare["status"] == "可入场"
    assert bare["fib_ext"] is None and bare["fib_text"].endswith("—")


def test_dual_neck_up_down_and_flat():
    # up-sloping: y2 > y1, class 上翘 -> tier1=sloped neck, tier2=y1
    up = {"y1": 100.0, "y2": 103.0, "neck": 102.5, "neck_class_cn": "上翘",
          "neck_s2": 0.25, "neck_angle_deg": 7.2, "stop": 110.0, "status": "可入场"}
    before = dict(up)
    scan_15m.attach_dual_neck(up)
    assert up["stop"] == before["stop"] and up["status"] == before["status"]
    assert up["dual_neck_tier1"] == pytest.approx(102.5)
    assert up["dual_neck_tier2"] == pytest.approx(100.0)
    assert "上翘约+7°" in up["dual_neck_text"]
    assert "第一档 斜颈线" in up["dual_neck_text"] and "先轻仓" in up["dual_neck_text"]
    assert "第二档 水平颈线" in up["dual_neck_text"] and "再加仓" in up["dual_neck_text"]

    # down-sloping: y2 < y1, class 下斜 -> tier1=y2, tier2=sloped neck
    down = {"y1": 103.0, "y2": 100.0, "neck": 100.8, "neck_class_cn": "下斜",
            "neck_s2": -0.25, "neck_angle_deg": -8.0, "stop": 110.0}
    scan_15m.attach_dual_neck(down)
    assert down["dual_neck_tier1"] == pytest.approx(100.0)
    assert down["dual_neck_tier2"] == pytest.approx(100.8)
    assert "下翘约-8°" in down["dual_neck_text"]
    assert "第一档 水平颈线" in down["dual_neck_text"]
    assert "第二档 斜颈线" in down["dual_neck_text"]

    # near-flat class 平 (at the existing NECK_S2_FLAT band)
    flat = {"y1": 100.0, "y2": 100.5, "neck": 100.3, "neck_class_cn": "平", "neck_s2": 0.05}
    scan_15m.attach_dual_neck(flat)
    assert flat["dual_neck_tier1"] == flat["dual_neck_tier2"] == pytest.approx(100.3)
    assert flat["dual_neck_text"] == "双颈线（只作参考）：颈线近乎水平，双颈线基本重合"

    # class derived from s2 when neck_class_cn is missing; |s2|<=0.10 is 平
    derived = {"y1": 100.0, "y2": 101.0, "neck": 100.5, "neck_s2": 0.10}
    scan_15m.attach_dual_neck(derived)
    assert "基本重合" in derived["dual_neck_text"]


def test_attach_does_not_change_signal_selection_fields():
    df = _hs_frame()
    row = _row(df, neck=100.2, official_entry=100.1, rr=1.8, status="可入场")
    before = {k: row.get(k) for k in TRADE_KEYS}
    scan_15m.attach_neck_slope(row, df)
    scan_15m.attach_shoulder_prop(row, df)  # also hooks chart refs
    assert {k: row.get(k) for k in TRADE_KEYS} == before
    assert row["fib_ext"] and row["fib_text"]
    assert row["dual_neck_text"]


def test_plot_setup_draws_fib_dual_and_macd_lines(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    df = _hs_frame()
    # tilt the right trough up so the neckline is clearly 上翘
    row = _row(df, y2=float(df["low"][150]) + 1.8, neck=float(df["low"][130]) + 0.4)
    captured = {}
    orig = plt.Figure.savefig

    def spy(self, *a, **k):
        ax, _ax_vol, _ax_rsi, ax_mh = self.axes[:4]
        captured["labels"] = [ln.get_label() for a in self.axes for ln in a.get_lines()]
        captured["texts"] = [t.get_text() for t in ax.texts]
        captured["mh_labels"] = [ln.get_label() for ln in ax_mh.get_lines()]
        return orig(self, *a, **k)

    monkeypatch.setattr(plt.Figure, "savefig", spy)
    out = tmp_path / "fibdual.png"
    scan_15m.plot_setup(df, row, out)
    assert out.exists() and out.stat().st_size > 10_000
    assert "水平颈线" in captured["labels"]
    assert "DIF" in captured["mh_labels"] and "DEA" in captured["mh_labels"]
    blob = "\n".join(captured["texts"])
    assert "Fib 1.0" in blob and "Fib 1.272" in blob
    assert "斐波那契扩展（只作参考）" in blob
    assert "双颈线（只作参考）" in blob
    assert row["fib_ext"] and row["dual_neck_text"]


def test_plot_setup_survives_missing_macd_lines(tmp_path):
    df = _hs_frame().drop(columns=["macd", "macd_sig", "macd_hist", "rsi"])
    row = _row(df)
    out = tmp_path / "nomacd.png"
    scan_15m.plot_setup(df, row, out)
    assert out.exists()


def test_htf_live_row_and_payload_carry_display_fields():
    rows = scan_htf.live_rows("TESTUSDT", "1h", synth_hs(), None)
    assert rows and rows[0]["status"] == "可入场"
    r = rows[0]
    for k in ("fib_ext", "fib_text", "dual_neck_tier1", "dual_neck_tier2", "dual_neck_text"):
        assert r.get(k) is not None, k
    assert "斐波那契扩展（只作参考）" in r["fib_text"]
    assert "双颈线（只作参考）" in r["dual_neck_text"]
    # trade fields still present and untouched in meaning
    assert r["stop"] > r["official_entry"] > r["target_1h"]
    p = scan_runner.build_htf_payload(rows, "1h", "2026-11-12 22:02", "2026-11-12 21:00")
    s = p["signals"][0]
    assert s["fib_ext"]["1.0"] == r["fib_ext"]["1.0"]
    assert s["fib_text"] == r["fib_text"]
    assert s["dual_neck_text"] == r["dual_neck_text"]
