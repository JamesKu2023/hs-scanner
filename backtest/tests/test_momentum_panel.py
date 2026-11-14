"""Chart-only momentum panels (RSI / MACD hist). Must not change signals or raise."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import scan_15m  # noqa: E402
from hs_backtest import add_indicators  # noqa: E402


def _hs_frame():
    # synthetic: two left highs (weak 20, strong 40), head 60, strong RS 80, weak RS 95
    # 80-bar zig-zag warm-up so RSI/MACD are settled, then the pattern
    n = 220
    x = np.arange(n, dtype=float)
    base = 100 + 0.3 * np.sin(x / 2.0)
    bumps = {100: 1.0, 120: 3.0, 140: 5.0, 160: 3.0, 175: 1.2}
    hi = base.copy()
    for c, a in bumps.items():
        hi += a * np.exp(-0.5 * ((x - c) / 2.5) ** 2)
    close = hi - 0.2
    df = pd.DataFrame({
        "dt": pd.date_range("2026-11-10", periods=n, freq="15min", tz="UTC"),
        "open": np.r_[close[0], close[:-1]], "high": hi + 0.05, "low": close - 0.3,
        "close": close, "volume": 1.0,
    })
    return add_indicators(df)


def test_no_momentum_pivot_finder_anymore():
    # 三栋 2026-11-12: indicators must not be used to FIND shoulders
    assert not hasattr(scan_15m, "momentum_side_peaks")


def test_pivot_confirmation_reads_at_given_pivots():
    df = _hs_frame()
    c = scan_15m.pivot_confirmation(df, 120, 140, 160)
    assert [c["pivots"][k]["i"] for k in ("LS", "H", "RS")] == [120, 140, 160]
    for k in ("LS", "H", "RS"):
        assert c["pivots"][k]["rsi"] is not None and c["pivots"][k]["hist"] is not None
    # flat volume -> no strict step-down, no volume divergence
    assert c["vol_stepdown"] is False and c["div"]["vol"] is False
    # divergence flags consistent with the numbers
    P = c["pivots"]
    assert c["div"]["rsi"] == (P["H"]["rsi"] < P["LS"]["rsi"])
    assert c["div"]["hist"] == (P["H"]["hist"] < P["LS"]["hist"])


def test_pivot_confirmation_volume_stepdown_and_divergence():
    df = _hs_frame()
    v = np.ones(len(df))
    v[119:122] = 3.0; v[139:142] = 2.0; v[159:162] = 1.5
    df["volume"] = v
    c = scan_15m.pivot_confirmation(df, 120, 140, 160)
    assert c["vol_stepdown"] is True and c["div"]["vol"] is True
    t = scan_15m.confirmation_text(c)
    assert "量能递减 LS>H>RS: 是" in t and "量是" in t


def test_pivot_confirmation_rsi_divergence_detected():
    df = _hs_frame()
    rsi = df["rsi"].to_numpy().copy()
    rsi[118:123] = 75.0; rsi[138:143] = 65.0
    df["rsi"] = rsi
    c = scan_15m.pivot_confirmation(df, 120, 140, 160)
    assert c["div"]["rsi"] is True and "RSI是" in scan_15m.confirmation_text(c)


def test_pivot_confirmation_missing_columns_is_safe():
    df = _hs_frame().drop(columns=["rsi", "macd_hist"])
    c = scan_15m.pivot_confirmation(df, 120, 140, 160)
    assert c["pivots"]["LS"]["rsi"] is None and c["div"]["rsi"] is None and c["div"]["hist"] is None
    assert "RSI?" in scan_15m.confirmation_text(c)


def test_plot_setup_renders_four_panels(tmp_path, monkeypatch):
    df = _hs_frame()
    row = {"symbol": "TESTUSDT", "status": "观望", "L": 120, "H": 140, "R": 160,
           "lt": 130, "y1": float(df["low"][130]), "rt": 150, "y2": float(df["low"][150]),
           "stop": 104.0, "target_1h": 98.0, "target_15h": 96.0, "exchange": "okx"}
    import matplotlib.pyplot as plt
    seen = {}
    orig = plt.subplots

    def spy(*a, **k):
        fig, axes = orig(*a, **k)
        seen["n"] = len(np.atleast_1d(axes))
        return fig, axes
    monkeypatch.setattr(scan_15m.plt, "subplots", spy)
    out = tmp_path / "c.png"
    scan_15m.plot_setup(df, dict(row), out)
    assert out.exists() and out.stat().st_size > 10_000
    assert seen["n"] == 4  # price(+Bollinger) / volume / RSI / MACD hist


def test_plot_setup_survives_broken_indicator_data(tmp_path):
    df = _hs_frame().drop(columns=["rsi", "macd_hist"])
    row = {"symbol": "TESTUSDT", "status": "观察", "L": 120, "H": 140, "R": 160,
           "lt": 130, "y1": float(df["low"][130]), "rt": 150, "y2": float(df["low"][150]),
           "stop": None, "target_1h": 98.0, "target_15h": 96.0, "exchange": "binance"}
    out = tmp_path / "c2.png"
    scan_15m.plot_setup(df, row, out)
    assert out.exists()


def test_plot_setup_does_not_mutate_row(tmp_path):
    df = _hs_frame()
    row = {"symbol": "TESTUSDT", "status": "观望", "L": 120, "H": 140, "R": 160,
           "lt": 130, "y1": float(df["low"][130]), "rt": 150, "y2": float(df["low"][150]),
           "stop": 104.0, "target_1h": 98.0, "target_15h": 96.0, "exchange": "binance"}
    before = dict(row)
    scan_15m.plot_setup(df, row, tmp_path / "c3.png")
    # display-only neck fields may be added; signal levels and indices must not change
    core = {k: v for k, v in row.items()
            if not k.startswith("neck_") and not k.startswith("prop_")
            and not k.startswith("fib_") and not k.startswith("dual_neck_")}
    assert core == before
    assert "neck_s2" in row and "neck_class_cn" in row and "neck_angle_deg" in row
    assert "prop_text" in row and "prop_class_cn" in row
    assert "fib_ext" in row and "fib_text" in row
    assert "dual_neck_tier1" in row and "dual_neck_text" in row


def test_bollinger_values():
    df = _hs_frame()
    mid, up, lo = scan_15m.bollinger(df)
    c = df["close"].to_numpy()
    assert np.isnan(mid[18]) and np.isfinite(mid[19])
    assert abs(mid[50] - c[31:51].mean()) < 1e-9
    assert abs((up[50] - mid[50]) - 2 * c[31:51].std()) < 1e-9 and abs((mid[50] - lo[50]) - 2 * c[31:51].std()) < 1e-9


def test_plot_setup_draws_bollinger_and_volume(tmp_path, monkeypatch):
    df = _hs_frame()
    df["volume"] = np.linspace(1, 3, len(df))
    row = {"symbol": "TESTUSDT", "status": "观望", "L": 120, "H": 140, "R": 160,
           "lt": 130, "y1": float(df["low"][130]), "rt": 150, "y2": float(df["low"][150]),
           "stop": 104.0, "target_1h": 98.0, "target_15h": 96.0, "exchange": "binance"}
    import matplotlib.pyplot as plt
    seen = {}
    orig_close = plt.close

    def spy_close(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            seen["labels"] = [ln.get_label() for a in fig.axes for ln in a.get_lines()]
            seen["vol_bars"] = len(fig.axes[1].patches)
        return orig_close(fig)
    monkeypatch.setattr(scan_15m.plt, "close", spy_close)
    scan_15m.plot_setup(df, dict(row), tmp_path / "bb.png")
    assert any("BB20" in l for l in seen["labels"])
    assert "MA20" in seen["labels"]
    assert seen["vol_bars"] > 100


def test_plot_setup_survives_missing_volume(tmp_path):
    df = _hs_frame().drop(columns=["volume"])
    row = {"symbol": "TESTUSDT", "status": "观察", "L": 120, "H": 140, "R": 160,
           "lt": 130, "y1": float(df["low"][130]), "rt": 150, "y2": float(df["low"][150]),
           "stop": None, "target_1h": 98.0, "target_15h": 96.0, "exchange": "okx"}
    out = tmp_path / "nv.png"
    scan_15m.plot_setup(df, row, out)
    assert out.exists()


def test_neck_s2_formula_and_class_bands():
    # s2 = (RS trough − LS trough) / (head − neck at head)
    # neck_at is linear between troughs; put the head midway so neck@H = mid(y1, y2)
    lt, rt, H = 0, 10, 5
    y1, y2 = 100.0, 101.0  # RS trough 1 above LS
    # if neck at H is 100.5 and head is 110.5, height=10, s2=0.10 → 平 (inclusive)
    head = 100.5 + 10.0
    # verify helper agrees with the explicit formula
    from hs_backtest import neck_at
    hgt = head - neck_at(lt, y1, rt, y2, H)
    expected = (y2 - y1) / hgt
    assert scan_15m.neck_s2(lt, y1, rt, y2, H, head) == pytest.approx(expected)
    assert abs(expected - 0.10) < 1e-12
    assert scan_15m.neck_class_cn(expected) == "平"
    assert scan_15m.neck_class_cn(-0.10) == "平"
    assert scan_15m.neck_class_cn(-0.10 - 1e-9) == "下斜"
    assert scan_15m.neck_class_cn(0.10 + 1e-9) == "上翘"
    assert scan_15m.neck_class_cn(0.0) == "平"
    assert scan_15m.neck_class_cn(None) is None
    assert scan_15m.neck_s2(lt, y1, rt, y2, H, y1) is None  # non-positive height
    row = {"L": 0, "H": H, "R": 10, "lt": lt, "y1": y1, "rt": rt, "y2": y2}
    scan_15m.attach_neck_slope(row, head_high=head + 1.0)  # height 11 → s2≈0.0909 平
    assert row["neck_class_cn"] == "平" and row["neck_s2"] == pytest.approx(1.0 / 11.0, abs=1e-4)
    scan_15m.attach_neck_slope(row, head_high=head - 1.0)  # height 9 → s2≈0.1111 上翘
    assert row["neck_class_cn"] == "上翘"
    # downward neck
    row2 = {"L": 0, "H": H, "R": 10, "lt": lt, "y1": 101.0, "rt": rt, "y2": 100.0}
    scan_15m.attach_neck_slope(row2, head_high=110.5)
    # neck mid=100.5, height=10, s2=-0.10 → 平; make it steeper
    scan_15m.attach_neck_slope(row2, head_high=109.5)  # height 9, s2≈-0.111 下斜
    assert row2["neck_class_cn"] == "下斜"
    assert "下斜" in scan_15m.neck_info_text(row2["neck_s2"], row2["neck_class_cn"], -8.2)
    assert scan_15m.neck_info_text(0.11, "上翘", 7.4) == "颈线 s2=+0.11（上翘）｜图上约 +7°"


def test_plot_setup_records_visual_neck_angle(tmp_path, monkeypatch):
    import math
    import matplotlib.pyplot as plt
    df = _hs_frame()
    # tilt the right trough down a bit so the neckline is visibly sloped
    row = {"symbol": "TESTUSDT", "status": "观望", "L": 120, "H": 140, "R": 160,
           "lt": 130, "y1": float(df["low"][130]), "rt": 150,
           "y2": float(df["low"][150]) - 1.5,
           "stop": 104.0, "target_1h": 98.0, "target_15h": 96.0, "exchange": "binance"}
    captured = {}
    orig = plt.Figure.savefig

    def spy(self, *a, **k):
        ax = self.axes[0]
        p1 = ax.transData.transform((row["lt"], row["y1"]))
        p2 = ax.transData.transform((row["rt"], row["y2"]))
        captured["ang"] = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        captured["texts"] = [t.get_text() for t in ax.texts]
        return orig(self, *a, **k)

    monkeypatch.setattr(plt.Figure, "savefig", spy)
    scan_15m.plot_setup(df, row, tmp_path / "neck.png")
    assert "ang" in captured
    assert row["neck_angle_deg"] == pytest.approx(round(captured["ang"], 1))
    assert row["neck_class_cn"] in {"下斜", "平", "上翘"}
    assert row["neck_s2"] is not None
    blob = "\n".join(captured["texts"])
    assert "颈线 s2=" in blob and "图上约" in blob
    assert row["neck_class_cn"] in blob



def test_prop_class_bands_and_text():
    # time 0.25–1.50, height 0.30–1.00 inclusive
    assert scan_15m.prop_class_cn(0.25, 1.50, 0.30, 1.00) == "比例协调"
    assert scan_15m.prop_class_cn(0.80, 0.70, 0.60, 0.50) == "比例协调"
    assert scan_15m.prop_class_cn(0.25 - 1e-12, 0.80, 0.50, 0.50) == "比例偏小"
    assert scan_15m.prop_class_cn(0.80, 0.20, 0.50, 0.50) == "比例偏小"
    assert scan_15m.prop_class_cn(0.80, 0.70, 0.20, 0.50) == "比例偏小"
    assert scan_15m.prop_class_cn(1.50 + 1e-12, 0.80, 0.50, 0.50) == "比例失调"
    assert scan_15m.prop_class_cn(0.10, 2.00, 0.50, 0.50) == "比例失调"  # mixed
    assert scan_15m.prop_class_cn(None, None, None, None) is None
    t = scan_15m.prop_info_text(0.40, 0.60, 0.80, 0.90)
    assert t.startswith("左肩跨度0.4×头、右肩0.6×头；") and t.endswith("比例协调")
    t2 = scan_15m.prop_info_text(0.20, 0.60, 0.80, 0.90)
    assert "左肩跨度0.2×头" in t2 and t2.endswith("左肩跨度偏小")
    t3 = scan_15m.prop_info_text(0.80, 2.00, 0.80, 0.90)
    assert t3.endswith("右肩跨度失调")
    assert scan_15m.prop_info_text(None, None, None, None) == "肩头比例 —"


def test_shoulder_prop_metrics_formula():
    # Flat neck at 100. p0 at 0, L=4, lt=8, H=12, rt=16, R=20; post-RS hits neck at 24.
    n = 30
    high = np.full(n, 102.0)
    low = np.full(n, 101.0)   # above the neck so only the true troughs qualify
    high[4], high[12], high[20] = 110.0, 120.0, 110.0
    low[0] = 99.0    # pre-LS trough p0
    low[8] = 100.0   # LS trough
    low[16] = 100.0  # RS trough
    low[21:24] = 101.0
    low[24] = 100.0  # following trough at/below neck
    # raise bars around peaks so troughs stay unique
    for i in (4, 12, 20):
        if i - 1 >= 0:
            high[i - 1] = max(high[i - 1], 102.0)
    m = scan_15m.shoulder_prop_metrics(high, low, 4, 12, 20, 8, 16, 100.0, 100.0)
    assert m["prop_ls_time"] == pytest.approx(8 / 8)   # 8-0 over 16-8
    assert m["prop_rs_time"] == pytest.approx(8 / 8)   # 24-16 over 16-8
    # v6 amplitude: LS (110-99 + 110-100)/2 = 10.5; head (120-100 + 120-100)/2 = 20;
    # RS (110-100 + 110-100)/2 = 10
    assert m["prop_ls_height"] == pytest.approx(10.5 / 20)
    assert m["prop_rs_height"] == pytest.approx(10 / 20)
    row = {"L": 4, "H": 12, "R": 20, "lt": 8, "rt": 16, "y1": 100.0, "y2": 100.0,
           "stop": 121.0, "target_1h": 90.0, "status": "可入场"}
    before = dict(row)
    scan_15m.attach_shoulder_prop(row, high=high, low=low)
    for k in ("L", "H", "R", "lt", "rt", "y1", "y2", "stop", "target_1h", "status"):
        assert row[k] == before[k]
    assert row["prop_class_cn"] == "比例协调"
    assert "比例协调" in row["prop_text"]
    # no data -> display placeholder, still no crash and no trade-field change
    bare = {"L": 4, "H": 12, "R": 20, "lt": 8, "rt": 16, "y1": 100.0, "y2": 100.0, "stop": 1.0}
    scan_15m.attach_shoulder_prop(bare)
    assert bare["stop"] == 1.0 and bare["prop_text"] == "肩头比例 —" and bare["prop_class_cn"] is None


def test_plot_setup_draws_prop_text(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    df = _hs_frame()
    row = {"symbol": "TESTUSDT", "status": "观望", "L": 120, "H": 140, "R": 160,
           "lt": 130, "y1": float(df["low"][130]), "rt": 150, "y2": float(df["low"][150]),
           "stop": 104.0, "target_1h": 98.0, "target_15h": 96.0, "exchange": "binance"}
    captured = {}
    orig = plt.Figure.savefig

    def spy(self, *a, **k):
        ax = self.axes[0]
        captured["texts"] = [t.get_text() for t in ax.texts]
        return orig(self, *a, **k)

    monkeypatch.setattr(plt.Figure, "savefig", spy)
    scan_15m.plot_setup(df, row, tmp_path / "prop.png")
    blob = "\n".join(captured["texts"])
    assert row["prop_text"] and row["prop_text"] in blob
    assert "左肩跨度" in blob
    assert "颈线 s2=" in blob  # still on the line above
