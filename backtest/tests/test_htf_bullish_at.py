import pandas as pd
import numpy as np
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from hs_backtest_v2_3lines import _searchsorted_right, _ts, htf_bullish_at


def _ms_series(ms_list):
    return pd.Series(pd.to_datetime(ms_list, unit="ms", utc=True))


def test_searchsorted_accepts_now_against_ms_bars():
    t = _ms_series([1_700_000_000_000, 1_700_000_360_000, 1_700_000_720_000])
    ts = pd.Timestamp("2023-11-14 22:13:40.123456", tz="UTC")  # us precision, not on a ms boundary
    # Should not raise, and land after the first bar (22:13:20) / before or at later ones
    idx = _searchsorted_right(t, ts)
    assert idx == 1  # after 22:13:20, before 22:19:20
    # 1_700_000_000_000 = 2023-11-14 22:13:20 UTC
    # 1_700_000_360_000 = 22:19:20
    # ts = 22:13:40 → right-search insertion point is 1


def test_searchsorted_now_style_does_not_crash():
    t = _ms_series([1_700_000_000_000, 1_700_000_360_000])
    ts = pd.Timestamp.now(tz="UTC")
    # now is far after 2023, so insertion point is len(t)
    assert _searchsorted_right(t, ts) == 2


def test_searchsorted_matches_on_exact_ms_boundary():
    t = _ms_series([1_700_000_000_000, 1_700_000_360_000, 1_700_000_720_000])
    ts = pd.Timestamp("2023-11-14 22:19:20", tz="UTC")  # exact second bar
    assert _searchsorted_right(t, ts) == int(t.searchsorted(ts.as_unit("ms"), side="right"))


def test_htf_bullish_unknown_if_empty():
    assert htf_bullish_at(None, pd.Timestamp.now(tz="UTC"), pd.Timedelta(hours=1)) is True
    assert htf_bullish_at(pd.DataFrame({"dt": []}), pd.Timestamp.now(tz="UTC"), pd.Timedelta(hours=1)) is True


def test_htf_bullish_uses_closed_bar_only():
    # Build a 1H frame with last closed bar ending before ts
    times = pd.to_datetime(
        [1_700_000_000_000, 1_700_003_600_000, 1_700_007_200_000], unit="ms", utc=True
    )
    # is_4h_bullish needs ema20, ema60, close
    df = pd.DataFrame({
        "dt": times,
        "open": [100.0, 101.0, 102.0],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
        "close": [100.5, 101.5, 90.0],  # last not clearly bullish
        "volume": [1.0, 1.0, 1.0],
        "ema20": [90.0, 90.0, 95.0],
        "ema60": [100.0, 100.0, 100.0],  # ema20 < ema60 → not bullish
    })
    # Query time: 1 hour after the last open, so last bar is closed
    ts = times[-1] + pd.Timedelta(hours=1, milliseconds=123) + pd.Timedelta(microseconds=456)
    assert htf_bullish_at(df, ts, pd.Timedelta(hours=1)) is False
