"""1H/4H (tier htf) scan: detector, payload, scheduling, separate books, dedup. Never POSTs."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paper_journal as pj  # noqa: E402
import scan_htf  # noqa: E402
import scan_runner  # noqa: E402
from hs_backtest import add_indicators  # noqa: E402

TZ = ZoneInfo("Asia/Shanghai")


def synth_hs(break_bar: bool = True) -> pd.DataFrame:
    """Flat base, rise to LS 110, pullback 104, head 118, back to 104, RS 110, break below 104."""
    pts = [(0, 100), (100, 100), (110, 110), (116, 104), (124, 118), (132, 104), (138, 110), (146, 103)]
    path = []
    for (i0, p0), (i1, p1) in zip(pts, pts[1:]):
        for i in range(i0, i1):
            path.append(p0 + (p1 - p0) * (i - i0) / (i1 - i0))
    path.append(pts[-1][1])
    closes = np.array(path, dtype=float)
    # flat base wiggle so ATR is not zero and swings are strict
    closes[:100] += 0.15 * np.sin(np.arange(100))
    opens = np.r_[closes[0], closes[:-1]]
    highs = np.maximum(opens, closes) + 0.3
    lows = np.minimum(opens, closes) - 0.3
    highs[[111, 125, 139]] += 0.5  # strict shoulder/head peaks
    n = len(closes)
    df = pd.DataFrame({
        "dt": pd.date_range("2026-11-01", periods=n, freq="1h", tz="UTC"),
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000.0,
    })
    if not break_bar:
        df = df.iloc[:143].reset_index(drop=True)
    return add_indicators(df)


def test_detect_finds_entry_with_price_only_pivots():
    df = synth_hs()
    evs = scan_htf.detect(df)
    ents = [e for e in evs if e["kind"] == "entry"]
    assert len(ents) == 1
    e = ents[0]
    assert (e["L"], e["H"], e["R"]) == (111, 125, 139)
    assert e["stop"] > e["entry"] > e["t1"] > e["t2"] > 0
    assert e["rr"] >= 1.5


def test_volume_is_not_a_gate():
    df = synth_hs()
    df["volume"] = 1.0  # flat volume: no shrink at RS, no break volume spike
    df = add_indicators(df)
    assert any(e["kind"] == "entry" for e in scan_htf.detect(df))


def test_live_rows_watch_then_entry():
    rows = scan_htf.live_rows("TESTUSDT", "1h", synth_hs(break_bar=False), None)
    assert [r["status"] for r in rows] == ["观望"]
    r = rows[0]
    assert r["tier"] == "htf" and r["timeframe"] == "1h" and r["validated"] is False
    assert r["suggested_risk_pct"] == 0.5
    rows = scan_htf.live_rows("TESTUSDT", "1h", synth_hs(), None)
    assert [r["status"] for r in rows] == ["可入场"]
    r = rows[0]
    for k in ("official_entry", "official_entry_time", "stop", "target_1h", "target_15h", "rr",
              "leverage_cap", "valid_until_shanghai", "neck_s2", "neck_class_cn", "prop_text", "confirm_text"):
        assert r.get(k) is not None, k


def test_live_rows_expired_break_is_not_emitted():
    """A break older than fresh_bars (1H: 2) is not a live 可入场, even if price is still valid."""
    df = synth_hs()  # break is the last bar
    # still fresh with one extra closed bar
    extra = df.iloc[[-1]].copy()
    extra["dt"] = extra["dt"] + pd.Timedelta(hours=1)
    one = pd.concat([df, extra], ignore_index=True)
    assert any(r["status"] == "可入场" for r in scan_htf.live_rows("TESTUSDT", "1h", one, None))
    # three extra bars: ev.t < n-1-2 -> dropped
    more = df
    last = df.iloc[-1]
    for i in range(1, 4):
        row = last.copy()
        row["dt"] = last["dt"] + pd.Timedelta(hours=i)
        more = pd.concat([more, pd.DataFrame([row])], ignore_index=True)
    rows = scan_htf.live_rows("TESTUSDT", "1h", more, None)
    assert [r["status"] for r in rows] == []


def test_ls_pullback_uses_closes_not_wicks():
    # LS at 1; next bar has a long lower wick to 90 but closes at 99.5; next higher high at 3.
    high = np.array([95, 100, 99.8, 101], float)
    close = np.array([94, 99.0, 99.5, 100.5], float)
    nge = scan_htf.next_greater(high)
    assert scan_htf.ls_pullback_close(high, close, nge, 1, 3) == pytest.approx(1.0)


def test_leverage_cap_keeps_liq_at_2x_stop():
    lev = scan_htf.leverage_cap(100.0, 102.0)
    assert lev == 22
    assert (1.0 / lev - scan_htf.MMR) >= 2 * 0.02
    assert scan_htf.leverage_cap(100.0, 99.0) is None


def test_htf_due_schedule():
    d = lambda h, m: datetime(2026, 11, 12, h, m, tzinfo=TZ)
    assert scan_runner.htf_due(d(14, 2)) == ["1h"]
    assert scan_runner.htf_due(d(16, 2)) == ["1h", "4h"]
    assert scan_runner.htf_due(d(0, 2)) == ["1h", "4h"]
    assert scan_runner.htf_due(d(8, 2)) == ["1h", "4h"]
    assert scan_runner.htf_due(d(14, 17)) == []
    assert scan_runner.htf_due(d(16, 47)) == []


def htf_sig(tf="1h", status="可入场", symbol="BTCUSDT", exchange="binance", H="2026-11-12 01:00"):
    return {
        "symbol": symbol, "exchange": exchange, "coin": "BTC", "status": status, "tier": "htf", "timeframe": tf,
        "L_time": "2026-11-11 18:00", "H_time": H, "R_time": "2026-11-12 08:00",
        "official_entry": 100.0, "official_entry_time": "2026-11-12 14:00", "suggest_entry": 100.0,
        "stop": 102.0, "target_1h": 96.0, "target_15h": 94.0, "current": 100.0, "rr": 2.0, "neck": 100.5,
        "leverage_cap": 22, "valid_until_shanghai": "2026-11-12 17:00", "neck_s2": 0.05, "neck_class_cn": "平",
        "neck_angle_deg": 2.0, "prop_text": "比例协调", "chart": "/tmp/x.png", "suggested_risk_pct": 0.5,
    }


def test_pattern_id_htf_separate_from_15m_and_shared_with_journal():
    s = htf_sig()
    assert scan_runner.pattern_id(s) == pj.pattern_id(s) == "1h|BTCUSDT|可入场|2026-11-12 01:00"
    s4 = htf_sig(tf="4h")
    assert scan_runner.pattern_id(s4) != scan_runner.pattern_id(s)
    s15 = dict(s); s15.pop("timeframe")
    assert scan_runner.pattern_id(s15) == "BTCUSDT|可入场|2026-11-11 18:00|2026-11-12 01:00|2026-11-12 08:00"


def test_build_htf_payload_fields_and_night():
    rows = [htf_sig(), htf_sig(symbol="BTC-USDT-SWAP", exchange="okx")]
    night = datetime(2026, 11, 12, 3, 2, tzinfo=TZ)
    p = scan_runner.build_htf_payload(rows, "1h", "2026-11-12 03:02", "2026-11-12 02:00", now=night)
    assert p["tier"] == "htf" and p["timeframe"] == "1h" and p["validated"] is False
    assert p["suggested_risk_pct"] == 0.5 and p["counts_toward_breaker"] is False
    assert p["quiet"] is True and p["night"] is True
    assert p["exchanges_in_payload"] == ["binance", "okx"]
    s = p["signals"][0]
    for k in ("exchange", "exchange_cn", "symbol", "coin", "entry", "stop", "target_1h", "target_15h", "rr",
              "leverage_cap", "valid_until_shanghai", "neck_s2", "neck_class_cn", "neck_angle_deg",
              "prop_text", "chart", "timeframe", "tier"):
        assert k in s, k
    assert s["tier"] == "htf"
    day = datetime(2026, 11, 12, 14, 2, tzinfo=TZ)
    assert scan_runner.build_htf_payload(rows, "4h", "x", "y", now=day)["quiet"] is False


def test_htf_journal_books_separate_and_never_trip(tmp_path):
    paths = pj.JournalPaths.under(tmp_path, book=pj.BOOK_HTF_1H)
    assert paths.json_path.name == "paper_htf_1h_journal.json"
    assert pj.JournalPaths.under(tmp_path, book=pj.BOOK_HTF_4H).json_path.name == "paper_htf_4h_journal.json"
    # three stopped-out trades in a row: htf book must not trip any breaker
    bars = pd.DataFrame({
        "dt": pd.date_range("2026-11-12 06:00", periods=8, freq="15min", tz="UTC"),
        "open": 100.0, "high": 103.0, "low": 99.5, "close": 101.0, "volume": 1.0,
    })
    sigs = []
    for i in range(3):
        s = htf_sig(H=f"2026-11-12 0{i}:00", symbol=f"C{i}USDT")
        s["official_entry_time"] = "2026-11-12 14:00"
        sigs.append(s)
    j = pj.update_from_scan({"signals": sigs}, now=datetime(2026, 11, 12, 16, 2, tzinfo=TZ),
                            paths=paths, fetch_klines_fn=lambda sym, start: bars, book=pj.BOOK_HTF_1H)
    assert j["book"] == "htf1h" and j["risk_pct"] == pytest.approx(0.005)
    assert not j["breaker"]["tripped"]
    assert all(t["timeframe"] == "1h" for t in j["trades"])
    assert "breaker=n/a" in pj.journal_log_line(j)
    assert not (tmp_path / "paper_journal.json").exists()


def test_run_htf_posts_once_dedups_and_ignores_15m_breaker(tmp_path, monkeypatch):
    out = tmp_path / "htf"
    (out / "1h").mkdir(parents=True)
    data = {"asof_shanghai": "2026-11-12 15:02", "last_bar_shanghai": "2026-11-12 14:00", "n_entry": 1, "n_watch": 1,
            "exchanges": ["binance"], "okx_ok": True,
            "signals": [htf_sig(), htf_sig(status="观望", symbol="ETHUSDT", H="2026-11-12 05:00")]}
    (out / "1h" / "signals.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(scan_runner, "HTF_OUT_DIR", out)
    monkeypatch.setattr(scan_runner, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(scan_runner, "LOG_PATH", tmp_path / "log.txt")
    monkeypatch.setattr(scan_runner, "run_htf_scan", lambda tfs: 0)
    monkeypatch.setattr(pj, "OUT_DIR", tmp_path)
    monkeypatch.setattr(pj.JournalPaths, "htf", classmethod(lambda cls, tf: cls.under(tmp_path, book=pj.HTF_BOOKS[tf])))
    monkeypatch.setattr(pj, "default_fetch_klines", lambda sym, start=None: pd.DataFrame())
    # a tripped 15m formal breaker must not matter here
    monkeypatch.setattr(pj, "is_tripped", lambda *a, **k: True)
    posts = []
    monkeypatch.setattr(scan_runner, "post_webhook", lambda payload, styles=None: (posts.append(payload) or (200, "bearer", "")))
    monkeypatch.setattr(scan_runner, "remembered_auth_style", lambda: "bearer")
    assert scan_runner.run_htf(["1h"]) == 0
    assert len(posts) == 1
    p = posts[0]
    assert p["tier"] == "htf" and p["timeframe"] == "1h" and len(p["signals"]) == 1
    assert p["signals"][0]["symbol"] == "BTCUSDT" and p["n_entry"] == 1 and p["n_watch"] == 0
    assert scan_runner.run_htf(["1h"]) == 0
    assert len(posts) == 1  # same heads: not pushed again
    assert (tmp_path / "paper_htf_1h_journal.json").exists()
    st = json.loads((tmp_path / "state.json").read_text())
    assert all(v["tier"] == "htf" for v in st["pushed"].values())
    assert all(v["status"] == "可入场" for v in st["pushed"].values())
    # pre-break 观望 was not marked pushed; when the same head later breaks, 可入场 is sent
    broken = htf_sig(status="可入场", symbol="ETHUSDT", H="2026-11-12 05:00")
    data2 = dict(data)
    data2["signals"] = [htf_sig(), broken]
    data2["n_entry"], data2["n_watch"] = 2, 0
    (out / "1h" / "signals.json").write_text(json.dumps(data2, ensure_ascii=False), encoding="utf-8")
    assert scan_runner.run_htf(["1h"]) == 0
    assert len(posts) == 2
    assert [s["symbol"] for s in posts[1]["signals"]] == ["ETHUSDT"]


def test_run_htf_dry_run_never_posts(tmp_path, monkeypatch):
    out = tmp_path / "htf"
    (out / "4h").mkdir(parents=True)
    (out / "4h" / "signals.json").write_text(json.dumps({"signals": [htf_sig(tf="4h")]}), encoding="utf-8")
    monkeypatch.setattr(scan_runner, "HTF_OUT_DIR", out)
    monkeypatch.setattr(scan_runner, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(scan_runner, "LOG_PATH", tmp_path / "log.txt")
    monkeypatch.setattr(scan_runner, "run_htf_scan", lambda tfs: 0)
    monkeypatch.setattr(pj, "default_fetch_klines", lambda sym, start=None: pd.DataFrame())

    def boom(*a, **k):
        raise AssertionError("must not post in dry run")
    monkeypatch.setattr(scan_runner, "post_webhook", boom)
    paths = {pj.BOOK_HTF_4H: pj.JournalPaths.under(tmp_path, book=pj.BOOK_HTF_4H)}
    assert scan_runner.run_htf(["4h"], dry_run=True, journal_paths=paths) == 0
    assert not (tmp_path / "state.json").exists()


def test_run_once_runs_htf_after_15m(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(scan_runner, "LOCK_PATH", tmp_path / "lock")
    monkeypatch.setattr(scan_runner, "LOG_PATH", tmp_path / "log.txt")
    monkeypatch.setattr(scan_runner, "_cycle_15m", lambda **k: order.append("15m") or 0)
    monkeypatch.setattr(scan_runner, "run_htf", lambda tfs, **k: order.append(tuple(tfs)) or 0)
    assert scan_runner.run_once(htf_tfs=["1h", "4h"]) == 0
    assert order == ["15m", ("1h", "4h")]
    order.clear()
    assert scan_runner.run_once() == 0
    assert order == ["15m"]



def _write_tf_signals(root: Path, tf: str, last_bar: str, extra: dict | None = None):
    d = root / tf
    d.mkdir(parents=True, exist_ok=True)
    data = {"last_bar_shanghai": last_bar, "asof_shanghai": "2026-11-12 15:08", "signals": []}
    if extra:
        data.update(extra)
    (d / "signals.json").write_text(__import__("json").dumps(data), encoding="utf-8")


def test_latest_closed_bar_open_times():
    """Bars are labeled by open time; the 4H close at 16:00 Beijing is last_bar=12:00."""
    d = lambda h, m, s=0, day=12: datetime(2026, 11, day, h, m, s, tzinfo=TZ)
    # 17:04 (the missed-16:02 aftermath): 1H last closed is 16:00, 4H is 12:00–16:00, 15m is 16:45
    now = d(17, 4)
    assert scan_runner.latest_closed_bar("1h", now) == d(16, 0)
    assert scan_runner.latest_closed_bar("4h", now) == d(12, 0)
    assert scan_runner.latest_closed_bar("15m", now) == d(16, 45)
    # exact 16:00 close: the 12:00 4H bar has just closed; 15:00 1H bar has just closed
    at_close = d(16, 0)
    assert scan_runner.latest_closed_bar("4h", at_close) == d(12, 0)
    assert scan_runner.latest_closed_bar("1h", at_close) == d(15, 0)
    # 15:08, before the 16:00 4H close: still the 08:00–12:00 bar
    assert scan_runner.latest_closed_bar("4h", d(15, 8)) == d(8, 0)
    # 20:00 close and midnight wrap
    assert scan_runner.latest_closed_bar("4h", d(20, 0)) == d(16, 0)
    assert scan_runner.latest_closed_bar("4h", d(0, 2, day=13)) == d(20, 0)
    # 15m boundaries
    assert scan_runner.latest_closed_bar("15m", d(17, 15)) == d(17, 0)
    assert scan_runner.latest_closed_bar("15m", d(17, 14, 59)) == d(16, 45)
    # naive datetime is treated as Shanghai
    naive = datetime(2026, 11, 12, 17, 4)
    assert scan_runner.latest_closed_bar("4h", naive) == d(12, 0)


def test_htf_catchup_stale_4h_after_missed_1602(tmp_path):
    """The 16:02 4H slot was missed; last 4H scan is still the 08:00 bar. 1H is current."""
    _write_tf_signals(tmp_path, "4h", "2026-11-12 08:00")
    _write_tf_signals(tmp_path, "1h", "2026-11-12 16:00")
    now = datetime(2026, 11, 12, 17, 4, tzinfo=TZ)
    assert scan_runner.htf_catchup(now, signals_dir=tmp_path) == ["4h"]
    # 17:02 is a 1H-only slot; catch-up still adds 4h
    assert scan_runner.htf_due(datetime(2026, 11, 12, 17, 2, tzinfo=TZ)) == ["1h"]
    assert scan_runner.resolve_htf_tfs(["1h"], now=now, signals_dir=tmp_path) == ["1h", "4h"]
    # a later :17/:32/:47 slot has no scheduled HTF, but 4H still catch-up
    assert scan_runner.htf_due(datetime(2026, 11, 12, 17, 17, tzinfo=TZ)) == []
    assert scan_runner.resolve_htf_tfs([], now=now, signals_dir=tmp_path) == ["4h"]


def test_htf_catchup_noop_when_current(tmp_path):
    now = datetime(2026, 11, 12, 17, 4, tzinfo=TZ)
    _write_tf_signals(tmp_path, "4h", "2026-11-12 12:00")  # bar that closed at 16:00
    _write_tf_signals(tmp_path, "1h", "2026-11-12 16:00")
    assert scan_runner.htf_catchup(now, signals_dir=tmp_path) == []
    assert scan_runner.resolve_htf_tfs(["1h"], now=now, signals_dir=tmp_path) == ["1h"]
    # a last_bar in the future / equal is not treated as lagging
    _write_tf_signals(tmp_path, "4h", "2026-11-12 16:00")
    assert scan_runner.htf_catchup(now, signals_dir=tmp_path) == []


def test_htf_catchup_missing_or_bad_signals(tmp_path):
    now = datetime(2026, 11, 12, 17, 4, tzinfo=TZ)
    # no files at all
    assert scan_runner.htf_catchup(now, signals_dir=tmp_path) == ["1h", "4h"]
    # empty / unreadable last_bar counts as unscanned
    _write_tf_signals(tmp_path, "1h", "")
    (tmp_path / "4h").mkdir(parents=True, exist_ok=True)
    (tmp_path / "4h" / "signals.json").write_text("not-json", encoding="utf-8")
    assert scan_runner.htf_catchup(now, signals_dir=tmp_path) == ["1h", "4h"]
    _write_tf_signals(tmp_path, "1h", "2026-11-12 16:00")
    assert scan_runner.htf_catchup(now, signals_dir=tmp_path) == ["4h"]


def test_m15_and_catchup_plan_after_missed_slot(tmp_path):
    now = datetime(2026, 11, 12, 17, 20, tzinfo=TZ)
    _write_tf_signals(tmp_path, "1h", "2026-11-12 16:00")
    _write_tf_signals(tmp_path, "4h", "2026-11-12 08:00")
    m15 = tmp_path / "m15_signals.json"
    m15.write_text('{"last_bar_shanghai": "2026-11-12 16:45"}', encoding="utf-8")
    plan = scan_runner.catchup_plan(now=now, signals_dir=tmp_path, m15_path=m15)
    assert plan["htf_tfs"] == ["4h"]
    assert plan["need_15m"] is True  # latest closed 15m is 17:00
    m15.write_text('{"last_bar_shanghai": "2026-11-12 17:00"}', encoding="utf-8")
    plan2 = scan_runner.catchup_plan(now=now, signals_dir=tmp_path, m15_path=m15)
    assert plan2["need_15m"] is False
    assert plan2["htf_tfs"] == ["4h"]
    # all current -> empty plan
    _write_tf_signals(tmp_path, "4h", "2026-11-12 12:00")
    plan3 = scan_runner.catchup_plan(now=now, signals_dir=tmp_path, m15_path=m15)
    assert plan3 == {"htf_tfs": [], "need_15m": False}


def test_run_catchup_if_needed_noop_and_dry_run_never_posts(tmp_path, monkeypatch):
    """Catch-up must not touch live logs or POST. Current bars are a no-op; dry-run never posts."""
    log_path = tmp_path / "runner.log"
    monkeypatch.setattr(scan_runner, "LOG_PATH", log_path)
    monkeypatch.setattr(scan_runner, "LOCK_PATH", tmp_path / "lock")
    monkeypatch.setattr(scan_runner, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(scan_runner, "HEARTBEAT_PATH", tmp_path / "hb.json")

    def boom(*a, **k):
        raise AssertionError("must not POST")

    monkeypatch.setattr(scan_runner, "post_webhook", boom)
    now = datetime(2026, 11, 12, 17, 4, tzinfo=TZ)
    _write_tf_signals(tmp_path, "1h", "2026-11-12 16:00")
    _write_tf_signals(tmp_path, "4h", "2026-11-12 12:00")
    m15 = tmp_path / "m15_signals.json"
    m15.write_text('{"last_bar_shanghai": "2026-11-12 16:45"}', encoding="utf-8")

    called = []

    def fake_run_once(**k):
        called.append(k)
        return 0

    monkeypatch.setattr(scan_runner, "run_once", fake_run_once)
    # all current: no run_once, no log write required
    assert scan_runner.run_catchup_if_needed(now=now, signals_dir=tmp_path, m15_path=m15) == 0
    assert called == []
    assert not log_path.exists()

    # stale 4H: dry-run still never posts (post_webhook is boom)
    _write_tf_signals(tmp_path, "4h", "2026-11-12 08:00")
    assert scan_runner.run_catchup_if_needed(
        now=now, signals_dir=tmp_path, m15_path=m15, dry_run=True
    ) == 0
    assert len(called) == 1
    assert called[0]["htf_tfs"] == ["4h"]
    assert called[0]["skip_15m"] is True
    assert called[0]["dry_run"] is True
    # log went only to the temp path
    assert log_path.exists()
    assert "catch-up scan" in log_path.read_text(encoding="utf-8")


def test_run_catchup_includes_15m_when_slot_missed(tmp_path, monkeypatch):
    monkeypatch.setattr(scan_runner, "LOG_PATH", tmp_path / "runner.log")
    monkeypatch.setattr(scan_runner, "post_webhook", lambda *a, **k: (_ for _ in ()).throw(AssertionError("POST")))
    now = datetime(2026, 11, 12, 16, 40, tzinfo=TZ)  # missed 16:02/16:17/16:32; latest 15m is 16:30
    _write_tf_signals(tmp_path, "1h", "2026-11-12 14:00")  # missed the 16:02 scan of the 15:00 1H bar
    _write_tf_signals(tmp_path, "4h", "2026-11-12 08:00")
    m15 = tmp_path / "m15_signals.json"
    m15.write_text('{"last_bar_shanghai": "2026-11-12 16:00"}', encoding="utf-8")
    called = []
    monkeypatch.setattr(scan_runner, "run_once", lambda **k: called.append(k) or 0)
    assert scan_runner.run_catchup_if_needed(now=now, signals_dir=tmp_path, m15_path=m15) == 0
    assert called[0]["htf_tfs"] == ["1h", "4h"]
    assert called[0]["skip_15m"] is False


def synth_hs_deep_break():
    """Same pattern as synth_hs, but the break bar closes deep so close-R:R < 1.5."""
    df = synth_hs()
    evs = [e for e in scan_htf.detect(df) if e["kind"] == "entry"]
    assert evs, "baseline synth_hs must still produce an entry"
    e = evs[0]
    t, brk, stop, t1 = int(e["t"]), float(e["brk"]), float(e["stop"]), float(e["t1"])
    cap = (t1 + 1.5 * stop) / 2.5
    deep = min(brk, cap) - (min(brk, cap) - t1) * 0.2
    df = df.copy()
    df.loc[t, "close"] = deep
    df.loc[t, "open"] = max(float(df.loc[t, "open"]), brk + 0.05)
    df.loc[t, "low"] = min(float(df.loc[t, "low"]), deep) - 0.05
    df.loc[t, "high"] = min(float(df.loc[t, "high"]), (brk + stop) / 2)
    return df


def test_detect_emits_retest_when_close_rr_fails():
    df = synth_hs_deep_break()
    evs = scan_htf.detect(df)
    rets = [e for e in evs if e["kind"] == "retest"]
    ents = [e for e in evs if e["kind"] == "entry"]
    assert len(rets) == 1 and ents == []
    e = rets[0]
    assert e["entry"] == pytest.approx(e["brk"])
    assert e["stop"] > e["brk"] > e["t1"] and e["t2"] > 0
    assert e["rr"] >= 1.5
    # once per head
    assert len({e["H"] for e in rets}) == 1


def test_live_rows_retest_fields_freshness_and_invalidation():
    df = synth_hs_deep_break()
    rows = scan_htf.live_rows("TESTUSDT", "1h", df, None, nested=False)
    assert [r["status"] for r in rows] == [scan_htf.STATUS_RETEST]
    r = rows[0]
    assert r["status"] == "已破位·等回踩"
    assert r["tier"] == "htf" and r["suggested_risk_pct"] == 0.5
    assert r["official_entry"] == pytest.approx(r["break_level"])
    assert r["suggest_entry"] == pytest.approx(r["break_level"])
    assert r["rr"] >= 1.5
    assert r["valid_until_shanghai"]
    assert r["official_entry_time"] and r["break_bar_shanghai"]
    assert r["leverage_cap"] is not None
    assert "挂限价空单等回踩" in r["note"] and "过期撤单" in r["note"]
    # price already tagging the break level does not drop a fresh retest (limit would have filled)
    tagged = df.copy()
    last = tagged.iloc[-1].copy()
    last["dt"] = last["dt"] + pd.Timedelta(hours=1)
    last["high"] = max(float(last["high"]), float(r["break_level"]) + 0.01)
    last["close"] = float(r["break_level"]) - 0.2
    last["open"] = float(r["break_level"]) - 0.3
    last["low"] = min(float(last["low"]), float(last["close"]))
    # keep high below the stop
    last["high"] = min(float(last["high"]), float(r["stop"]) - 0.05)
    tagged = pd.concat([tagged, pd.DataFrame([last])], ignore_index=True)
    rows2 = scan_htf.live_rows("TESTUSDT", "1h", tagged, None, nested=False)
    assert [x["status"] for x in rows2] == [scan_htf.STATUS_RETEST]
    # stop hit after the break drops it
    stopped = df.copy()
    hit = stopped.iloc[-1].copy()
    hit["dt"] = hit["dt"] + pd.Timedelta(hours=1)
    hit["high"] = float(r["stop"]) + 0.2
    hit["close"] = float(r["official_entry"])
    stopped = pd.concat([stopped, pd.DataFrame([hit])], ignore_index=True)
    assert scan_htf.live_rows("TESTUSDT", "1h", stopped, None, nested=False) == []
    # stale beyond fresh_bars (1H: 2)
    stale = df
    last0 = df.iloc[-1]
    for i in range(1, 4):
        row = last0.copy()
        row["dt"] = last0["dt"] + pd.Timedelta(hours=i)
        stale = pd.concat([stale, pd.DataFrame([row])], ignore_index=True)
    assert scan_htf.live_rows("TESTUSDT", "1h", stale, None, nested=False) == []


def test_entry_preferred_over_retest_same_head():
    df = synth_hs()  # close-R:R already qualifies as 可入场
    rows = scan_htf.live_rows("TESTUSDT", "1h", df, None, nested=False)
    assert [r["status"] for r in rows] == ["可入场"]


def test_actionable_and_pattern_id_include_retest():
    assert scan_htf.STATUS_RETEST == "已破位·等回踩"
    assert "已破位·等回踩" in scan_runner.ACTIONABLE
    assert "已破位·等回踩" in scan_runner.HTF_ACTIONABLE
    assert "已破位·等回踩" in scan_runner.NESTED_ACTIONABLE
    s = htf_sig(status="已破位·等回踩")
    assert scan_runner.pattern_id(s) == pj.pattern_id(s) == "1h|BTCUSDT|已破位·等回踩|2026-11-12 01:00"
    assert scan_runner.pattern_id(s) != scan_runner.pattern_id(htf_sig())


def test_htf_payload_includes_retest_count():
    rows = [htf_sig(status="已破位·等回踩")]
    p = scan_runner.build_htf_payload(rows, "1h", "2026-11-12 22:04", "2026-11-12 21:00")
    assert p["n_retest"] == 1 and p["n_entry"] == 0
    assert p["signals"][0]["status"] == "已破位·等回踩"
    assert p["signals"][0]["entry"] == 100.0


def test_journal_skips_retest_limit_signals(tmp_path):
    """Paper ledger still assumes a market fill; retest (limit at brk) is not ingested."""
    paths = pj.JournalPaths.under(tmp_path, book=pj.BOOK_HTF_1H)
    s = htf_sig(status="已破位·等回踩")
    j = pj.update_from_scan(
        {"signals": [s]},
        now=datetime(2026, 11, 12, 22, 4, tzinfo=TZ),
        paths=paths,
        book=pj.BOOK_HTF_1H,
        fetch_klines_fn=lambda sym, start=None: pd.DataFrame(),
    )
    assert j["trades"] == []
