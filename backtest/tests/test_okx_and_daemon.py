"""Tests for OKX support and the stall-proof daemon sleep. No network, never POSTs."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import okx_market  # noqa: E402
import paper_journal as pj  # noqa: E402
import scan_15m  # noqa: E402
import scan_runner  # noqa: E402
from hs_backtest import slippage_for  # noqa: E402

TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 11, 11, 11, 2, tzinfo=TZ)


# ---------------- naming ----------------

def test_coin_key_matches_across_exchanges():
    assert okx_market.coin_key("1000PEPEUSDT", "binance") == "PEPE"
    assert okx_market.coin_key("PEPE-USDT-SWAP", "okx") == "PEPE"
    assert okx_market.coin_key("BTCUSDT", "binance") == okx_market.coin_key("BTC-USDT-SWAP", "okx") == "BTC"
    assert okx_market.coin_key("1INCHUSDT", "binance") == "1INCH"  # not stripped (no '1000' prefix)
    assert okx_market.binance_equiv("ETH-USDT-SWAP") == "ETHUSDT"


def test_okx_slippage_uses_major_table():
    assert pj._cost_symbol("BTC-USDT-SWAP") == "BTCUSDT"
    assert slippage_for(pj._cost_symbol("BTC-USDT-SWAP")) == slippage_for("BTCUSDT")
    assert slippage_for(pj._cost_symbol("PEPE-USDT-SWAP")) == slippage_for("1000PEPEUSDT")
    assert pj.exchange_of("BTC-USDT-SWAP") == "okx"
    assert pj.exchange_of({"symbol": "BTCUSDT"}) == "binance"
    assert pj.exchange_of({"symbol": "X", "exchange": "okx"}) == "okx"


# ---------------- OKX candles parsing ----------------

def _okx_rows(start_ms, n, step=15 * 60 * 1000, last_unconfirmed=True):
    rows = []
    for i in range(n):
        ts = start_ms + i * step
        c = "0" if (last_unconfirmed and i == n - 1) else "1"
        rows.append([str(ts), "10", "11", "9", "10.5", "100", "1", "10.5", c])
    return list(reversed(rows))  # OKX returns newest first


def test_fetch_okx_klines_drops_open_bar_and_orders(monkeypatch):
    now_ms = int(pd.Timestamp.now(tz="UTC").floor("15min").timestamp() * 1000)
    start = now_ms - 9 * 15 * 60 * 1000
    rows = _okx_rows(start, 10)
    calls = []

    def fake_get(path, params, limiter, retries=2):
        calls.append((path, dict(params)))
        if "after" in params:
            return []
        return rows

    monkeypatch.setattr(okx_market, "_get", fake_get)
    df = okx_market.fetch_okx_klines("BTC-USDT-SWAP", "15m", 5)
    assert len(df) == 5
    assert df["open_time"].is_monotonic_increasing
    assert int(df["open_time"].iloc[-1]) == start + 8 * 15 * 60 * 1000  # unconfirmed bar dropped
    assert {"open", "high", "low", "close", "volume", "dt", "close_time"} <= set(df.columns)
    assert df["volume"].iloc[0] == 1.0  # base-coin volume (volCcy)


def test_okx_get_raises_on_error_code(monkeypatch):
    class R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"code": "51001", "msg": "Instrument ID does not exist", "data": []}
    monkeypatch.setattr(okx_market.SESSION, "get", lambda *a, **k: R())
    with pytest.raises(okx_market.OkxError):
        okx_market._get("/x", {}, okx_market.RateLimiter(100, 1.0))


def test_rate_limiter_caps_calls():
    import time
    rl = okx_market.RateLimiter(5, 0.5)
    t0 = time.monotonic()
    for _ in range(11):
        rl.wait()
    assert time.monotonic() - t0 >= 0.9  # 11 calls at 5/0.5s need >= 2 full windows


def test_okx_pool_filters(monkeypatch):
    tickers = [
        {"instId": "BTC-USDT-SWAP", "volCcy24h": "1000", "last": "70000"},
        {"instId": "NVDA-USDT-SWAP", "volCcy24h": "1e9", "last": "100"},   # stock -> drop
        {"instId": "USDC-USDT-SWAP", "volCcy24h": "1e9", "last": "1"},     # stable -> drop
        {"instId": "ETH-USD-SWAP", "volCcy24h": "1e9", "last": "3000"},    # coin-margined -> drop
        {"instId": "NEW-USDT-SWAP", "volCcy24h": "1e9", "last": "1"},      # listed 1 day ago -> drop
        {"instId": "DOGE-USDT-SWAP", "volCcy24h": "1e6", "last": "0.2"},
    ]
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    old = str(now_ms - 400 * 86400000)
    insts = [
        {"instId": "BTC-USDT-SWAP", "instCategory": "1", "state": "live", "settleCcy": "USDT", "listTime": old},
        {"instId": "NVDA-USDT-SWAP", "instCategory": "3", "state": "live", "settleCcy": "USDT", "listTime": old},
        {"instId": "USDC-USDT-SWAP", "instCategory": "1", "state": "live", "settleCcy": "USDT", "listTime": old},
        {"instId": "NEW-USDT-SWAP", "instCategory": "1", "state": "live", "settleCcy": "USDT",
         "listTime": str(now_ms - 86400000)},
        {"instId": "DOGE-USDT-SWAP", "instCategory": "1", "state": "live", "settleCcy": "USDT", "listTime": old},
    ]

    def fake_get(path, params, limiter, retries=2):
        return {"/api/v5/market/tickers": tickers, "/api/v5/public/instruments": insts}[path]

    monkeypatch.setattr(okx_market, "_get", fake_get)
    pool = okx_market.build_okx_pool(top_n=30, check_funding=False)
    assert [s["instId"] for s in pool["symbols"]] == ["BTC-USDT-SWAP", "DOGE-USDT-SWAP"]


# ---------------- scan: exchange tagging, charts, OKX failure isolation ----------------

def test_chart_names_do_not_collide():
    b = {"symbol": "BTCUSDT", "status": "可入场", "exchange": "binance"}
    o = {"symbol": "BTC-USDT-SWAP", "status": "可入场", "exchange": "okx"}
    ob = {"symbol": "BTC-USDT-SWAP", "status": "观察", "exchange": "okx"}
    assert scan_15m.chart_path(b).name == "BTCUSDT.png"
    assert scan_15m.chart_path(o).name == "okx_BTC-USDT-SWAP.png"
    assert scan_15m.chart_path(ob).name == "okx_BTC-USDT-SWAP_observe.png"


def test_same_coin_tagging():
    rows = [
        {"symbol": "1000PEPEUSDT", "exchange": "binance", "coin": "PEPE", "status": "可入场", "tier": "formal"},
        {"symbol": "PEPE-USDT-SWAP", "exchange": "okx", "coin": "PEPE", "status": "观察", "tier": "watch"},
        {"symbol": "SOL-USDT-SWAP", "exchange": "okx", "coin": "SOL", "status": "观望", "tier": "formal"},
    ]
    assert scan_15m.tag_same_coin(rows) == 2
    assert rows[0]["same_coin_on_other_exchange"]["symbol"] == "PEPE-USDT-SWAP"
    assert rows[0]["same_coin_on_other_exchange"]["exchange_cn"] == "欧易"
    assert rows[1]["same_coin_on_other_exchange"]["exchange"] == "binance"
    assert rows[2]["same_coin_on_other_exchange"] is None


def _fake_row(sym, exchange, status="可入场"):
    return {
        "symbol": sym, "status": status, "tier": "watch" if status == "观察" else "formal",
        "L": 1, "H": 2, "R": 3, "lt": 1, "y1": 1.0, "rt": 2, "y2": 1.0,
        "L_time": "a", "H_time": "b", "R_time": "c", "neck": 1.0, "suggest_entry": 1.0,
        "official_entry": 1.0, "official_entry_time": "2026-11-11 10:00", "stop": 1.1,
        "target_1h": 0.9, "target_15h": 0.85, "current": 1.0, "rr": 1.6, "dist_stop_pct": 10.0,
        "dist_t1_pct": 10.0, "note": "",
        "exchange": exchange, "exchange_cn": scan_15m.EXCHANGE_CN[exchange],
        "coin": okx_market.coin_key(sym, exchange),
    }


def _patch_scan(monkeypatch, tmp_path, okx_fails=False, okx_pool=("BTC-USDT-SWAP",)):
    import json
    pool = tmp_path / "core_pool.json"
    pool.write_text(json.dumps({"symbols": ["BTCUSDT"]}))
    monkeypatch.setattr(scan_15m, "POOL_PATH", pool)
    monkeypatch.setattr(scan_15m, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(scan_15m, "make_demo_chart", lambda: "demo skipped (test)")
    monkeypatch.setattr(scan_15m, "plot_setup", lambda df, r, p, title_extra="": None)
    bars = pd.DataFrame({"dt": pd.date_range("2026-11-01", periods=200, freq="15min", tz="UTC")})
    monkeypatch.setattr(scan_15m, "fetch_klines", lambda s, i, l: bars)
    monkeypatch.setattr(okx_market, "load_okx_pool", lambda **k: list(okx_pool))

    def fake_okx(sym, interval, limit):
        if okx_fails:
            raise okx_market.OkxError("HTTP 503")
        return bars

    monkeypatch.setattr(scan_15m, "fetch_okx", fake_okx)

    def fake_scan_symbol(exchange, sym, df15, df1, now):
        return [_fake_row(sym, exchange)], bars

    monkeypatch.setattr(scan_15m, "scan_symbol", fake_scan_symbol)


def test_scan_labels_both_exchanges(monkeypatch, tmp_path):
    import json
    _patch_scan(monkeypatch, tmp_path)
    assert scan_15m.main() == 0
    data = json.loads((tmp_path / "out" / "signals.json").read_text())
    ex = sorted(s["exchange"] for s in data["signals"])
    assert ex == ["binance", "okx"]
    okx = [s for s in data["signals"] if s["exchange"] == "okx"][0]
    assert okx["symbol"] == "BTC-USDT-SWAP" and okx["chart"].endswith("okx_BTC-USDT-SWAP.png")
    assert okx["same_coin_on_other_exchange"]["symbol"] == "BTCUSDT"
    assert data["okx_ok"] is True and data["pool_size_okx"] == 1
    assert "scan_seconds_total" in data
    md = (tmp_path / "out" / "scan.md").read_text()
    assert "| 欧易 | BTC-USDT-SWAP |" in md and "| 币安 | BTCUSDT |" in md
    assert "同一个币两所同时出信号" in md


def test_okx_failure_does_not_break_binance(monkeypatch, tmp_path):
    import json
    _patch_scan(monkeypatch, tmp_path, okx_fails=True, okx_pool=tuple(f"C{i}-USDT-SWAP" for i in range(30)))
    assert scan_15m.main() == 0
    data = json.loads((tmp_path / "out" / "signals.json").read_text())
    assert [s["exchange"] for s in data["signals"]] == ["binance"]
    assert data["okx_ok"] is False and data["okx_error"]


def test_okx_pool_unavailable_skips(monkeypatch, tmp_path):
    import json
    _patch_scan(monkeypatch, tmp_path, okx_pool=())
    assert scan_15m.main() == 0
    data = json.loads((tmp_path / "out" / "signals.json").read_text())
    assert [s["exchange"] for s in data["signals"]] == ["binance"]
    assert data["okx_ok"] is False


# ---------------- webhook payload ----------------

def test_payload_carries_exchange_fields():
    b = _fake_row("BTCUSDT", "binance")
    o = _fake_row("BTC-USDT-SWAP", "okx")
    scan_15m.tag_same_coin([b, o])
    payload = scan_runner.build_payload([b, o], "formal", "2026-11-11 19:02", "2026-11-11 18:45",
                                        scan_meta={"exchanges": ["binance", "okx"], "okx_ok": True})
    sig_o = [s for s in payload["signals"] if s["exchange"] == "okx"][0]
    assert sig_o["exchange_cn"] == "欧易" and sig_o["symbol"] == "BTC-USDT-SWAP"
    assert sig_o["same_coin_on_other_exchange"]["symbol"] == "BTCUSDT"
    assert payload["exchanges_in_payload_cn"] == ["币安", "欧易"]
    assert payload["exchanges_scanned"] == ["binance", "okx"]
    # legacy rows without an exchange field default to binance
    legacy = dict(b); legacy.pop("exchange")
    assert scan_runner.summarize(legacy)["exchange"] == "binance"


def test_summarize_carries_neck_angle_fields():
    row = _fake_row("BTCUSDT", "binance")
    row.update({"neck_s2": 0.11, "neck_class_cn": "上翘", "neck_angle_deg": 7.2})
    s = scan_runner.summarize(row)
    assert s["neck_s2"] == 0.11 and s["neck_class_cn"] == "上翘" and s["neck_angle_deg"] == 7.2
    payload = scan_runner.build_payload([row], "formal", "2026-11-12 14:02", "2026-11-12 13:45")
    assert payload["signals"][0]["neck_class_cn"] == "上翘"
    # missing keys stay None (display-only, never required to fire a signal)
    bare = scan_runner.summarize(_fake_row("ETHUSDT", "binance"))
    assert bare["neck_s2"] is None and bare["neck_class_cn"] is None and bare["neck_angle_deg"] is None


def test_summarize_carries_shoulder_prop_fields():
    row = _fake_row("BTCUSDT", "binance")
    row.update({
        "prop_ls_time": 0.4, "prop_rs_time": 0.6,
        "prop_ls_height": 0.8, "prop_rs_height": 0.9,
        "prop_class_cn": "比例偏小",
        "prop_text": "左肩跨度0.4×头、右肩0.6×头；左肩高度0.8×头、右肩0.9×头，左肩跨度偏小",
    })
    s = scan_runner.summarize(row)
    assert s["prop_ls_time"] == 0.4 and s["prop_rs_time"] == 0.6
    assert s["prop_ls_height"] == 0.8 and s["prop_rs_height"] == 0.9
    assert s["prop_class_cn"] == "比例偏小"
    assert "左肩跨度偏小" in s["prop_text"]
    payload = scan_runner.build_payload([row], "watch", "2026-11-12 14:17", "2026-11-12 14:00")
    assert payload["signals"][0]["prop_class_cn"] == "比例偏小"
    bare = scan_runner.summarize(_fake_row("ETHUSDT", "okx"))
    assert bare["prop_ls_time"] is None and bare["prop_class_cn"] is None and bare["prop_text"] is None


def test_watch_has_no_daily_cap_across_exchanges():
    today = "2026-11-11"
    pushed = {
        "a": {"tier": "watch", "first_pushed": f"{today} 10:00:00", "exchange": "binance"},
        "b": {"tier": "watch", "first_pushed": f"{today} 11:00:00", "exchange": "okx"},
    }
    fresh = [_fake_row("ETH-USDT-SWAP", "okx", "观察"), _fake_row("SOLUSDT", "binance", "观察")]
    f_rows, w_rows, w_over = scan_runner.split_fresh(fresh, pushed, today, cap=3)
    assert f_rows == []
    assert len(w_rows) == 2 and w_over == []
    assert [s["exchange"] for s in w_rows] == ["okx", "binance"]


def test_journal_records_exchange_and_shared_breaker(tmp_path):
    def sh_bar(ts, o, h, l, c):
        return {"dt": pd.Timestamp(ts, tz=TZ).tz_convert("UTC"), "open": o, "high": h, "low": l, "close": c,
                "volume": 1.0}

    def fetch(symbol, start_utc=None):
        return pd.DataFrame([
            sh_bar("2026-11-11 10:00", 100.0, 100.5, 99.5, 100.2),
            sh_bar("2026-11-11 10:15", 100.2, 103.0, 100.0, 102.5),  # stop 102 hit
        ])

    def s(sym, ex, R):
        return {"symbol": sym, "exchange": ex, "status": "可入场", "tier": "formal",
                "L_time": "x", "H_time": "y", "R_time": R, "official_entry": 100.0,
                "official_entry_time": "2026-11-11 10:00", "stop": 102.0, "target_1h": 97.0,
                "target_15h": 95.5}

    paths = pj.JournalPaths.under(tmp_path)
    j = pj.update_from_scan({"signals": [s("BTCUSDT", "binance", "r1"), s("BTC-USDT-SWAP", "okx", "r2")]},
                            now=NOW, paths=paths, fetch_klines_fn=fetch)
    exs = sorted(t["exchange"] for t in j["trades"])
    assert exs == ["binance", "okx"]
    # both stopped out in the same shared account -> streak counts both exchanges
    assert pj.loss_streak(j) == 2
    assert "exchange" in paths.csv_path.read_text().splitlines()[0]
    assert "欧易 BTC-USDT-SWAP" in paths.md_path.read_text()


# ---------------- daemon sleep / stall handling ----------------

class FakeClock:
    def __init__(self, start):
        self.wall = start
        self.mono = 0.0
        self.pause_at = None  # (mono_threshold, wall_jump_sec)

    def now(self):
        return self.wall

    def monotonic(self):
        return self.mono

    def sleep(self, sec):
        self.mono += sec
        self.wall += timedelta(seconds=sec)
        if self.pause_at and self.mono >= self.pause_at[0]:
            self.wall += timedelta(seconds=self.pause_at[1])  # VM paused: wall moves, monotonic doesn't
            self.pause_at = None


def test_wait_until_short_chunks_and_detects_pause():
    start = datetime(2026, 11, 11, 17, 17, 39, tzinfo=TZ)
    target = datetime(2026, 11, 11, 17, 32, 0, tzinfo=TZ)
    fc = FakeClock(start)
    fc.pause_at = (60, 3600)  # after 60 s of sleeping, the box freezes for an hour
    steps = []
    orig = fc.sleep
    fc.sleep = lambda s: (steps.append(s), orig(s))
    res = scan_runner.wait_until(target, chunk=10, sleep=fc.sleep, clock=fc.now, mono=fc.monotonic)
    assert max(steps) <= 10
    assert fc.mono <= 70 + 1e-9  # woke right after the resume, not after another 800 s
    assert res["late_sec"] > 2500 and res["jumps"]


def test_daemon_step_missed_and_night_still_runs():
    slot = datetime(2026, 11, 11, 17, 32, tzinfo=TZ)
    assert scan_runner.daemon_step(slot + timedelta(seconds=5), slot, 5) == "run"
    assert scan_runner.daemon_step(slot + timedelta(hours=1), slot, 3600) == "run_missed"
    late_night = datetime(2026, 11, 12, 2, 0, tzinfo=TZ)
    assert scan_runner.daemon_step(late_night, slot, 9000) == "run_missed"
    night_slot = datetime(2026, 11, 12, 2, 2, tzinfo=TZ)
    assert scan_runner.daemon_step(night_slot + timedelta(seconds=5), night_slot, 5) == "run"


def test_missed_slot_runs_once_then_next_future_slot():
    woke = datetime(2026, 11, 11, 18, 31, 5, tzinfo=TZ)  # missed 17:32, 17:47, 18:02, 18:17
    nxt = scan_runner.next_slot(woke)
    assert nxt == datetime(2026, 11, 11, 18, 32, tzinfo=TZ)  # only the next real slot; no burst


def test_next_slot_is_24h_and_wraps_to_0002():
    assert scan_runner.next_slot(datetime(2026, 11, 11, 8, 50, tzinfo=TZ)) == datetime(2026, 11, 11, 9, 2, tzinfo=TZ)
    assert scan_runner.next_slot(datetime(2026, 11, 11, 23, 47, tzinfo=TZ)) == datetime(2026, 11, 12, 0, 2, tzinfo=TZ)
    assert scan_runner.next_slot(datetime(2026, 11, 12, 0, 0, tzinfo=TZ)) == datetime(2026, 11, 12, 0, 2, tzinfo=TZ)
    assert scan_runner.next_slot(datetime(2026, 11, 12, 0, 2, tzinfo=TZ)) == datetime(2026, 11, 12, 0, 17, tzinfo=TZ)
    # exactly on a slot + 1s still advances to the next one
    assert scan_runner.next_slot(datetime(2026, 11, 11, 23, 47, 2, tzinfo=TZ)) == datetime(2026, 11, 12, 0, 2, tzinfo=TZ)


def test_payload_quiet_and_night_flags():
    row = _fake_row("BTCUSDT", "binance")
    day = datetime(2026, 11, 12, 9, 2, tzinfo=TZ)
    p_day = scan_runner.build_payload([row], "formal", "2026-11-12 09:02", "2026-11-12 08:45", now=day)
    assert p_day["quiet"] is False and p_day["night"] is False
    night = datetime(2026, 11, 12, 0, 2, tzinfo=TZ)
    p_night = scan_runner.build_payload([row], "formal", "2026-11-12 00:02", "2026-11-11 23:45", now=night)
    assert p_night["quiet"] is True and p_night["night"] is True
    # 08:59 is still quiet; 09:00 is not
    almost = datetime(2026, 11, 12, 8, 59, 59, tzinfo=TZ)
    p_almost = scan_runner.build_payload([row], "watch", "2026-11-12 08:59", "2026-11-12 08:45", now=almost)
    assert p_almost["quiet"] is True and p_almost["night"] is True
    nine = datetime(2026, 11, 12, 9, 0, tzinfo=TZ)
    p_nine = scan_runner.build_payload([row], "formal", "2026-11-12 09:00", "2026-11-12 08:45", now=nine)
    assert p_nine["quiet"] is False and p_nine["night"] is False
    assert scan_runner.is_quiet_hours(datetime(2026, 11, 12, 0, 0, tzinfo=TZ)) is True
    assert scan_runner.is_quiet_hours(datetime(2026, 11, 12, 8, 59, tzinfo=TZ)) is True
    assert scan_runner.is_quiet_hours(datetime(2026, 11, 12, 9, 0, tzinfo=TZ)) is False


def test_heartbeat_written(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(scan_runner, "HEARTBEAT_PATH", tmp_path / "hb.json")
    scan_runner.write_heartbeat("sleeping", datetime(2026, 11, 11, 18, 47, tzinfo=TZ))
    d = json.loads((tmp_path / "hb.json").read_text())
    assert d["state"] == "sleeping" and d["next_slot_shanghai"] == "2026-11-11 18:47:00"


def test_summarize_carries_fib_and_dual_neck_fields():
    row = _fake_row("BTCUSDT", "binance")
    row.update({
        "fib_ext": {"1.0": 96.0, "1.272": 94.5, "1.618": 92.8, "2.0": 90.0},
        "fib_text": "斐波那契扩展（只作参考）：1.0=96.0000｜1.272=94.5000｜1.618=92.8000｜2.0=90.0000",
        "dual_neck_tier1": 100.5, "dual_neck_tier2": 99.0,
        "dual_neck_text": "双颈线（只作参考）：颈线上翘约+7°，第一档 斜颈线 100.5000 先轻仓，第二档 水平颈线 99.0000 破了再加仓，止损不变",
    })
    s = scan_runner.summarize(row)
    assert s["fib_ext"]["1.272"] == 94.5 and "1.618=" in s["fib_text"]
    assert s["dual_neck_tier1"] == 100.5 and s["dual_neck_tier2"] == 99.0
    assert "先轻仓" in s["dual_neck_text"]
    payload = scan_runner.build_payload([row], "formal", "2026-11-12 22:02", "2026-11-12 21:45")
    assert payload["signals"][0]["fib_ext"]["2.0"] == 90.0
    assert "双颈线" in payload["signals"][0]["dual_neck_text"]
    bare = scan_runner.summarize(_fake_row("ETHUSDT", "okx"))
    assert bare["fib_ext"] is None and bare["fib_text"] is None
    assert bare["dual_neck_tier1"] is None and bare["dual_neck_tier2"] is None
    assert bare["dual_neck_text"] is None
