"""GitHub Actions port: data-source fallback, webhook secrets handling, ids, chart re-render,
state round-trip. No network, never POSTs."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import binance_data as BD  # noqa: E402
import gha_runner  # noqa: E402
import render_from_payload as RP  # noqa: E402
import scan_15m  # noqa: E402
import scan_htf  # noqa: E402
import scan_runner as SR  # noqa: E402
from test_htf import synth_hs  # noqa: E402

TZ = ZoneInfo("Asia/Shanghai")


def _raw_frame(n=5, start=1_700_000_000_000, step=900_000, px=1.0):
    rows = [[start + i * step, px, px * 1.01, px * 0.99, px, 10.0, start + (i + 1) * step - 1, 10.0, 1, 0, 0, 0]
            for i in range(n)]
    return BD._frame(rows)


@pytest.fixture(autouse=True)
def _reset_bd(monkeypatch):
    monkeypatch.setitem(BD.STATE, "fapi_blocked", False)
    BD.SOURCE_BY_SYMBOL.clear()
    BD.NO_SPOT.clear()
    monkeypatch.delenv("HS_BINANCE_SOURCE", raising=False)
    monkeypatch.delenv("HS_SIM_FAPI_451", raising=False)
    yield


def test_spot_symbol_mapping():
    assert BD.spot_symbol("BTCUSDT") == ("BTCUSDT", 1.0)
    assert BD.spot_symbol("1000PEPEUSDT") == ("PEPEUSDT", 1000.0)
    assert BD.okx_inst("1000PEPEUSDT") == "PEPE-USDT-SWAP"
    assert BD.okx_inst("EDGEUSDT") == "EDGE-USDT-SWAP"


def test_fapi_451_falls_back_to_spot_and_sticks(monkeypatch):
    monkeypatch.setenv("HS_SIM_FAPI_451", "1")
    calls = []
    monkeypatch.setattr(BD, "_spot", lambda s, p: calls.append(s) or _raw_frame())
    df = BD.fetch("BTCUSDT", "15m", 800)
    assert len(df) == 5 and BD.STATE["fapi_blocked"]
    assert BD.source_of("BTCUSDT") == BD.SRC_SPOT
    monkeypatch.setattr(BD, "_fapi", lambda *a: pytest.fail("fapi must not be retried after 451"))
    BD.fetch("ETHUSDT", "1h", 300)
    assert calls == ["BTCUSDT", "ETHUSDT"]


def test_spot_scales_1000_prefix(monkeypatch):
    class R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return [[1, "0.00001", "0.00002", "0.000005", "0.00001", "5000000", 2, "50", 1, 0, 0, 0]]
    seen = {}
    monkeypatch.setattr(BD.SESSION, "get", lambda url, params, timeout: seen.update(params) or R())
    df = BD._spot("1000PEPEUSDT", {"interval": "15m", "limit": 1500})
    assert seen["symbol"] == "PEPEUSDT" and seen["limit"] == 1000
    assert df["close"].iloc[0] == pytest.approx(0.01)
    assert df["volume"].iloc[0] == pytest.approx(5000.0)


def test_perp_only_coin_uses_okx_proxy_unless_in_okx_pool(monkeypatch):
    monkeypatch.setenv("HS_BINANCE_SOURCE", "spot")
    def no_spot(s, p):
        raise BD.SourceUnavailable("no spot pair")
    monkeypatch.setattr(BD, "_spot", no_spot)
    import okx_market
    monkeypatch.setattr(okx_market, "fetch_okx_klines", lambda inst, bar, limit, start_ms=None: _raw_frame())
    monkeypatch.setattr(BD, "okx_pool_set", lambda: set())
    BD.fetch("ARIAUSDT", "15m", 800)
    assert BD.source_of("ARIAUSDT") == BD.SRC_OKX_PROXY
    monkeypatch.setattr(BD, "okx_pool_set", lambda: {"EDGE-USDT-SWAP"})
    with pytest.raises(BD.SourceUnavailable):
        BD.fetch("EDGEUSDT", "15m", 800)


def test_post_webhook_skips_without_secret(monkeypatch, tmp_path):
    monkeypatch.delenv("HS_WEBHOOK_KEY", raising=False)
    monkeypatch.setattr(SR, "ENV_FILE", tmp_path / "none.env")
    monkeypatch.setattr(SR, "LOG_PATH", tmp_path / "log.txt")
    monkeypatch.setattr(SR, "PUSH_DISABLED_FLAG", tmp_path / "flag")
    monkeypatch.setattr(SR.requests, "post", lambda *a, **k: pytest.fail("must not POST"))
    status, style, _ = SR.post_webhook({"kind": "signals", "signals": []})
    assert status == 0 and style == "no-key"
    assert "跳过推送" in (tmp_path / "log.txt").read_text()


def test_box_standby_flag_blocks_push(monkeypatch, tmp_path):
    monkeypatch.setenv("HS_WEBHOOK_KEY", "x")
    flag = tmp_path / "box_push_disabled"
    flag.write_text("1")
    monkeypatch.setattr(SR, "PUSH_DISABLED_FLAG", flag)
    monkeypatch.setattr(SR, "LOG_PATH", tmp_path / "log.txt")
    monkeypatch.setattr(SR.requests, "post", lambda *a, **k: pytest.fail("must not POST"))
    assert SR.post_webhook({"signals": []})[0] == 0


def test_signal_id_and_summary_fields():
    sig = {"exchange": "binance", "symbol": "SOLUSDT", "timeframe": "1h", "H_time": "2026-11-12 01:00",
           "status": "已破位·等回踩", "data_source": "binance_spot", "chart_spec": {"v": 1}}
    d = SR.summarize_htf(sig)
    assert d["signal_id"] == "binance|SOLUSDT|1h|2026-11-12 01:00|已破位·等回踩"
    assert d["data_source"] == "binance_spot" and d["chart_spec"] == {"v": 1}
    p = SR.decorate_payload({"signals": [d]})
    assert p["data_source_note"] == SR.DATA_SOURCE_NOTE_SPOT
    assert SR.signal_id({"symbol": "X", "status": "可入场"}).split("|")[2] == "15m"
    assert SR.signal_id({"symbol": "X", "nested_signal": True, "timeframe": "4h"}).split("|")[2] == "nested-4h"


def test_night_flag_uses_beijing_time_even_from_utc_clock():
    utc = ZoneInfo("UTC")
    assert SR.is_quiet_hours(datetime(2026, 11, 13, 17, 30, tzinfo=utc))      # 01:30 Beijing
    assert not SR.is_quiet_hours(datetime(2026, 11, 13, 2, 30, tzinfo=utc))   # 10:30 Beijing


def test_pool_age_from_asof(tmp_path):
    p = tmp_path / "pool.json"
    p.write_text(json.dumps({"asof_utc": "2020-01-01T00:00:00Z"}))
    assert SR.pool_age_sec(p) > 365 * 86400   # mtime is now, asof wins


def test_chart_spec_roundtrip_renders(tmp_path, monkeypatch):
    df = synth_hs()
    df["open_time"] = (df["dt"].astype("int64") // 1_000_000).astype("int64")
    rows = scan_htf.live_rows("TESTUSDT", "1h", df, None)
    assert rows
    r = rows[0]
    r["exchange"] = "binance"
    spec = scan_15m.chart_spec(df, r, "1h", tf_label="1H")
    json.dumps(spec)  # JSON-safe
    sig = SR.summarize_htf({**{k: v for k, v in r.items() if k not in ("L", "H", "R", "lt", "rt", "y1", "y2")},
                            "chart_spec": spec, "data_source": "binance_fapi"})
    raw = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    monkeypatch.setattr(RP, "fetch_frame", lambda s, sp: raw.assign(
        dt=pd.to_datetime(raw["open_time"], unit="ms", utc=True)))
    out = RP.render_one(json.loads(json.dumps(sig, ensure_ascii=False, default=str)), tmp_path)
    assert out.exists() and out.stat().st_size > 10_000


def test_gha_state_roundtrip(tmp_path, monkeypatch):
    here, state = tmp_path / "backtest", tmp_path / "state"
    (here / "scan_out" / "htf" / "1h").mkdir(parents=True)
    monkeypatch.setattr(gha_runner, "HERE", here)
    monkeypatch.setattr(gha_runner, "STATE", state)
    (here / "pushed_state.json").write_text('{"pushed": {"a": {}}}')
    (here / "scan_out" / "paper_journal.json").write_text("{}")
    (here / "scan_out" / "htf" / "1h" / "signals.json").write_text('{"last_bar_shanghai": "2026-11-13 16:00"}')
    bars = gha_runner.save_state()
    assert bars["1h"] == "2026-11-13 16:00" and (state / "scan_out" / "paper_journal.json").exists()
    import shutil
    shutil.rmtree(here)
    gha_runner.restore_state()
    assert json.loads((here / "pushed_state.json").read_text())["pushed"] == {"a": {}}
    stub = json.loads((here / "scan_out" / "htf" / "1h" / "signals.json").read_text())
    assert stub["last_bar_shanghai"] == "2026-11-13 16:00"
    assert "paper_journal.json" in " ".join(gha_runner.STATE_FILES)
    assert "paper_trades.csv" in " ".join(gha_runner.STATE_FILES)
