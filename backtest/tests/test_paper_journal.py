"""Synthetic 15m-bar tests for the paper journal and circuit breaker.

Never POSTs to the live webhook.
"""
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

import paper_journal as pj  # noqa: E402
import scan_runner  # noqa: E402
from hs_backtest import TAKER_FEE, slippage_for  # noqa: E402

TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 11, 11, 11, 2, tzinfo=TZ)


def sh_bar(ts: str, o: float, h: float, l: float, c: float) -> dict:
    dt = pd.Timestamp(ts, tz=TZ).tz_convert("UTC")
    return {"dt": dt, "open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": 1.0}


def frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def provider(mapping: dict[str, pd.DataFrame]):
    def fetch(symbol: str, start_utc=None):
        df = mapping.get(symbol)
        return df.copy() if df is not None else pd.DataFrame()

    return fetch


def sig(
    symbol: str,
    entry_time: str,
    *,
    entry: float = 100.0,
    stop: float = 102.0,
    t1: float = 97.0,
    t15: float = 95.5,
    L: str | None = None,
    H: str | None = None,
    R: str | None = None,
    status: str = "可入场",
) -> dict:
    # Unique L/H/R per symbol+entry by default so pattern_ids do not collide.
    return {
        "symbol": symbol,
        "status": status,
        "L_time": L or "2026-11-11 07:00",
        "H_time": H or "2026-11-11 08:00",
        "R_time": R or f"2026-11-11 08:30 {symbol}",
        "official_entry": entry,
        "official_entry_time": entry_time,
        "stop": stop,
        "target_1h": t1,
        "target_15h": t15,
        "neck": 100.2,
        "current": 99.8,
    }


def run(tmp_path: Path, signals: list[dict], bars: dict[str, pd.DataFrame], now=NOW, extra_trades=None, seed_journal=None):
    paths = pj.JournalPaths.under(tmp_path)
    if seed_journal is not None:
        pj.save_journal(seed_journal, paths=paths, now=now)
    elif extra_trades:
        j = pj.empty_journal(now)
        j["trades"] = list(extra_trades)
        j["equity"] = seed_equity(extra_trades)
        pj.evaluate_breaker(j, now)
        pj.save_journal(j, paths=paths, now=now)
    data = {"signals": signals, "asof_shanghai": now.strftime("%Y-%m-%d %H:%M")}
    return pj.update_from_scan(data, now=now, paths=paths, fetch_klines_fn=provider(bars))


def seed_equity(trades: list[dict]) -> float:
    eq = 1.0
    for t in trades:
        if t.get("account_taken") and t.get("status") == "closed" and t.get("pnl_R") is not None:
            eq *= 1.0 + float(t["pnl_R"]) * 0.01
            t["equity_applied"] = True
    return eq


def closed(
    symbol: str,
    pnl_R: float,
    exit_time: str,
    *,
    account_taken: bool = True,
    suffix: str = "",
) -> dict:
    pid = f"{symbol}|可入场|2026-11-10 07:00|2026-11-10 08:00|{exit_time}{suffix}"
    return {
        "id": pid,
        "pattern_id": pid,
        "symbol": symbol,
        "status": "closed",
        "first_seen": exit_time + ("" if ":" in exit_time[11:] and len(exit_time) > 16 else ":00"),
        "alerted": True,
        "account_taken": account_taken,
        "account_reason": "",
        "official_entry": 100.0,
        "official_entry_time": "2026-11-10 10:00",
        "fill_entry": 99.95,
        "stop": 102.0,
        "target_1h": 97.0,
        "target_15h": 95.5,
        "L_time": "2026-11-10 07:00",
        "H_time": "2026-11-10 08:00",
        "R_time": f"{exit_time}{suffix}",
        "exit_time": exit_time,
        "exit_px": 102.0,
        "hit": "stop",
        "pnl_R": pnl_R,
        "t1_hit": False,
        "remaining": 0.0,
        "bars_used": 2,
        "note": "",
        "equity_applied": True,
    }


def test_pattern_id_matches_runner():
    s = sig("BTCUSDT", "2026-11-11 10:00", L="2026-11-11 06:00", H="2026-11-11 07:00", R="2026-11-11 08:00")
    assert pj.pattern_id(s) == scan_runner.pattern_id(s)


def test_stop_first_when_stop_and_tp1_same_bar(tmp_path):
    # Entry bar trades through both the stop and 1.0H; adverse assumption: stop first.
    bars = {
        "AAAUSDT": frame(
            sh_bar("2026-11-11 09:45", 100.2, 100.4, 99.9, 100.1),
            sh_bar("2026-11-11 10:00", 100.0, 103.0, 96.0, 99.0),
        )
    }
    journal = run(tmp_path, [sig("AAAUSDT", "2026-11-11 10:00")], bars)
    t = journal["trades"][0]
    assert t["status"] == "closed"
    assert t["hit"] == "stop_samebar_priority"
    assert t["t1_hit"] is False
    assert t["remaining"] == 0.0
    assert t["account_taken"] is True
    slip = slippage_for("AAAUSDT")
    fill = 100.0 * (1 - slip)
    exit_px = 102.0 * (1 + slip)
    raw = (fill - exit_px) / fill
    net = raw - 2 * TAKER_FEE
    r_dist = (102.0 - fill) / fill
    assert t["pnl_R"] == pytest.approx(net / r_dist, rel=1e-9)
    assert t["pnl_R"] < 0


def test_partial_tp1_then_remainder_stopped_at_breakeven(tmp_path):
    bars = {
        "AAVEUSDT": frame(
            sh_bar("2026-11-11 10:00", 100.0, 101.0, 96.8, 97.2),  # hits 1.0H only
            sh_bar("2026-11-11 10:15", 97.5, 101.0, 97.2, 100.5),  # remainder hits breakeven
        )
    }
    journal = run(tmp_path, [sig("AAVEUSDT", "2026-11-11 10:00")], bars)
    t = journal["trades"][0]
    assert t["status"] == "closed"
    assert t["t1_hit"] is True
    assert t["remaining"] == 0.0
    assert t["hit"] == "stop_breakeven"
    slip = slippage_for("AAVEUSDT")
    fill = 100.0 * (1 - slip)
    t1_px = 97.0 * (1 - slip)  # min(open=100, target=97)
    be_px = max(97.5, fill) * (1 + slip)
    exit_px = 0.60 * t1_px + 0.40 * be_px
    raw = (fill - exit_px) / fill
    net = raw - 2 * TAKER_FEE
    r_dist = (102.0 - fill) / fill
    assert t["exit_px"] == pytest.approx(exit_px, rel=1e-9)
    assert t["pnl_R"] == pytest.approx(net / r_dist, rel=1e-9)
    # 0.6R-ish after costs, not a full 1.0H winner and not a full stop.
    assert 0.3 < t["pnl_R"] < 0.95
    # Equity only moves on account_taken close.
    assert journal["equity"] == pytest.approx(1.0 * (1 + t["pnl_R"] * 0.01), rel=1e-9)


def test_open_trade_left_open_no_forced_eod(tmp_path):
    bars = {
        "BNBUSDT": frame(
            sh_bar("2026-11-11 10:00", 100.0, 101.0, 99.0, 99.6),
            sh_bar("2026-11-11 10:15", 99.6, 100.4, 98.8, 99.2),
            sh_bar("2026-11-11 10:30", 99.2, 100.1, 98.5, 98.9),
        )
    }
    journal = run(tmp_path, [sig("BNBUSDT", "2026-11-11 10:00")], bars)
    t = journal["trades"][0]
    assert t["status"] == "open"
    assert t["exit_time"] is None
    assert t["exit_px"] is None
    assert t["hit"] is None
    assert t["remaining"] == pytest.approx(1.0)
    assert t["t1_hit"] is False
    assert t["fill_entry"] is not None
    assert "eod" not in str(t.get("hit") or "").lower()
    assert journal["equity"] == pytest.approx(1.0)
    # Snapshot must not claim a forced close.
    md = (tmp_path / "paper_journal.md").read_text(encoding="utf-8")
    assert "持有中" in md


def test_pending_entry_until_official_bar_appears(tmp_path):
    bars = {
        "SOLUSDT": frame(sh_bar("2026-11-11 09:45", 100.2, 100.5, 99.8, 100.0))
    }
    journal = run(tmp_path, [sig("SOLUSDT", "2026-11-11 10:00")], bars)
    t = journal["trades"][0]
    assert t["status"] == "pending_entry"
    assert t["fill_entry"] is None
    # Later the official bar arrives and does not hit levels: becomes open.
    bars["SOLUSDT"] = frame(
        sh_bar("2026-11-11 09:45", 100.2, 100.5, 99.8, 100.0),
        sh_bar("2026-11-11 10:00", 100.0, 100.8, 99.4, 99.7),
    )
    journal = run(tmp_path, [sig("SOLUSDT", "2026-11-11 10:00")], bars)
    t = journal["trades"][0]
    assert t["status"] == "open"
    assert t["fill_entry"] == pytest.approx(100.0 * (1 - slippage_for("SOLUSDT")))
    assert len(journal["trades"]) == 1


def test_daily_minus_3r_trips_and_suppresses_new_alert(tmp_path):
    prior = [
        closed("BTCUSDT", -1.6, "2026-11-11 09:30", suffix="-a"),
        closed("ETHUSDT", -1.7, "2026-11-11 10:00", suffix="-b"),
    ]
    bars = {
        "NEARUSDT": frame(sh_bar("2026-11-11 10:30", 100.0, 100.6, 99.5, 99.8)),
        "BTCUSDT": pd.DataFrame(),
        "ETHUSDT": pd.DataFrame(),
    }
    journal = run(
        tmp_path,
        [sig("NEARUSDT", "2026-11-11 10:30", L="2026-11-11 08:00", H="2026-11-11 09:00", R="2026-11-11 09:45")],
        bars,
        extra_trades=prior,
    )
    b = journal["breaker"]
    assert b["tripped"] is True
    assert b["trip_date"] == "2026-11-11"
    assert b["day_R"] == pytest.approx(-3.3)
    new = [t for t in journal["trades"] if t["symbol"] == "NEARUSDT"][0]
    assert new["status"] in {"open", "pending_entry"}
    assert new["alerted"] is False
    assert "熔断" in (new.get("note") or "")
    plan = pj.notification_plan(journal, now=NOW)
    assert plan["suppress_trading"] is True
    assert plan["send_breaker_notice"] is True
    assert plan["payload"]["test"] is False
    assert plan["payload"]["kind"] == "circuit_breaker"
    assert plan["payload"]["day_R"] == pytest.approx(-3.3)
    # One-time notice: after marking sent, do not re-spam.
    pj.mark_notice_sent(journal, now=NOW, paths=pj.JournalPaths.under(tmp_path))
    plan2 = pj.notification_plan(journal, now=NOW)
    assert plan2["suppress_trading"] is True
    assert plan2["send_breaker_notice"] is False
    assert plan2["payload"] is None


def test_three_consecutive_losses_trip_breaker(tmp_path):
    prior = [
        closed("BTCUSDT", -1.1, "2026-11-10 16:00", suffix="-a"),
        closed("ETHUSDT", -0.8, "2026-11-11 09:20", suffix="-b"),
    ]
    # New signal stops out on its official entry bar → third account loss today.
    bars = {
        "SOLUSDT": frame(sh_bar("2026-11-11 10:00", 100.0, 103.0, 99.5, 102.2)),
        "BTCUSDT": pd.DataFrame(),
        "ETHUSDT": pd.DataFrame(),
    }
    journal = run(tmp_path, [sig("SOLUSDT", "2026-11-11 10:00")], bars, extra_trades=prior)
    new = [t for t in journal["trades"] if t["symbol"] == "SOLUSDT"][0]
    assert new["status"] == "closed"
    assert new["account_taken"] is True
    assert new["pnl_R"] < 0
    assert journal["breaker"]["tripped"] is True
    assert journal["breaker"]["loss_streak"] == 3
    assert "3 笔" in (journal["breaker"]["reason"] or "")
    plan = pj.notification_plan(journal, now=NOW)
    assert plan["suppress_trading"] is True
    assert plan["send_breaker_notice"] is True


def test_auto_resume_next_shanghai_day(tmp_path):
    prior = [
        closed("BTCUSDT", -1.1, "2026-11-10 14:00", suffix="-a"),
        closed("ETHUSDT", -1.0, "2026-11-10 16:00", suffix="-b"),
        closed("SOLUSDT", -1.2, "2026-11-10 18:00", suffix="-c"),
    ]
    j = pj.empty_journal(datetime(2026, 11, 10, 18, 30, tzinfo=TZ))
    j["trades"] = prior
    j["equity"] = seed_equity(prior)
    j["breaker"]["tripped"] = True
    j["breaker"]["trip_date"] = "2026-11-10"
    j["breaker"]["notice_sent_date"] = "2026-11-10"
    j["breaker"]["reason"] = "账户内最近 3 笔已平仓均为亏损，触发熔断"
    j["breaker"]["loss_streak"] = 3
    j["breaker"]["day_R"] = -3.3
    paths = pj.JournalPaths.under(tmp_path)
    pj.save_journal(j, paths=paths, now=datetime(2026, 11, 10, 18, 30, tzinfo=TZ))

    nxt = datetime(2026, 11, 11, 9, 2, tzinfo=TZ)
    bars = {"NEARUSDT": frame(sh_bar("2026-11-11 09:00", 100.0, 100.5, 99.6, 99.8))}
    journal = pj.update_from_scan(
        {"signals": [sig("NEARUSDT", "2026-11-11 09:00", R="2026-11-11 08:45 NEAR")], "asof_shanghai": "2026-11-11 09:02"},
        now=nxt,
        paths=paths,
        fetch_klines_fn=provider(bars),
    )
    assert journal["breaker"]["tripped"] is False
    assert journal["breaker"]["loss_streak"] == 3  # not erased until a later winning close
    assert journal["breaker"]["day_R"] == pytest.approx(0.0)  # fresh calendar day, no closes yet
    new = [t for t in journal["trades"] if t["symbol"] == "NEARUSDT"][0]
    assert new["account_taken"] is True
    assert "熔断" not in (new.get("note") or "")
    plan = pj.notification_plan(journal, now=nxt)
    assert plan["suppress_trading"] is False
    assert plan["send_breaker_notice"] is False
    # A later winning close would clear the streak; a new loss today would trip again.
    loss_today = closed("ADAUSDT", -0.5, "2026-11-11 10:00", suffix="-d")
    journal["trades"].append(loss_today)
    pj.evaluate_breaker(journal, NOW)
    assert journal["breaker"]["tripped"] is True
    assert journal["breaker"]["trip_date"] == "2026-11-11"


def test_breaker_stays_tripped_overnight_until_0902(tmp_path):
    """24h scans must not resume a prior day's trip at 00:02; first resume is still 09:02."""
    prior = [
        closed("BTCUSDT", -1.1, "2026-11-10 14:00", suffix="-a"),
        closed("ETHUSDT", -1.0, "2026-11-10 16:00", suffix="-b"),
        closed("SOLUSDT", -1.2, "2026-11-10 18:00", suffix="-c"),
    ]
    j = pj.empty_journal(datetime(2026, 11, 10, 18, 30, tzinfo=TZ))
    j["trades"] = prior
    j["equity"] = seed_equity(prior)
    j["breaker"]["tripped"] = True
    j["breaker"]["trip_date"] = "2026-11-10"
    j["breaker"]["notice_sent_date"] = "2026-11-10"
    j["breaker"]["reason"] = "账户内最近 3 笔已平仓均为亏损，触发熔断"
    j["breaker"]["loss_streak"] = 3
    j["breaker"]["day_R"] = -3.3
    paths = pj.JournalPaths.under(tmp_path)
    pj.save_journal(j, paths=paths, now=datetime(2026, 11, 10, 18, 30, tzinfo=TZ))

    for hh, mm in ((0, 2), (8, 59)):
        t = datetime(2026, 11, 11, hh, mm, tzinfo=TZ)
        journal = pj.load_journal(paths)
        assert pj.maybe_resume_breaker(journal, t) is False
        pj.evaluate_breaker(journal, t)
        assert journal["breaker"]["tripped"] is True
        assert journal["breaker"]["trip_date"] == "2026-11-10"
        plan = pj.notification_plan(journal, now=t)
        assert plan["suppress_trading"] is True
        assert plan["send_breaker_notice"] is False  # do not re-ping the previous day's trip

    nxt = datetime(2026, 11, 11, 9, 2, tzinfo=TZ)
    journal = pj.load_journal(paths)
    assert pj.maybe_resume_breaker(journal, nxt) is True
    assert journal["breaker"]["tripped"] is False
    plan = pj.notification_plan(journal, now=nxt)
    assert plan["suppress_trading"] is False
    assert plan["send_breaker_notice"] is False


def test_fourth_concurrent_is_capacity_skipped_but_journaled(tmp_path):
    def open_bars(shift_h: int) -> pd.DataFrame:
        # Four symbols, each official entry at 10:00 + 0 bars; path stays inside the range.
        return frame(
            sh_bar("2026-11-11 10:00", 100.0, 100.8, 99.2, 99.7),
            sh_bar("2026-11-11 10:15", 99.7, 100.5, 99.0, 99.4),
        )

    symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"]
    signals = [
        sig(sym, "2026-11-11 10:00", R=f"2026-11-11 08:30 {sym}")
        for sym in symbols
    ]
    bars = {sym: open_bars(i) for i, sym in enumerate(symbols)}
    journal = run(tmp_path, signals, bars)
    assert len(journal["trades"]) == 4
    taken = [t for t in journal["trades"] if t.get("account_taken")]
    skipped = [t for t in journal["trades"] if not t.get("account_taken")]
    assert len(taken) == 2
    assert len(skipped) == 2
    fourth = journal["trades"][3]
    assert fourth["account_taken"] is False
    assert fourth["account_reason"] == "capacity"
    assert fourth["status"] in {"open", "pending_entry", "closed", "skipped_capacity"}
    assert fourth["status"] == "open"  # still tracked on the unconstrained path
    # Capacity trades must not move equity or trip the breaker by themselves.
    assert journal["equity"] == pytest.approx(1.0)
    # Watching a signal must not open a paper trade.
    watch = sig("EEEUSDT", "2026-11-11 10:00", status="观望", R="2026-11-11 08:30 EEE")
    journal2 = run(tmp_path, signals + [watch], bars)
    assert all(t["symbol"] != "EEEUSDT" for t in journal2["trades"])
    assert len(journal2["trades"]) == 4


def test_watch_only_scan_still_marks_open_trades(tmp_path):
    bars = {
        "BNBUSDT": frame(
            sh_bar("2026-11-11 10:00", 100.0, 101.0, 99.0, 99.6),
            sh_bar("2026-11-11 10:15", 99.6, 100.2, 99.1, 99.4),
        )
    }
    run(tmp_path, [sig("BNBUSDT", "2026-11-11 10:00")], bars)
    # Next scan: no 可入场, only 观望, but one more bar hits 1.0H then stays open on remainder.
    bars["BNBUSDT"] = frame(
        sh_bar("2026-11-11 10:00", 100.0, 101.0, 99.0, 99.6),
        sh_bar("2026-11-11 10:15", 99.6, 100.2, 99.1, 99.4),
        sh_bar("2026-11-11 10:30", 99.4, 99.8, 96.5, 97.1),
    )
    journal = run(tmp_path, [sig("BNBUSDT", "2026-11-11 10:00", status="观望")], bars)
    assert len(journal["trades"]) == 1
    t = journal["trades"][0]
    assert t["status"] == "open"
    assert t["t1_hit"] is True
    assert t["remaining"] == pytest.approx(0.4)
    assert t["hit"] == "target_1.0H"


def test_historical_outcome_from_official_entry_even_if_seen_late(tmp_path):
    # By the time we first see the signal, both the entry bar and a later stop bar already exist.
    bars = {
        "ONUSDT": frame(
            sh_bar("2026-11-11 10:00", 100.0, 100.5, 99.4, 99.6),
            sh_bar("2026-11-11 10:15", 99.6, 103.0, 99.5, 102.4),
        )
    }
    late = datetime(2026, 11, 11, 10, 32, tzinfo=TZ)
    journal = run(tmp_path, [sig("ONUSDT", "2026-11-11 10:00")], bars, now=late)
    t = journal["trades"][0]
    assert t["status"] == "closed"
    assert t["hit"] == "stop"
    assert t["fill_time"] == "2026-11-11 10:00"
    assert t["exit_time"] == "2026-11-11 10:15"


def test_csv_and_snapshot_written(tmp_path):
    bars = {"AAAUSDT": frame(sh_bar("2026-11-11 10:00", 100.0, 100.4, 99.5, 99.7))}
    run(tmp_path, [sig("AAAUSDT", "2026-11-11 10:00")], bars)
    assert (tmp_path / "paper_journal.json").is_file()
    csv_text = (tmp_path / "paper_trades.csv").read_text(encoding="utf-8")
    assert "AAAUSDT" in csv_text
    assert "official_entry" in csv_text
    md = (tmp_path / "paper_journal.md").read_text(encoding="utf-8")
    assert "不下单" in md
    assert "AAAUSDT" in md
