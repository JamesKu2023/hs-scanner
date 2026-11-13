"""Tests for the 观察 (near-miss, watch) tier. Never POSTs to the live webhook."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import hs_nearmiss as nm  # noqa: E402
import paper_journal as pj  # noqa: E402
import scan_15m  # noqa: E402
import scan_runner  # noqa: E402

TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 11, 11, 11, 2, tzinfo=TZ)


def sh_bar(ts, o, h, l, c):
    return {"dt": pd.Timestamp(ts, tz=TZ).tz_convert("UTC"), "open": o, "high": h, "low": l, "close": c, "volume": 1.0}


def sig(symbol, status="观察", entry_time="2026-11-11 10:00", R="2026-11-11 08:30"):
    d = {
        "symbol": symbol, "status": status,
        "L_time": "2026-11-11 06:00", "H_time": "2026-11-11 07:15", "R_time": f"{R} {symbol}",
        "official_entry": 100.0, "official_entry_time": entry_time, "suggest_entry": 100.0,
        "stop": 102.0, "target_1h": 97.0, "target_15h": 95.5, "current": 100.0, "rr": 1.5, "neck": 99.0,
    }
    if status == "观察":
        d.update({"tier": "watch", "missed_condition": "shoulder_sym", "missed_text": "两肩高度差 0.9 ATR",
                  "suggested_risk_pct": 0.5})
    else:
        d.update({"tier": "formal", "suggested_risk_pct": 1.0})
    return d


def fetch_stop(symbol, start_utc=None):
    # entry bar then a bar that hits the stop (102)
    return pd.DataFrame([
        sh_bar("2026-11-11 10:00", 100.0, 100.5, 99.5, 100.2),
        sh_bar("2026-11-11 10:15", 100.2, 103.0, 100.0, 102.5),
    ])


# ---------------- hs_nearmiss definition ----------------

def test_soft_eval_bands_and_tolerances():
    n = 40
    high = np.full(n, 100.0)
    L, H, R = 5, 15, 25
    high[L], high[H], high[R] = 103.0, 106.0, 103.9  # sym=0.9 ATR (near), head_rs=2.1 (pass)
    vol = np.ones(n)
    vol[L - 1:L + 2] = 3.0
    vol[H - 1:H + 2] = 2.0
    vol[R - 1:R + 2] = 1.0
    atr = np.ones(n)
    ml = np.zeros(n); ms = np.zeros(n); mh = np.zeros(n)
    mh[R - 2:R + 1] = [3.0, 2.0, 1.0]  # shrinking hist -> macd_rs ok
    ev = nm.soft_eval(high, vol, ml, ms, mh, atr, L, H, R, 95.0, 95.5)  # neck 0.5 (pass)
    st = ev["status"]
    assert st["shoulder_sym"] == "near"
    assert st["neck_flat"] == st["head_rs"] == st["rs_volume"] == st["rs_macd"] == "pass"
    # beyond tolerance -> fail
    high[R] = 104.2
    assert nm.soft_eval(high, vol, ml, ms, mh, atr, L, H, R, 95.0, 95.5)["status"]["shoulder_sym"] == "fail"
    # neck 0.9 -> near ; 1.1 -> fail
    high[R] = 103.0
    assert nm.soft_eval(high, vol, ml, ms, mh, atr, L, H, R, 95.0, 95.9)["status"]["neck_flat"] == "near"
    assert nm.soft_eval(high, vol, ml, ms, mh, atr, L, H, R, 95.0, 96.1)["status"]["neck_flat"] == "fail"


def test_locked_tolerances_match_meta():
    meta = json.loads((Path(__file__).resolve().parents[2] / "research" / "hs_nearmiss_meta.json").read_text())
    assert meta["locked_before_results"] is True
    assert nm.NM_NECK_FLAT_MAX == 1.0 and nm.NM_SYM_MAX == 1.0
    assert nm.NM_HEAD_RS_MIN == 0.75 and nm.NM_RS_VOL_RATIO_MAX == 1.2
    assert nm.SUGGESTED_RISK_PCT == 0.5


def test_miss_text_mentions_amount():
    assert "超出 0.15" in nm.miss_text("shoulder_sym", 0.90)
    assert "差 0.20" in nm.miss_text("head_rs", 0.80)
    assert nm.miss_amount("rs_volume", 1.1) == pytest.approx(0.1)


def test_merge_tiers_formal_priority():
    ents = [{"symbol": "BTCUSDT", "status": "可入场"}]
    obs = [{"symbol": "BTCUSDT", "status": "观察"}, {"symbol": "ETHUSDT", "status": "观察"}]
    assert [r["symbol"] for r in scan_15m.merge_tiers(ents, obs)] == ["ETHUSDT"]


# ---------------- watch paper book ----------------

def test_watch_book_separate_and_never_trips(tmp_path):
    fp = pj.JournalPaths.under(tmp_path)
    wp = pj.JournalPaths.under(tmp_path, book=pj.BOOK_WATCH)
    signals = [sig("AAAUSDT"), sig("BBBUSDT", R="2026-11-11 08:31"), sig("CCCUSDT", R="2026-11-11 08:32"),
               sig("DDDUSDT", R="2026-11-11 08:33")]
    later = datetime(2026, 11, 11, 10, 32, tzinfo=TZ)
    f = pj.update_from_scan({"signals": signals}, now=later, paths=fp, fetch_klines_fn=fetch_stop)
    w = pj.update_from_scan({"signals": signals}, now=later, paths=wp, fetch_klines_fn=fetch_stop,
                            book=pj.BOOK_WATCH)
    assert f["trades"] == []  # formal book ignores 观察
    assert len(w["trades"]) == 4 and w["book"] == "watch"
    assert w["risk_pct"] == pytest.approx(0.005)
    assert all(t["status"] == "closed" and t["hit"] == "stop" for t in w["trades"])
    assert w["breaker"]["tripped"] is False  # 4 straight losses today, still never trips in watch book
    exp = 1.0
    for t in w["trades"]:
        if t.get("account_taken"):
            exp *= 1 + float(t["pnl_R"]) * 0.005  # 0.5% risk per trade
    assert w["equity"] == pytest.approx(exp) and w["equity"] < 1.0
    assert (tmp_path / "paper_watch_trades.csv").is_file()
    assert "missed_condition" in (tmp_path / "paper_watch_trades.csv").read_text(encoding="utf-8")
    assert "观察档" in (tmp_path / "paper_watch_journal.md").read_text(encoding="utf-8")
    assert not (tmp_path / "paper_trades.csv").read_text(encoding="utf-8").count("AAAUSDT")
    # formal breaker unaffected
    assert pj.notification_plan(pj.load_journal(fp), now=later)["suppress_trading"] is False


def test_formal_book_ignores_watch_and_watch_ignores_formal(tmp_path):
    wp = pj.JournalPaths.under(tmp_path, book=pj.BOOK_WATCH)
    w = pj.update_from_scan({"signals": [sig("AAAUSDT", status="可入场")]}, now=NOW, paths=wp,
                            fetch_klines_fn=fetch_stop, book=pj.BOOK_WATCH)
    assert w["trades"] == []


# ---------------- runner: payload / breaker (no daily watch cap) ----------------

def test_split_fresh_sends_all_watch_with_no_daily_cap():
    today = "2026-11-11"
    pushed = {
        "a": {"first_pushed": "2026-11-11 09:17:00", "tier": "watch"},
        "b": {"first_pushed": "2026-11-11 10:17:00", "tier": "watch"},
        "c": {"first_pushed": "2026-11-11 10:17:00", "tier": "watch", "suppressed": "daily_cap"},
        "d": {"first_pushed": "2026-11-10 22:17:00", "tier": "watch"},
        "e": {"first_pushed": "2026-11-11 10:17:00", "tier": "formal"},
    }
    assert scan_runner.watch_pushed_today(pushed, today) == 2  # informational count only
    fresh = [sig("X1USDT"), sig("X2USDT"), sig("X3USDT"), sig("F1USDT", status="可入场"),
             sig("F2USDT", status="可入场")]
    f, w, over = scan_runner.split_fresh(fresh, pushed, today)
    assert len(f) == 2
    assert [s["symbol"] for s in w] == ["X1USDT", "X2USDT", "X3USDT"]
    assert over == []
    # even an explicit old cap argument must not truncate
    _, w2, over2 = scan_runner.split_fresh(fresh, pushed, today, cap=3)
    assert [s["symbol"] for s in w2] == ["X1USDT", "X2USDT", "X3USDT"] and over2 == []


def test_payload_tier_fields():
    p = scan_runner.build_payload([sig("X1USDT")], "watch", "2026-11-11 11:02", "2026-11-11 10:45", 1)
    assert p["tier"] == "watch" and p["validated"] is False and p["suggested_risk_pct"] == 0.5
    assert p["n_observe"] == 1 and p["watch_daily_cap"] is None
    assert p["watch_alerts_today"] == 1
    s = p["signals"][0]
    for k in ("tier", "status", "missed_condition", "missed_text", "suggested_risk_pct", "entry", "stop",
              "target_1h", "target_15h", "chart"):
        assert k in s
    assert s["tier"] == "watch" and s["status"] == "观察"
    pf = scan_runner.build_payload([sig("F1USDT", status="可入场")], "formal", "t", "t")
    assert pf["tier"] == "formal" and pf["validated"] is True and pf["signals"][0]["tier"] == "formal"


@pytest.fixture
def runner_env(tmp_path, monkeypatch):
    monkeypatch.setattr(scan_runner, "run_scan", lambda: 0)
    monkeypatch.setattr(scan_runner, "SIGNALS_PATH", tmp_path / "signals.json")
    monkeypatch.setattr(scan_runner, "STATE_PATH", tmp_path / "pushed_state.json")
    monkeypatch.setattr(scan_runner, "LOCK_PATH", tmp_path / "lock")
    monkeypatch.setattr(scan_runner, "LOG_PATH", tmp_path / "runner.log")
    monkeypatch.setattr(pj, "OUT_DIR", tmp_path)
    monkeypatch.setattr(pj, "JOURNAL_JSON", tmp_path / "paper_journal.json")
    monkeypatch.setattr(pj, "JOURNAL_CSV", tmp_path / "paper_trades.csv")
    monkeypatch.setattr(pj, "JOURNAL_MD", tmp_path / "paper_journal.md")
    monkeypatch.setattr(pj, "WATCH_JSON", tmp_path / "paper_watch_journal.json")
    monkeypatch.setattr(pj, "WATCH_CSV", tmp_path / "paper_watch_trades.csv")
    monkeypatch.setattr(pj, "WATCH_MD", tmp_path / "paper_watch_journal.md")
    monkeypatch.setattr(pj, "default_fetch_klines", lambda s, start_utc=None: pd.DataFrame())
    posts = []

    def fake_post(payload, styles=None):
        posts.append(payload)
        return 200, "bearer", "ok"

    monkeypatch.setattr(scan_runner, "post_webhook", fake_post)
    monkeypatch.setattr(scan_runner, "remembered_auth_style", lambda: "bearer")
    return tmp_path, posts


def write_signals(tmp, rows):
    (tmp / "signals.json").write_text(json.dumps({"asof_shanghai": "2026-11-11 11:02",
                                                  "last_bar_shanghai": "2026-11-11 10:45",
                                                  "signals": rows}, ensure_ascii=False), encoding="utf-8")


def test_run_once_posts_formal_then_watch(runner_env):
    tmp, posts = runner_env
    write_signals(tmp, [sig("F1USDT", status="可入场"), sig("W1USDT"), sig("W2USDT"), sig("W3USDT"),
                        sig("W4USDT")])
    assert scan_runner.run_once() == 0
    assert [p["tier"] for p in posts] == ["formal", "watch"]
    assert len(posts[1]["signals"]) == 4  # no daily cap; all 观察 are sent
    assert posts[1]["watch_daily_cap"] is None
    assert posts[1]["watch_alerts_today"] == 4
    state = json.loads((tmp / "pushed_state.json").read_text(encoding="utf-8"))["pushed"]
    assert sum(1 for v in state.values() if v.get("suppressed") == "daily_cap") == 0
    wj = json.loads((tmp / "paper_watch_journal.json").read_text(encoding="utf-8"))
    assert len(wj["trades"]) == 4
    assert sum(1 for t in wj["trades"] if t.get("alerted")) == 4
    fj = json.loads((tmp / "paper_journal.json").read_text(encoding="utf-8"))
    assert [t["symbol"] for t in fj["trades"]] == ["F1USDT"]
    # second run: nothing new
    posts.clear()
    assert scan_runner.run_once() == 0
    assert posts == []


def test_run_once_breaker_suppresses_watch(runner_env):
    tmp, posts = runner_env
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    j = pj.empty_journal()
    for i in range(3):
        j["trades"].append({"id": f"x{i}", "pattern_id": f"x{i}", "symbol": "ZZZUSDT", "status": "closed",
                            "account_taken": True, "pnl_R": -1.0, "exit_time": f"{today} 0{i + 1}:00"})
    (tmp / "paper_journal.json").write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    write_signals(tmp, [sig("W1USDT")])
    assert scan_runner.run_once() == 0
    assert all(p.get("kind") == "circuit_breaker" for p in posts)  # only the breaker notice
    wj = json.loads((tmp / "paper_watch_journal.json").read_text(encoding="utf-8"))
    assert len(wj["trades"]) == 1 and not wj["trades"][0].get("alerted")


def test_dry_run_never_posts(runner_env):
    tmp, posts = runner_env
    write_signals(tmp, [sig("F1USDT", status="可入场"), sig("W1USDT")])
    d = tmp / "dry"
    paths = {"formal": pj.JournalPaths.under(d), "watch": pj.JournalPaths.under(d, book=pj.BOOK_WATCH)}
    assert scan_runner.run_once(dry_run=True, journal_paths=paths) == 0
    assert posts == []
    assert not (tmp / "pushed_state.json").exists()
    assert not (tmp / "paper_journal.json").exists()


def test_15m_prebreak_watch_is_not_posted(runner_env):
    """15m formal 观望 is pre-break and must not go out on the webhook.
    可入场 (formal) and 观察 (near-miss, already post-break) still do."""
    tmp, posts = runner_env
    write_signals(tmp, [
        sig("F1USDT", status="可入场"),
        sig("WATCHUSDT", status="观望"),
        sig("W1USDT"),  # 观察
    ])
    assert scan_runner.run_once() == 0
    assert [p["tier"] for p in posts] == ["formal", "watch"]
    assert [s["symbol"] for s in posts[0]["signals"]] == ["F1USDT"]
    assert [s["symbol"] for s in posts[1]["signals"]] == ["W1USDT"]
    state = json.loads((tmp / "pushed_state.json").read_text(encoding="utf-8"))["pushed"]
    assert not any("观望" in k or v.get("status") == "观望" for k, v in state.items())
    # later the same LS/H/RS breaks -> 可入场 is a new pattern_id and is pushed
    posts.clear()
    write_signals(tmp, [
        sig("F1USDT", status="可入场"),
        sig("WATCHUSDT", status="可入场", R="2026-11-11 08:30 WATCHUSDT"),
        sig("W1USDT"),
    ])
    assert scan_runner.run_once() == 0
    assert len(posts) == 1 and posts[0]["tier"] == "formal"
    assert [s["symbol"] for s in posts[0]["signals"]] == ["WATCHUSDT"]



def test_15m_collect_lowrr_and_chart_path():
    from types import SimpleNamespace
    from hs_opt_strategy import measured_ok

    ok = SimpleNamespace(entry_px=100.0, stop=102.0, target=97.0, rr=1.5)
    bad = SimpleNamespace(entry_px=100.0, stop=102.0, target=97.0, rr=1.0)
    assert measured_ok(ok) and not measured_ok(bad)
    recs = [
        {"sig_breakout": ok, "q_breakout": True},
        {"sig_breakout": bad, "q_breakout": True},
        {"sig_breakout": bad, "q_breakout": False},
        {"sig_breakout": None, "q_breakout": True},
    ]
    got = scan_15m.collect_lowrr_breakouts(recs)
    assert got == [bad]

    formal = {"symbol": "SOLUSDT", "status": "已破位·等回踩", "exchange": "binance", "tier": "formal"}
    assert scan_15m.chart_path(formal).name == "SOLUSDT_retest.png"
    watch = dict(formal, tier="watch")
    assert scan_15m.chart_path(watch).name == "SOLUSDT_observe_retest.png"
    nest = dict(formal, nested_signal=True)
    assert scan_15m.chart_path(nest).name == "SOLUSDT_nested_retest.png"


def test_15m_payload_and_dedup_retest():
    assert "已破位·等回踩" in scan_runner.ACTIONABLE
    s = sig("SOLUSDT", status="已破位·等回踩")
    s["break_level"] = 82.0
    s["suggest_entry"] = 82.0
    s["official_entry"] = 82.0
    s["valid_until_shanghai"] = "2026-11-13 16:00"
    s["note"] = "15m 已收盘跌破颈线，但按收盘价追的盈亏比不足 1.5；在 82.0000 附近挂限价空单等回踩，有效至 2026-11-13 16:00，过期撤单。"
    p = scan_runner.build_payload([s], "formal", "2026-11-12 22:04", "2026-11-12 22:00")
    assert p["n_retest"] == 1 and p["n_entry"] == 0
    sm = p["signals"][0]
    assert sm["status"] == "已破位·等回踩"
    assert sm["entry"] == 82.0 and sm["break_level"] == 82.0
    assert sm["valid_until_shanghai"] == "2026-11-13 16:00"
    # pattern_id includes status so a later 可入场 on the same pivots is a new key
    assert "已破位·等回踩" in scan_runner.pattern_id(s)
    assert scan_runner.pattern_id(s) != scan_runner.pattern_id(sig("SOLUSDT", status="可入场"))
