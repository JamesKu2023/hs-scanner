"""Nested (complex) H&S, 2026-11-12 三栋 UNI 4H case. Never POSTs."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nested_hs as N  # noqa: E402
import paper_journal as pj  # noqa: E402
import scan_htf  # noqa: E402
import scan_runner  # noqa: E402
import scan_15m as SC  # noqa: E402
from hs_backtest import add_indicators, to_shanghai  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "uni_4h_20261112.csv"


@pytest.fixture(scope="module")
def uni():
    d = pd.read_csv(FIX)
    d["dt"] = pd.to_datetime(d["open_time"], unit="ms", utc=True)
    return add_indicators(d)


def _t(d, i):
    return to_shanghai(d["dt"].iloc[i])


def test_trace_does_not_change_detect(uni):
    tr = {}
    assert scan_htf.detect(uni) == scan_htf.detect(uni, trace=tr)
    assert tr, "trace should collect outer patterns"


def test_uni_inner_matches_annotation(uni):
    _, outers = N.outer_patterns(uni)
    o = [o for o in outers if _t(uni, o["H"]) == "2026-11-05 04:00" and _t(uni, o["R"]) == "2026-11-08 08:00"]
    assert o, "outer UNI H&S (H 11-05 04:00, RS 11-08 08:00) must be identified"
    _, structs = N.nested_scan(uni, o)
    hist = structs[(o[0]["H"], o[0]["R"])]
    assert hist
    g = hist[-1][1]
    assert _t(uni, g["L"]) == "2026-11-06 20:00"      # 小左肩 (三栋: ~11-07)
    assert _t(uni, g["H"]) == "2026-11-08 08:00"      # 小头 = 大右肩
    assert _t(uni, g["R"]) == "2026-11-10 00:00"      # 小右肩 (三栋: 11-10~11)


def _uni_outer(uni):
    _, outers = N.outer_patterns(uni)
    o = [o for o in outers if _t(uni, o["H"]) == "2026-11-05 04:00" and _t(uni, o["R"]) == "2026-11-08 08:00"]
    assert o
    return o[0]


def test_uni_v6_pivots_and_no_break_below_sloped_neck(uni):
    """2026-11-14 三栋 v6: LS = deepest independent pullback on the rising leg (11-01 20:00),
    RS = highest bar of the rebound segment (11-08 08:00). The v6 neckline joins the two
    pullback lows (11-03 08:00 → 11-05 20:00, sloping down); UNI never CLOSED below it up to
    11-12 08:00, so v6 emits no 可入场 / 已破位 for this head (the old horizontal-plateau
    break at 11-11 12:00 no longer counts)."""
    o = _uni_outer(uni)
    assert _t(uni, o["L"]) == "2026-11-01 20:00"
    assert _t(uni, o["lt"]) == "2026-11-03 08:00" and _t(uni, o["rt"]) == "2026-11-05 20:00"
    evs = [e for e in scan_htf.detect(uni) if e["H"] == o["H"]]
    assert not [e for e in evs if e["kind"] in ("entry", "retest")]
    for cut in (494, 499):
        rows = scan_htf.live_rows("UNIUSDT", "4h", uni.iloc[:cut].reset_index(drop=True), None)
        assert not [r for r in rows if r.get("H_time") == "2026-11-05 04:00" and r["status"] in ("可入场", "已破位·等回踩")]


def test_v6_htf_row_has_score_and_confirm(uni):
    o = _uni_outer(uni)
    n = len(uni); t = n - 1
    ev = dict(o, kind="forming", t=t, neck=6.08, brk=6.04, stop=7.4, t1=4.9, t2=4.3, entry=6.04, rr=1.6)
    r = scan_htf.make_row("UNIUSDT", "4h", uni, ev, None)
    assert r["v6"] is True and r["tier_cn"] == "4H·v6 三栋认肩·未验证" and r["suggested_risk_pct"] == 0.5
    assert isinstance(r["prop_score"], int) and 0 <= r["prop_score"] <= 100
    assert r["confirm_x3_text"].startswith("确认 ") and r["confirm_total"] == 3
    assert r["prop_text"].startswith("比例分 ")


def test_nested_rows_not_serialized_private(uni, tmp_path):
    o = _uni_outer(uni)
    _, structs = N.nested_scan(uni, [o])
    g = structs[(o["H"], o["R"])][-1][1]
    n = len(uni)
    ev = dict(o, kind="forming", t=n - 1, neck=6.08, brk=6.04, stop=7.4, t1=4.9, t2=4.3, entry=6.04, rr=1.6)
    r = dict(scan_htf.make_row("UNIUSDT", "4h", uni, ev, None), exchange="binance", exchange_cn="币安",
             nested_role="outer", _nested_other=g)
    p = tmp_path / "c.png"
    SC.plot_setup(uni, r, p, tf_label="4H")
    assert p.exists() and p.stat().st_size > 10000
    assert "_nested_other" in r


def _nested_sig(status="可入场", tf="4h"):
    return {"symbol": "UNIUSDT", "status": status, "timeframe": tf, "nested_signal": True, "nested": True,
            "nested_role": "inner", "H_time": "2026-11-08 08:00", "L_time": "2026-11-06 20:00",
            "R_time": "2026-11-10 00:00", "official_entry": 6.5, "official_entry_time": "2026-11-11 16:00",
            "stop": 7.2, "target_1h": 5.8, "target_15h": 5.5, "tier": "htf", "exchange": "binance",
            "nested_text": "大头肩的右肩内又成一个小头肩", "parent": {"neck_now": 6.2}}


def test_pattern_id_namespaces_match():
    s = _nested_sig()
    assert pj.pattern_id(s) == scan_runner.pattern_id(s) == "nested|4h|UNIUSDT|可入场|2026-11-08 08:00"
    s2 = dict(s, nested_signal=False)
    assert pj.pattern_id(s2) == "4h|UNIUSDT|可入场|2026-11-08 08:00"
    s15 = _nested_sig(tf="15m")
    assert scan_runner.pattern_id(s15).startswith("nested|15m|")


def test_books_separate(tmp_path):
    for tf in ("15m", "1h", "4h"):
        p = pj.JournalPaths.under(tmp_path, book=pj.NESTED_BOOKS[tf])
        assert p.json_path.name == f"paper_nested_{tf}_journal.json"
    data = {"signals": [_nested_sig()]}
    fetch = lambda sym, start: pd.DataFrame()  # noqa: E731
    j_htf = pj.update_from_scan(data, paths=pj.JournalPaths.under(tmp_path, book=pj.BOOK_HTF_4H),
                                book=pj.BOOK_HTF_4H, fetch_klines_fn=fetch)
    j_formal = pj.update_from_scan(data, paths=pj.JournalPaths.under(tmp_path), fetch_klines_fn=fetch)
    j_n = pj.update_from_scan(data, paths=pj.JournalPaths.under(tmp_path, book=pj.BOOK_NESTED_4H),
                              book=pj.BOOK_NESTED_4H, fetch_klines_fn=fetch)
    assert not j_htf["trades"] and not j_formal["trades"]
    assert len(j_n["trades"]) == 1 and j_n["breaker"]["tripped"] is False


def test_strip_nested_and_payload():
    reg = {"symbol": "X", "status": "可入场"}
    data, nested = scan_runner.strip_nested({"signals": [reg, _nested_sig()], "okx_ok": True})
    assert data["signals"] == [reg] and len(nested) == 1
    p = scan_runner.build_nested_payload(nested, "4h", "2026-11-12 16:02", "2026-11-12 12:00")
    assert p["nested"] is True and p["counts_toward_breaker"] is False and p["suggested_risk_pct"] == 0.5
    s = p["signals"][0]
    assert s["nested_role"] == "inner" and s["nested_text"] and s["parent"]["neck_now"] == 6.2
    p15 = scan_runner.build_nested_payload([_nested_sig(tf="15m")], "15m", "t", "b")
    assert p15["tier"] == "watch"
    assert "HS_WEBHOOK_KEY" not in str(p)


def test_run_nested_skips_prebreak_watch(tmp_path, monkeypatch):
    """Nested inner 观望 is kept off the webhook; only a close below the small neckline is posted."""
    posts = []
    monkeypatch.setattr(scan_runner, "post_webhook", lambda *a, **k: posts.append(a[0] if a else k) or (200, "bearer", ""))
    monkeypatch.setattr(scan_runner, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(scan_runner, "LOG_PATH", tmp_path / "log.txt")
    monkeypatch.setattr(scan_runner, "remembered_auth_style", lambda: "bearer")
    monkeypatch.setattr(pj, "default_fetch_klines", lambda sym, start=None: pd.DataFrame())
    monkeypatch.setattr(pj, "OUT_DIR", tmp_path)
    paths = {pj.BOOK_NESTED_4H: pj.JournalPaths.under(tmp_path, book=pj.BOOK_NESTED_4H)}
    watch = _nested_sig(status="观望")
    rc = scan_runner.run_nested([watch], "4h", {"signals": [watch], "asof_shanghai": "2026-11-12 16:02"},
                                journal_paths=paths)
    assert rc == 0 and posts == []
    assert not (tmp_path / "state.json").exists()
    entry = _nested_sig(status="可入场")
    rc = scan_runner.run_nested([entry], "4h", {"signals": [entry], "asof_shanghai": "2026-11-12 16:02"},
                                journal_paths=paths)
    assert rc == 0 and len(posts) == 1
    assert posts[0]["nested"] is True and posts[0]["n_entry"] == 1 and posts[0]["n_watch"] == 0
    assert posts[0]["signals"][0]["status"] == "可入场"


def test_run_nested_dry_run_no_post(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(scan_runner, "post_webhook", lambda *a, **k: called.append(1) or (200, "bearer", ""))
    monkeypatch.setattr(scan_runner, "load_state", lambda: {"pushed": {}})
    monkeypatch.setattr(scan_runner, "log", lambda msg: None)  # never write the live log
    monkeypatch.setattr(pj, "default_fetch_klines", lambda sym, start=None: pd.DataFrame())
    paths = {pj.BOOK_NESTED_4H: pj.JournalPaths.under(tmp_path, book=pj.BOOK_NESTED_4H)}
    rc = scan_runner.run_nested([_nested_sig()], "4h", {"signals": []}, dry_run=True, journal_paths=paths)
    assert rc == 0 and not called


def test_summarize_regular_row_has_nested_false():
    s = scan_runner.summarize({"symbol": "BTCUSDT", "status": "可入场"})
    assert s["nested"] is False and s["nested_role"] is None


def test_wrap_cn_keeps_prices():
    t = SC._wrap_cn("嵌套头肩：右肩内又成一个小头肩（小LS 11-06 20:00 7.1300 / 小H 11-08 08:00 7.2420），小颈线 6.3558。", 20)
    assert "7.1300" in t.replace("\n", "") and all("7.13\n" not in t for _ in [0])


def test_nested_retest_is_actionable_and_payload():
    s = _nested_sig(status="已破位·等回踩")
    assert scan_runner.pattern_id(s) == "nested|4h|UNIUSDT|已破位·等回踩|2026-11-08 08:00"
    p = scan_runner.build_nested_payload([s], "4h", "2026-11-12 22:04", "2026-11-12 20:00")
    assert p["n_retest"] == 1 and p["signals"][0]["status"] == "已破位·等回踩"
