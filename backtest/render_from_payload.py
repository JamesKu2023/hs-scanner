#!/workspace/venv/bin/python
"""Re-draw H&S alert charts from a webhook payload (charts made on GitHub are not on the box).

Usage:
  render_from_payload.py --payload /path/payload.json [--out-dir DIR]
  cat payload.json | render_from_payload.py --payload -
Input: the full webhook payload ({"signals": [...]}), a list of signals, or one signal.
For every signal carrying chart_spec it re-fetches public klines from the same source the scan
used (binance_fapi / binance_spot / okx_swap / okx_swap_proxy), ending at the same last bar,
and calls scan_15m.plot_setup with the same row. Prints one line per signal:
  CHART <signal_id> <png path>     or     NOCHART <signal_id> <reason>
and finally a JSON line {"charts": {signal_id: path}}. Public market data only, no keys.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

import binance_data as BD  # noqa: E402
import okx_market  # noqa: E402
import scan_15m as SC  # noqa: E402
from hs_backtest import add_indicators  # noqa: E402

BIN_INTERVAL = {"15m": "15m", "1h": "1h", "4h": "4h"}
OKX_BAR = {"15m": "15m", "1h": "1H", "4h": "4H"}
BAR_MS = {"15m": 15 * 60_000, "1h": 3_600_000, "4h": 4 * 3_600_000}
DEFAULT_OUT = HERE / "scan_out" / "gha"
STATUS_EN = {"已破位·等回踩": "retest", "可入场": "entry", "观察": "observe", "观望": "watch"}


def _signals(obj) -> list[dict]:
    if isinstance(obj, dict) and isinstance(obj.get("signals"), list):
        return [s for s in obj["signals"] if isinstance(s, dict)]
    if isinstance(obj, list):
        return [s for s in obj if isinstance(s, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def fetch_frame(sig: dict, spec: dict) -> pd.DataFrame:
    tf = spec["tf"]
    n = int(spec["n_bars"])
    last = int(spec["last_open_ms"])
    step = BAR_MS[tf]
    symbol = str(sig.get("symbol") or "")
    src = sig.get("data_source") or ("okx_swap" if sig.get("exchange") == "okx" else "binance_fapi")
    if src == "okx_swap" or src == BD.SRC_OKX_PROXY:
        inst = symbol if symbol.endswith("-SWAP") else BD.okx_inst(symbol)
        elapsed = max(0, int((time.time() * 1000 - last) // step))
        df = okx_market.fetch_okx_klines(inst, OKX_BAR[tf], min(n + elapsed + 2, 5000))
    else:
        params = {"interval": BIN_INTERVAL[tf], "limit": n, "endTime": last + step - 1}
        if src == BD.SRC_SPOT:
            df = BD._spot(symbol, params)
        else:
            try:
                df = BD._fapi(symbol, params)
            except Exception:  # noqa: BLE001  fapi blocked here too -> spot
                df = BD._spot(symbol, params)
    df = df[df["open_time"].astype("int64") <= last].tail(n).copy()
    df["dt"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    return df.reset_index(drop=True)


def render_one(sig: dict, out_dir: Path) -> Path:
    spec = sig.get("chart_spec")
    if not isinstance(spec, dict):
        raise ValueError("no chart_spec in payload")
    df = fetch_frame(sig, spec)
    if df.empty:
        raise ValueError("no klines")
    dfi = add_indicators(df)
    pos = {int(t): i for i, t in enumerate(dfi["open_time"].astype("int64"))}
    row = dict(spec.get("row") or {})
    for k, ms in spec["idx_ms"].items():
        if int(ms) not in pos:
            raise ValueError(f"bar {k} ({ms}) not in re-fetched window")
        row[k] = pos[int(ms)]
    row["y1"], row["y2"] = float(spec["y1"]), float(spec["y2"])
    if spec.get("other_idx_ms"):
        other = {k: pos[int(ms)] for k, ms in spec["other_idx_ms"].items() if int(ms) in pos}
        if len(other) == len(spec["other_idx_ms"]):
            other["y1"], other["y2"] = (float(x) for x in spec["other_y"])
            row["_nested_other"] = other
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = sig.get("signal_id") or "|".join(str(sig.get(k) or "") for k in ("exchange", "symbol", "timeframe",
                                                                            "H_time", "status"))
    for cn, en in STATUS_EN.items():
        sid = sid.replace(cn, en)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sid).strip("_") or "chart"
    out = out_dir / f"{safe}.png"
    SC.plot_setup(dfi, row, out, title_extra=spec.get("title_extra") or "", tf_label=spec.get("tf_label") or "15m")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payload", required=True, help="payload JSON file, or - for stdin")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    text = sys.stdin.read() if a.payload == "-" else Path(a.payload).read_text(encoding="utf-8")
    sigs = _signals(json.loads(text))
    charts = {}
    for s in sigs:
        sid = s.get("signal_id") or s.get("symbol")
        try:
            p = render_one(s, Path(a.out_dir))
            charts[sid] = str(p)
            print(f"CHART {sid} {p}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"NOCHART {sid} {type(e).__name__}: {e}", flush=True)
    print(json.dumps({"charts": charts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
