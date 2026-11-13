#!/usr/bin/env python3
"""Build the USDT-M core symbol pool from public Binance futures APIs.

Dependencies: requests
Idempotent: overwrites /workspace/trading/core_pool.json and core_pool.md
Does not place orders.

Usage:
    python3 build_core_pool.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "core_pool.json"
OUT_MD = ROOT / "core_pool.md"

FAPI = "https://fapi.binance.com"
UA = {"User-Agent": "hs-core-pool/1.0"}
SH = timezone(timedelta(hours=8))

# Fiat / stable bases that are not directional shorts we want in this pool.
STABLE_BASES = {
    "USDT", "USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USDP", "USDE", "USTC", "UST",
    "USD1", "USDD", "PYUSD", "GUSD", "EUR", "AEUR", "EURI", "BFUSD", "XUSD",
    "PAX", "TUSD", "USD",
}

TOP_N = 30
MAX_ABS_FUNDING = 0.001  # 0.1% per 8h
MIN_HISTORY_DAYS = 14
MAJORS = {"BTCUSDT", "ETHUSDT"}


def get_json(path: str, params=None):
    r = requests.get(f"{FAPI}{path}", params=params, headers=UA, timeout=(8, 30))
    r.raise_for_status()
    return r.json()


def slippage_bps(symbol: str) -> int:
    return 2 if symbol in MAJORS else 5


def main() -> int:
    now = datetime.now(timezone.utc)
    info = get_json("/fapi/v1/exchangeInfo")
    tickers = {t["symbol"]: t for t in get_json("/fapi/v1/ticker/24hr")}
    prem = {p["symbol"]: p for p in get_json("/fapi/v1/premiumIndex")}

    eligible = []
    excluded = {"not_usdt_perp": 0, "stable": 0, "not_trading": 0,
                "short_history": 0, "extreme_funding": 0, "no_ticker": 0}

    for s in info.get("symbols", []):
        symbol = s.get("symbol", "")
        if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT":
            excluded["not_usdt_perp"] += 1
            continue
        if s.get("status") != "TRADING":
            excluded["not_trading"] += 1
            continue
        base = s.get("baseAsset", "")
        if base in STABLE_BASES:
            excluded["stable"] += 1
            continue
        onboard = s.get("onboardDate") or s.get("deliveryDate") or 0
        try:
            onboard_ms = int(onboard)
        except (TypeError, ValueError):
            onboard_ms = 0
        if onboard_ms > 0:
            age_days = (now.timestamp() * 1000 - onboard_ms) / 86400000.0
            if age_days < MIN_HISTORY_DAYS:
                excluded["short_history"] += 1
                continue
        else:
            excluded["short_history"] += 1
            continue
        tk = tickers.get(symbol)
        if not tk:
            excluded["no_ticker"] += 1
            continue
        fr_raw = prem.get(symbol, {}).get("lastFundingRate")
        try:
            funding = float(fr_raw) if fr_raw is not None else None
        except (TypeError, ValueError):
            funding = None
        if funding is None or abs(funding) > MAX_ABS_FUNDING:
            excluded["extreme_funding"] += 1
            continue
        try:
            qvol = float(tk.get("quoteVolume") or 0.0)
        except (TypeError, ValueError):
            qvol = 0.0
        eligible.append({
            "symbol": symbol,
            "base_asset": base,
            "quote_volume_24h": qvol,
            "funding_rate": funding,
            "funding_rate_pct_8h": funding * 100.0,
            "slippage_bps": slippage_bps(symbol),
            "onboard_date_utc": datetime.fromtimestamp(onboard_ms / 1000, tz=timezone.utc)
            .strftime("%Y-%m-%d"),
        })

    eligible.sort(key=lambda x: x["quote_volume_24h"], reverse=True)
    picked = eligible[:TOP_N]
    for i, row in enumerate(picked, 1):
        row["rank"] = i

    payload = {
        "asof_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "asof_shanghai": now.astimezone(SH).strftime("%Y-%m-%d %H:%M"),
        "rules": {
            "market": "Binance USDT-M perpetual",
            "exclude_stable_bases": sorted(STABLE_BASES),
            "top_n_by_24h_quote_volume": TOP_N,
            "max_abs_funding_per_8h": MAX_ABS_FUNDING,
            "min_history_days": MIN_HISTORY_DAYS,
            "slippage_bps": {"BTCUSDT": 2, "ETHUSDT": 2, "others": 5},
        },
        "excluded_counts": excluded,
        "n_eligible_before_top": len(eligible),
        "n": len(picked),
        "symbols": picked,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 头肩扫描核心池",
        "",
        f"生成时间：{payload['asof_shanghai']}（北京时间，Asia/Shanghai）",
        f"数据来源：币安公开接口 `exchangeInfo`、`ticker/24hr`、`premiumIndex`（USDT-M）。",
        "",
        "## 筛选标准",
        "",
        f"- 只保留币安 **USDT 永续**、状态为 TRADING 的合约。",
        f"- 排除稳定币/法币本位（如 USDC、FDUSD、DAI、USDE、EUR 等，共 {len(STABLE_BASES)} 个代码）。",
        f"- 上市至少 **{MIN_HISTORY_DAYS} 天**，以保证 15m 形态跨度（最多 80 根）和 1H EMA60 能算出来。",
        f"- 当前资金费率绝对值不超过 **0.1% / 8h**（即 `|lastFundingRate| ≤ 0.001`），避开极端拥挤费率。",
        f"- 在通过以上过滤的合约里，按 24h 报价货币成交额（USDT）取前 **{TOP_N}**。",
        "- 滑点档与回测一致：BTC/ETH **2 bps**，其他 **5 bps**。",
        "",
        f"过滤后还剩 {len(eligible)} 个合约，从中取前 {len(picked)} 个写入核心池。",
        "",
        "## 当前名单",
        "",
        "| 排名 | 合约 | 24h 成交额 (USDT) | 资金费率 (/8h) | 滑点 | 上市日期 (UTC) |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in picked:
        lines.append(
            f"| {row['rank']} | {row['symbol']} | {row['quote_volume_24h']:,.0f} | "
            f"{row['funding_rate_pct_8h']:+.4f}% | {row['slippage_bps']} bps | {row['onboard_date_utc']} |"
        )
    lines.extend([
        "",
        "机器可读文件：`/workspace/trading/core_pool.json`。",
        "本文件每次运行 `build_core_pool.py` 都会覆盖重写，适合定时任务。不自动下单。",
        "",
    ])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_JSON} n={len(picked)} eligible={len(eligible)}")
    print(f"wrote {OUT_MD}")
    for row in picked[:5]:
        print(f"  #{row['rank']} {row['symbol']:16s} vol={row['quote_volume_24h']:.0f} "
              f"fr={row['funding_rate_pct_8h']:+.4f}% slip={row['slippage_bps']}bps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
